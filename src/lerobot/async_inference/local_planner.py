# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Local, single-process async planners for sim evaluation of action-chunk policies.

Mirrors the observation -> chunk -> aggregated-queue semantics of the gRPC
`RobotClient` in a single process, so sim rollouts can exercise async-style
control (high-frequency re-prediction, chunk aggregation, staleness tracking)
before bringing up the robot path.

Drop-in replacements for `policy.select_action` inside `rollout()`.

Two planners are provided:

- ``LocalAsyncPlanner``: synchronous only. Inference runs on the caller thread.
  Useful for deterministic ablations of the queue/threshold machinery.
- ``BackgroundAsyncPlanner``: runs inference on a background worker thread.
  ``select_action`` still returns synchronously, but in steady-state the main
  thread only drains results and pops queued actions. This exposes real
  inference-latency-driven staleness, not just threshold-driven virtual
  staleness.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, fields
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy

AggregateFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def latest_only(_old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    return new


def weighted_average(old: torch.Tensor, new: torch.Tensor, w_new: float = 0.7) -> torch.Tensor:
    return (1.0 - w_new) * old + w_new * new


@dataclass
class PlannerMetrics:
    """Flat scalars. `as_dict()` output is safe for `wandb_logger.log_dict`.

    The ``num_in_flight_skips`` / ``sum_receive_latency_steps`` /
    ``sum_wallclock_inference_s`` fields are only populated by
    ``BackgroundAsyncPlanner``; they stay at 0 for the sync planner.
    """

    num_steps: int = 0
    num_chunks_generated: int = 0
    num_aggregated_actions: int = 0
    sum_staleness_steps: int = 0
    sum_queue_fill: float = 0.0
    inference_time_s: float = 0.0
    # Background-only counters.
    num_in_flight_skips: int = 0
    sum_receive_latency_steps: int = 0
    sum_wallclock_inference_s: float = 0.0
    num_stale_actions_dropped: int = 0

    def as_dict(self) -> dict[str, float]:
        n = max(self.num_steps, 1)
        n_chunks = max(self.num_chunks_generated, 1)
        return {
            "async_chunks_per_step": self.num_chunks_generated / n,
            "async_aggregated_rate": self.num_aggregated_actions / n,
            "async_avg_staleness_steps": self.sum_staleness_steps / n,
            "async_avg_queue_fill": self.sum_queue_fill / n,
            "async_inference_s_total": self.inference_time_s,
            "async_num_in_flight_skips": float(self.num_in_flight_skips),
            "async_avg_receive_latency_steps": self.sum_receive_latency_steps / n_chunks,
            "async_wallclock_inference_s_total": self.sum_wallclock_inference_s,
            "async_num_stale_actions_dropped": float(self.num_stale_actions_dropped),
        }

    def raw_dict(self) -> dict[str, float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_raw_dict(cls, raw: dict[str, float]) -> PlannerMetrics:
        defaults = cls()
        kwargs = {}
        for metric_field in fields(cls):
            default = getattr(defaults, metric_field.name)
            value = raw.get(metric_field.name, default)
            kwargs[metric_field.name] = int(value) if isinstance(default, int) else float(value)
        return cls(**kwargs)


class LocalAsyncPlanner:
    """Single-process replacement for `policy.select_action` with an async-style queue.

    Args:
        policy: any `PreTrainedPolicy` whose `predict_action_chunk` is
            self-contained (accepts a cold observation batch).
        actions_per_chunk: how many actions from each predicted chunk to keep.
            Lower -> more refreshes, closer to MPC. Clamped to <= chunk length
            returned by the policy.
        chunk_size_threshold: trigger a fresh prediction when
            `len(queue)/last_chunk_size <= threshold`. 1.0 means refresh every
            step; 0.5 matches `RobotClientConfig` default.
        aggregate_fn: `(old, new) -> merged` used when an incoming chunk
            overlaps existing queued actions. Defaults to `latest_only` (drop
            old, keep fresh).
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        *,
        actions_per_chunk: int,
        chunk_size_threshold: float = 0.9,
        aggregate_fn: AggregateFn | None = None,
    ):
        if not 0.0 <= chunk_size_threshold <= 1.0:
            raise ValueError(f"chunk_size_threshold must be in [0,1], got {chunk_size_threshold}")
        if actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {actions_per_chunk}")

        self.policy = policy
        self.actions_per_chunk = actions_per_chunk
        self.threshold = chunk_size_threshold
        self.aggregate_fn = aggregate_fn or latest_only

        # Action queue keyed by global timestep so aggregation is trivial.
        # OrderedDict keeps insertion order; we re-sort on merge for determinism.
        self._queue: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._origin: dict[int, int] = {}
        self._last_chunk_size: int = actions_per_chunk
        self._current_step: int = 0
        self._did_predict_this_step: bool = False
        self._lock = threading.Lock()

        self.metrics = PlannerMetrics()

    # ---------- lifecycle ----------

    def reset(self) -> None:
        self.policy.reset()
        with self._lock:
            self._queue.clear()
            self._origin.clear()
            self._current_step = 0
            self._last_chunk_size = self.actions_per_chunk
            self._did_predict_this_step = False
        self.metrics = PlannerMetrics()

    # ---------- drop-in policy API ----------

    def select_action(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pop the action for `current_step`, predicting first if the queue is stale."""
        self._did_predict_this_step = False
        if self._should_predict():
            self._predict_and_merge(observation)
            self._did_predict_this_step = True

        with self._lock:
            if self._current_step not in self._queue:
                raise RuntimeError(
                    f"LocalAsyncPlanner: no action for step {self._current_step} after prediction. "
                    f"Queue steps: {list(self._queue.keys())[:8]}..."
                )
            return self._pop_current_step_locked()

    # ---------- eval-timing hooks (match PreTrainedPolicy API) ----------

    def get_eval_timing_context(self) -> dict[str, int | bool]:
        # Record whether this step will trigger a fresh prediction. Captured
        # BEFORE select_action, consumed by `is_chunk_generation_step` AFTER.
        # This matches both the cold-queue case and the threshold-driven prefetch.
        return {
            "will_predict": self._should_predict(),
            "action_queue_len": len(self._queue),
        }

    def is_chunk_generation_step(self, timing_context: dict | None) -> bool | None:
        if timing_context is None:
            return None
        return bool(timing_context.get("will_predict", False))

    # ---------- internals ----------

    def _should_predict_locked(self) -> bool:
        if not self._queue or self._current_step not in self._queue:
            return True
        fill = len(self._queue) / max(self._last_chunk_size, 1)
        return fill <= self.threshold

    def _should_predict(self) -> bool:
        with self._lock:
            return self._should_predict_locked()

    def _pop_current_step_locked(self) -> torch.Tensor:
        action = self._queue.pop(self._current_step)
        origin = self._origin.pop(self._current_step, self._current_step)
        self.metrics.num_steps += 1
        self.metrics.sum_staleness_steps += self._current_step - origin
        self.metrics.sum_queue_fill += len(self._queue) / max(self._last_chunk_size, 1)
        self._current_step += 1
        return action

    def _merge_chunk_locked(self, chunk: torch.Tensor, origin: int) -> int:
        """Merge a sliced chunk into the queue. Caller must hold ``self._lock``.

        ``chunk`` must have shape [B, K, D] (already sliced to
        ``actions_per_chunk`` and detached). Steps strictly before
        ``self._current_step`` are dropped (they've already been consumed, so
        re-queueing them would either overwrite future steps or leak memory).

        Returns the number of stale actions dropped.
        """
        chunk_len = chunk.shape[1]
        dropped = 0
        for k in range(chunk_len):
            step = origin + k
            if step < self._current_step:
                dropped += 1
                continue
            new_a = chunk[:, k]
            if step in self._queue:
                self._queue[step] = self.aggregate_fn(self._queue[step], new_a)
                self.metrics.num_aggregated_actions += 1
            else:
                self._queue[step] = new_a
            self._origin[step] = origin
        # Keep the queue ordered so `_should_predict_locked` and debug prints
        # see steps in ascending order.
        self._queue = OrderedDict(sorted(self._queue.items()))
        self._last_chunk_size = chunk_len
        self.metrics.num_chunks_generated += 1
        self.metrics.num_stale_actions_dropped += dropped
        return dropped

    def _validate_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 3:
            raise ValueError(f"predict_action_chunk must return [B, K, D], got shape {tuple(chunk.shape)}")
        return chunk[:, : self.actions_per_chunk].detach()

    def _predict_and_merge(self, observation: dict[str, torch.Tensor]) -> None:
        t0 = time.perf_counter()
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(observation)  # [B, K_full, D]
        dt = time.perf_counter() - t0
        chunk = self._validate_chunk(chunk)
        with self._lock:
            self._merge_chunk_locked(chunk, origin=self._current_step)
            self.metrics.inference_time_s += dt
            self.metrics.sum_wallclock_inference_s += dt


# ---------- background-thread variant ----------


_STOP_SENTINEL = object()


@dataclass
class _InboxItem:
    observation: dict[str, Any]
    origin_step: int
    epoch: int


@dataclass
class _OutboxChunk:
    chunk: torch.Tensor
    origin_step: int
    inference_s: float
    epoch: int


@dataclass
class _OutboxError:
    exc: BaseException
    epoch: int
    # `traceback` kept for logging if callers want it.
    tb: str = field(default="")


class BackgroundAsyncPlanner(LocalAsyncPlanner):
    """Real-async planner: inference runs on a background worker thread.

    Contract preserved vs ``LocalAsyncPlanner``:
      * ``select_action(obs)`` returns a tensor for the current step, synchronously.
      * ``reset()`` clears all state and is safe to call between episodes.

    Scheduling rules:
      * Single-flight: at most one prediction in the worker at any time.
      * If the threshold check fires but the worker is busy, the request is
        dropped and ``metrics.num_in_flight_skips`` is incremented.
      * On the first step (cold queue), we launch a prediction and block on the
        worker until the result arrives. This keeps inference off the main
        thread even for the bootstrap — there is no second code path that calls
        ``policy.predict_action_chunk`` from the main thread.
      * When a chunk arrives late, any action whose timestep has already been
        consumed is discarded (see ``_merge_chunk_locked``). This is the async
        equivalent of the `RobotClient` aggregate path.

    Epoch tag:
      * Each ``reset()`` bumps ``self._epoch``. Items queued with a stale epoch
        are dropped. Combined with worker join-on-reset, this is belt-and-
        suspenders for the "in-flight result from last episode" race.

    Worker errors:
      * Any exception inside ``policy.predict_action_chunk`` is forwarded to
        the main thread via the outbox; the next ``select_action`` re-raises it.
        This avoids silent deadlocks on bootstrap.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        *,
        actions_per_chunk: int,
        chunk_size_threshold: float = 0.9,
        aggregate_fn: AggregateFn | None = None,
        bootstrap_timeout_s: float = 120.0,
    ):
        super().__init__(
            policy=policy,
            actions_per_chunk=actions_per_chunk,
            chunk_size_threshold=chunk_size_threshold,
            aggregate_fn=aggregate_fn,
        )
        if bootstrap_timeout_s <= 0:
            raise ValueError(f"bootstrap_timeout_s must be positive, got {bootstrap_timeout_s}")
        self._bootstrap_timeout_s = bootstrap_timeout_s

        self._inbox: queue.Queue = queue.Queue()
        self._outbox: queue.Queue = queue.Queue()
        self._in_flight: bool = False
        self._epoch: int = 0
        self._worker: threading.Thread | None = None
        self._worker_exc: BaseException | None = None
        self._closed: bool = False

        self._start_worker()

    # ---------- lifecycle ----------

    def _start_worker(self) -> None:
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"BackgroundAsyncPlanner-worker-{id(self):x}",
            daemon=True,
        )
        self._worker.start()

    def _stop_worker(self) -> None:
        """Joins the worker so policy state can be touched safely by the caller."""
        if self._worker is None:
            return
        # Flush any queued inbox items so the worker reaches the sentinel quickly.
        try:
            while True:
                self._inbox.get_nowait()
        except queue.Empty:
            pass
        self._inbox.put(_STOP_SENTINEL)
        self._worker.join(timeout=30.0)
        if self._worker.is_alive():
            # Don't hang forever on a truly stuck worker; degrade to daemon GC.
            # This should be exceedingly rare (stuck CUDA kernel, etc.).
            pass
        self._worker = None

    def reset(self) -> None:
        # Stop worker first so the subsequent `policy.reset()` (inside
        # super().reset()) does not race with a mid-prediction worker.
        self._stop_worker()
        # Drop anything the worker produced between join and here (usually empty).
        self._drain_outbox(discard_all=True)
        super().reset()  # calls policy.reset(), clears queue/origin/current_step, resets metrics
        with self._lock:
            self._in_flight = False
            self._worker_exc = None
            self._epoch += 1
        self._start_worker()

    def close(self) -> None:
        """Idempotent shutdown. Safe to call from tests or owner scripts."""
        if self._closed:
            return
        self._closed = True
        self._stop_worker()
        self._drain_outbox(discard_all=True)

    def __del__(self):
        with suppress(Exception):
            self.close()

    # ---------- drop-in policy API ----------

    def select_action(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        self._raise_if_worker_died()
        self._drain_outbox()

        # Threshold check: try to launch; if worker is busy, count the skip.
        self._maybe_launch(observation)

        # If we don't have an action for this step, block on the worker. This
        # covers bootstrap (cold queue on step 0) AND the case where inference
        # is slower than the queue drains.
        if not self._has_action_for_current_step():
            if not self._is_in_flight():
                # Either bootstrap (nothing ever launched) or the threshold
                # held us off but the queue actually ran dry. Force-launch.
                self._force_launch(observation)
            self._wait_for_current_step()

        with self._lock:
            if self._current_step not in self._queue:
                raise RuntimeError(
                    f"BackgroundAsyncPlanner: no action for step {self._current_step} "
                    f"after wait. Queue steps: {list(self._queue.keys())[:8]}..."
                )
            return self._pop_current_step_locked()

    # ---------- eval-timing hooks (override) ----------

    def get_eval_timing_context(self) -> dict[str, int | bool]:
        # Main-thread wall-clock measurement around select_action does NOT
        # capture inference time in the background path (inference ran on the
        # worker). Return None-ish signals so lerobot_eval.rollout() skips
        # per-step chunk timing accumulation; the true wall-clock is in
        # ``metrics.inference_time_s`` / ``sum_wallclock_inference_s``.
        return {
            "will_predict": False,
            "action_queue_len": len(self._queue),
        }

    def is_chunk_generation_step(self, _timing_context: dict | None) -> bool | None:
        # Intentionally return None: the main-thread step time doesn't reflect
        # inference cost in async mode. Chunk-generation accounting comes from
        # the planner's own metrics instead.
        return None

    # ---------- worker ----------

    def _worker_loop(self) -> None:
        # torch.inference_mode is a thread-local guard. Keep it for the life of
        # the worker rather than re-entering on every call.
        with torch.inference_mode():
            while True:
                item = self._inbox.get()
                if item is _STOP_SENTINEL:
                    return
                assert isinstance(item, _InboxItem)
                t0 = time.perf_counter()
                try:
                    chunk = self.policy.predict_action_chunk(item.observation)
                except BaseException as exc:  # noqa: BLE001
                    import traceback

                    self._outbox.put(
                        _OutboxError(exc=exc, epoch=item.epoch, tb=traceback.format_exc())
                    )
                    continue
                dt = time.perf_counter() - t0
                self._outbox.put(
                    _OutboxChunk(
                        chunk=chunk.detach(),
                        origin_step=item.origin_step,
                        inference_s=dt,
                        epoch=item.epoch,
                    )
                )

    # ---------- launch helpers ----------

    def _maybe_launch(self, observation: dict[str, torch.Tensor]) -> None:
        """Launch a prediction if threshold says so and worker is idle. Otherwise count skips."""
        with self._lock:
            want = self._should_predict_locked()
            if not want:
                return
            if self._in_flight:
                # Threshold would have fired, but a prior prediction is still
                # running. Record the skip and return.
                self.metrics.num_in_flight_skips += 1
                return
            self._in_flight = True
            origin_step = self._current_step
            epoch = self._epoch
        obs_clone = _clone_observation(observation)
        self._inbox.put(_InboxItem(observation=obs_clone, origin_step=origin_step, epoch=epoch))

    def _force_launch(self, observation: dict[str, torch.Tensor]) -> None:
        """Unconditional launch (used for bootstrap / rescuing an empty queue).

        No-op if a prediction is already in flight — the in-flight one will
        eventually deliver the chunk we're waiting on.
        """
        with self._lock:
            if self._in_flight:
                return
            self._in_flight = True
            origin_step = self._current_step
            epoch = self._epoch
        obs_clone = _clone_observation(observation)
        self._inbox.put(_InboxItem(observation=obs_clone, origin_step=origin_step, epoch=epoch))

    # ---------- outbox drain / wait ----------

    def _drain_outbox(self, discard_all: bool = False) -> bool:
        """Pop all currently-available items. Returns True if a valid chunk was merged."""
        merged_any = False
        while True:
            try:
                item = self._outbox.get_nowait()
            except queue.Empty:
                return merged_any
            if discard_all:
                continue
            if self._process_outbox_item(item):
                merged_any = True

    def _wait_for_current_step(self) -> None:
        """Block on the outbox until the queue contains ``current_step``.

        Raises ``TimeoutError`` if no chunk covering the current step arrives
        within ``bootstrap_timeout_s``. Re-raises worker exceptions immediately.
        """
        deadline = time.perf_counter() + self._bootstrap_timeout_s
        while True:
            self._raise_if_worker_died()
            if self._has_action_for_current_step():
                return
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    f"BackgroundAsyncPlanner: no action for step {self._current_step} "
                    f"after {self._bootstrap_timeout_s}s; worker may be stuck."
                )
            try:
                item = self._outbox.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            self._process_outbox_item(item)

    def _process_outbox_item(self, item: _OutboxChunk | _OutboxError) -> bool:
        """Merge a chunk or surface an error. Returns True iff a chunk was merged."""
        if isinstance(item, _OutboxError):
            # Always clear the in-flight flag — the worker finished the request
            # (unsuccessfully). Stash the exception for re-raise on next call.
            with self._lock:
                self._in_flight = False
                if item.epoch == self._epoch:
                    self._worker_exc = item.exc
            return False
        assert isinstance(item, _OutboxChunk)
        with self._lock:
            # Worker finished this request; clear single-flight flag.
            self._in_flight = False
            if item.epoch != self._epoch:
                # Result from an episode that was already reset. Drop it.
                return False
            try:
                sliced = self._validate_chunk(item.chunk)
            except ValueError as exc:
                self._worker_exc = exc
                return False
            receive_step = self._current_step
            self._merge_chunk_locked(sliced, origin=item.origin_step)
            self.metrics.sum_receive_latency_steps += receive_step - item.origin_step
            self.metrics.inference_time_s += item.inference_s
            self.metrics.sum_wallclock_inference_s += item.inference_s
            return True

    # ---------- state accessors ----------

    def _has_action_for_current_step(self) -> bool:
        with self._lock:
            return self._current_step in self._queue

    def _is_in_flight(self) -> bool:
        with self._lock:
            return self._in_flight

    def _raise_if_worker_died(self) -> None:
        with self._lock:
            exc = self._worker_exc
            self._worker_exc = None
        if exc is not None:
            raise RuntimeError("BackgroundAsyncPlanner worker raised during prediction") from exc


def _clone_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Deep-clone tensors; pass-through non-tensor values (strings, lists, etc.).

    We must NOT share tensor storage between the caller and the worker — the
    main thread may in-place mutate the obs dict on the next step (processors
    reuse buffers), which would corrupt the worker's mid-flight input.
    """
    cloned: dict[str, Any] = {}
    for k, v in observation.items():
        if isinstance(v, torch.Tensor):
            cloned[k] = v.detach().clone()
        else:
            # Strings, lists of strings (the "task" key), scalars. These are
            # expected to be immutable or not reused across steps in the
            # rollout; shallow copy is fine.
            cloned[k] = v
    return cloned
