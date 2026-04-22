# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Local, single-process async planner for sim evaluation of action-chunk policies.

Mirrors the observation -> chunk -> aggregated-queue semantics of the gRPC
`RobotClient` in a single process, so sim rollouts can exercise async-style
control (high-frequency re-prediction, chunk aggregation, staleness tracking)
before bringing up the robot path.

Drop-in replacement for `policy.select_action` inside `rollout()`.

Current version is synchronous-only: inference runs on the caller thread. A
background-thread mode can be layered in later to emulate robot wall-clock
latency without changing the public interface.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import torch

from lerobot.policies.pretrained import PreTrainedPolicy

AggregateFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def latest_only(_old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    return new


def weighted_average(old: torch.Tensor, new: torch.Tensor, w_new: float = 0.7) -> torch.Tensor:
    return (1.0 - w_new) * old + w_new * new


@dataclass
class PlannerMetrics:
    """Flat scalars. `as_dict()` output is safe for `wandb_logger.log_dict`."""

    num_steps: int = 0
    num_chunks_generated: int = 0
    num_aggregated_actions: int = 0
    sum_staleness_steps: int = 0
    sum_queue_fill: float = 0.0
    inference_time_s: float = 0.0

    def as_dict(self) -> dict[str, float]:
        n = max(self.num_steps, 1)
        return {
            "async_chunks_per_step": self.num_chunks_generated / n,
            "async_aggregated_rate": self.num_aggregated_actions / n,
            "async_avg_staleness_steps": self.sum_staleness_steps / n,
            "async_avg_queue_fill": self.sum_queue_fill / n,
            "async_inference_s_total": self.inference_time_s,
        }


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
            action = self._queue.pop(self._current_step)
            origin = self._origin.pop(self._current_step, self._current_step)
            self.metrics.num_steps += 1
            self.metrics.sum_staleness_steps += self._current_step - origin
            self.metrics.sum_queue_fill += len(self._queue) / max(self._last_chunk_size, 1)
            self._current_step += 1
            return action

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

    def _should_predict(self) -> bool:
        with self._lock:
            if not self._queue or self._current_step not in self._queue:
                return True
            fill = len(self._queue) / max(self._last_chunk_size, 1)
            return fill <= self.threshold

    def _predict_and_merge(self, observation: dict[str, torch.Tensor]) -> None:
        t0 = time.perf_counter()
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(observation)  # [B, K_full, D]
        dt = time.perf_counter() - t0

        if chunk.ndim != 3:
            raise ValueError(f"predict_action_chunk must return [B, K, D], got {chunk.shape}")
        chunk = chunk[:, : self.actions_per_chunk]
        K = chunk.shape[1]

        with self._lock:
            origin = self._current_step
            for k in range(K):
                step = origin + k
                new_a = chunk[:, k].detach()
                if step in self._queue:
                    self._queue[step] = self.aggregate_fn(self._queue[step], new_a)
                    self.metrics.num_aggregated_actions += 1
                else:
                    self._queue[step] = new_a
                self._origin[step] = origin  # track the most recent chunk origin per step
            self._queue = OrderedDict(sorted(self._queue.items()))
            self._last_chunk_size = K
            self.metrics.num_chunks_generated += 1
            self.metrics.inference_time_s += dt
