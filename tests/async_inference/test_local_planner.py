# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Smoke tests for ``BackgroundAsyncPlanner`` (and the base merge helper).

These tests deliberately avoid loading a real policy: we stub
``predict_action_chunk`` with a controllable fake so the tests exercise the
planner's concurrency + queue logic without GPU/hub access. That also keeps
them out of the ``[async]`` extra gate (no gRPC import here).
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from lerobot.async_inference.local_planner import (
    BackgroundAsyncPlanner,
    LocalAsyncPlanner,
)


# ---------- fakes ----------


class FakeChunkPolicy:
    """Policy stub with a configurable delay and a call counter.

    Action encoding (for debug readability):
      chunk[0, k, 0] = call_id   # 1-indexed
      chunk[0, k, 1] = k         # offset within chunk
    """

    def __init__(
        self,
        *,
        chunk_len: int = 4,
        action_dim: int = 2,
        inference_s: float = 0.0,
        gate: threading.Event | None = None,
    ):
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.inference_s = inference_s
        # If `gate` is provided, predictions block on it (caller decides when
        # each prediction can return). We still record the call_id BEFORE
        # blocking so the test can observe concurrency.
        self.gate = gate
        self._lock = threading.Lock()
        self.call_count = 0
        self.reset_count = 0
        # Recorded observations, keyed by 1-indexed call_id.
        self.obs_log: list[dict] = []

    def predict_action_chunk(self, batch):
        with self._lock:
            self.call_count += 1
            cid = self.call_count
            self.obs_log.append({k: v for k, v in batch.items()})
        if self.gate is not None:
            self.gate.wait()
        if self.inference_s:
            time.sleep(self.inference_s)
        chunk = torch.zeros(1, self.chunk_len, self.action_dim)
        chunk[0, :, 0] = float(cid)
        chunk[0, :, 1] = torch.arange(self.chunk_len, dtype=torch.float32)
        return chunk

    def reset(self):
        self.reset_count += 1


def _fake_obs(value: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.full((1, 2), value),
        "task": ["test"],  # non-tensor value, must survive clone
    }


# ---------- tests for the shared merge helper ----------


def test_merge_chunk_drops_stale_steps():
    """A chunk whose origin is in the past must only re-enqueue future steps."""
    policy = FakeChunkPolicy(chunk_len=8)
    planner = LocalAsyncPlanner(policy=policy, actions_per_chunk=8, chunk_size_threshold=0.5)
    planner._current_step = 5
    chunk = torch.zeros(1, 8, 2)
    for k in range(8):
        chunk[0, k, 0] = float(k)

    with planner._lock:
        dropped = planner._merge_chunk_locked(chunk, origin=0)

    assert dropped == 5, "steps 0..4 should be dropped as stale"
    assert list(planner._queue.keys()) == [5, 6, 7]
    assert planner.metrics.num_stale_actions_dropped == 5


# ---------- bootstrap ----------


def test_bootstrap_goes_through_worker_not_main_thread():
    policy = FakeChunkPolicy(chunk_len=4, inference_s=0.02)
    planner = BackgroundAsyncPlanner(
        policy=policy,
        actions_per_chunk=4,
        chunk_size_threshold=0.5,
        bootstrap_timeout_s=5.0,
    )
    try:
        main_tid = threading.get_ident()
        action = planner.select_action(_fake_obs(0.1))
        assert action.shape == (1, 2)
        # First chunk's call_id is 1.
        assert action[0, 0].item() == 1.0
        # Policy was called exactly once (bootstrap only).
        assert policy.call_count == 1
        # And it was not called from the main thread — the worker is the only
        # code path that touches `predict_action_chunk`.
        assert planner._worker is not None
        assert planner._worker.ident is not None
        assert planner._worker.ident != main_tid
    finally:
        planner.close()


# ---------- single-flight ----------


def test_single_flight_skips_when_worker_busy():
    """With a gated policy, the first prediction blocks; subsequent select_action
    calls that would trigger a threshold launch must be skipped (not queued).
    """
    gate = threading.Event()  # not set -> first predict blocks
    policy = FakeChunkPolicy(chunk_len=4, gate=gate)
    planner = BackgroundAsyncPlanner(
        policy=policy,
        actions_per_chunk=4,
        chunk_size_threshold=1.0,  # predict every step if idle
        bootstrap_timeout_s=5.0,
    )
    try:
        # Boot with gate closed -> the bootstrap call is blocked on gate.wait().
        # We open the gate once to let bootstrap return so we have actions.
        gate.set()
        planner.select_action(_fake_obs())  # bootstrap
        assert policy.call_count == 1
        gate.clear()  # future predictions block again

        # Threshold=1.0 would want to predict every step. Run 3 more steps
        # while the worker is stuck. Each step should:
        #   - try to launch (threshold says yes)
        #   - see in_flight=True (the threshold-launch from the previous step)
        #   - skip, incrementing num_in_flight_skips
        # ...except the very first of these, which successfully launches (1
        # new prediction in-flight).
        before_skips = planner.metrics.num_in_flight_skips
        for _ in range(3):
            planner.select_action(_fake_obs())
        # At most one additional predict was launched (single-flight).
        assert policy.call_count <= 2
        # And the threshold fired at least twice without launching.
        assert planner.metrics.num_in_flight_skips - before_skips >= 2
        gate.set()  # let the stuck worker finish so close() can join
    finally:
        gate.set()
        planner.close()


# ---------- reset isolation ----------


def test_reset_discards_in_flight_result_from_previous_episode():
    """An in-flight prediction at reset time must not leak into the new episode."""
    gate = threading.Event()
    policy = FakeChunkPolicy(chunk_len=4, gate=gate)
    planner = BackgroundAsyncPlanner(
        policy=policy,
        actions_per_chunk=4,
        chunk_size_threshold=0.5,
        bootstrap_timeout_s=5.0,
    )
    try:
        # Episode 1: bootstrap, then drain queue to force another launch.
        gate.set()
        planner.select_action(_fake_obs())
        gate.clear()
        # Chunk covers steps 0..3. Pop steps 1 and 2; at step 2, fill drops to
        # 2/4 <= 0.5 so another predict is launched (and blocks on gate).
        planner.select_action(_fake_obs())
        planner.select_action(_fake_obs())
        assert planner._is_in_flight(), "a prediction should be in flight before reset"
        pre_reset_calls = policy.call_count

        # Reset MID-flight. The blocked worker must be stopped so policy.reset()
        # can touch policy state safely. We release the gate so the blocked
        # worker can unblock and exit via the stop sentinel.
        gate.set()
        planner.reset()
        gate.clear()

        # Episode 2: bootstrap again. The new chunk's call_id must be strictly
        # greater than the pre-reset count, and the action must reflect the
        # freshly-launched prediction — not a leftover from episode 1.
        gate.set()
        action = planner.select_action(_fake_obs(0.5))
        new_cid = int(action[0, 0].item())
        assert new_cid > pre_reset_calls, (
            f"expected new prediction after reset, got call_id={new_cid} "
            f"(pre-reset count was {pre_reset_calls})"
        )
        # policy.reset was called exactly once during the planner reset.
        assert policy.reset_count >= 1
    finally:
        gate.set()
        planner.close()


# ---------- worker error propagation ----------


class _ExplodingPolicy:
    def __init__(self):
        self.reset_count = 0

    def predict_action_chunk(self, batch):
        raise RuntimeError("boom from worker")

    def reset(self):
        self.reset_count += 1


def test_worker_exception_is_surfaced_to_main_thread():
    policy = _ExplodingPolicy()
    planner = BackgroundAsyncPlanner(
        policy=policy,
        actions_per_chunk=4,
        chunk_size_threshold=0.5,
        bootstrap_timeout_s=2.0,
    )
    try:
        with pytest.raises(RuntimeError, match="worker raised"):
            planner.select_action(_fake_obs())
    finally:
        planner.close()


# ---------- observation isolation ----------


def test_observation_tensors_are_cloned_for_worker():
    """Mutating the caller-side obs dict after select_action must not affect what
    the worker already processed (i.e. the planner must clone tensors on the way
    into the inbox)."""
    policy = FakeChunkPolicy(chunk_len=4)
    planner = BackgroundAsyncPlanner(
        policy=policy,
        actions_per_chunk=4,
        chunk_size_threshold=0.5,
        bootstrap_timeout_s=5.0,
    )
    try:
        obs = _fake_obs(0.25)
        original_value = obs["observation.state"].clone()
        planner.select_action(obs)
        # Mutate the caller-side tensor in place; then do a second step. The
        # first recorded obs inside the worker must still match the original
        # value (proving a clone happened at handoff time).
        obs["observation.state"].fill_(99.0)
        planner.select_action(obs)
        recorded_first = policy.obs_log[0]["observation.state"]
        assert torch.equal(recorded_first, original_value), (
            "worker saw mutated obs; planner failed to clone tensors"
        )
    finally:
        planner.close()
