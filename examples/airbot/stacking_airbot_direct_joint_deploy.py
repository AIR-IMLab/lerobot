#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deploy the stacking drifting checkpoint on AIRBOT Play with direct joint targets.

This script intentionally does not use Cartesian pose actions or IK. The stacking
dataset action is:

    [arm_joint_0, ..., arm_joint_5, g2_parallel_mit]

The arm command is sent as a direct AIRBOT joint-position target. The G2 command
is sent through the AIRBOT websocket `submit_g2_mit_command` path using the
dataset's parallel MIT value as the commanded G2 position.

Example, dry-run from a converted dataset sample:

    uv run python examples/airbot/stacking_airbot_direct_joint_deploy.py \
      --policy-path outputs/local_models/stacking_output_v30_drifting_200k/pretrained_model \
      --dataset-root /home/developer/Documents/VS_ws/Stacking/stacking_output_v30 \
      --dataset-repo-id local/stacking_output_v30 \
      --dry-run --run-seconds 2 --device cuda

Example, hardware through airbot-play-ws:

    cargo run --manifest-path ../rollio-ng/third_party/airbot-play-rust/Cargo.toml \
      --bin airbot-play-ws -- --interface can0 --bind 127.0.0.1:9002

    uv run python examples/airbot/stacking_airbot_direct_joint_deploy.py \
      --policy-path outputs/local_models/stacking_output_v30_drifting_200k/pretrained_model \
      --websocket-url ws://127.0.0.1:9002 \
      --image observation.images.camera_3__color=opencv:/dev/video4:1920x1080@30 \
      --image observation.images.camera_4__color=opencv:/dev/video6:1920x1080@30 \
      --initial-state 0.25,-1.09,0.98,0.73,-0.10,-0.96,0.068 \
      --run-seconds 30 --device cuda
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import signal
import time
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2  # type: ignore
import numpy as np
import torch

from lerobot.cameras import Camera, make_cameras_from_configs
from lerobot.cameras.configs import CameraConfig, Cv2Backends
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import get_policy_class, make_pre_post_processors, prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

LOGGER = logging.getLogger("stacking_airbot_direct_joint_deploy")

ARM_DOF = 6
ACTION_DIM = 7
DEFAULT_FPS = 30.0
DEFAULT_G2_MIT_KP = 10.0
DEFAULT_G2_MIT_KD = 0.5
DEFAULT_MAX_JOINT_STEP_RAD = 0.03
DEFAULT_MAX_G2_STEP = 0.003


@dataclass(frozen=True)
class ImageSpec:
    feature_key: str
    backend: Literal["opencv", "realsense"]
    source: str
    width: int
    height: int
    fps: int


class AirbotBackend:
    async def connect(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

    async def read_state(self) -> np.ndarray:
        raise NotImplementedError

    async def send_action(self, action: np.ndarray) -> None:
        raise NotImplementedError


class DryRunBackend(AirbotBackend):
    def __init__(self, initial_state: np.ndarray) -> None:
        self.state = np.asarray(initial_state, dtype=np.float32).copy()

    async def connect(self) -> None:
        LOGGER.info("Using dry-run backend; no AIRBOT commands will be sent.")

    async def close(self) -> None:
        return None

    async def read_state(self) -> np.ndarray:
        return self.state.copy()

    async def send_action(self, action: np.ndarray) -> None:
        self.state = np.asarray(action, dtype=np.float32).copy()


class DatasetReplayBackend(AirbotBackend):
    """Dry backend that feeds observations from an existing LeRobot dataset."""

    def __init__(self, dataset: LeRobotDataset, start_index: int) -> None:
        if start_index < 0 or start_index >= len(dataset):
            raise ValueError(f"start_index={start_index} is outside dataset length {len(dataset)}.")
        self.dataset = dataset
        self.index = start_index
        self.last_action: np.ndarray | None = None

    async def connect(self) -> None:
        LOGGER.info("Using dataset replay backend from frame index %s.", self.index)

    async def close(self) -> None:
        return None

    async def read_state(self) -> np.ndarray:
        sample = self.dataset[min(self.index, len(self.dataset) - 1)]
        state = sample[OBS_STATE]
        if isinstance(state, torch.Tensor):
            state = state[-1] if state.ndim == 2 else state
            return state.detach().cpu().numpy().astype(np.float32)
        return np.asarray(state, dtype=np.float32).reshape(-1)

    def read_observation(self) -> dict[str, np.ndarray]:
        sample = self.dataset[min(self.index, len(self.dataset) - 1)]
        observation: dict[str, np.ndarray] = {}
        for key, value in sample.items():
            if (key == OBS_STATE or key.startswith("observation.images.")) and isinstance(value, torch.Tensor):
                observation[key] = value.detach().cpu().numpy()
        return observation

    async def send_action(self, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self.index = min(self.index + 1, len(self.dataset) - 1)


class LenientOpenCVCamera(OpenCVCamera):
    """OpenCV camera wrapper for V4L2 devices that report failed property sets.

    Some UVC/V4L2 cameras return `False` from `VideoCapture.set()` even when the
    requested width/height is already active. The base LeRobot camera treats that
    as fatal; for deployment we only require the actual frame size to match.
    """

    def _validate_width_and_height(self) -> None:
        if self.videocapture is None:
            raise RuntimeError(f"{self} videocapture is not initialized")
        if self.capture_width is None or self.capture_height is None:
            raise ValueError(f"{self} capture_width or capture_height is not set")

        width_success = self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
        height_success = self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))

        actual_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.capture_width != actual_width or self.capture_height != actual_height:
            raise RuntimeError(
                f"{self} failed to set {self.capture_width}x{self.capture_height} "
                f"(actual={actual_width}x{actual_height}, {width_success=}, {height_success=})."
            )
        if not width_success or not height_success:
            LOGGER.warning(
                "%s reported failed width/height set, but actual frame size is %sx%s; continuing.",
                self,
                actual_width,
                actual_height,
            )

    def _validate_fps(self) -> None:
        if self.videocapture is None:
            raise RuntimeError(f"{self} videocapture is not initialized")
        if self.fps is None:
            raise ValueError(f"{self} FPS is not set")

        success = self.videocapture.set(cv2.CAP_PROP_FPS, float(self.fps))
        actual_fps = float(self.videocapture.get(cv2.CAP_PROP_FPS))
        if success and math.isclose(float(self.fps), actual_fps, rel_tol=1e-3):
            return
        if actual_fps > 0:
            LOGGER.warning(
                "%s requested fps=%s but camera reports actual_fps=%.3f (set_success=%s); continuing.",
                self,
                self.fps,
                actual_fps,
                success,
            )
            self.fps = int(round(actual_fps))
            return
        raise RuntimeError(f"{self} failed to set fps={self.fps} ({actual_fps=}, {success=}).")


class AirbotWebSocketBackend(AirbotBackend):
    """AIRBOT Play websocket backend.

    The websocket server is provided by `airbot-play-ws` from rollio-ng. It
    exposes direct joint targets with `submit_joint_target`; no Cartesian
    `submit_task_target` is used here.
    """

    def __init__(
        self,
        url: str,
        *,
        g2_mit_kp: float,
        g2_mit_kd: float,
        g2_effort: float,
        g2_velocity: float,
        g2_current_threshold: float,
        feedback_timeout_s: float,
    ) -> None:
        self.url = url
        self.g2_mit_kp = float(g2_mit_kp)
        self.g2_mit_kd = float(g2_mit_kd)
        self.g2_effort = float(g2_effort)
        self.g2_velocity = float(g2_velocity)
        self.g2_current_threshold = float(g2_current_threshold)
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.ws: Any = None
        self.session: Any = None
        self.last_arm_feedback: dict[str, Any] | None = None
        self.last_eef_feedback: dict[str, Any] | None = None
        self.reader_task: asyncio.Task | None = None

    async def connect(self) -> None:
        try:
            import aiohttp
        except ImportError as exc:
            raise RuntimeError("aiohttp is required for AIRBOT websocket deployment.") from exc

        self.session = aiohttp.ClientSession()
        try:
            self.ws = await self.session.ws_connect(self.url)
            await self._send({"type": "hello", "access_mode": "control"})
            await self._send({"type": "subscribe_arm_feedback"})
            await self._send({"type": "subscribe_eef_feedback"})
            self.reader_task = asyncio.create_task(self._reader())
            await self._send({"type": "set_arm_state", "state": "command_following"})
            await self._send({"type": "set_eef_state", "state": "enabled"})
        except Exception:
            await self.close()
            raise
        LOGGER.info("Connected to AIRBOT websocket %s.", self.url)

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()
        if self.reader_task is not None:
            self.reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.reader_task
        session = getattr(self, "session", None)
        if session is not None:
            await session.close()

    async def _send(self, message: dict[str, Any]) -> None:
        await self.ws.send_str(json.dumps(message))

    async def _reader(self) -> None:
        async for message in self.ws:
            if message.type.name == "TEXT":
                payload = json.loads(message.data)
                msg_type = payload.get("type")
                if msg_type == "arm_feedback":
                    self.last_arm_feedback = payload["feedback"]
                elif msg_type == "eef_feedback":
                    self.last_eef_feedback = payload["feedback"]
                elif msg_type == "error":
                    LOGGER.error("AIRBOT websocket error: %s", payload.get("message"))
                elif msg_type not in {"ack", "connected", "joint_target_accepted"}:
                    LOGGER.debug("AIRBOT websocket message: %s", payload)
            elif message.type.name == "ERROR":
                raise RuntimeError(f"AIRBOT websocket failed: {self.ws.exception()}")

    async def read_state(self) -> np.ndarray:
        deadline = time.perf_counter() + self.feedback_timeout_s
        while self.last_arm_feedback is None and time.perf_counter() < deadline:
            await asyncio.sleep(0.005)
        if self.last_arm_feedback is None:
            raise RuntimeError("No AIRBOT arm feedback received.")

        arm = np.asarray(self.last_arm_feedback["positions"], dtype=np.float32)
        g2 = 0.0
        if self.last_eef_feedback is not None and self.last_eef_feedback.get("valid", True):
            g2 = float(self.last_eef_feedback["position"])
        return np.concatenate([arm[:ARM_DOF], np.asarray([g2], dtype=np.float32)])

    async def send_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Expected action shape {(ACTION_DIM,)}, got {action.shape}.")
        await self._send({"type": "submit_joint_target", "positions": action[:ARM_DOF].tolist()})
        await self._send(
            {
                "type": "submit_g2_mit_command",
                "command": {
                    "position": float(action[6]),
                    "velocity": self.g2_velocity,
                    "effort": self.g2_effort,
                    "mit_kp": self.g2_mit_kp,
                    "mit_kd": self.g2_mit_kd,
                    "current_threshold": self.g2_current_threshold,
                },
            }
        )


def parse_width_height(text: str) -> tuple[int, int]:
    width_text, height_text = text.lower().split("x", maxsplit=1)
    return int(width_text), int(height_text)


def parse_image_spec(text: str) -> ImageSpec:
    try:
        key, rest = text.split("=", maxsplit=1)
        backend, source, size_fps = rest.split(":", maxsplit=2)
        size, fps_text = size_fps.split("@", maxsplit=1)
        width, height = parse_width_height(size)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected KEY=opencv:/dev/videoX:WIDTHxHEIGHT@FPS or KEY=realsense:SERIAL:WIDTHxHEIGHT@FPS."
        ) from exc
    if backend not in {"opencv", "realsense"}:
        raise argparse.ArgumentTypeError(f"Unsupported image backend {backend!r}.")
    return ImageSpec(
        feature_key=key,
        backend=backend,  # type: ignore[arg-type]
        source=source,
        width=width,
        height=height,
        fps=int(fps_text),
    )


def parse_state(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != ACTION_DIM:
        raise argparse.ArgumentTypeError(f"Expected {ACTION_DIM} comma-separated values, got {len(values)}.")
    return np.asarray(values, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="Path to checkpoint pretrained_model.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda or cpu.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--run-seconds", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run policy loop without sending hardware commands.")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--websocket-url",
        default=None,
        help="AIRBOT Play websocket URL from airbot-play-ws, e.g. ws://127.0.0.1:9002.",
    )
    parser.add_argument("--feedback-timeout-s", type=float, default=2.0)
    parser.add_argument("--g2-mit-kp", type=float, default=DEFAULT_G2_MIT_KP)
    parser.add_argument("--g2-mit-kd", type=float, default=DEFAULT_G2_MIT_KD)
    parser.add_argument("--g2-effort", type=float, default=0.0)
    parser.add_argument("--g2-velocity", type=float, default=0.0)
    parser.add_argument("--g2-current-threshold", type=float, default=0.0)

    parser.add_argument(
        "--image",
        action="append",
        default=[],
        type=parse_image_spec,
        help="Camera binding. Repeat for each policy image feature.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional converted stacking dataset root for dry-run/replay observations.",
    )
    parser.add_argument("--dataset-repo-id", default="local/stacking_output_v30")
    parser.add_argument("--dataset-start-index", type=int, default=1)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--initial-state", type=parse_state, default=None)
    parser.add_argument("--task", default="")
    parser.add_argument("--robot-type", default="airbot_play_stacking")
    parser.add_argument("--max-joint-step-rad", type=float, default=DEFAULT_MAX_JOINT_STEP_RAD)
    parser.add_argument("--max-g2-step", type=float, default=DEFAULT_MAX_G2_STEP)
    parser.add_argument("--no-action-limits", action="store_true")
    return parser.parse_args()


def load_policy(policy_path: str, device_override: str | None):
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    if device_override is not None:
        cfg.device = device_override
        cfg.use_amp = cfg.use_amp and torch.device(device_override).type == "cuda"
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=cfg)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
    )
    return cfg, policy, preprocessor, postprocessor


def validate_policy_contract(cfg: PreTrainedConfig, image_specs: list[ImageSpec]) -> None:
    if cfg.type != "drifting":
        raise ValueError(f"Expected a drifting checkpoint, got {cfg.type!r}.")
    if cfg.robot_state_feature is None or cfg.robot_state_feature.shape != (ACTION_DIM,):
        raise ValueError(f"Expected 7D AIRBOT stacking state, got {cfg.robot_state_feature}.")
    if cfg.action_feature is None or cfg.action_feature.shape != (ACTION_DIM,):
        raise ValueError(f"Expected 7D AIRBOT stacking action, got {cfg.action_feature}.")
    expected_images = set(cfg.image_features)
    provided_images = {spec.feature_key for spec in image_specs}
    if image_specs and provided_images != expected_images:
        raise ValueError(f"Image features mismatch: expected {sorted(expected_images)}, got {sorted(provided_images)}.")


def camera_config_from_spec(spec: ImageSpec) -> CameraConfig:
    if spec.backend == "opencv":
        source: int | str
        source = int(spec.source) if spec.source.isdigit() else spec.source
        return OpenCVCameraConfig(
            index_or_path=source,
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
            fourcc="MJPG",
            backend=Cv2Backends.V4L2,
        )
    return RealSenseCameraConfig(
        serial_number_or_name=spec.source,
        width=spec.width,
        height=spec.height,
        fps=spec.fps,
    )


def connect_cameras(image_specs: list[ImageSpec]) -> dict[str, Camera]:
    cameras: dict[str, Camera] = {}
    realsense_specs: list[ImageSpec] = []
    for spec in image_specs:
        if spec.backend == "opencv":
            cameras[spec.feature_key] = LenientOpenCVCamera(camera_config_from_spec(spec))  # type: ignore[arg-type]
        else:
            realsense_specs.append(spec)
    if realsense_specs:
        cameras.update(
            make_cameras_from_configs(
                {spec.feature_key: camera_config_from_spec(spec) for spec in realsense_specs}
            )
        )
    for camera in cameras.values():
        camera.connect()
    return cameras


def disconnect_cameras(cameras: dict[str, Camera]) -> None:
    for camera in cameras.values():
        with suppress(Exception):
            camera.disconnect()


def read_camera_observation(cameras: dict[str, Camera]) -> dict[str, np.ndarray]:
    observation = {}
    for key, camera in cameras.items():
        observation[key] = camera.read()
    return observation


def tensor_sample_to_numpy_observation(sample: dict[str, Any], cfg: PreTrainedConfig) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {}
    for key in [OBS_STATE, *cfg.image_features]:
        value = sample[key]
        if isinstance(value, torch.Tensor):
            observation[key] = value.detach().cpu().numpy()
        else:
            observation[key] = np.asarray(value)
    return observation


def make_temporal_model_batch(processed: dict[str, Any], cfg: PreTrainedConfig) -> dict[str, torch.Tensor]:
    model_batch = {key: value for key, value in processed.items() if isinstance(value, torch.Tensor) and key != ACTION}
    if cfg.image_features:
        model_batch[OBS_IMAGES] = torch.stack([model_batch[key] for key in cfg.image_features], dim=-4)
    return model_batch


def predict_action_chunk(
    *,
    observation: dict[str, np.ndarray],
    cfg: PreTrainedConfig,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    task: str,
    robot_type: str,
) -> np.ndarray:
    device = torch.device(str(cfg.device))
    with torch.inference_mode(), torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
        policy.reset()
        prepared = prepare_observation_for_inference(observation, device, task, robot_type)
        processed = preprocessor(prepared)
        raw_chunk = policy.predict_action_chunk(processed)
        chunk = postprocessor(raw_chunk)
    return chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)


def predict_action_chunk_from_dataset_sample(
    *,
    sample: dict[str, Any],
    cfg: PreTrainedConfig,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
) -> np.ndarray:
    batch = {}
    keys = set(cfg.input_features or {}) | set(cfg.output_features or {})
    for key in keys:
        if key not in sample:
            continue
        value = sample[key]
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
    if isinstance(sample.get("task"), str):
        batch["task"] = [sample["task"]]
    processed = preprocessor(batch)
    model_batch = make_temporal_model_batch(processed, cfg)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=torch.device(str(cfg.device)).type) if cfg.use_amp else nullcontext(),
    ):
        policy.reset()
        raw_chunk = policy.drifting.generate_actions(model_batch)
        chunk = postprocessor(raw_chunk)
    return chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)


def apply_action_limits(
    target: np.ndarray,
    current: np.ndarray,
    *,
    max_joint_step_rad: float,
    max_g2_step: float,
    enabled: bool,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).reshape(ACTION_DIM)
    current = np.asarray(current, dtype=np.float32).reshape(ACTION_DIM)
    if not enabled:
        return target
    limited = target.copy()
    joint_delta = np.clip(
        limited[:ARM_DOF] - current[:ARM_DOF],
        -float(max_joint_step_rad),
        float(max_joint_step_rad),
    )
    limited[:ARM_DOF] = current[:ARM_DOF] + joint_delta
    g2_delta = np.clip(limited[6] - current[6], -float(max_g2_step), float(max_g2_step))
    limited[6] = current[6] + g2_delta
    return limited


def load_dataset_for_replay(args: argparse.Namespace, cfg: PreTrainedConfig) -> LeRobotDataset | None:
    if args.dataset_root is None:
        return None
    ds_meta = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, video_backend=args.video_backend).meta
    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)
    if delta_timestamps is None:
        raise ValueError("Policy config did not resolve any delta_timestamps for dataset replay.")
    return LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )


async def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    cfg, policy, preprocessor, postprocessor = load_policy(args.policy_path, args.device)
    validate_policy_contract(cfg, args.image)
    LOGGER.info(
        "Loaded policy type=%s n_obs_steps=%s horizon=%s n_action_steps=%s device=%s.",
        cfg.type,
        cfg.n_obs_steps,
        cfg.horizon,
        cfg.n_action_steps,
        cfg.device,
    )

    dataset = load_dataset_for_replay(args, cfg)
    cameras: dict[str, Camera] = {}
    if dataset is not None:
        backend: AirbotBackend = DatasetReplayBackend(dataset, args.dataset_start_index)
    elif args.dry_run:
        initial_state = args.initial_state
        if initial_state is None:
            raise ValueError("--initial-state is required for --dry-run without --dataset-root.")
        backend = DryRunBackend(initial_state)
        if args.image:
            cameras = connect_cameras(args.image)
    else:
        if args.websocket_url is None:
            raise ValueError("--websocket-url is required unless --dry-run or --dataset-root is used.")
        if not args.image:
            raise ValueError("Hardware deployment needs --image bindings for all policy camera features.")
        cameras = connect_cameras(args.image)
        backend = AirbotWebSocketBackend(
            args.websocket_url,
            g2_mit_kp=args.g2_mit_kp,
            g2_mit_kd=args.g2_mit_kd,
            g2_effort=args.g2_effort,
            g2_velocity=args.g2_velocity,
            g2_current_threshold=args.g2_current_threshold,
            feedback_timeout_s=args.feedback_timeout_s,
        )

    stop_event = asyncio.Event()

    def _stop(*_: object) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _stop)

    await backend.connect()
    period_s = 1.0 / args.fps
    deadline = None if args.run_seconds is None else time.perf_counter() + args.run_seconds
    step = 0
    next_tick = time.perf_counter()
    try:
        while not stop_event.is_set():
            if deadline is not None and time.perf_counter() >= deadline:
                break
            current_state = await backend.read_state()

            if isinstance(backend, DatasetReplayBackend):
                sample = backend.dataset[min(backend.index, len(backend.dataset) - 1)]
                action_chunk = predict_action_chunk_from_dataset_sample(
                    sample=sample,
                    cfg=cfg,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
            else:
                observation = {OBS_STATE: current_state}
                observation.update(read_camera_observation(cameras))
                action_chunk = predict_action_chunk(
                    observation=observation,
                    cfg=cfg,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task=args.task,
                    robot_type=args.robot_type,
                )

            action = apply_action_limits(
                action_chunk[0],
                current_state,
                max_joint_step_rad=args.max_joint_step_rad,
                max_g2_step=args.max_g2_step,
                enabled=not args.no_action_limits,
            )
            await backend.send_action(action)
            if step % max(1, args.print_every) == 0:
                LOGGER.info(
                    "step=%s state=%s action=%s raw_first=%s",
                    step,
                    np.array2string(current_state, precision=4, suppress_small=True),
                    np.array2string(action, precision=4, suppress_small=True),
                    np.array2string(action_chunk[0], precision=4, suppress_small=True),
                )
            step += 1
            next_tick += period_s
            await asyncio.sleep(max(0.0, next_tick - time.perf_counter()))
    finally:
        await backend.close()
        disconnect_cameras(cameras)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
