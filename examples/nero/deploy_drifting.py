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

"""Deploy a LeRobot Drifting checkpoint on an AGX Nero arm.

This script reuses the Rollio Nero driver for the hardware pieces:

* pyAgxArm connection / enable handshake
* Nero gravity model
* Cartesian IK from target pose7 to 7 arm joints
* AIRBOT-aligned pose adapter used by Rollio datasets

Expected policy contract:

* observation.state is [joint_0, ..., joint_6, gripper_width_m] by default
* action is [x, y, z, qx, qy, qz, qw, gripper_width_m]
* pose7 uses scalar-last xyzw quaternions
* positions and gripper width are in meters

Example:

uv run python examples/nero/deploy_drifting.py \
  --policy-path /data/checkpoints/nero_drifting/pretrained_model \
  --interface can0 \
  --image observation.images.front=opencv:/dev/video0:640x480@30 \
  --fps 10 \
  --pose-frame aligned

Read-only preview:

uv run python examples/nero/deploy_drifting.py \
  --policy-path /data/checkpoints/nero_drifting/pretrained_model \
  --interface can0 \
  --image observation.images.realsense_color=realsense:SERIAL:1920x1080@30 \
  --preview-only \
  --preview-rerun

Camera-only inference, no Nero connected:

uv run python examples/nero/deploy_drifting.py \
  --policy-path /data/checkpoints/nero_drifting/pretrained_model \
  --image observation.images.realsense_color=realsense:SERIAL:1280x720@30 \
  --camera-only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import signal
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from lerobot.cameras import Camera, make_cameras_from_configs
from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.visualization_utils import init_rerun, shutdown_rerun

LOGGER = logging.getLogger("deploy_drifting_nero")

ARM_DOF = 7
POSE7_DIM = 7
POSE7_GRIPPER_DIM = 8
DEFAULT_POLICY_FPS = 10.0
DEFAULT_TRACKING_KP = 20.0
DEFAULT_TRACKING_KD = 1.0
DEFAULT_GRIPPER_HZ = 20.0
DEFAULT_GRIPPER_FORCE_N = 1.0
DEFAULT_MAX_EE_STEP_M = 0.05
DEFAULT_MAX_JOINT_DELTA_RAD = math.pi / 12.0
TRIAL_POLICY_FPS = 3.0
TRIAL_TRACKING_KP = 10.0
TRIAL_TRACKING_KD = 0.8
TRIAL_GRIPPER_HZ = 5.0
TRIAL_GRIPPER_FORCE_N = 0.3
TRIAL_MAX_EE_STEP_M = 0.005
TRIAL_MAX_JOINT_DELTA_RAD = 0.015
TRIAL_MAX_JOINT_VEL_RAD_S = 0.15
TRIAL_MAX_CARTESIAN_VEL_M_S = 0.02
TRIAL_MAX_ANGULAR_VEL_RAD_S = 0.3
TRIAL_MAX_GRIPPER_VEL_M_S = 0.01
DEFAULT_HANDOVER_STATE = np.asarray(
    [
        -0.01491420789079838,
        0.5375512154928302,
        0.016529188605555842,
        1.5734635288003889,
        0.010142373681970938,
        -0.04286711606385838,
        -0.4292580009383019,
        0.06720915155165022,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class ImageSpec:
    feature_key: str
    backend: Literal["opencv", "realsense"]
    source: str
    width: int
    height: int
    fps: int


@dataclass
class SharedTargets:
    target_q: np.ndarray
    gripper_width_m: float | None
    gripper_force_n: float
    last_policy_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True)
class RollioNeroRuntime:
    create_robot: Any
    enable_robot: Any
    init_gripper: Any
    AgxArmBackend: Any
    AgxGripperBackend: Any
    NeroModel: Any
    solve_ik: Any
    apply_publish_pose_fix: Any
    apply_command_pose_fix: Any
    disabled_hold_q: np.ndarray


@dataclass(frozen=True)
class PolicyPreview:
    observation: dict[str, np.ndarray]
    current_pose7: np.ndarray
    target_pose7_raw: np.ndarray
    target_pose7: np.ndarray
    gripper_width_m: float
    target_gripper_width_m: float | None
    q_target: np.ndarray
    ik_converged: bool
    ik_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a LeRobot Drifting policy on AGX Nero using Rollio's Nero hardware driver. "
            "The policy checkpoint must include config.json and the saved policy processors."
        )
    )
    parser.add_argument("--policy-path", default=None, help="Local path or Hugging Face repo id.")
    parser.add_argument("--interface", default="can0", help="CAN interface for AGX Nero.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda, cuda:0, cpu.")
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_POLICY_FPS, help="Policy observation/action rate."
    )
    parser.add_argument("--control-hz", type=float, default=250.0, help="MIT command loop frequency.")
    parser.add_argument(
        "--gripper-hz", type=float, default=DEFAULT_GRIPPER_HZ, help="Gripper command frequency."
    )
    parser.add_argument(
        "--gripper-force", type=float, default=DEFAULT_GRIPPER_FORCE_N, help="AGX gripper force in newtons."
    )
    parser.add_argument("--gripper-min", type=float, default=0.0, help="Minimum gripper width in meters.")
    parser.add_argument("--gripper-max", type=float, default=0.10, help="Maximum gripper width in meters.")
    parser.add_argument("--no-gripper", action="store_true", help="Run arm only and ignore action[7].")
    parser.add_argument(
        "--state-layout",
        choices=("joints_gripper", "pose7_gripper"),
        default="joints_gripper",
        help=(
            "How to build observation.state. The current handover checkpoint was trained with "
            "joint_position.0..6 + gripper_position. Use pose7_gripper only for datasets recorded that way."
        ),
    )
    parser.add_argument(
        "--pose-frame",
        choices=("aligned", "native"),
        default="aligned",
        help=(
            "Frame used by the checkpoint for pose7. Use 'aligned' for Rollio/AIRBOT-aligned datasets; "
            "use 'native' only if the model was trained on raw Nero FK poses."
        ),
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="FEATURE=BACKEND:SOURCE:WIDTHxHEIGHT@FPS",
        help=(
            "Camera source for a visual policy feature. Example: "
            "observation.images.front=opencv:/dev/video0:640x480@30 or "
            "observation.images.wrist=realsense:123456:640x480@30. Repeat for multiple cameras."
        ),
    )
    parser.add_argument(
        "--policy-image-size",
        type=parse_width_height,
        default=None,
        metavar="WIDTHxHEIGHT",
        help=(
            "Resize camera frames to this size before sending them to the policy, e.g. 854x480. "
            "Default keeps the captured camera size."
        ),
    )
    parser.add_argument("--camera-max-age-ms", type=int, default=500)
    parser.add_argument("--task", default="", help="Optional task string passed to LeRobot processors.")
    parser.add_argument("--robot-type", default="agx_nero", help="Optional robot type string.")
    parser.add_argument(
        "--max-ee-step-m",
        type=float,
        default=DEFAULT_MAX_EE_STEP_M,
        help="Clamp target TCP translation to this many meters per policy tick. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max-cartesian-vel-m-s",
        type=float,
        default=0.0,
        help="Low-pass policy target translation in TCP space. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max-angular-vel-rad-s",
        type=float,
        default=0.0,
        help="Low-pass policy target orientation with quaternion slerp. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max-gripper-vel-m-s",
        type=float,
        default=0.0,
        help="Limit commanded gripper width speed. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max-ik-error",
        type=float,
        default=0.05,
        help="Skip a target update when final IK twist norm exceeds this value. Set <=0 to disable.",
    )
    parser.add_argument(
        "--max-joint-delta-rad",
        type=float,
        default=DEFAULT_MAX_JOINT_DELTA_RAD,
        help="Per-control-tick joint target clamp relative to measured q.",
    )
    parser.add_argument(
        "--max-joint-vel-rad-s",
        type=float,
        default=0.0,
        help="Interpolate joint targets at this max speed before MIT commands. Set <=0 to disable.",
    )
    parser.add_argument("--tracking-kp", type=float, default=DEFAULT_TRACKING_KP)
    parser.add_argument("--tracking-kd", type=float, default=DEFAULT_TRACKING_KD)
    parser.add_argument("--run-seconds", type=float, default=None, help="Optional automatic stop time.")
    parser.add_argument(
        "--trial-control",
        action="store_true",
        help=(
            "Enable conservative first-contact control defaults and interpolation. "
            "Explicitly supplied speed/gain args still override these defaults."
        ),
    )
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help=(
            "Run the policy from camera frames and a fixed observation.state without connecting Nero. "
            "Useful for checking the 8D drifting action output before the arm is attached."
        ),
    )
    parser.add_argument(
        "--camera-only-steps",
        type=int,
        default=1,
        help="Number of camera-only inference steps to run. Use 0 to run until Ctrl-C or --run-seconds.",
    )
    parser.add_argument(
        "--fixed-state",
        default=None,
        metavar="V0,V1,...,V7",
        help=(
            "Comma-separated observation.state used by --camera-only. Defaults to the q50 state from "
            "the current handover dataset: 7 joint positions + gripper width."
        ),
    )
    parser.add_argument(
        "--state-stats-path",
        type=Path,
        default=None,
        help="Optional stats.json path. In --camera-only, reads observation.state.q50 from this file.",
    )
    parser.add_argument(
        "--camera-only-save-image",
        default="outputs/captured_images/drifting_camera_only_input.png",
        help="Where to save the latest camera frame used for camera-only inference. Set empty string to disable.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help=(
            "Read cameras and robot feedback, run the policy, and print/log predicted targets without "
            "enabling motors or sending any arm/gripper command."
        ),
    )
    parser.add_argument(
        "--preview-rerun",
        action="store_true",
        help="Visualize current TCP and predicted policy target in Rerun 3D.",
    )
    parser.add_argument(
        "--preview-rerun-images",
        action="store_true",
        help="Also log camera images to Rerun preview. This can be heavy at 1080p.",
    )
    parser.add_argument(
        "--preview-rerun-image-every",
        type=int,
        default=5,
        help="Log one camera image to Rerun every N policy ticks. Use 1 for every tick.",
    )
    parser.add_argument(
        "--preview-rerun-image-max-width",
        type=int,
        default=640,
        help="Resize Rerun camera images to this max width before logging. Use 0 to keep full resolution.",
    )
    parser.add_argument(
        "--preview-print-every",
        type=int,
        default=1,
        help="Print one preview line every N policy ticks.",
    )
    parser.add_argument(
        "--home-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ramp to Rollio's disabled hold pose before disconnecting.",
    )
    parser.add_argument("--home-duration-s", type=float, default=3.0)
    parser.add_argument("--home-settle-s", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the checkpoint and validate feature wiring, but do not connect hardware or cameras.",
    )
    parser.add_argument(
        "--inspect-policy",
        action="store_true",
        help="Print checkpoint feature config and exit before loading weights, cameras, or hardware.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def import_rollio_nero() -> RollioNeroRuntime:
    try:
        from rollio_device_nero.airbot_aligned_pose import apply_command_pose_fix, apply_publish_pose_fix
        from rollio_device_nero.gravity import NeroModel
        from rollio_device_nero.ik import solve as solve_ik
        from rollio_device_nero.runtime.agx_backend import (
            AgxArmBackend,
            AgxGripperBackend,
            create_robot,
            enable_robot,
            init_gripper,
        )
        from rollio_device_nero.runtime.arm import DISABLED_HOLD_Q
    except Exception as exc:
        raise RuntimeError(
            "Rollio's Nero package is required on the AGX host. Install it from rollio-ng with:\n"
            "  git submodule update --init third_party/iceoryx2 third_party/pyAgxArm\n"
            "  uv pip install -e robots/nero"
        ) from exc

    return RollioNeroRuntime(
        create_robot=create_robot,
        enable_robot=enable_robot,
        init_gripper=init_gripper,
        AgxArmBackend=AgxArmBackend,
        AgxGripperBackend=AgxGripperBackend,
        NeroModel=NeroModel,
        solve_ik=solve_ik,
        apply_publish_pose_fix=apply_publish_pose_fix,
        apply_command_pose_fix=apply_command_pose_fix,
        disabled_hold_q=np.asarray(DISABLED_HOLD_Q, dtype=float),
    )


def parse_width_height(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid size {value!r}; expected WIDTHxHEIGHT, e.g. 854x480."
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"Invalid size {value!r}; width and height must be positive.")
    return width, height


def parse_image_spec(value: str) -> ImageSpec:
    try:
        feature_key, camera_spec = value.split("=", 1)
        backend, rest = camera_spec.split(":", 1)
        source, profile = rest.rsplit(":", 1)
        size, fps_text = profile.split("@", 1)
        width, height = parse_width_height(size)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --image value {value!r}; expected FEATURE=BACKEND:SOURCE:WIDTHxHEIGHT@FPS."
        ) from exc

    backend = backend.lower()
    if backend == "intelrealsense":
        backend = "realsense"
    if backend not in {"opencv", "realsense"}:
        raise argparse.ArgumentTypeError(f"Unsupported camera backend {backend!r}.")

    return ImageSpec(
        feature_key=feature_key,
        backend=backend,
        source=source,
        width=width,
        height=height,
        fps=int(fps_text),
    )


def make_camera_configs(image_specs: list[ImageSpec]) -> dict[str, CameraConfig]:
    camera_configs: dict[str, CameraConfig] = {}
    for spec in image_specs:
        if spec.feature_key in camera_configs:
            raise ValueError(f"Duplicate camera feature key: {spec.feature_key}")

        if spec.backend == "opencv":
            source: int | Path
            source = int(spec.source) if spec.source.isdecimal() else Path(spec.source)
            camera_configs[spec.feature_key] = OpenCVCameraConfig(
                index_or_path=source,
                width=spec.width,
                height=spec.height,
                fps=spec.fps,
            )
        else:
            camera_configs[spec.feature_key] = RealSenseCameraConfig(
                serial_number_or_name=spec.source,
                width=spec.width,
                height=spec.height,
                fps=spec.fps,
            )
    return camera_configs


def load_policy(policy_path: str, device_override: str | None):
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    if device_override is not None:
        cfg.device = device_override
        cfg.use_amp = cfg.use_amp and torch.device(device_override).type == "cuda"

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
    )
    policy.reset()
    return cfg, policy, preprocessor, postprocessor


def log_policy_contract(cfg: PreTrainedConfig) -> None:
    lines = [f"Policy type: {cfg.type}", f"Device from config: {cfg.device}", "Input features:"]
    for key, feature in (cfg.input_features or {}).items():
        lines.append(f"  {key}: type={feature.type} shape={feature.shape}")
    lines.append("Output features:")
    for key, feature in (cfg.output_features or {}).items():
        lines.append(f"  {key}: type={feature.type} shape={feature.shape}")
    lines.extend(
        [
            f"Image feature keys: {sorted(cfg.image_features)}",
            f"{OBS_STATE} dim: {None if cfg.robot_state_feature is None else cfg.robot_state_feature.shape}",
            f"{ACTION} dim: {None if cfg.action_feature is None else cfg.action_feature.shape}",
        ]
    )
    print("\n".join(lines))


def _is_default(value: float, default: float) -> bool:
    return math.isclose(float(value), float(default), rel_tol=1e-9, abs_tol=1e-12)


def apply_trial_control_defaults(args: argparse.Namespace) -> None:
    if not args.trial_control:
        return

    if _is_default(args.fps, DEFAULT_POLICY_FPS):
        args.fps = TRIAL_POLICY_FPS
    if _is_default(args.tracking_kp, DEFAULT_TRACKING_KP):
        args.tracking_kp = TRIAL_TRACKING_KP
    if _is_default(args.tracking_kd, DEFAULT_TRACKING_KD):
        args.tracking_kd = TRIAL_TRACKING_KD
    if _is_default(args.gripper_hz, DEFAULT_GRIPPER_HZ):
        args.gripper_hz = TRIAL_GRIPPER_HZ
    if _is_default(args.gripper_force, DEFAULT_GRIPPER_FORCE_N):
        args.gripper_force = TRIAL_GRIPPER_FORCE_N
    if _is_default(args.max_ee_step_m, DEFAULT_MAX_EE_STEP_M):
        args.max_ee_step_m = TRIAL_MAX_EE_STEP_M
    if _is_default(args.max_joint_delta_rad, DEFAULT_MAX_JOINT_DELTA_RAD):
        args.max_joint_delta_rad = TRIAL_MAX_JOINT_DELTA_RAD
    if args.max_joint_vel_rad_s <= 0.0:
        args.max_joint_vel_rad_s = TRIAL_MAX_JOINT_VEL_RAD_S
    if args.max_cartesian_vel_m_s <= 0.0:
        args.max_cartesian_vel_m_s = TRIAL_MAX_CARTESIAN_VEL_M_S
    if args.max_angular_vel_rad_s <= 0.0:
        args.max_angular_vel_rad_s = TRIAL_MAX_ANGULAR_VEL_RAD_S
    if args.max_gripper_vel_m_s <= 0.0:
        args.max_gripper_vel_m_s = TRIAL_MAX_GRIPPER_VEL_M_S
    if args.run_seconds is None and not args.preview_only:
        args.run_seconds = 10.0

    summary = (
        "Trial control enabled: "
        f"fps={args.fps:.2f}, kp={args.tracking_kp:.3f}, kd={args.tracking_kd:.3f}, "
        f"max_joint_vel={args.max_joint_vel_rad_s:.3f} rad/s, "
        f"max_cart_vel={args.max_cartesian_vel_m_s:.3f} m/s, "
        f"max_ang_vel={args.max_angular_vel_rad_s:.3f} rad/s, "
        f"max_gripper_vel={args.max_gripper_vel_m_s:.3f} m/s, "
        f"max_joint_delta={args.max_joint_delta_rad:.4f} rad/tick, "
        f"gripper_force={args.gripper_force:.3f} N, run_seconds={args.run_seconds}"
    )
    LOGGER.info("%s", summary)
    print(summary, flush=True)


def validate_policy_features(cfg: PreTrainedConfig, provided_image_keys: set[str], no_gripper: bool) -> None:
    if cfg.robot_state_feature is None:
        raise ValueError(f"This deploy script expects a {OBS_STATE!r} robot-state feature.")
    state_dim = cfg.robot_state_feature.shape[0]
    if state_dim != POSE7_GRIPPER_DIM:
        raise ValueError(
            f"This deploy script currently builds {OBS_STATE} as 8 dims, "
            f"but the checkpoint expects {state_dim} dims."
        )

    if cfg.action_feature is None:
        raise ValueError(f"This deploy script expects an {ACTION!r} action feature.")
    action_dim = cfg.action_feature.shape[0]
    min_action_dim = POSE7_DIM if no_gripper else POSE7_GRIPPER_DIM
    if action_dim < min_action_dim:
        raise ValueError(
            f"Checkpoint action dim is {action_dim}, but Nero pose7"
            f"{'' if no_gripper else '+gripper'} control needs at least {min_action_dim} dims."
        )
    if action_dim > POSE7_GRIPPER_DIM:
        LOGGER.warning("Checkpoint action dim is %s; using the first 8 dims as pose7+gripper.", action_dim)

    expected_images = set(cfg.image_features)
    missing_images = expected_images - provided_image_keys
    extra_images = provided_image_keys - expected_images
    if missing_images:
        raise ValueError(
            "Missing camera sources for policy image features: "
            f"{sorted(missing_images)}. Pass one --image per feature key."
        )
    if extra_images:
        raise ValueError(
            "Camera sources were provided for non-policy image features: "
            f"{sorted(extra_images)}. Check spelling against checkpoint config.json."
        )


def parse_fixed_state(text: str) -> np.ndarray:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if len(values) != POSE7_GRIPPER_DIM:
        raise ValueError(
            f"--fixed-state must contain {POSE7_GRIPPER_DIM} comma-separated values, got {len(values)}"
        )
    return np.asarray(values, dtype=np.float32)


def load_camera_only_state(args: argparse.Namespace) -> np.ndarray:
    if args.fixed_state:
        return parse_fixed_state(args.fixed_state)
    if args.state_stats_path is not None:
        with args.state_stats_path.open() as f:
            stats = json.load(f)
        return np.asarray(stats[OBS_STATE]["q50"], dtype=np.float32)
    return DEFAULT_HANDOVER_STATE.copy()


def connect_cameras(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    cameras = make_cameras_from_configs(camera_configs)
    for key, camera in cameras.items():
        LOGGER.info("Connecting camera %s", key)
        camera.connect()
    return cameras


def disconnect_cameras(cameras: dict[str, Camera]) -> None:
    for key, camera in cameras.items():
        LOGGER.info("Disconnecting camera %s", key)
        with suppress(Exception):
            camera.disconnect()


def save_camera_only_image(image: np.ndarray, path: str | Path | None) -> None:
    if path is None or str(path) == "":
        return
    path = Path(path)
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def format_camera_only_action(step: int, action: np.ndarray) -> str:
    pose7 = action[:POSE7_DIM]
    quat = pose7[3:7]
    quat_norm = float(np.linalg.norm(quat))
    gripper = float("nan") if action.shape[0] <= POSE7_DIM else float(action[POSE7_DIM])
    return (
        f"step={step:06d} "
        f"action_8d={np.array2string(action[:POSE7_GRIPPER_DIM], precision=6, suppress_small=False)} "
        f"target_xyz_m={np.array2string(pose7[:3], precision=6, suppress_small=False)} "
        f"target_quat_xyzw={np.array2string(quat, precision=6, suppress_small=False)} "
        f"quat_norm={quat_norm:.6f} "
        f"target_gripper_m={gripper:.6f}"
    )


def resize_image_to_size(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if image.ndim < 2 or (image.shape[1] == width and image.shape[0] == height):
        return image

    import cv2

    interpolation = cv2.INTER_AREA if width < image.shape[1] or height < image.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_policy_image(image: np.ndarray, policy_image_size: tuple[int, int] | None) -> np.ndarray:
    if policy_image_size is None:
        return image
    return resize_image_to_size(image, policy_image_size)


def prepare_rerun_image(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or image.ndim < 2 or image.shape[1] <= max_width:
        return image

    scale = float(max_width) / float(image.shape[1])
    height = max(1, int(round(image.shape[0] * scale)))
    return resize_image_to_size(image, (max_width, height))


def should_log_rerun_image(step: int, every: int) -> bool:
    return step % max(1, every) == 0


def log_camera_only_rerun(
    step: int,
    observation: dict[str, np.ndarray],
    action: np.ndarray,
    *,
    include_images: bool,
    image_every: int,
    image_max_width: int,
) -> None:
    import rerun as rr

    target_xyz = action[:3]
    rr.set_time("step", sequence=step)
    rr.log(
        "camera_only/policy_target",
        rr.Points3D([target_xyz], colors=[[255, 120, 0]], radii=[0.014]),
    )
    rr.log("camera_only/scalars/target_gripper_m", rr.Scalars(float(action[POSE7_DIM])))
    if include_images and should_log_rerun_image(step, image_every):
        for feature_key, image in observation.items():
            if "image" in feature_key and isinstance(image, np.ndarray):
                rr.log(feature_key, rr.Image(prepare_rerun_image(image, image_max_width)).compress())


def run_camera_only_loop(
    *,
    args: argparse.Namespace,
    cfg: PreTrainedConfig,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    cameras: dict[str, Camera],
) -> None:
    state = load_camera_only_state(args)
    period_s = 1.0 / args.fps
    started_at = time.perf_counter()
    next_tick = started_at
    step = 0

    if args.preview_rerun:
        init_preview_rerun()

    try:
        print("Camera-only mode: no Nero connection; observation.state is fixed.", flush=True)
        print(f"fixed_state={np.array2string(state, precision=6, suppress_small=False)}", flush=True)
        while True:
            if args.run_seconds is not None and time.perf_counter() - started_at >= args.run_seconds:
                break
            if args.camera_only_steps > 0 and step >= args.camera_only_steps:
                break

            observation: dict[str, np.ndarray] = {OBS_STATE: state.copy()}
            for feature_key, camera in cameras.items():
                image = camera.read_latest(max_age_ms=args.camera_max_age_ms)
                observation[feature_key] = resize_policy_image(image, args.policy_image_size)

            first_image = next((value for key, value in observation.items() if "image" in key), None)
            if isinstance(first_image, np.ndarray) and step == 0:
                save_camera_only_image(first_image, args.camera_only_save_image)
                print(f"input_image_shape_hwc={first_image.shape}", flush=True)
                if args.camera_only_save_image is not None and str(args.camera_only_save_image) != "":
                    print(f"saved_input_image={args.camera_only_save_image}", flush=True)

            action = predict_action(
                observation=observation,
                policy=policy,
                device=torch.device(str(cfg.device)),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=cfg.use_amp,
                task=args.task,
                robot_type=args.robot_type,
            )
            action_np = action.squeeze(0).detach().cpu().numpy().astype(float).reshape(-1)
            print(format_camera_only_action(step, action_np), flush=True)

            if args.preview_rerun:
                log_camera_only_rerun(
                    step,
                    observation,
                    action_np,
                    include_images=args.preview_rerun_images,
                    image_every=args.preview_rerun_image_every,
                    image_max_width=args.preview_rerun_image_max_width,
                )

            step += 1
            next_tick += period_s
            sleep_until(next_tick)
    finally:
        if args.preview_rerun:
            shutdown_preview_rerun()


def wait_for_joint_feedback(arm: Any, timeout_s: float = 5.0) -> np.ndarray:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        q_meas = arm.get_joint_angles_array()
        if q_meas is not None:
            q_meas = np.asarray(q_meas, dtype=float)
            if q_meas.shape == (ARM_DOF,):
                return q_meas
        time.sleep(0.01)
    raise TimeoutError(f"No AGX Nero joint feedback received within {timeout_s:.1f}s.")


def normalize_pose7_xyzw(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=float).reshape(POSE7_DIM).copy()
    quat = pose7[3:7]
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        pose7[3:7] = np.asarray([0.0, 0.0, 0.0, 1.0])
    else:
        pose7[3:7] = quat / norm
    return pose7


def quat_dot_xyzw(q0: np.ndarray, q1: np.ndarray) -> float:
    return float(np.dot(q0, q1))


def quat_slerp_xyzw(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=float).reshape(4)
    q1 = np.asarray(q1, dtype=float).reshape(4)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-8)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-8)

    dot = quat_dot_xyzw(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        out = q0 + fraction * (q1 - q0)
        return out / max(float(np.linalg.norm(out)), 1e-8)

    theta_0 = math.acos(dot)
    theta = theta_0 * float(np.clip(fraction, 0.0, 1.0))
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    out = s0 * q0 + s1 * q1
    return out / max(float(np.linalg.norm(out)), 1e-8)


def quat_angle_xyzw(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = np.asarray(q0, dtype=float).reshape(4)
    q1 = np.asarray(q1, dtype=float).reshape(4)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-8)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-8)
    dot = abs(quat_dot_xyzw(q0, q1))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def _positive_min(*values: float) -> float:
    positive_values = [value for value in values if value > 0.0]
    if not positive_values:
        return 0.0
    return min(positive_values)


def limit_scalar_step(current: float, target: float, max_step: float) -> float:
    if max_step <= 0.0:
        return float(target)
    return float(current + np.clip(target - current, -max_step, max_step))


def limit_pose7_step(
    current_pose7: np.ndarray,
    target_pose7: np.ndarray,
    *,
    max_translation_step_m: float,
    max_angular_step_rad: float,
) -> np.ndarray:
    current_pose7 = normalize_pose7_xyzw(current_pose7)
    target_pose7 = normalize_pose7_xyzw(target_pose7)
    out = target_pose7.copy()

    if max_translation_step_m > 0.0:
        delta = target_pose7[:3] - current_pose7[:3]
        dist = float(np.linalg.norm(delta))
        if dist > max_translation_step_m:
            out[:3] = current_pose7[:3] + delta * (max_translation_step_m / dist)

    if max_angular_step_rad > 0.0:
        angle = quat_angle_xyzw(current_pose7[3:7], target_pose7[3:7])
        if angle > max_angular_step_rad:
            out[3:7] = quat_slerp_xyzw(current_pose7[3:7], target_pose7[3:7], max_angular_step_rad / angle)

    return normalize_pose7_xyzw(out)


def build_observation(
    *,
    q_meas: np.ndarray,
    gripper_width_m: float,
    cameras: dict[str, Camera],
    nero: Any,
    runtime: RollioNeroRuntime,
    pose_frame: str,
    state_layout: str,
    camera_max_age_ms: int,
    policy_image_size: tuple[int, int] | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    native_pose7 = np.asarray(nero.end_effector_pose7(q_meas), dtype=float)
    if pose_frame == "aligned":
        obs_pose7 = np.asarray(runtime.apply_publish_pose_fix(native_pose7), dtype=float)
    else:
        obs_pose7 = native_pose7
    obs_pose7 = normalize_pose7_xyzw(obs_pose7)

    if state_layout == "joints_gripper":
        observation_state = np.asarray([*q_meas, gripper_width_m], dtype=np.float32)
    elif state_layout == "pose7_gripper":
        observation_state = np.asarray([*obs_pose7, gripper_width_m], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported state layout: {state_layout}")

    observation: dict[str, np.ndarray] = {OBS_STATE: observation_state}
    for feature_key, camera in cameras.items():
        image = camera.read_latest(max_age_ms=camera_max_age_ms)
        observation[feature_key] = resize_policy_image(image, policy_image_size)
    return observation, obs_pose7


def read_gripper_width(gripper: Any | None, gripper_min: float, gripper_max: float) -> float:
    if gripper is None:
        return 0.0
    reported_width = gripper.get_gripper_position_m()
    if reported_width is None:
        return 0.0
    return float(np.clip(reported_width, gripper_min, gripper_max))


def predict_nero_policy_target(
    *,
    args: argparse.Namespace,
    cfg: PreTrainedConfig,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    q_meas: np.ndarray,
    gripper_width_m: float,
    nero: Any,
    runtime: RollioNeroRuntime,
    cameras: dict[str, Camera],
    ik_seed: np.ndarray,
    policy_period_s: float,
    pose_filter_start: np.ndarray | None = None,
    gripper_filter_start: float | None = None,
) -> PolicyPreview:
    observation, current_pose7 = build_observation(
        q_meas=q_meas,
        gripper_width_m=gripper_width_m,
        cameras=cameras,
        nero=nero,
        runtime=runtime,
        pose_frame=args.pose_frame,
        state_layout=args.state_layout,
        camera_max_age_ms=args.camera_max_age_ms,
        policy_image_size=args.policy_image_size,
    )

    action = predict_action(
        observation=observation,
        policy=policy,
        device=torch.device(str(cfg.device)),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=cfg.use_amp,
        task=args.task,
        robot_type=args.robot_type,
    )
    action_np = action.squeeze(0).detach().cpu().numpy().astype(float).reshape(-1)
    target_pose7_raw = normalize_pose7_xyzw(action_np[:POSE7_DIM])
    pose_step_start = current_pose7 if pose_filter_start is None else normalize_pose7_xyzw(pose_filter_start)
    max_translation_step_m = _positive_min(
        args.max_ee_step_m,
        args.max_cartesian_vel_m_s * policy_period_s if args.max_cartesian_vel_m_s > 0.0 else 0.0,
    )
    max_angular_step_rad = (
        args.max_angular_vel_rad_s * policy_period_s if args.max_angular_vel_rad_s > 0.0 else 0.0
    )
    target_pose7 = limit_pose7_step(
        pose_step_start,
        target_pose7_raw,
        max_translation_step_m=max_translation_step_m,
        max_angular_step_rad=max_angular_step_rad,
    )
    if args.pose_frame == "aligned":
        target_pose7_native = np.asarray(runtime.apply_command_pose_fix(target_pose7), dtype=float)
    else:
        target_pose7_native = target_pose7
    target_pose7_native = normalize_pose7_xyzw(target_pose7_native)

    q_target, ik_converged, ik_error = runtime.solve_ik(
        nero,
        target_pose7_native,
        q0=ik_seed,
        q_anchor=q_meas,
    )
    target_gripper_width_m = None
    if not args.no_gripper:
        target_gripper_width_m = float(np.clip(action_np[POSE7_DIM], args.gripper_min, args.gripper_max))
        if gripper_filter_start is not None and args.max_gripper_vel_m_s > 0.0:
            target_gripper_width_m = limit_scalar_step(
                float(gripper_filter_start),
                target_gripper_width_m,
                args.max_gripper_vel_m_s * policy_period_s,
            )

    return PolicyPreview(
        observation=observation,
        current_pose7=current_pose7,
        target_pose7_raw=target_pose7_raw,
        target_pose7=target_pose7,
        gripper_width_m=gripper_width_m,
        target_gripper_width_m=target_gripper_width_m,
        q_target=np.asarray(q_target, dtype=float),
        ik_converged=bool(ik_converged),
        ik_error=float(ik_error),
    )


def format_pose_preview(step: int, elapsed_s: float, preview: PolicyPreview) -> str:
    current_xyz = preview.current_pose7[:3]
    target_xyz = preview.target_pose7[:3]
    raw_xyz = preview.target_pose7_raw[:3]
    delta_xyz = target_xyz - current_xyz
    target_quat = preview.target_pose7[3:7]
    target_gripper = (
        float("nan") if preview.target_gripper_width_m is None else preview.target_gripper_width_m
    )
    return (
        f"step={step:06d} t={elapsed_s:8.3f}s "
        f"cur_xyz=({current_xyz[0]:+.4f},{current_xyz[1]:+.4f},{current_xyz[2]:+.4f}) "
        f"target_xyz=({target_xyz[0]:+.4f},{target_xyz[1]:+.4f},{target_xyz[2]:+.4f}) "
        f"raw_xyz=({raw_xyz[0]:+.4f},{raw_xyz[1]:+.4f},{raw_xyz[2]:+.4f}) "
        f"dxyz=({delta_xyz[0]:+.4f},{delta_xyz[1]:+.4f},{delta_xyz[2]:+.4f}) "
        f"quat_xyzw=({target_quat[0]:+.4f},{target_quat[1]:+.4f},{target_quat[2]:+.4f},{target_quat[3]:+.4f}) "
        f"grip={preview.gripper_width_m:.4f}->{target_gripper:.4f} "
        f"ik={'ok' if preview.ik_converged else 'partial'} err={preview.ik_error:.5f}"
    )


def init_preview_rerun() -> None:
    init_rerun(session_name="nero_drifting_preview")


def shutdown_preview_rerun() -> None:
    with suppress(Exception):
        shutdown_rerun()


def log_preview_rerun(
    step: int,
    preview: PolicyPreview,
    *,
    include_images: bool,
    image_every: int,
    image_max_width: int,
) -> None:
    import rerun as rr

    current_xyz = preview.current_pose7[:3]
    target_xyz = preview.target_pose7[:3]
    raw_target_xyz = preview.target_pose7_raw[:3]
    delta_xyz = target_xyz - current_xyz

    rr.set_time("step", sequence=step)
    rr.log(
        "nero/current_tcp",
        rr.Points3D([current_xyz], colors=[[0, 180, 255]], radii=[0.012]),
    )
    rr.log(
        "nero/policy_target",
        rr.Points3D([target_xyz], colors=[[255, 120, 0]], radii=[0.014]),
    )
    rr.log(
        "nero/raw_policy_target",
        rr.Points3D([raw_target_xyz], colors=[[255, 40, 40]], radii=[0.009]),
    )
    rr.log(
        "nero/current_to_target",
        rr.Arrows3D(origins=[current_xyz], vectors=[delta_xyz], colors=[[255, 180, 0]]),
    )
    rr.log("nero/scalars/target_gripper_m", rr.Scalars(preview.target_gripper_width_m or 0.0))
    rr.log("nero/scalars/ik_error", rr.Scalars(preview.ik_error))

    if include_images and should_log_rerun_image(step, image_every):
        for feature_key, image in preview.observation.items():
            if "image" in feature_key and isinstance(image, np.ndarray):
                rr.log(feature_key, rr.Image(prepare_rerun_image(image, image_max_width)).compress())


def command_control_tick(
    *,
    arm: Any,
    nero: Any,
    q_meas: np.ndarray,
    target_q: np.ndarray,
    tracking_kp: float,
    tracking_kd: float,
    max_joint_delta_rad: float,
) -> np.ndarray:
    bounded_q = q_meas + np.clip(target_q - q_meas, -max_joint_delta_rad, max_joint_delta_rad)
    gravity_ff = nero.gravity_torques_clipped(q_meas)
    for joint_index, (p_des, t_ff) in enumerate(zip(bounded_q, gravity_ff, strict=True), start=1):
        arm.move_mit(
            joint_index=joint_index,
            p_des=float(p_des),
            v_des=0.0,
            kp=float(tracking_kp),
            kd=float(tracking_kd),
            t_ff=float(t_ff),
        )
    return bounded_q


def sleep_until(next_time: float) -> float:
    now = time.perf_counter()
    sleep_s = next_time - now
    if sleep_s > 0:
        time.sleep(sleep_s)
    return time.perf_counter()


def run_control_loop(
    *,
    arm: Any,
    gripper: Any | None,
    nero: Any,
    shared: SharedTargets,
    stop_event: threading.Event,
    control_hz: float,
    gripper_hz: float,
    tracking_kp: float,
    tracking_kd: float,
    max_joint_delta_rad: float,
    max_joint_vel_rad_s: float,
    max_gripper_vel_m_s: float,
    gripper_min: float,
    gripper_max: float,
) -> None:
    control_period_s = 1.0 / control_hz
    gripper_period_s = 1.0 / gripper_hz if gripper_hz > 0 else float("inf")
    next_tick = time.perf_counter()
    next_gripper_tick = next_tick
    last_warn_at = 0.0
    last_status_at = 0.0
    command_q: np.ndarray | None = None
    command_gripper_width_m: float | None = None

    while not stop_event.is_set():
        tick_started = time.perf_counter()
        q_meas = arm.get_joint_angles_array()
        if q_meas is None:
            time.sleep(0.005)
            continue
        q_meas = np.asarray(q_meas, dtype=float)

        with shared.lock:
            target_q = shared.target_q.copy()
            gripper_width_m = shared.gripper_width_m
            gripper_force_n = shared.gripper_force_n
            last_policy_at = shared.last_policy_at

        if command_q is None:
            command_q = q_meas.copy()
        if max_joint_vel_rad_s > 0.0:
            joint_step = max_joint_vel_rad_s * control_period_s
            command_q = command_q + np.clip(target_q - command_q, -joint_step, joint_step)
        else:
            command_q = target_q

        bounded_q = command_control_tick(
            arm=arm,
            nero=nero,
            q_meas=q_meas,
            target_q=command_q,
            tracking_kp=tracking_kp,
            tracking_kd=tracking_kd,
            max_joint_delta_rad=max_joint_delta_rad,
        )

        now = time.perf_counter()
        if now - last_status_at >= 1.0:
            LOGGER.info(
                "Control tracking: shared_q_err=%.4f rad, command_q_err=%.4f rad, sent_q_err=%.4f rad, "
                "policy_age=%.2fs",
                float(np.linalg.norm(target_q - q_meas)),
                float(np.linalg.norm(command_q - q_meas)),
                float(np.linalg.norm(bounded_q - q_meas)),
                float("nan") if last_policy_at <= 0.0 else now - last_policy_at,
            )
            last_status_at = now

        if gripper is not None and gripper_width_m is not None and now >= next_gripper_tick:
            if command_gripper_width_m is None:
                command_gripper_width_m = float(np.clip(gripper_width_m, gripper_min, gripper_max))
            else:
                command_gripper_width_m = limit_scalar_step(
                    command_gripper_width_m,
                    float(np.clip(gripper_width_m, gripper_min, gripper_max)),
                    max_gripper_vel_m_s * gripper_period_s if max_gripper_vel_m_s > 0.0 else 0.0,
                )
            gripper.move_gripper_m(
                value=command_gripper_width_m,
                force=float(gripper_force_n),
            )
            next_gripper_tick = now + gripper_period_s

        next_tick += control_period_s
        now = sleep_until(next_tick)
        lateness_s = now - next_tick
        if lateness_s > control_period_s and now - last_warn_at > 1.0:
            LOGGER.warning(
                "Control loop is late by %.1f ms; AGX host may be overloaded.",
                lateness_s * 1e3,
            )
            last_warn_at = now
            next_tick = time.perf_counter()

        if tick_started > next_tick + 5.0 * control_period_s:
            next_tick = time.perf_counter()


def run_policy_loop(
    *,
    args: argparse.Namespace,
    cfg: PreTrainedConfig,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    arm: Any,
    gripper: Any | None,
    nero: Any,
    runtime: RollioNeroRuntime,
    cameras: dict[str, Camera],
    shared: SharedTargets,
    stop_event: threading.Event,
) -> None:
    policy_period_s = 1.0 / args.fps
    next_tick = time.perf_counter()
    last_ik_seed = wait_for_joint_feedback(arm)
    last_warn_at = 0.0
    step = 0
    filtered_pose7: np.ndarray | None = None
    filtered_gripper_width_m: float | None = None

    while not stop_event.is_set():
        q_meas = arm.get_joint_angles_array()
        if q_meas is None:
            time.sleep(0.01)
            continue
        q_meas = np.asarray(q_meas, dtype=float)

        with shared.lock:
            ik_seed = shared.target_q.copy()
        preview = predict_nero_policy_target(
            args=args,
            cfg=cfg,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            q_meas=q_meas,
            gripper_width_m=read_gripper_width(gripper, args.gripper_min, args.gripper_max),
            nero=nero,
            runtime=runtime,
            cameras=cameras,
            ik_seed=ik_seed if ik_seed is not None else last_ik_seed,
            policy_period_s=policy_period_s,
            pose_filter_start=filtered_pose7,
            gripper_filter_start=filtered_gripper_width_m,
        )
        last_ik_seed = preview.q_target

        now = time.perf_counter()
        if (args.max_ik_error <= 0.0) or preview.ik_converged or preview.ik_error <= args.max_ik_error:
            filtered_pose7 = preview.target_pose7
            filtered_gripper_width_m = preview.target_gripper_width_m
            with shared.lock:
                shared.target_q = preview.q_target
                shared.gripper_width_m = preview.target_gripper_width_m
                shared.last_policy_at = now
        elif now - last_warn_at > 1.0:
            LOGGER.warning(
                "Skipping policy target: IK did not converge enough (err=%.4f, converged=%s).",
                preview.ik_error,
                preview.ik_converged,
            )
            last_warn_at = now

        if args.preview_rerun:
            log_preview_rerun(
                step,
                preview,
                include_images=args.preview_rerun_images,
                image_every=args.preview_rerun_image_every,
                image_max_width=args.preview_rerun_image_max_width,
            )

        step += 1
        next_tick += policy_period_s
        now = sleep_until(next_tick)
        if now - next_tick > policy_period_s:
            LOGGER.warning("Policy loop overran its %.1f Hz budget.", args.fps)
            next_tick = time.perf_counter()


def run_preview_loop(
    *,
    args: argparse.Namespace,
    cfg: PreTrainedConfig,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    arm: Any,
    gripper: Any | None,
    nero: Any,
    runtime: RollioNeroRuntime,
    cameras: dict[str, Camera],
    stop_event: threading.Event,
) -> None:
    policy_period_s = 1.0 / args.fps
    started_at = time.perf_counter()
    next_tick = started_at
    deadline = None if args.run_seconds is None else started_at + args.run_seconds
    ik_seed = wait_for_joint_feedback(arm)
    step = 0
    filtered_pose7: np.ndarray | None = None
    filtered_gripper_width_m: float | None = None

    if args.preview_rerun:
        init_preview_rerun()

    print("Preview-only mode: motors are not enabled and no commands will be sent.", flush=True)
    while not stop_event.is_set():
        if deadline is not None and time.perf_counter() >= deadline:
            break

        q_meas = arm.get_joint_angles_array()
        if q_meas is None:
            time.sleep(0.01)
            continue
        q_meas = np.asarray(q_meas, dtype=float)

        preview = predict_nero_policy_target(
            args=args,
            cfg=cfg,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            q_meas=q_meas,
            gripper_width_m=read_gripper_width(gripper, args.gripper_min, args.gripper_max),
            nero=nero,
            runtime=runtime,
            cameras=cameras,
            ik_seed=ik_seed,
            policy_period_s=policy_period_s,
            pose_filter_start=filtered_pose7,
            gripper_filter_start=filtered_gripper_width_m,
        )
        ik_seed = preview.q_target
        filtered_pose7 = preview.target_pose7
        filtered_gripper_width_m = preview.target_gripper_width_m

        if step % max(1, args.preview_print_every) == 0:
            print(format_pose_preview(step, time.perf_counter() - started_at, preview), flush=True)
        if args.preview_rerun:
            log_preview_rerun(
                step,
                preview,
                include_images=args.preview_rerun_images,
                image_every=args.preview_rerun_image_every,
                image_max_width=args.preview_rerun_image_max_width,
            )

        step += 1
        next_tick += policy_period_s
        now = sleep_until(next_tick)
        if now - next_tick > policy_period_s:
            LOGGER.warning("Preview loop overran its %.1f Hz budget.", args.fps)
            next_tick = time.perf_counter()


def home_on_exit(
    *,
    arm: Any,
    nero: Any,
    target_q: np.ndarray,
    control_hz: float,
    tracking_kp: float,
    tracking_kd: float,
    duration_s: float,
    settle_s: float,
) -> None:
    q_start = wait_for_joint_feedback(arm, timeout_s=1.0)
    q_end = np.asarray(target_q, dtype=float)
    period_s = 1.0 / control_hz
    started_at = time.perf_counter()
    finished_at = started_at + max(0.0, duration_s) + max(0.0, settle_s)
    next_tick = started_at

    LOGGER.info("Homing Nero arm to disabled hold pose.")
    while time.perf_counter() < finished_at:
        now = time.perf_counter()
        alpha = 1.0 if duration_s <= 0 else min(1.0, max(0.0, (now - started_at) / duration_s))
        desired_q = q_start + (q_end - q_start) * alpha
        desired_v = np.zeros_like(desired_q) if alpha >= 1.0 else (q_end - q_start) / duration_s
        q_meas = arm.get_joint_angles_array()
        if q_meas is None:
            time.sleep(0.01)
            continue
        q_meas = np.asarray(q_meas, dtype=float)
        gravity_ff = nero.gravity_torques_clipped(q_meas)
        for joint_index, (p_des, v_des, t_ff) in enumerate(
            zip(desired_q, desired_v, gravity_ff, strict=True),
            start=1,
        ):
            arm.move_mit(
                joint_index=joint_index,
                p_des=float(p_des),
                v_des=float(v_des),
                kp=float(tracking_kp),
                kd=float(tracking_kd),
                t_ff=float(t_ff),
            )
        next_tick += period_s
        sleep_until(next_tick)


def thread_entry(
    name: str,
    fn: Any,
    failures: queue.SimpleQueue[tuple[str, BaseException]],
    stop_event: threading.Event,
    **kwargs: Any,
) -> None:
    try:
        fn(stop_event=stop_event, **kwargs)
    except BaseException as exc:
        failures.put((name, exc))
        stop_event.set()
        LOGGER.exception("%s thread failed.", name)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s"
    )
    apply_trial_control_defaults(args)

    if args.policy_path is None:
        raise ValueError("--policy-path is required.")

    image_specs = [parse_image_spec(value) for value in args.image]
    camera_configs = make_camera_configs(image_specs)

    if args.inspect_policy:
        cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        log_policy_contract(cfg)
        return

    LOGGER.info("Loading policy from %s", args.policy_path)
    cfg, policy, preprocessor, postprocessor = load_policy(args.policy_path, args.device)
    validate_policy_features(cfg, set(camera_configs), args.no_gripper)
    log_policy_contract(cfg)
    if args.policy_image_size is not None:
        width, height = args.policy_image_size
        LOGGER.info(
            "Resizing camera observations before policy inference to %sx%s (WIDTHxHEIGHT).", width, height
        )
    LOGGER.info("Loaded %s policy on %s", cfg.type, cfg.device)

    if args.dry_run:
        LOGGER.info("Dry run complete; hardware and cameras were not opened.")
        return

    if args.camera_only:
        cameras = connect_cameras(camera_configs)
        try:
            run_camera_only_loop(
                args=args,
                cfg=cfg,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                cameras=cameras,
            )
        finally:
            disconnect_cameras(cameras)
        return

    runtime = import_rollio_nero()
    cameras: dict[str, Camera] = {}
    robot: Any | None = None
    stop_event = threading.Event()
    failures: queue.SimpleQueue[tuple[str, BaseException]] = queue.SimpleQueue()
    threads: list[threading.Thread] = []

    def stop_from_signal(signum: int, _frame: Any) -> None:
        LOGGER.info("Received signal %s; stopping.", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, stop_from_signal)
    signal.signal(signal.SIGTERM, stop_from_signal)

    try:
        cameras = connect_cameras(camera_configs)

        LOGGER.info("Connecting AGX Nero on %s", args.interface)
        robot = runtime.create_robot(args.interface)
        if args.preview_only:
            LOGGER.info("Preview-only mode: skipping robot enable and all command threads.")
        else:
            enabled = runtime.enable_robot(robot)
            if not enabled:
                LOGGER.warning("AGX Nero enable() did not report success; continuing in MIT mode.")

        arm = runtime.AgxArmBackend(robot)
        raw_gripper = None if args.no_gripper else runtime.init_gripper(robot)
        gripper = None if raw_gripper is None else runtime.AgxGripperBackend(raw_gripper)
        nero = runtime.NeroModel(with_gripper=not args.no_gripper)

        if args.preview_only:
            run_preview_loop(
                args=args,
                cfg=cfg,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                arm=arm,
                gripper=gripper,
                nero=nero,
                runtime=runtime,
                cameras=cameras,
                stop_event=stop_event,
            )
            return

        initial_q = wait_for_joint_feedback(arm)
        shared = SharedTargets(
            target_q=initial_q.copy(),
            gripper_width_m=None,
            gripper_force_n=float(args.gripper_force),
        )

        if args.preview_rerun:
            init_preview_rerun()
            LOGGER.info("Rerun visualization enabled for deployment loop.")

        threads = [
            threading.Thread(
                target=thread_entry,
                name="nero-control",
                kwargs={
                    "name": "control",
                    "fn": run_control_loop,
                    "failures": failures,
                    "stop_event": stop_event,
                    "arm": arm,
                    "gripper": gripper,
                    "nero": nero,
                    "shared": shared,
                    "control_hz": args.control_hz,
                    "gripper_hz": args.gripper_hz,
                    "tracking_kp": args.tracking_kp,
                    "tracking_kd": args.tracking_kd,
                    "max_joint_delta_rad": args.max_joint_delta_rad,
                    "max_joint_vel_rad_s": args.max_joint_vel_rad_s,
                    "max_gripper_vel_m_s": args.max_gripper_vel_m_s,
                    "gripper_min": args.gripper_min,
                    "gripper_max": args.gripper_max,
                },
                daemon=True,
            ),
            threading.Thread(
                target=thread_entry,
                name="nero-policy",
                kwargs={
                    "name": "policy",
                    "fn": run_policy_loop,
                    "failures": failures,
                    "stop_event": stop_event,
                    "args": args,
                    "cfg": cfg,
                    "policy": policy,
                    "preprocessor": preprocessor,
                    "postprocessor": postprocessor,
                    "arm": arm,
                    "gripper": gripper,
                    "nero": nero,
                    "runtime": runtime,
                    "cameras": cameras,
                    "shared": shared,
                },
                daemon=True,
            ),
        ]

        for thread in threads:
            thread.start()

        deadline = None if args.run_seconds is None else time.perf_counter() + args.run_seconds
        LOGGER.info("Deployment loop started. Press Ctrl-C to stop.")
        while not stop_event.is_set():
            if not failures.empty():
                break
            if deadline is not None and time.perf_counter() >= deadline:
                LOGGER.info("run-seconds elapsed; stopping.")
                stop_event.set()
                break
            time.sleep(0.1)

        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)

        if not failures.empty():
            thread_name, exc = failures.get()
            raise RuntimeError(f"{thread_name} thread failed") from exc

        if args.home_on_exit:
            home_on_exit(
                arm=arm,
                nero=nero,
                target_q=runtime.disabled_hold_q,
                control_hz=args.control_hz,
                tracking_kp=args.tracking_kp,
                tracking_kd=args.tracking_kd,
                duration_s=args.home_duration_s,
                settle_s=args.home_settle_s,
            )
    finally:
        stop_event.set()
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        if args.preview_rerun:
            shutdown_preview_rerun()
        disconnect_cameras(cameras)
        if robot is not None:
            LOGGER.info("Disconnecting AGX Nero.")
            with suppress(Exception):
                robot.disconnect()


if __name__ == "__main__":
    main()
