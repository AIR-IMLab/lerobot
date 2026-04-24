#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.transforms import ImageTransformsConfig
from lerobot.utils.import_utils import get_safe_default_codec


@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
    repo_id: str
    # Root directory for a concrete local dataset tree (e.g. 'dataset/path'). If None, local datasets are
    # looked up under $HF_LEROBOT_HOME/repo_id and Hub downloads use a revision-safe cache under $HF_LEROBOT_HOME/hub.
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False

    def __post_init__(self) -> None:
        if self.episodes is not None:
            if any(ep < 0 for ep in self.episodes):
                raise ValueError(
                    f"Episode indices must be non-negative, got: {[ep for ep in self.episodes if ep < 0]}"
                )
            if len(self.episodes) != len(set(self.episodes)):
                duplicates = sorted({ep for ep in self.episodes if self.episodes.count(ep) > 1})
                raise ValueError(f"Episode indices contain duplicates: {duplicates}")


@dataclass
class WandBConfig:
    enable: bool = False
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'
    add_tags: bool = True  # If True, save configuration as tags in the WandB run.


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    # Set to 0 for auto-tuning based on available CPU cores and n_episodes.
    batch_size: int = 0
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    # Defaults to True; automatically downgraded to SyncVectorEnv when batch_size=1.
    use_async_envs: bool = True
    # Async policy evaluation mode. "none" calls policy.select_action directly.
    # "local" uses the action queue on the caller thread; "background" runs chunk prediction on a worker thread.
    async_policy: str = "none"
    # Number of actions to keep from each predicted chunk. Set to 0 to infer from policy.config.n_action_steps.
    async_actions_per_chunk: int = 0
    async_chunk_size_threshold: float = 0.5
    async_aggregate_fn_name: str = "latest_only"
    async_bootstrap_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.batch_size == 0:
            self.batch_size = self._auto_batch_size()
        if self.batch_size > self.n_episodes:
            self.batch_size = self.n_episodes
        self.async_policy = self._normalize_async_policy(self.async_policy)
        if self.async_actions_per_chunk < 0:
            raise ValueError(
                f"async_actions_per_chunk must be non-negative, got {self.async_actions_per_chunk}"
            )
        if not 0.0 <= self.async_chunk_size_threshold <= 1.0:
            raise ValueError(
                "async_chunk_size_threshold must be in [0, 1], "
                f"got {self.async_chunk_size_threshold}"
            )
        if self.async_bootstrap_timeout_s <= 0:
            raise ValueError(
                f"async_bootstrap_timeout_s must be positive, got {self.async_bootstrap_timeout_s}"
            )

    def _normalize_async_policy(self, mode: str) -> str:
        normalized = mode.lower().replace("-", "_")
        aliases = {
            "": "none",
            "off": "none",
            "false": "none",
            "sync": "none",
            "backgroundasync": "background",
            "background_async": "background",
        }
        normalized = aliases.get(normalized, normalized)
        valid_modes = {"none", "local", "background"}
        if normalized not in valid_modes:
            raise ValueError(f"async_policy must be one of {sorted(valid_modes)}, got {mode!r}")
        return normalized

    def _auto_batch_size(self) -> int:
        """Pick batch_size based on CPU cores, capped by n_episodes."""
        import math
        import os

        cpu_cores = os.cpu_count() or 4
        # Each async env worker needs ~1 core; leave headroom for main process + inference.
        by_cpu = max(1, math.floor(cpu_cores * 0.7))
        return min(by_cpu, self.n_episodes, 64)


@dataclass
class PeftConfig:
    # PEFT offers many fine-tuning methods, layer adapters being the most common and currently also the most
    # effective methods so we'll focus on those in this high-level config interface.

    # Either a string (module name suffix or 'all-linear'), a list of module name suffixes or a regular expression
    # describing module names to target with the configured PEFT method. Some policies have a default value for this
    # so that you don't *have* to choose which layers to adapt but it might still be worthwhile depending on your case.
    target_modules: list[str] | str | None = None

    # Names/suffixes of modules to fully fine-tune and store alongside adapter weights. Useful for layers that are
    # not part of a pre-trained model (e.g., action state projections). Depending on the policy this defaults to layers
    # that are newly created in pre-trained policies. If you're fine-tuning an already trained policy you might want
    # to set this to `[]`. Corresponds to PEFT's `modules_to_save`.
    full_training_modules: list[str] | None = None

    # The PEFT (adapter) method to apply to the policy. Needs to be a valid PEFT type.
    method_type: str = "LORA"

    # Adapter initialization method. Look at the specific PEFT adapter documentation for defaults.
    init_type: str | None = None

    # We expect that all PEFT adapters are in some way doing rank-decomposition therefore this parameter specifies
    # the rank used for the adapter. In general a higher rank means more trainable parameters and closer to full
    # fine-tuning.
    r: int = 16
