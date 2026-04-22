"""Drifting Policy.

Single-step (t=0) action-chunk generator trained with the drifting force-matching
loss. Reuses LeRobot's diffusion vision encoder and conditional UNet1D backbone.
"""

from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from ..diffusion.modeling_diffusion import DiffusionConditionalUnet1d, DiffusionRgbEncoder
from ..pretrained import PreTrainedPolicy
from ..utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from .configuration_drifting import DriftingConfig


def _cdist(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Pairwise L2 distance: [B, N, D] x [B, M, D] -> [B, N, M]."""
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def drift_loss(
    gen: Tensor,
    fixed_pos: Tensor,
    fixed_neg: Tensor | None = None,
    weight_gen: Tensor | None = None,
    weight_pos: Tensor | None = None,
    weight_neg: Tensor | None = None,
    R_list: tuple[float, ...] = (0.02, 0.05, 0.2),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Drifting loss (port of phamtrongthang123/drifting_policy drift_loss, JAX -> PyTorch).

    gen: [B, C_g, S], fixed_pos: [B, C_p, S], fixed_neg: [B, C_n, S] (optional).
    Returns per-batch loss [B] and a dict of scalar diagnostics.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = gen.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = gen.new_ones(B, C_n)

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    with torch.no_grad():
        info: dict[str, Tensor] = {}
        dist = _cdist(old_gen, targets)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean() / targets_w.mean()
        info["scale"] = scale

        scale_inputs = torch.clamp(scale / (S**0.5), min=1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs

        dist_normed = dist / torch.clamp(scale, min=1e-3)

        # Mask self-connections in the gen block.
        diag_mask = torch.eye(C_g, device=gen.device, dtype=gen.dtype)
        block_mask = F.pad(diag_mask, (0, C_n + C_p)).unsqueeze(0)
        dist_normed = dist_normed + block_mask * 100.0

        force_across_R = torch.zeros_like(old_gen_scaled)
        for R in R_list:
            logits = -dist_normed / R
            affinity = torch.softmax(logits, dim=-1)
            aff_t = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * aff_t, min=1e-6))
            affinity = affinity * targets_w[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg

            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)
            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
            total_coeffs = R_coeff.sum(dim=-1)
            total_force_R = total_force_R - total_coeffs.unsqueeze(-1) * old_gen_scaled

            f_norm_val = (total_force_R**2).mean()
            info[f"loss_{R}"] = f_norm_val
            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R

    gen_scaled = gen / scale_inputs.detach()
    diff = gen_scaled - goal_scaled.detach()
    loss = (diff**2).mean(dim=(-1, -2))
    info = {k: v.mean() for k, v in info.items()}
    return loss, info


class DriftingPolicy(PreTrainedPolicy):
    """Drifting policy: single-step generator over action chunks."""

    config_class = DriftingConfig
    name = "drifting"

    def __init__(self, config: DriftingConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.drifting = DriftingModel(config)
        self.reset()

    def get_optim_params(self):
        return self.drifting.parameters()

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    def _prepare_batch_for_queue(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Normalize an incoming observation batch so it can feed the obs queues.

        - Shallow-copies so callers aren't mutated.
        - Drops ACTION if present (offline eval batches include it).
        - Stacks per-camera image keys into OBS_IMAGES along a new camera dim.
        """
        batch = dict(batch)
        batch.pop(ACTION, None)
        if self.config.image_features:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return batch

    def _stack_from_queues(self) -> dict[str, Tensor]:
        return {k: torch.stack(list(self._queues[k]), dim=1) for k in self._queues if k != ACTION}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Self-contained action-chunk prediction.

        Accepts a cold batch (no prior `select_action` call required): stacks the
        camera keys, populates the observation queues, then runs single-step
        generation. Safe to call from the async PolicyServer path.
        """
        batch = self._prepare_batch_for_queue(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self.drifting.generate_actions(self._stack_from_queues(), noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = self._prepare_batch_for_queue(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self.drifting.generate_actions(self._stack_from_queues(), noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, None]:
        """Forward pass.

        reduction:
          - "mean" (default): scalar loss for standard training.
          - "none": per-sample loss tensor of shape [B] for the RA-BC weighting
            path used by `lerobot_train.py`.
        """
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss, info = self.drifting.compute_loss(batch, reduction=reduction)
        output_dict = {f"drift_{k}": float(v.detach()) for k, v in info.items()}
        return loss, output_dict


class DriftingModel(nn.Module):
    def __init__(self, config: DriftingConfig):
        super().__init__()
        self.config = config

        global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        # DiffusionConditionalUnet1d only reads attributes also defined on DriftingConfig.
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        feats = [batch[OBS_STATE]]
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [enc(im) for enc, im in zip(self.rgb_encoder, images_per_camera, strict=True)]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            feats.append(img_features)
        if self.config.env_state_feature:
            feats.append(batch[OBS_ENV_STATE])
        return torch.cat(feats, dim=-1).flatten(start_dim=1)

    def _single_step_predict(self, noise: Tensor, global_cond: Tensor) -> Tensor:
        timesteps = torch.zeros(noise.shape[0], dtype=torch.long, device=noise.device)
        return self.unet(noise, timesteps, global_cond=global_cond)

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        if noise is None:
            noise = torch.randn(
                batch_size, self.config.horizon, self.config.action_feature.shape[0],
                device=device, dtype=dtype,
            )
        sample = self._single_step_predict(noise, global_cond)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return sample[:, start:end]

    def compute_loss(
        self, batch: dict[str, Tensor], reduction: str = "mean"
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)  # [B, cond]
        actions = batch[ACTION]  # [B, T, D]
        B, T, D = actions.shape
        G = self.config.gen_per_label

        global_cond_rep = global_cond.repeat_interleave(G, dim=0)  # [B*G, cond]
        noise = torch.randn(B * G, T, D, device=actions.device, dtype=actions.dtype)
        pred = self._single_step_predict(noise, global_cond_rep)  # [B*G, T, D]
        pred = pred.reshape(B, G, T, D)

        R_list = tuple(self.config.temperatures)
        pad_mask = None
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError("`action_is_pad` required when do_mask_loss_for_padding=True")
            pad_mask = (~batch["action_is_pad"]).to(actions.dtype)  # [B, T]

        info_out: dict[str, Tensor] = {}
        if self.config.per_timestep_loss:
            # Per-timestep mode: drift loss is computed independently at each step
            # and exactly zeroed at padded timesteps. Returns [B] when reduction="none"
            # by accumulating per-sample losses across valid timesteps.
            per_sample = actions.new_zeros(B)
            acc_info: dict[str, Tensor] = {}
            for t in range(T):
                gen_t = pred[:, :, t, :]                      # [B, G, D]
                pos_t = actions[:, t, :].unsqueeze(1)         # [B, 1, D]
                loss_t, info_t = drift_loss(gen_t, pos_t, R_list=R_list)  # [B]
                if pad_mask is not None:
                    loss_t = loss_t * pad_mask[:, t]
                per_sample = per_sample + loss_t
                for k, v in info_t.items():
                    acc_info[k] = acc_info.get(k, 0.0) + v / T
            per_sample = per_sample / T
            info_out = acc_info
        else:
            # Flattened mode: drift loss operates on the whole [T*D] chunk, so it
            # cannot be exactly per-step masked. We approximate by weighting the
            # per-sample loss by the fraction of valid (non-padded) steps. This
            # is a LeRobot-specific adaptation; the upstream drifting reference
            # does not implement padding masks at all.
            gen = pred.reshape(B, G, T * D)
            pos = actions.reshape(B, 1, T * D)
            per_sample, info_out = drift_loss(gen, pos, R_list=R_list)  # [B]
            if pad_mask is not None:
                per_sample = per_sample * pad_mask.mean(dim=1)

        if reduction == "none":
            return per_sample, info_out
        if reduction == "mean":
            return per_sample.mean(), info_out
        if reduction == "sum":
            return per_sample.sum(), info_out
        raise ValueError(f"Unsupported reduction: {reduction}")
