"""Frozen low-level VAE policy utilities for hierarchical RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch import Tensor

from src.policy.vae import ConditionalTrajectoryVAE, MODEL_TYPE, VAEConfig

DEFAULT_LOW_LEVEL_RSL_RL_CHECKPOINT = Path("/home/yizhao/yi/D2H/logs/rsl_rl/anyreorient/model_14999.pt")


def _torch_load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class LowLevelVAEPolicy:
    """Inference wrapper around a frozen conditional trajectory VAE."""

    model: ConditionalTrajectoryVAE
    config: VAEConfig

    @property
    def state_dim(self) -> int:
        return int(self.config.state_dim)

    @property
    def action_dim(self) -> int:
        return int(self.config.action_dim)

    @property
    def latent_dim(self) -> int:
        return int(self.config.latent_dim)

    @property
    def chunk_length(self) -> int:
        return int(self.config.future_length)

    @torch.inference_mode()
    def decode(self, context: Tensor, z: Tensor) -> Tensor:
        """Decode a latent into a low-level hand-action chunk."""
        if context.shape[-1] != self.config.context_dim:
            raise ValueError(f"Expected VAE context dim {self.config.context_dim}, got {context.shape[-1]}.")
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {z.shape[-1]}.")
        condition = self.model.context_encoder(context)
        return self.model.decode(condition, z)


@dataclass
class LowLevelRslRlPolicy:
    """Inference wrapper around a frozen RSL-RL actor checkpoint."""

    model: nn.Module
    obs_mean: Tensor | None
    obs_std: Tensor | None
    obs_dim: int
    action_dim: int
    normalization_eps: float = 1e-2

    @torch.inference_mode()
    def act(self, obs: Tensor) -> Tensor:
        """Compute one low-level hand action from the reorient policy observation."""
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected RSL-RL observation dim {self.obs_dim}, got {obs.shape[-1]}.")
        param = next(self.model.parameters())
        obs = obs.to(device=param.device, dtype=param.dtype)
        if self.obs_mean is not None and self.obs_std is not None:
            obs = (obs - self.obs_mean) / (self.obs_std + self.normalization_eps)
        return self.model(obs)


def load_low_level_vae(
    checkpoint_path: str | Path,
    device: torch.device | str,
    expected_action_dim: int | None = None,
) -> LowLevelVAEPolicy:
    """Load a frozen VAE checkpoint for low-level hand-action decoding."""
    path = Path(checkpoint_path).expanduser()
    checkpoint = _torch_load_checkpoint(path, map_location=device)
    model_type = checkpoint.get("model_type")
    if model_type != MODEL_TYPE:
        raise ValueError(f"Expected checkpoint model_type {MODEL_TYPE!r}, got {model_type!r}.")
    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint is missing VAE config: {path}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {path}")

    config = VAEConfig(**checkpoint["config"])
    if expected_action_dim is not None and int(expected_action_dim) != int(config.action_dim):
        raise ValueError(f"Expected low-level action dim {expected_action_dim}, checkpoint has {config.action_dim}.")

    model = ConditionalTrajectoryVAE(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return LowLevelVAEPolicy(model=model, config=config)


def _build_rsl_rl_actor_from_state_dict(state_dict: dict[str, Tensor]) -> nn.Sequential:
    weight_items = []
    for key, value in state_dict.items():
        if key.startswith("actor.") and key.endswith(".weight"):
            layer_idx = int(key.split(".")[1])
            weight_items.append((layer_idx, value))
    if not weight_items:
        raise KeyError("Checkpoint is missing actor weights.")

    layers: list[nn.Module] = []
    for idx, (_, weight) in enumerate(sorted(weight_items)):
        in_features = int(weight.shape[1])
        out_features = int(weight.shape[0])
        layers.append(nn.Linear(in_features, out_features))
        if idx < len(weight_items) - 1:
            layers.append(nn.ELU())
    model = nn.Sequential(*layers)
    actor_state_dict = {key.removeprefix("actor."): value for key, value in state_dict.items() if key.startswith("actor.")}
    model.load_state_dict(actor_state_dict)
    return model


def load_low_level_rsl_rl_policy(
    checkpoint_path: str | Path = DEFAULT_LOW_LEVEL_RSL_RL_CHECKPOINT,
    device: torch.device | str = "cpu",
    expected_obs_dim: int | None = None,
    expected_action_dim: int | None = None,
) -> LowLevelRslRlPolicy:
    """Load a frozen RSL-RL actor checkpoint for one-step low-level hand control."""
    path = Path(checkpoint_path).expanduser()
    checkpoint = _torch_load_checkpoint(path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {path}")

    state_dict = checkpoint["model_state_dict"]
    actor = _build_rsl_rl_actor_from_state_dict(state_dict).to(device)
    actor.eval()
    for param in actor.parameters():
        param.requires_grad_(False)

    first_linear = next(module for module in actor.modules() if isinstance(module, nn.Linear))
    last_linear = next(module for module in reversed(list(actor.modules())) if isinstance(module, nn.Linear))
    obs_dim = int(first_linear.in_features)
    action_dim = int(last_linear.out_features)
    if expected_obs_dim is not None and int(expected_obs_dim) != obs_dim:
        raise ValueError(f"Expected low-level observation dim {expected_obs_dim}, checkpoint has {obs_dim}.")
    if expected_action_dim is not None and int(expected_action_dim) != action_dim:
        raise ValueError(f"Expected low-level action dim {expected_action_dim}, checkpoint has {action_dim}.")

    obs_mean = state_dict.get("actor_obs_normalizer._mean")
    obs_std = state_dict.get("actor_obs_normalizer._std")
    if obs_mean is not None:
        obs_mean = obs_mean.to(device=device)
    if obs_std is not None:
        obs_std = obs_std.to(device=device)

    return LowLevelRslRlPolicy(
        model=actor,
        obs_mean=obs_mean,
        obs_std=obs_std,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
