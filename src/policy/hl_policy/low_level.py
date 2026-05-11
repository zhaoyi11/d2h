"""Frozen low-level VAE policy utilities for hierarchical RL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.policy.vae import ConditionalTrajectoryVAE, MODEL_TYPE, VAEConfig
from src.policy.goal_conditioned import (
    GoalConditionedChunkConfig,
    GoalConditionedChunkPolicy,
    MODEL_TYPE as GOAL_CONDITIONED_MODEL_TYPE,
)


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
class LowLevelGoalConditionedPolicy:
    """Inference wrapper around a frozen goal-conditioned action-chunk policy."""

    model: GoalConditionedChunkPolicy
    config: GoalConditionedChunkConfig

    @property
    def state_dim(self) -> int:
        return int(self.config.state_dim)

    @property
    def action_dim(self) -> int:
        return int(self.config.action_dim)

    @property
    def goal_dim(self) -> int:
        return int(self.config.goal_dim)

    @property
    def chunk_length(self) -> int:
        return int(self.config.future_length)

    @torch.inference_mode()
    def predict(self, context: Tensor, goal: Tensor) -> Tensor:
        """Predict a low-level hand-action chunk from context and goal error."""
        return self.model(context, goal)


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


def load_low_level_goal_conditioned(
    checkpoint_path: str | Path,
    device: torch.device | str,
    expected_action_dim: int | None = None,
) -> LowLevelGoalConditionedPolicy:
    """Load a frozen goal-conditioned low-level hand-action policy checkpoint."""
    path = Path(checkpoint_path).expanduser()
    checkpoint = _torch_load_checkpoint(path, map_location=device)
    model_type = checkpoint.get("model_type")
    if model_type != GOAL_CONDITIONED_MODEL_TYPE:
        raise ValueError(f"Expected checkpoint model_type {GOAL_CONDITIONED_MODEL_TYPE!r}, got {model_type!r}.")
    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint is missing goal-conditioned policy config: {path}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {path}")

    config = GoalConditionedChunkConfig(**checkpoint["config"])
    if expected_action_dim is not None and int(expected_action_dim) != int(config.action_dim):
        raise ValueError(f"Expected low-level action dim {expected_action_dim}, checkpoint has {config.action_dim}.")

    model = GoalConditionedChunkPolicy(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return LowLevelGoalConditionedPolicy(model=model, config=config)
