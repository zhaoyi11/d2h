"""Direct frozen RSL-RL low-level policy loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import torch
from torch import Tensor, nn


DEFAULT_DIRECT_RSL_RL_CHECKPOINT = (
    Path(__file__).resolve().parents[3] / "logs" / "rsl_rl" / "anyreorient" / "model_14999.pt"
)

_ACTOR_WEIGHT_RE = re.compile(r"^actor\.(\d+)\.weight$")


def _torch_load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint to be a dict, got {type(checkpoint).__name__}.")
    return checkpoint


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported RSL-RL actor activation: {name!r}.")


def _actor_layer_indices(state_dict: dict[str, Tensor]) -> list[int]:
    indices = sorted(
        int(match.group(1))
        for key in state_dict
        if (match := _ACTOR_WEIGHT_RE.match(key)) is not None
    )
    if not indices:
        raise KeyError("Checkpoint model_state_dict does not contain actor.*.weight tensors.")
    return indices


def _build_actor(state_dict: dict[str, Tensor], activation: str) -> nn.Sequential:
    indices = _actor_layer_indices(state_dict)
    layers: list[nn.Module] = []
    for layer_idx in indices:
        weight_key = f"actor.{layer_idx}.weight"
        bias_key = f"actor.{layer_idx}.bias"
        if bias_key not in state_dict:
            raise KeyError(f"Checkpoint model_state_dict is missing {bias_key!r}.")

        weight = state_dict[weight_key]
        if weight.ndim != 2:
            raise ValueError(f"Expected {weight_key} to be 2D, got shape {tuple(weight.shape)}.")
        layers.append(nn.Linear(int(weight.shape[1]), int(weight.shape[0])))
        if layer_idx != indices[-1]:
            layers.append(_activation(activation))

    actor = nn.Sequential(*layers)
    actor_state = {key.removeprefix("actor."): value for key, value in state_dict.items() if key.startswith("actor.")}
    actor.load_state_dict(actor_state, strict=True)
    return actor


def _normalizer_tensors(state_dict: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    mean_key = "actor_obs_normalizer._mean"
    std_key = "actor_obs_normalizer._std"
    var_key = "actor_obs_normalizer._var"
    if mean_key not in state_dict:
        raise KeyError(f"Checkpoint model_state_dict is missing {mean_key!r}.")
    if std_key in state_dict:
        obs_std = state_dict[std_key]
    elif var_key in state_dict:
        obs_std = torch.sqrt(torch.clamp(state_dict[var_key], min=0.0))
    else:
        raise KeyError(f"Checkpoint model_state_dict is missing {std_key!r} or {var_key!r}.")

    obs_mean = state_dict[mean_key].reshape(1, -1)
    obs_std = obs_std.reshape(1, -1)
    if obs_mean.shape != obs_std.shape:
        raise ValueError(
            f"Actor observation normalizer mean/std shapes differ: {tuple(obs_mean.shape)} vs {tuple(obs_std.shape)}."
        )
    return obs_mean, obs_std


@dataclass
class DirectRslRlLowLevelPolicy:
    """Inference wrapper around a frozen RSL-RL actor."""

    model: nn.Sequential
    obs_mean: Tensor
    obs_std: Tensor
    eps: float = 0.01

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def obs_dim(self) -> int:
        return int(self.obs_mean.shape[-1])

    @property
    def action_dim(self) -> int:
        final_linear = next(module for module in reversed(self.model) if isinstance(module, nn.Linear))
        return int(final_linear.out_features)

    @torch.inference_mode()
    def act(self, obs: Tensor) -> Tensor:
        """Return deterministic hand actions from low-level observations."""
        obs = obs.to(device=self.device, dtype=self.obs_mean.dtype)
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected direct RSL-RL obs dim {self.obs_dim}, got {obs.shape[-1]}.")
        return self.model((obs - self.obs_mean) / (self.obs_std + self.eps))


def load_direct_rsl_rl_policy(
    checkpoint_path: str | Path = DEFAULT_DIRECT_RSL_RL_CHECKPOINT,
    device: torch.device | str = "cpu",
    expected_obs_dim: int | None = None,
    expected_action_dim: int | None = None,
    activation: str = "elu",
    normalizer_eps: float = 0.01,
) -> DirectRslRlLowLevelPolicy:
    """Load a frozen deterministic RSL-RL actor from an OnPolicyRunner checkpoint."""
    path = Path(checkpoint_path).expanduser()
    checkpoint = _torch_load_checkpoint(path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {path}")

    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected model_state_dict to be a dict, got {type(state_dict).__name__}.")

    model = _build_actor(state_dict, activation=activation).to(device)
    obs_mean, obs_std = _normalizer_tensors(state_dict)
    obs_mean = obs_mean.to(device=device, dtype=next(model.parameters()).dtype)
    obs_std = obs_std.to(device=device, dtype=next(model.parameters()).dtype)

    policy = DirectRslRlLowLevelPolicy(model=model, obs_mean=obs_mean, obs_std=obs_std, eps=normalizer_eps)
    if expected_obs_dim is not None and int(expected_obs_dim) != policy.obs_dim:
        raise ValueError(f"Expected direct RSL-RL obs dim {expected_obs_dim}, checkpoint has {policy.obs_dim}.")
    if expected_action_dim is not None and int(expected_action_dim) != policy.action_dim:
        raise ValueError(f"Expected direct RSL-RL action dim {expected_action_dim}, checkpoint has {policy.action_dim}.")

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return policy
