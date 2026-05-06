"""DiT-based trajectory autoencoder with SIGReg latent regularization."""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import tyro
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, get_worker_info

from src.policy.diffusion_autoencoder.fm import FlowMatchingObjective
from src.policy.diffusion_autoencoder.transformer import SinusoidalPosEmb, modulate
from src.policy.vae import STATE_KEYS, _move_batch
from src.policy.vae_sigreg import SIGReg


MODEL_TYPE = "conditional_trajectory_diffusion_autoencoder"
DEFAULT_DATASET_DIR = Path("/home/yizhao/yi/D2H/datasets/Reorient_Play-v0/20260504_001812")
DEFAULT_WANDB_PROJECT = "d2h-diffusion-autoencoder"


@dataclass
class DiffusionAutoencoderConfigArgs:
    past_length: int = 4
    future_length: int = 8
    latent_dim: int = 64
    condition_dim: int = 256
    hidden_dim: int = 1024
    transformer_hidden_dim: int = 512
    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_feedforward_dim: int = 1024
    transformer_dropout: float = 0.1
    timestep_embed_dim: int = 128
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    sigma_min: float = 1e-4
    num_sample_steps: int = 20
    timestep_scale: float = 1000.0
    integration_method: Literal["euler", "heun", "rk4"] = "euler"

    def build(self, state_dim: int, action_dim: int) -> DiffusionAutoencoderConfig:
        return DiffusionAutoencoderConfig(
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            **asdict(self),
        )


@dataclass(kw_only=True)
class DiffusionAutoencoderConfig(DiffusionAutoencoderConfigArgs):
    state_dim: int
    action_dim: int

    @property
    def token_dim(self) -> int:
        return self.state_dim + self.action_dim

    @property
    def context_dim(self) -> int:
        return self.token_dim

    @property
    def encoder_input_dim(self) -> int:
        return self.encoder_length * self.token_dim

    @property
    def context_input_dim(self) -> int:
        return self.past_length * self.context_dim

    @property
    def target_length(self) -> int:
        return self.future_length

    @property
    def encoder_length(self) -> int:
        return self.past_length + self.future_length


@dataclass
class DiffusionAutoencoderTrainArgs:
    dataset_dir: str = str(DEFAULT_DATASET_DIR)
    output_dir: str | None = None
    diffusion_autoencoder_config: DiffusionAutoencoderConfigArgs = field(
        default_factory=DiffusionAutoencoderConfigArgs
    )
    batch_size: int = 256
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float | None = None
    num_workers: int = 0
    val_ratio: float = 0.1
    train_windows_per_epoch: int = 100_000
    val_windows: int = 10_000
    log_every_steps: int = 100
    checkpoint_every_epochs: int = 10
    resume: str | None = None
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_run_name: str | None = None
    amp: bool = False
    seed: int = 0
    device: str = "cuda"


###### Transformer Blocks ######
def _transformer_encoder(config: DiffusionAutoencoderConfig) -> nn.TransformerEncoder:
    if config.hidden_dim % config.transformer_heads != 0:
        raise ValueError("hidden_dim must be divisible by transformer_heads")
    layer = nn.TransformerEncoderLayer(
        d_model=config.hidden_dim,
        nhead=config.transformer_heads,
        dim_feedforward=config.transformer_feedforward_dim,
        dropout=config.transformer_dropout,
        activation="gelu",
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(layer, num_layers=config.transformer_layers)


class SequenceEncoder(nn.Module):
    def __init__(self, config: DiffusionAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.token_dim, config.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.pos_embedding = nn.Parameter(
            torch.empty(1, 1 + config.encoder_length, config.hidden_dim).normal_(std=0.02)
        )
        self.transformer = _transformer_encoder(config)
        self.latent_proj = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, encoder_input: Tensor) -> Tensor:
        tokens = encoder_input.reshape(-1, self.config.encoder_length, self.config.token_dim)
        hidden = self.input_proj(tokens)
        cls = self.cls_token.expand(hidden.size(0), -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = hidden + self.pos_embedding[:, : hidden.size(1)]
        encoded = self.transformer(hidden)
        return self.latent_proj(encoded[:, 0])


class ContextEncoder(nn.Module):
    def __init__(self, config: DiffusionAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.context_dim, config.hidden_dim)
        self.pos_embedding = nn.Parameter(
            torch.empty(1, config.past_length, config.hidden_dim).normal_(std=0.02)
        )
        self.transformer = _transformer_encoder(config)
        self.condition_proj = nn.Linear(config.hidden_dim, config.condition_dim)

    def forward(self, context: Tensor) -> Tensor:
        tokens = context.reshape(-1, self.config.past_length, self.config.context_dim)
        hidden = self.input_proj(tokens)
        hidden = hidden + self.pos_embedding[:, : hidden.size(1)]
        encoded = self.transformer(hidden)
        return self.condition_proj(encoded)


class SeparateConditionTransformerBlock(nn.Module):
    """DiT block with AdaLN time/latent conditioning and context cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        time_dim: int,
        context_dim: int,
        latent_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.context_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
            kdim=context_dim,
            vdim=context_dim,
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 6 * hidden_size, bias=True))
        self.latent_modulation = nn.Sequential(nn.SiLU(), nn.Linear(latent_dim, 6 * hidden_size, bias=True))

    def forward(self, x: Tensor, time_features: Tensor, context_features: Tensor, latent: Tensor) -> Tensor:
        modulation = self.time_modulation(time_features) + self.latent_modulation(latent)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        attn_input = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        attn_out, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + gate_msa.unsqueeze(1) * attn_out

        context_out, _ = self.context_attn(self.norm_context(x), context_features, context_features)
        x = x + context_out

        mlp_input = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return x


class TrajectoryDiTDecoder(nn.Module):
    def __init__(self, config: DiffusionAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.action_dim, config.transformer_hidden_dim)
        self.pos_embedding = nn.Parameter(
            torch.empty(1, config.target_length, config.transformer_hidden_dim).normal_(std=0.02)
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, 2 * config.timestep_embed_dim),
            nn.GELU(),
            nn.Linear(2 * config.timestep_embed_dim, config.timestep_embed_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                SeparateConditionTransformerBlock(
                    hidden_size=config.transformer_hidden_dim,
                    num_heads=config.transformer_heads,
                    time_dim=config.timestep_embed_dim,
                    context_dim=config.condition_dim,
                    latent_dim=config.latent_dim,
                    dropout=config.transformer_dropout,
                )
                for _ in range(config.transformer_layers)
            ]
        )
        self.output_proj = nn.Linear(config.transformer_hidden_dim, config.action_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for block in self.blocks:
            for branch in (block.time_modulation, block.latent_modulation):
                nn.init.constant_(branch[-1].weight, 0)
                nn.init.constant_(branch[-1].bias, 0)

    def forward(self, actions: Tensor, timestep: Tensor, context_features: Tensor, latent: Tensor) -> Tensor:
        _, seq_len, _ = actions.shape
        hidden = self.input_proj(actions)
        hidden = hidden + self.pos_embedding[:, :seq_len]
        time_features = self.time_mlp(timestep * self.config.timestep_scale)

        for block in self.blocks:
            hidden = block(hidden, time_features, context_features, latent)
        return self.output_proj(hidden)


class ConditionalTrajectoryDiffusionAutoencoder(nn.Module):
    def __init__(self, config: DiffusionAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SequenceEncoder(config)
        self.context_encoder = ContextEncoder(config)
        self.decoder = TrajectoryDiTDecoder(config)
        self.sigreg = SIGReg(knots=config.sigreg_knots, num_proj=config.sigreg_num_proj)
        self.latent_prior = nn.Parameter(torch.zeros(config.latent_dim))
        self.flow_matching = FlowMatchingObjective(config, config.action_dim, config.target_length)

    def encode(self, encoder_input: Tensor) -> Tensor:
        return self.encoder(encoder_input)

    def _flow_matching_forward(
        self,
        context_features: Tensor,
        latent: Tensor,
        target_actions: Tensor,
        noise: Tensor | None = None,
        timestep: Tensor | None = None,
    ) -> dict[str, Tensor]:
        noised, timestep, target = self.flow_matching.prepare_training_sample(
            target_actions,
            noise=noise,
            timestep=timestep,
        )
        prediction = self.decoder(noised, timestep, context_features, latent)
        return {"prediction": prediction, "target": target, "objective_loss_key": "flow_matching_loss"}

    def forward(
        self,
        encoder_input: Tensor,
        context: Tensor,
        target_actions: Tensor,
        flow_noise: Tensor | None = None,
        flow_timestep: Tensor | None = None,
    ) -> dict[str, Tensor]:
        latent = self.encode(encoder_input)
        context_features = self.context_encoder(context)

        output = self._flow_matching_forward(
            context_features,
            latent,
            target_actions,
            noise=flow_noise,
            timestep=flow_timestep,
        )
        output["latent"] = latent
        return output

    def loss(self, output: dict[str, Tensor], target_actions: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        del target_actions
        objective_key = str(output["objective_loss_key"])
        objective_loss = F.mse_loss(output["prediction"], output["target"])
        sigreg_loss = self.sigreg(output["latent"])
        loss = objective_loss + self.config.sigreg_weight * sigreg_loss
        return loss, {
            objective_key: objective_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        context: Tensor,
        latent: Tensor | None = None,
        initial_sample: Tensor | None = None,
    ) -> Tensor:
        batch_size = context.shape[0]
        if latent is None:
            raise ValueError("sample() requires an encoded latent; no trained latent prior is available.")
        context_features = self.context_encoder(context)

        if initial_sample is None:
            sample = torch.randn(
                batch_size,
                self.config.target_length,
                self.config.action_dim,
                device=context.device,
                dtype=context.dtype,
            )
        else:
            sample = initial_sample

        return self._sample_flow_matching(sample, context_features, latent)

    def _sample_flow_matching(self, sample: Tensor, context_features: Tensor, latent: Tensor) -> Tensor:
        return self.flow_matching.integrate(
            lambda x, t: self.decoder(x, t, context_features, latent),
            sample,
        )


@dataclass(frozen=True)
class _EpisodeSpec:
    path: Path
    length: int
    max_start: int


@dataclass(frozen=True)
class _WindowDatasetInfo:
    state_dim: int
    action_dim: int
    state_keys: tuple[str, ...]
    eligible_episodes: int
    train_episodes: int
    val_episodes: int


class _BaseTrajectoryWindowDataset(Dataset):
    def __init__(
        self,
        episode_specs: list[_EpisodeSpec],
        state_keys: tuple[str, ...],
        past_length: int,
        future_length: int,
    ) -> None:
        if not episode_specs:
            raise ValueError("episode_specs must not be empty")
        self.episode_specs = episode_specs
        self.state_keys = state_keys
        self.past_length = int(past_length)
        self.future_length = int(future_length)
        self.required_length = self.past_length + self.future_length

    def _load_window(self, spec: _EpisodeSpec, start: int) -> dict[str, Tensor]:
        end = start + self.required_length
        with np.load(spec.path) as episode:
            states = self._states_from_episode(episode)[start:end]
            actions = self._actions_from_episode(episode)[start:end]

        encoder_input = np.concatenate([states, actions], axis=-1).reshape(-1)
        context = np.concatenate([states[: self.past_length], actions[: self.past_length]], axis=-1).reshape(-1)
        target_actions = actions[self.past_length :]
        return {
            "encoder_input": torch.from_numpy(encoder_input.astype(np.float32, copy=False)),
            "context": torch.from_numpy(context.astype(np.float32, copy=False)),
            "target_actions": torch.from_numpy(target_actions.astype(np.float32, copy=False)),
        }

    def _states_from_episode(self, episode: np.lib.npyio.NpzFile) -> np.ndarray:
        state_parts = []
        for key in self.state_keys:
            arr = np.asarray(episode[key], dtype=np.float32)
            state_parts.append(arr.reshape(arr.shape[0], -1))
        return np.concatenate(state_parts, axis=-1)

    @staticmethod
    def _actions_from_episode(episode: np.lib.npyio.NpzFile) -> np.ndarray:
        actions = np.asarray(episode["action"], dtype=np.float32)
        return actions.reshape(actions.shape[0], -1)


class RandomTrajectoryWindowDataset(_BaseTrajectoryWindowDataset):
    def __init__(
        self,
        episode_specs: list[_EpisodeSpec],
        state_keys: tuple[str, ...],
        past_length: int,
        future_length: int,
        windows_per_epoch: int,
        seed: int,
    ) -> None:
        super().__init__(episode_specs, state_keys, past_length, future_length)
        if windows_per_epoch < 1:
            raise ValueError("windows_per_epoch must be at least 1")
        self.windows_per_epoch = int(windows_per_epoch)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._worker_id: int | None = None

    def __len__(self) -> int:
        return self.windows_per_epoch

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        del idx
        worker = get_worker_info()
        worker_id = None if worker is None else int(worker.id)
        if worker_id != self._worker_id:
            self._worker_id = worker_id
            self._rng = np.random.default_rng(self.seed if worker_id is None else self.seed + worker_id + 1)

        spec = self.episode_specs[int(self._rng.integers(0, len(self.episode_specs)))]
        start = int(self._rng.integers(0, spec.max_start + 1)) if spec.max_start > 0 else 0
        return self._load_window(spec, start)


class FixedTrajectoryWindowDataset(_BaseTrajectoryWindowDataset):
    def __init__(
        self,
        episode_specs: list[_EpisodeSpec],
        state_keys: tuple[str, ...],
        past_length: int,
        future_length: int,
        windows: list[tuple[int, int]],
    ) -> None:
        super().__init__(episode_specs, state_keys, past_length, future_length)
        if not windows:
            raise ValueError("windows must not be empty")
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        spec_idx, start = self.windows[idx]
        return self._load_window(self.episode_specs[spec_idx], start)


def _state_keys_from_metadata(dataset_dir: Path, state_keys: tuple[str, ...] | None) -> tuple[str, ...]:
    if state_keys is not None:
        return tuple(state_keys)

    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return STATE_KEYS

    with metadata_path.open() as f:
        metadata = json.load(f)
    groups = metadata.get("obs_groups")
    if not groups:
        return STATE_KEYS
    return tuple(f"observation.{group}" for group in groups)


def _discover_episode_specs(
    dataset_dir: str | Path,
    past_length: int,
    future_length: int,
    state_keys: tuple[str, ...] | None = None,
) -> tuple[list[_EpisodeSpec], _WindowDatasetInfo]:
    dataset_path = Path(dataset_dir).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    resolved_state_keys = _state_keys_from_metadata(dataset_path, state_keys)
    required_length = int(past_length) + int(future_length)
    specs: list[_EpisodeSpec] = []
    state_dim: int | None = None
    action_dim: int | None = None

    for episode_path in sorted(dataset_path.glob("*.npz")):
        with np.load(episode_path) as episode:
            for key in (*resolved_state_keys, "action"):
                if key not in episode:
                    raise KeyError(f"{episode_path} is missing required array {key!r}")

            length = int(episode["action"].shape[0])
            if length < required_length:
                print(
                    f"[WARN] Skipping {episode_path}: length {length} < required_length {required_length}",
                    file=sys.stderr,
                )
                continue

            state_parts = []
            for key in resolved_state_keys:
                arr = np.asarray(episode[key], dtype=np.float32)
                state_parts.append(arr.reshape(arr.shape[0], -1))
            states = np.concatenate(state_parts, axis=-1)
            actions = np.asarray(episode["action"], dtype=np.float32).reshape(length, -1)
            if states.shape[0] < length:
                raise ValueError(
                    f"{episode_path} has {states.shape[0]} states for {length} actions; expected at least {length}"
                )

            if state_dim is None:
                state_dim = int(states.shape[-1])
                action_dim = int(actions.shape[-1])
            elif state_dim != states.shape[-1] or action_dim != actions.shape[-1]:
                raise ValueError(f"{episode_path} has dimensions inconsistent with earlier episodes")

            specs.append(_EpisodeSpec(path=episode_path, length=length, max_start=length - required_length))

    if not specs:
        raise RuntimeError(f"No eligible episodes found in {dataset_path} for required_length {required_length}")

    assert state_dim is not None
    assert action_dim is not None
    info = _WindowDatasetInfo(
        state_dim=state_dim,
        action_dim=action_dim,
        state_keys=resolved_state_keys,
        eligible_episodes=len(specs),
        train_episodes=0,
        val_episodes=0,
    )
    return specs, info


def _split_episode_specs(
    specs: list[_EpisodeSpec],
    val_ratio: float,
    seed: int,
) -> tuple[list[_EpisodeSpec], list[_EpisodeSpec]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if len(specs) == 1:
        return specs, specs

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(specs))
    val_count = 0
    if val_ratio > 0:
        val_count = max(1, int(round(len(specs) * val_ratio)))
        val_count = min(val_count, len(specs) - 1)
    val_indices = set(int(i) for i in order[:val_count])
    train_specs = [spec for i, spec in enumerate(specs) if i not in val_indices]
    val_specs = [spec for i, spec in enumerate(specs) if i in val_indices]
    if not val_specs:
        val_specs = train_specs
    return train_specs, val_specs


def _sample_fixed_windows(specs: list[_EpisodeSpec], val_windows: int, seed: int) -> list[tuple[int, int]]:
    if val_windows < 1:
        raise ValueError("val_windows must be at least 1")
    rng = np.random.default_rng(seed)
    windows = []
    for _ in range(int(val_windows)):
        spec_idx = int(rng.integers(0, len(specs)))
        spec = specs[spec_idx]
        start = int(rng.integers(0, spec.max_start + 1)) if spec.max_start > 0 else 0
        windows.append((spec_idx, start))
    return windows


def _build_window_datasets(
    dataset_dir: str | Path,
    past_length: int,
    future_length: int,
    val_ratio: float,
    train_windows_per_epoch: int,
    val_windows: int,
    seed: int,
    state_keys: tuple[str, ...] | None = None,
) -> tuple[RandomTrajectoryWindowDataset, FixedTrajectoryWindowDataset, _WindowDatasetInfo]:
    specs, info = _discover_episode_specs(dataset_dir, past_length, future_length, state_keys=state_keys)
    train_specs, val_specs = _split_episode_specs(specs, val_ratio=val_ratio, seed=seed)
    train_dataset = RandomTrajectoryWindowDataset(
        train_specs,
        info.state_keys,
        past_length=past_length,
        future_length=future_length,
        windows_per_epoch=train_windows_per_epoch,
        seed=seed,
    )
    val_dataset = FixedTrajectoryWindowDataset(
        val_specs,
        info.state_keys,
        past_length=past_length,
        future_length=future_length,
        windows=_sample_fixed_windows(val_specs, val_windows=val_windows, seed=seed + 1),
    )
    return (
        train_dataset,
        val_dataset,
        _WindowDatasetInfo(
            state_dim=info.state_dim,
            action_dim=info.action_dim,
            state_keys=info.state_keys,
            eligible_episodes=info.eligible_episodes,
            train_episodes=len(train_specs),
            val_episodes=len(val_specs),
        ),
    )


def _default_output_dir(dataset_dir: str | Path) -> Path:
    dataset_path = Path(dataset_dir).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = dataset_path.parent.name
    dataset_name = dataset_path.name
    return Path("logs") / "diffusion_autoencoder" / task_name / dataset_name / timestamp


def _default_wandb_run_name(dataset_dir: str | Path) -> str:
    dataset_path = Path(dataset_dir).expanduser()
    return f"{dataset_path.parent.name}_{dataset_path.name}"


def _torch_load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(
    path: Path,
    model: ConditionalTrajectoryDiffusionAutoencoder,
    optimizer: torch.optim.Optimizer,
    config: DiffusionAutoencoderConfig,
    train_args: dict[str, Any],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    best_val_sample_mse: float,
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    payload: dict[str, Any] = {
        "config": asdict(config),
        "model_type": MODEL_TYPE,
        "training": train_args,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "best_val_sample_mse": best_val_sample_mse,
        "best_checkpoint_metric": "val_sample_mse",
        "rng_state": _rng_state(),
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def _init_wandb(
    wandb_mode: str,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_run_name: str,
    config_payload: dict[str, Any],
    output_path: Path,
):
    if wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging was requested but wandb is not installed. "
            "Install wandb or pass --wandb-mode disabled."
        ) from exc
    return wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        mode=wandb_mode,
        config=config_payload,
        dir=str(output_path),
    )


def _wandb_log(wandb_run, payload: dict[str, float | int], step: int) -> None:
    if wandb_run is not None:
        wandb_run.log(payload, step=step)


def _wandb_save(wandb_run, path: Path) -> None:
    if wandb_run is not None:
        wandb_run.save(str(path))


def _cuda_autocast(enabled: bool):
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _seeded_randn_like(reference: Tensor, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.randn(reference.shape, dtype=reference.dtype, generator=generator)
    return value.to(device=reference.device)


def _seeded_rand(
    shape: tuple[int, ...],
    reference: Tensor,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.rand(shape, dtype=reference.dtype, generator=generator)
    return value.to(device=reference.device)


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@torch.no_grad()
def _evaluate(
    model: ConditionalTrajectoryDiffusionAutoencoder,
    loader: DataLoader,
    torch_device: torch.device,
    amp_enabled: bool,
    validation_seed: int | None = None,
) -> dict[str, float]:
    model.eval()
    rng_state = _rng_state() if validation_seed is not None else None
    total_loss = 0.0
    total_flow_matching = 0.0
    total_sigreg = 0.0
    total_sample_mse = 0.0
    total_batches = 0
    try:
        for batch_idx, batch in enumerate(loader):
            batch = _move_batch(batch, torch_device)
            flow_noise = None
            flow_timestep = None
            initial_sample = None
            if validation_seed is not None:
                seed = int(validation_seed) + batch_idx * 4
                flow_noise = _seeded_randn_like(batch["target_actions"], seed)
                flow_timestep = _seeded_rand(
                    (batch["target_actions"].shape[0],),
                    batch["target_actions"],
                    seed + 1,
                )
                initial_sample = _seeded_randn_like(batch["target_actions"], seed + 2)
                _set_torch_seed(seed + 3)

            with _cuda_autocast(amp_enabled):
                output = model(
                    batch["encoder_input"],
                    batch["context"],
                    batch["target_actions"],
                    flow_noise=flow_noise,
                    flow_timestep=flow_timestep,
                )
                loss, metrics = model.loss(output, batch["target_actions"])
                clean_samples = model.sample(
                    batch["context"],
                    latent=output["latent"],
                    initial_sample=initial_sample,
                )
                sample_mse = F.mse_loss(clean_samples, batch["target_actions"])
            total_loss += float(loss.detach().cpu())
            total_flow_matching += float(metrics["flow_matching_loss"].cpu())
            total_sigreg += float(metrics["sigreg_loss"].cpu())
            total_sample_mse += float(sample_mse.detach().cpu())
            total_batches += 1
    finally:
        _restore_rng_state(rng_state)
    return {
        "loss": total_loss / total_batches,
        "flow_matching_loss": total_flow_matching / total_batches,
        "sigreg_loss": total_sigreg / total_batches,
        "sample_mse": total_sample_mse / total_batches,
    }


def train_diffusion_autoencoder(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_dir: str | Path | None = None,
    past_length: int = 4,
    future_length: int = 8,
    latent_dim: int = 64,
    condition_dim: int = 256,
    hidden_dim: int = 1024,
    transformer_hidden_dim: int = 512,
    transformer_layers: int = 4,
    transformer_heads: int = 4,
    transformer_feedforward_dim: int = 1024,
    transformer_dropout: float = 0.1,
    timestep_embed_dim: int = 128,
    batch_size: int = 256,
    epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    grad_clip_norm: float | None = None,
    sigreg_weight: float = 0.09,
    sigreg_knots: int = 17,
    sigreg_num_proj: int = 1024,
    sigma_min: float = 1e-4,
    num_sample_steps: int = 20,
    timestep_scale: float = 1000.0,
    integration_method: Literal["euler", "heun", "rk4"] = "euler",
    num_workers: int = 0,
    val_ratio: float = 0.1,
    train_windows_per_epoch: int = 100_000,
    val_windows: int = 10_000,
    log_every_steps: int = 100,
    checkpoint_every_epochs: int = 10,
    resume: str | Path | None = None,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
    wandb_run_name: str | None = None,
    amp: bool = False,
    seed: int = 0,
    device: str = "cuda",
) -> ConditionalTrajectoryDiffusionAutoencoder:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset, val_dataset, dataset_info = _build_window_datasets(
        dataset_dir,
        past_length=past_length,
        future_length=future_length,
        val_ratio=val_ratio,
        train_windows_per_epoch=train_windows_per_epoch,
        val_windows=val_windows,
        seed=seed,
    )
    config = DiffusionAutoencoderConfig(
        state_dim=int(dataset_info.state_dim),
        action_dim=int(dataset_info.action_dim),
        past_length=past_length,
        future_length=future_length,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        transformer_hidden_dim=transformer_hidden_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_feedforward_dim=transformer_feedforward_dim,
        transformer_dropout=transformer_dropout,
        timestep_embed_dim=timestep_embed_dim,
        sigreg_weight=sigreg_weight,
        sigreg_knots=sigreg_knots,
        sigreg_num_proj=sigreg_num_proj,
        sigma_min=sigma_min,
        num_sample_steps=num_sample_steps,
        timestep_scale=timestep_scale,
        integration_method=integration_method,
    )

    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch_device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    model = ConditionalTrajectoryDiffusionAutoencoder(config).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    amp_enabled = bool(amp and torch_device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None

    if output_dir is None and resume is not None:
        output_path = Path(resume).expanduser().parent
    elif output_dir is None:
        output_path = _default_output_dir(dataset_dir)
    else:
        output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    if wandb_run_name is None:
        wandb_run_name = _default_wandb_run_name(dataset_dir)

    train_args = {
        "dataset_dir": str(Path(dataset_dir).expanduser()),
        "output_dir": str(output_path),
        "past_length": past_length,
        "future_length": future_length,
        "latent_dim": latent_dim,
        "condition_dim": condition_dim,
        "hidden_dim": hidden_dim,
        "transformer_hidden_dim": transformer_hidden_dim,
        "transformer_layers": transformer_layers,
        "transformer_heads": transformer_heads,
        "transformer_feedforward_dim": transformer_feedforward_dim,
        "transformer_dropout": transformer_dropout,
        "timestep_embed_dim": timestep_embed_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip_norm,
        "sigreg_weight": sigreg_weight,
        "sigreg_knots": sigreg_knots,
        "sigreg_num_proj": sigreg_num_proj,
        "sigma_min": sigma_min,
        "num_sample_steps": num_sample_steps,
        "timestep_scale": timestep_scale,
        "integration_method": integration_method,
        "num_workers": num_workers,
        "val_ratio": val_ratio,
        "train_windows_per_epoch": train_windows_per_epoch,
        "val_windows": val_windows,
        "log_every_steps": log_every_steps,
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "resume": None if resume is None else str(Path(resume).expanduser()),
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_mode": wandb_mode,
        "wandb_run_name": wandb_run_name,
        "amp": amp,
        "seed": seed,
        "validation_seed": seed,
        "best_checkpoint_metric": "val_sample_mse",
        "device": str(torch_device),
        "state_keys": dataset_info.state_keys,
        "eligible_episodes": dataset_info.eligible_episodes,
        "train_episodes": dataset_info.train_episodes,
        "val_episodes": dataset_info.val_episodes,
    }
    config_payload = asdict(config) | {"model_type": MODEL_TYPE, "training": train_args}
    with (output_path / "config.json").open("w") as f:
        json.dump(config_payload, f, indent=2)

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    best_val_sample_mse = float("inf")
    if resume is not None:
        checkpoint = _torch_load_checkpoint(Path(resume).expanduser(), map_location=torch_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_val_sample_mse = float(checkpoint.get("best_val_sample_mse", best_val_sample_mse))
        _restore_rng_state(checkpoint.get("rng_state"))

    wandb_run = _init_wandb(
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        config_payload=config_payload,
        output_path=output_path,
    )

    log_path = output_path / "train_log.jsonl"
    log_mode = "a" if resume is not None and log_path.exists() else "w"
    try:
        with log_path.open(log_mode) as log_file:
            for epoch in range(start_epoch, epochs + 1):
                model.train()
                total_loss = 0.0
                total_flow_matching = 0.0
                total_sigreg = 0.0
                total_batches = 0

                for batch in train_loader:
                    batch = _move_batch(batch, torch_device)
                    optimizer.zero_grad(set_to_none=True)
                    with _cuda_autocast(amp_enabled):
                        output = model(batch["encoder_input"], batch["context"], batch["target_actions"])
                        loss, metrics = model.loss(output, batch["target_actions"])

                    if scaler is not None:
                        scaler.scale(loss).backward()
                        if grad_clip_norm is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        optimizer.step()

                    global_step += 1
                    total_loss += float(loss.detach().cpu())
                    total_flow_matching += float(metrics["flow_matching_loss"].cpu())
                    total_sigreg += float(metrics["sigreg_loss"].cpu())
                    total_batches += 1

                    if log_every_steps > 0 and global_step % log_every_steps == 0:
                        _wandb_log(
                            wandb_run,
                            {
                                "train/loss": float(loss.detach().cpu()),
                                "train/flow_matching_loss": float(metrics["flow_matching_loss"].cpu()),
                                "train/sigreg_loss": float(metrics["sigreg_loss"].cpu()),
                                "train/lr": float(optimizer.param_groups[0]["lr"]),
                                "epoch": epoch,
                            },
                            step=global_step,
                        )

                train_loss = total_loss / total_batches
                train_flow_matching = total_flow_matching / total_batches
                train_sigreg = total_sigreg / total_batches
                val_metrics = _evaluate(
                    model,
                    val_loader,
                    torch_device,
                    amp_enabled,
                    validation_seed=seed,
                )
                val_loss = val_metrics["loss"]
                val_sample_mse = val_metrics["sample_mse"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                is_best = val_sample_mse < best_val_sample_mse
                if is_best:
                    best_val_sample_mse = val_sample_mse

                row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": train_loss,
                    "flow_matching_loss": train_flow_matching,
                    "sigreg_loss": train_sigreg,
                    "val_loss": val_loss,
                    "val_flow_matching_loss": val_metrics["flow_matching_loss"],
                    "val_sigreg_loss": val_metrics["sigreg_loss"],
                    "val_sample_mse": val_sample_mse,
                    "best_val_loss": best_val_loss,
                    "best_val_sample_mse": best_val_sample_mse,
                    "best_checkpoint_metric": "val_sample_mse",
                }
                print(json.dumps(row), file=log_file, flush=True)
                print(
                    f"epoch {epoch:04d} step={global_step} loss={row['loss']:.6f} "
                    f"flow_matching_loss={row['flow_matching_loss']:.6f} sigreg={row['sigreg_loss']:.6f} "
                    f"val_loss={row['val_loss']:.6f} val_sample_mse={row['val_sample_mse']:.6f}",
                    flush=True,
                )
                _wandb_log(
                    wandb_run,
                    {
                        "epoch": epoch,
                        "train/epoch_loss": train_loss,
                        "train/epoch_flow_matching_loss": train_flow_matching,
                        "train/epoch_sigreg_loss": train_sigreg,
                        "val/loss": val_metrics["loss"],
                        "val/flow_matching_loss": val_metrics["flow_matching_loss"],
                        "val/sigreg_loss": val_metrics["sigreg_loss"],
                        "val/sample_mse": val_sample_mse,
                        "best_val_loss": best_val_loss,
                        "best_val_sample_mse": best_val_sample_mse,
                    },
                    step=global_step,
                )

                last_path = output_path / "last.pt"
                _save_checkpoint(
                    last_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    train_args=train_args,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    best_val_sample_mse=best_val_sample_mse,
                    scaler=scaler,
                )
                _wandb_save(wandb_run, last_path)

                if is_best:
                    best_path = output_path / "best.pt"
                    _save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        train_args=train_args,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        best_val_sample_mse=best_val_sample_mse,
                        scaler=scaler,
                    )
                    _wandb_save(wandb_run, best_path)

                if checkpoint_every_epochs > 0 and epoch % checkpoint_every_epochs == 0:
                    epoch_path = output_path / f"epoch_{epoch:04d}.pt"
                    _save_checkpoint(
                        epoch_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        train_args=train_args,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        best_val_sample_mse=best_val_sample_mse,
                        scaler=scaler,
                    )
                    _wandb_save(wandb_run, epoch_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    torch.save(
        {
            "config": asdict(config),
            "model_type": MODEL_TYPE,
            "model_state_dict": model.state_dict(),
        },
        output_path / "model.pt",
    )
    return model


def _parse_args(args: list[str] | None = None) -> DiffusionAutoencoderTrainArgs:
    return tyro.cli(DiffusionAutoencoderTrainArgs, args=args)


def _train_kwargs_from_args(args: DiffusionAutoencoderTrainArgs) -> dict[str, Any]:
    train_kwargs = asdict(args)
    config_kwargs = train_kwargs.pop("diffusion_autoencoder_config")
    train_kwargs.update(config_kwargs)
    return train_kwargs


def main() -> None:
    args = _parse_args()
    train_diffusion_autoencoder(
        **_train_kwargs_from_args(args),
    )


if __name__ == "__main__":
    main()
