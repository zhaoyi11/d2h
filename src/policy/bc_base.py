"""Basic behavior-cloning action-chunk policy."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, get_worker_info

from src.policy.vae import (
    _EpisodeSpec,
    DEFAULT_WANDB_PROJECT,
    _default_output_dir,
    _default_wandb_run_name,
    _move_batch,
    _init_wandb,
    _sample_fixed_windows,
    _split_episode_specs,
    _wandb_log,
    _wandb_save,
    _state_keys_from_metadata,
)


MODEL_TYPE = "basic_behavior_chunk_policy"


@dataclass
class BehaviorCloningChunkConfig:
    state_dim: int
    action_dim: int
    future_length: int = 4
    hidden_dim: int = 512
    num_layers: int = 4
    normalization_eps: float = 1e-6

    @property
    def target_length(self) -> int:
        return self.future_length


@dataclass
class BehaviorCloningNormalizationStats:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


class BehaviorCloningChunkPolicy(nn.Module):
    def __init__(self, config: BehaviorCloningChunkConfig) -> None:
        super().__init__()
        if int(config.num_layers) < 1:
            raise ValueError("num_layers must be at least 1")
        self.config = config
        layers: list[nn.Module] = []
        input_dim = config.state_dim
        for _ in range(config.num_layers):
            layers.extend(
                (
                    nn.Linear(input_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.ReLU(),
                )
            )
            input_dim = config.hidden_dim
        layers.append(nn.Linear(config.hidden_dim, config.future_length * config.action_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("obs_mean", torch.zeros(config.state_dim))
        self.register_buffer("obs_std", torch.ones(config.state_dim))
        self.register_buffer("action_mean", torch.zeros(config.action_dim))
        self.register_buffer("action_std", torch.ones(config.action_dim))

    def set_normalization(
        self,
        obs_mean: np.ndarray | Tensor,
        obs_std: np.ndarray | Tensor,
        action_mean: np.ndarray | Tensor,
        action_std: np.ndarray | Tensor,
    ) -> None:
        self._copy_normalization_buffer("obs_mean", obs_mean, (self.config.state_dim,))
        self._copy_normalization_buffer("obs_std", obs_std, (self.config.state_dim,))
        self._copy_normalization_buffer("action_mean", action_mean, (self.config.action_dim,))
        self._copy_normalization_buffer("action_std", action_std, (self.config.action_dim,))

    def _copy_normalization_buffer(self, name: str, value: np.ndarray | Tensor, shape: tuple[int, ...]) -> None:
        buffer = getattr(self, name)
        tensor = torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Expected {name} shape {shape}, got {tuple(tensor.shape)}.")
        buffer.copy_(tensor)

    def normalize_obs(self, obs: Tensor) -> Tensor:
        if obs.shape[-1] != self.config.state_dim:
            raise ValueError(f"Expected obs dim {self.config.state_dim}, got {obs.shape[-1]}.")
        return (obs - self.obs_mean) / (self.obs_std + self.config.normalization_eps)

    def normalize_actions(self, actions: Tensor) -> Tensor:
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError(f"Expected action dim {self.config.action_dim}, got {actions.shape[-1]}.")
        return (actions - self.action_mean) / (self.action_std + self.config.normalization_eps)

    def denormalize_actions(self, actions: Tensor) -> Tensor:
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError(f"Expected action dim {self.config.action_dim}, got {actions.shape[-1]}.")
        return actions * (self.action_std + self.config.normalization_eps) + self.action_mean

    def forward_normalized(self, obs: Tensor) -> Tensor:
        actions = self.net(self.normalize_obs(obs))
        return actions.reshape(-1, self.config.future_length, self.config.action_dim)

    def forward(self, obs: Tensor) -> Tensor:
        return self.denormalize_actions(self.forward_normalized(obs))


class BehaviorCloningWindowDataset(Dataset):
    """Fixed-length action chunks aligned with the current observation."""

    def __init__(
        self,
        episode_specs: list[_EpisodeSpec],
        state_keys: tuple[str, ...],
        future_length: int,
        windows: int | list[tuple[int, int]],
        seed: int = 0,
    ) -> None:
        if not episode_specs:
            raise ValueError("episode_specs must not be empty")
        if int(future_length) < 1:
            raise ValueError("future_length must be at least 1")
        self.episode_specs = episode_specs
        self.state_keys = state_keys
        self.future_length = int(future_length)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._worker_id: int | None = None
        self._windows = windows

    def __len__(self) -> int:
        return int(self._windows) if isinstance(self._windows, int) else len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if isinstance(self._windows, int):
            worker = get_worker_info()
            worker_id = None if worker is None else int(worker.id)
            if worker_id != self._worker_id:
                self._worker_id = worker_id
                self._rng = np.random.default_rng(self.seed if worker_id is None else self.seed + worker_id + 1)
            spec = self.episode_specs[int(self._rng.integers(0, len(self.episode_specs)))]
            start = int(self._rng.integers(0, spec.max_start + 1)) if spec.max_start > 0 else 0
        else:
            spec_idx, start = self._windows[idx]
            spec = self.episode_specs[spec_idx]
        return self._load_window(spec, start)

    def _load_window(self, spec: _EpisodeSpec, start: int) -> dict[str, Tensor]:
        end = start + self.future_length
        with np.load(spec.path) as episode:
            obs = self._states_from_episode(episode)[start]
            target_actions = self._actions_from_episode(episode)[start:end]
        return {
            "obs": torch.from_numpy(obs.astype(np.float32, copy=False)),
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

    @property
    def state_dim(self) -> int:
        spec = self.episode_specs[0]
        with np.load(spec.path) as episode:
            return int(self._states_from_episode(episode).shape[-1])

    @property
    def action_dim(self) -> int:
        spec = self.episode_specs[0]
        with np.load(spec.path) as episode:
            return int(self._actions_from_episode(episode).shape[-1])


def discover_behavior_cloning_specs(
    dataset_dir: str | Path,
    future_length: int,
    state_keys: tuple[str, ...] | None = None,
) -> tuple[list[_EpisodeSpec], tuple[str, ...], int, int]:
    dataset_path = Path(dataset_dir).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    if int(future_length) < 1:
        raise ValueError("future_length must be at least 1")

    resolved_state_keys = _state_keys_from_metadata(dataset_path, state_keys)
    specs: list[_EpisodeSpec] = []
    state_dim: int | None = None
    action_dim: int | None = None
    for episode_path in sorted(dataset_path.glob("*.npz")):
        with np.load(episode_path) as episode:
            for key in (*resolved_state_keys, "action"):
                if key not in episode:
                    raise KeyError(f"{episode_path} is missing required array {key!r}")
            length = int(episode["action"].shape[0])
            if length < int(future_length):
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
            specs.append(_EpisodeSpec(path=episode_path, length=length, max_start=length - int(future_length)))
    if not specs:
        raise RuntimeError(f"No eligible episodes found in {dataset_path} for future_length {future_length}")
    assert state_dim is not None and action_dim is not None
    return specs, resolved_state_keys, state_dim, action_dim


def _compute_behavior_cloning_normalization_stats(
    episode_specs: list[_EpisodeSpec],
    state_keys: tuple[str, ...],
    normalization_eps: float = 1e-6,
) -> BehaviorCloningNormalizationStats:
    if not episode_specs:
        raise ValueError("episode_specs must not be empty")

    obs_sum: np.ndarray | None = None
    obs_sq_sum: np.ndarray | None = None
    action_sum: np.ndarray | None = None
    action_sq_sum: np.ndarray | None = None
    total_steps = 0
    for spec in episode_specs:
        with np.load(spec.path) as episode:
            actions = np.asarray(episode["action"], dtype=np.float32).reshape(spec.length, -1)
            state_parts = []
            for key in state_keys:
                arr = np.asarray(episode[key], dtype=np.float32)
                state_parts.append(arr.reshape(arr.shape[0], -1)[: spec.length])
            states = np.concatenate(state_parts, axis=-1)

        states64 = states.astype(np.float64, copy=False)
        actions64 = actions.astype(np.float64, copy=False)
        if obs_sum is None:
            obs_sum = np.zeros(states64.shape[-1], dtype=np.float64)
            obs_sq_sum = np.zeros(states64.shape[-1], dtype=np.float64)
            action_sum = np.zeros(actions64.shape[-1], dtype=np.float64)
            action_sq_sum = np.zeros(actions64.shape[-1], dtype=np.float64)
        obs_sum += states64.sum(axis=0)
        obs_sq_sum += np.square(states64).sum(axis=0)
        action_sum += actions64.sum(axis=0)
        action_sq_sum += np.square(actions64).sum(axis=0)
        total_steps += spec.length

    assert obs_sum is not None and obs_sq_sum is not None
    assert action_sum is not None and action_sq_sum is not None
    obs_mean, obs_std = _mean_and_clamped_std(obs_sum, obs_sq_sum, total_steps, normalization_eps)
    action_mean, action_std = _mean_and_clamped_std(action_sum, action_sq_sum, total_steps, normalization_eps)
    return BehaviorCloningNormalizationStats(
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )


def _mean_and_clamped_std(
    total: np.ndarray,
    squared_total: np.ndarray,
    count: int,
    normalization_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = total / float(count)
    variance = np.maximum(squared_total / float(count) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < normalization_eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def build_behavior_cloning_datasets(
    dataset_dir: str | Path,
    future_length: int,
    train_windows_per_epoch: int,
    val_windows: int,
    val_ratio: float,
    seed: int,
    state_keys: tuple[str, ...] | None = None,
) -> tuple[BehaviorCloningWindowDataset, BehaviorCloningWindowDataset]:
    specs, resolved_state_keys, _, _ = discover_behavior_cloning_specs(
        dataset_dir, future_length, state_keys=state_keys
    )
    train_specs, val_specs = _split_episode_specs(specs, val_ratio=val_ratio, seed=seed)
    return (
        BehaviorCloningWindowDataset(
            train_specs,
            resolved_state_keys,
            future_length=future_length,
            windows=train_windows_per_epoch,
            seed=seed,
        ),
        BehaviorCloningWindowDataset(
            val_specs,
            resolved_state_keys,
            future_length=future_length,
            windows=_sample_fixed_windows(val_specs, val_windows=val_windows, seed=seed + 1),
            seed=seed + 1,
        ),
    )


def save_checkpoint(
    path: Path,
    model: BehaviorCloningChunkPolicy,
    optimizer: torch.optim.Optimizer,
    train_args: dict[str, Any],
    epoch: int,
    best_val_loss: float,
    state_keys: tuple[str, ...],
) -> None:
    torch.save(
        {
            "config": asdict(model.config),
            "model_type": MODEL_TYPE,
            "training": {**train_args, "state_keys": state_keys},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        },
        path,
    )


def train(
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    future_length: int = 4,
    hidden_dim: int = 512,
    num_layers: int = 4,
    normalization_eps: float = 1e-6,
    batch_size: int = 512,
    epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    val_ratio: float = 0.1,
    train_windows_per_epoch: int = 100_000,
    val_windows: int = 10_000,
    seed: int = 0,
    device: str = "cuda",
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
    wandb_run_name: str | None = None,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    out_dir = Path(output_dir).expanduser() if output_dir is not None else _default_output_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if wandb_run_name is None:
        wandb_run_name = _default_wandb_run_name(dataset_dir)
    train_dataset, val_dataset = build_behavior_cloning_datasets(
        dataset_dir,
        future_length=future_length,
        train_windows_per_epoch=train_windows_per_epoch,
        val_windows=val_windows,
        val_ratio=val_ratio,
        seed=seed,
    )
    config = BehaviorCloningChunkConfig(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        future_length=future_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        normalization_eps=normalization_eps,
    )
    model = BehaviorCloningChunkPolicy(config).to(device)
    normalization_stats = _compute_behavior_cloning_normalization_stats(
        train_dataset.episode_specs,
        train_dataset.state_keys,
        normalization_eps=normalization_eps,
    )
    model.set_normalization(
        normalization_stats.obs_mean,
        normalization_stats.obs_std,
        normalization_stats.action_mean,
        normalization_stats.action_std,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    normalization_metadata = {
        "enabled": True,
        "eps": normalization_eps,
        "source": "train_split",
    }
    train_args = {
        "dataset_dir": str(Path(dataset_dir).expanduser()),
        "output_dir": str(out_dir),
        "future_length": future_length,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "normalization": normalization_metadata,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "val_ratio": val_ratio,
        "train_windows_per_epoch": train_windows_per_epoch,
        "val_windows": val_windows,
        "seed": seed,
        "device": device,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_mode": wandb_mode,
        "wandb_run_name": wandb_run_name,
    }
    config_payload = {"model_type": MODEL_TYPE, "config": asdict(config), "training": train_args}
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    wandb_run = _init_wandb(
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        config_payload=config_payload,
        output_path=out_dir,
    )

    best_val_loss = float("inf")
    best_path = out_dir / "best.pt"
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                batch = _move_batch(batch, torch.device(device))
                pred = model.forward_normalized(batch["obs"])
                # target = model.normalize_actions(batch["target_actions"])
                target = batch["target_actions"]
                loss = F.mse_loss(pred, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(train_losses))
            model.eval()
            val_losses = []
            # val_raw_mses = []
            with torch.inference_mode():
                for batch in val_loader:
                    batch = _move_batch(batch, torch.device(device))
                    pred = model.forward_normalized(batch["obs"])
                    # target = model.normalize_actions(batch["target_actions"])
                    target = batch["target_actions"]
                    val_losses.append(F.mse_loss(pred, target).detach())
                    # val_raw_mses.append(F.mse_loss(model(batch["obs"]), batch["target_actions"]).detach())
            val_loss = float(torch.stack(val_losses).mean().cpu())
            # val_raw_mse = float(torch.stack(val_raw_mses).mean().cpu())
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    train_args=train_args,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    state_keys=train_dataset.state_keys,
                )
                _wandb_save(wandb_run, best_path)
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"best_val_loss={best_val_loss:.6f}"
            )
            _wandb_log(
                wandb_run,
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "best_val_loss": best_val_loss,
                },
                step=epoch,
            )

        final_path = out_dir / f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        save_checkpoint(
            final_path,
            model,
            optimizer,
            train_args=train_args,
            epoch=epochs,
            best_val_loss=best_val_loss,
            state_keys=train_dataset.state_keys,
        )
        _wandb_save(wandb_run, final_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return best_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a basic behavior-cloning action-chunk policy.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--future-length", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--normalization-eps", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--train-windows-per-epoch", type=int, default=100_000)
    parser.add_argument("--val-windows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        future_length=args.future_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        normalization_eps=args.normalization_eps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        train_windows_per_epoch=args.train_windows_per_epoch,
        val_windows=args.val_windows,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_run_name=args.wandb_run_name,
    )
