"""Goal-conditioned action-chunk policy for low-level in-hand reorientation."""

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


MODEL_TYPE = "goal_conditioned_trajectory_chunk_policy"
DEFAULT_GOAL_SLICE = (-23, -16)


@dataclass
class GoalConditionedChunkConfig:
    state_dim: int
    action_dim: int
    goal_dim: int = 7
    past_length: int = 4
    future_length: int = 8
    condition_dim: int = 256
    hidden_dim: int = 512

    @property
    def context_dim(self) -> int:
        return self.past_length * (self.state_dim + self.action_dim)

    @property
    def target_length(self) -> int:
        return self.future_length


class GoalConditionedChunkPolicy(nn.Module):
    def __init__(self, config: GoalConditionedChunkConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = nn.Sequential(
            nn.Linear(config.context_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, config.condition_dim),
            nn.LayerNorm(config.condition_dim),
            nn.ReLU(),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(config.goal_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, config.condition_dim),
            nn.LayerNorm(config.condition_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * config.condition_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, config.future_length * config.action_dim),
        )

    def forward(self, context: Tensor, goal: Tensor) -> Tensor:
        if context.shape[-1] != self.config.context_dim:
            raise ValueError(f"Expected context dim {self.config.context_dim}, got {context.shape[-1]}.")
        if goal.shape[-1] != self.config.goal_dim:
            raise ValueError(f"Expected goal dim {self.config.goal_dim}, got {goal.shape[-1]}.")
        condition = torch.cat((self.context_encoder(context), self.goal_encoder(goal)), dim=-1)
        actions = self.decoder(condition)
        return actions.reshape(-1, self.config.future_length, self.config.action_dim)


class GoalConditionedWindowDataset(Dataset):
    """Fixed-length action windows with goal error split out of the state."""

    def __init__(
        self,
        episode_specs: list[_EpisodeSpec],
        state_keys: tuple[str, ...],
        past_length: int,
        future_length: int,
        windows: int | list[tuple[int, int]],
        goal_slice: tuple[int | None, int | None] = DEFAULT_GOAL_SLICE,
        seed: int = 0,
    ) -> None:
        if not episode_specs:
            raise ValueError("episode_specs must not be empty")
        self.episode_specs = episode_specs
        self.state_keys = state_keys
        self.past_length = int(past_length)
        self.future_length = int(future_length)
        self.required_length = self.past_length + self.future_length
        self.goal_slice = slice(*goal_slice)
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
        end = start + self.required_length
        with np.load(spec.path) as episode:
            states = self._states_from_episode(episode)[start:end]
            actions = np.asarray(episode["action"], dtype=np.float32).reshape(-1, self.action_dim)[start:end]
        
        import ipdb; ipdb.set_trace()
        goal = states[self.past_length - 1, self.goal_slice]
        context_states = np.concatenate((states[: self.past_length, : self.goal_start], states[: self.past_length, self.goal_stop :]), axis=-1)
        context = np.concatenate((context_states, actions[: self.past_length]), axis=-1).reshape(-1)
        target_actions = actions[self.past_length :]
        return {
            "context": torch.from_numpy(context.astype(np.float32, copy=False)),
            "goal": torch.from_numpy(goal.astype(np.float32, copy=False)),
            "target_actions": torch.from_numpy(target_actions.astype(np.float32, copy=False)),
        }

    def _states_from_episode(self, episode: np.lib.npyio.NpzFile) -> np.ndarray:
        state_parts = []
        for key in self.state_keys:
            arr = np.asarray(episode[key], dtype=np.float32)
            state_parts.append(arr.reshape(arr.shape[0], -1))
        return np.concatenate(state_parts, axis=-1)

    @property
    def action_dim(self) -> int:
        spec = self.episode_specs[0]
        with np.load(spec.path) as episode:
            return int(np.asarray(episode["action"]).reshape(spec.length, -1).shape[-1])

    @property
    def full_state_dim(self) -> int:
        spec = self.episode_specs[0]
        with np.load(spec.path) as episode:
            return int(self._states_from_episode(episode).shape[-1])

    @property
    def goal_start(self) -> int:
        start, _, _ = self.goal_slice.indices(self.full_state_dim)
        return start

    @property
    def goal_stop(self) -> int:
        _, stop, _ = self.goal_slice.indices(self.full_state_dim)
        return stop

    @property
    def state_dim(self) -> int:
        return self.full_state_dim - (self.goal_stop - self.goal_start)

    @property
    def goal_dim(self) -> int:
        return self.goal_stop - self.goal_start


def discover_goal_conditioned_specs(
    dataset_dir: str | Path,
    past_length: int,
    future_length: int,
    state_keys: tuple[str, ...] | None = None,
) -> tuple[list[_EpisodeSpec], tuple[str, ...], int, int]:
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
                continue
            state_parts = []
            for key in resolved_state_keys:
                arr = np.asarray(episode[key], dtype=np.float32)
                state_parts.append(arr.reshape(arr.shape[0], -1)[:length])
            states = np.concatenate(state_parts, axis=-1)
            actions = np.asarray(episode["action"], dtype=np.float32).reshape(length, -1)
            if state_dim is None:
                state_dim = int(states.shape[-1])
                action_dim = int(actions.shape[-1])
            elif state_dim != states.shape[-1] or action_dim != actions.shape[-1]:
                raise ValueError(f"{episode_path} has dimensions inconsistent with earlier episodes")
            specs.append(_EpisodeSpec(path=episode_path, length=length, max_start=length - required_length))
    if not specs:
        raise RuntimeError(f"No eligible episodes found in {dataset_path} for required_length {required_length}")
    assert state_dim is not None and action_dim is not None
    return specs, resolved_state_keys, state_dim, action_dim


def build_goal_conditioned_datasets(
    dataset_dir: str | Path,
    past_length: int,
    future_length: int,
    train_windows_per_epoch: int,
    val_windows: int,
    val_ratio: float,
    seed: int,
    state_keys: tuple[str, ...] | None = None,
    goal_slice: tuple[int | None, int | None] = DEFAULT_GOAL_SLICE,
) -> tuple[GoalConditionedWindowDataset, GoalConditionedWindowDataset]:
    specs, resolved_state_keys, _, _ = discover_goal_conditioned_specs(
        dataset_dir, past_length, future_length, state_keys=state_keys
    )
    train_specs, val_specs = _split_episode_specs(specs, val_ratio=val_ratio, seed=seed)
    return (
        GoalConditionedWindowDataset(
            train_specs,
            resolved_state_keys,
            past_length=past_length,
            future_length=future_length,
            windows=train_windows_per_epoch,
            goal_slice=goal_slice,
            seed=seed,
        ),
        GoalConditionedWindowDataset(
            val_specs,
            resolved_state_keys,
            past_length=past_length,
            future_length=future_length,
            windows=_sample_fixed_windows(val_specs, val_windows=val_windows, seed=seed + 1),
            goal_slice=goal_slice,
            seed=seed + 1,
        ),
    )


def save_checkpoint(
    path: Path,
    model: GoalConditionedChunkPolicy,
    optimizer: torch.optim.Optimizer,
    train_args: dict[str, Any],
    epoch: int,
    best_val_loss: float,
    state_keys: tuple[str, ...],
    goal_slice: tuple[int | None, int | None],
) -> None:
    torch.save(
        {
            "config": asdict(model.config),
            "model_type": MODEL_TYPE,
            "training": {**train_args, "state_keys": state_keys, "goal_slice": goal_slice},
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
    past_length: int = 4,
    future_length: int = 8,
    condition_dim: int = 256,
    hidden_dim: int = 512,
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
    train_dataset, val_dataset = build_goal_conditioned_datasets(
        dataset_dir,
        past_length=past_length,
        future_length=future_length,
        train_windows_per_epoch=train_windows_per_epoch,
        val_windows=val_windows,
        val_ratio=val_ratio,
        seed=seed,
    )
    config = GoalConditionedChunkConfig(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        goal_dim=train_dataset.goal_dim,
        past_length=past_length,
        future_length=future_length,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
    )
    model = GoalConditionedChunkPolicy(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_args = {
        "dataset_dir": str(Path(dataset_dir).expanduser()),
        "output_dir": str(out_dir),
        "past_length": past_length,
        "future_length": future_length,
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
                pred = model(batch["context"], batch["goal"])
                loss = F.mse_loss(pred, batch["target_actions"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(train_losses))
            model.eval()
            val_losses = []
            with torch.inference_mode():
                for batch in val_loader:
                    batch = _move_batch(batch, torch.device(device))
                    val_losses.append(F.mse_loss(model(batch["context"], batch["goal"]), batch["target_actions"]).detach())
            val_loss = float(torch.stack(val_losses).mean().cpu())
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
                    goal_slice=DEFAULT_GOAL_SLICE,
                )
                _wandb_save(wandb_run, best_path)
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val_loss={best_val_loss:.6f}")
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
            goal_slice=DEFAULT_GOAL_SLICE,
        )
        _wandb_save(wandb_run, final_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return best_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a goal-conditioned low-level action-chunk policy.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--past-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=8)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
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
        past_length=args.past_length,
        future_length=args.future_length,
        condition_dim=args.condition_dim,
        hidden_dim=args.hidden_dim,
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
