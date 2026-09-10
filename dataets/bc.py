"""Train a compact behavior-cloning policy from recorded knob-rotation episodes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


MODEL_TYPE = "behavior_cloning_action_chunk"
OBS_KEY = "observation.bc"
ACTION_KEY = "action"
DEFAULT_DATASET_DIR = Path(__file__).with_name("rotate_object_bc")


@dataclass(frozen=True)
class BehaviorCloningConfig:
    state_dim: int
    action_dim: int
    future_length: int = 4
    hidden_dim: int = 256
    num_layers: int = 2
    normalization_eps: float = 1e-6


@dataclass(frozen=True)
class BehaviorCloningData:
    train_obs: np.ndarray
    train_actions: np.ndarray
    val_obs: np.ndarray
    val_actions: np.ndarray
    train_episodes: tuple[str, ...]
    val_episodes: tuple[str, ...]


class BehaviorCloningPolicy(nn.Module):
    """MLP that predicts a short chunk of raw robot actions."""

    def __init__(self, config: BehaviorCloningConfig) -> None:
        super().__init__()
        if config.num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.config = config
        layers: list[nn.Module] = []
        input_dim = config.state_dim
        for _ in range(config.num_layers):
            layers.extend(
                (nn.Linear(input_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.ReLU())
            )
            input_dim = config.hidden_dim
        layers.append(nn.Linear(input_dim, config.future_length * config.action_dim))
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
        for name, value, size in (
            ("obs_mean", obs_mean, self.config.state_dim),
            ("obs_std", obs_std, self.config.state_dim),
            ("action_mean", action_mean, self.config.action_dim),
            ("action_std", action_std, self.config.action_dim),
        ):
            target = getattr(self, name)
            tensor = torch.as_tensor(value, dtype=target.dtype, device=target.device)
            if tensor.shape != (size,):
                raise ValueError(f"Expected {name} shape {(size,)}, got {tuple(tensor.shape)}.")
            target.copy_(tensor)

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
        return actions.reshape(*obs.shape[:-1], self.config.future_length, self.config.action_dim)

    def forward(self, obs: Tensor) -> Tensor:
        return self.denormalize_actions(self.forward_normalized(obs))


def _load_episodes(
    dataset_dir: str | Path,
    future_length: int,
) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    dataset_path = Path(dataset_dir).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    episodes: list[tuple[Path, np.ndarray, np.ndarray]] = []
    dimensions: tuple[int, int] | None = None
    for path in sorted(dataset_path.glob("*.npz")):
        with np.load(path) as episode:
            missing = [key for key in (OBS_KEY, ACTION_KEY) if key not in episode]
            if missing:
                raise KeyError(f"{path} is missing required arrays: {missing}")
            obs = np.asarray(episode[OBS_KEY], dtype=np.float32)
            actions = np.asarray(episode[ACTION_KEY], dtype=np.float32)

        if obs.ndim != 2 or actions.ndim != 2 or len(obs) != len(actions):
            raise ValueError(f"{path} must contain aligned 2D observation and action arrays.")
        if len(actions) < future_length:
            raise ValueError(f"{path} has {len(actions)} steps; expected at least {future_length}.")
        if not np.isfinite(obs).all() or not np.isfinite(actions).all():
            raise ValueError(f"{path} contains non-finite observations or actions.")
        episode_dimensions = (obs.shape[1], actions.shape[1])
        if dimensions is None:
            dimensions = episode_dimensions
        elif episode_dimensions != dimensions:
            raise ValueError(f"{path} dimensions {episode_dimensions} do not match {dimensions}.")
        episodes.append((path, obs, actions))

    if len(episodes) < 2:
        raise ValueError(f"Expected at least two NPZ episodes in {dataset_path} for train/validation split.")
    return episodes


def _episode_windows(
    episodes: list[tuple[Path, np.ndarray, np.ndarray]],
    future_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    observations = []
    action_chunks = []
    for _, obs, actions in episodes:
        count = len(actions) - future_length + 1
        observations.append(obs[:count])
        action_chunks.append(np.stack([actions[start : start + future_length] for start in range(count)]))
    return np.concatenate(observations), np.concatenate(action_chunks)


def load_behavior_cloning_data(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    future_length: int = 4,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> BehaviorCloningData:
    if future_length < 1:
        raise ValueError("future_length must be at least 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    episodes = _load_episodes(dataset_dir, future_length)
    order = np.random.default_rng(seed).permutation(len(episodes))
    val_count = min(len(episodes) - 1, max(1, round(len(episodes) * val_ratio)))
    val_indices = set(order[:val_count].tolist())
    train_episodes = [episode for index, episode in enumerate(episodes) if index not in val_indices]
    val_episodes = [episode for index, episode in enumerate(episodes) if index in val_indices]
    train_obs, train_actions = _episode_windows(train_episodes, future_length)
    val_obs, val_actions = _episode_windows(val_episodes, future_length)
    return BehaviorCloningData(
        train_obs=train_obs,
        train_actions=train_actions,
        val_obs=val_obs,
        val_actions=val_actions,
        train_episodes=tuple(episode[0].name for episode in train_episodes),
        val_episodes=tuple(episode[0].name for episode in val_episodes),
    )


def _normalization(data: BehaviorCloningData, eps: float) -> tuple[np.ndarray, ...]:
    actions = data.train_actions.reshape(-1, data.train_actions.shape[-1])
    obs_mean = data.train_obs.mean(axis=0, dtype=np.float64).astype(np.float32)
    obs_std = data.train_obs.std(axis=0, dtype=np.float64).astype(np.float32)
    action_mean = actions.mean(axis=0, dtype=np.float64).astype(np.float32)
    action_std = actions.std(axis=0, dtype=np.float64).astype(np.float32)
    obs_std[obs_std < eps] = 1.0
    action_std[action_std < eps] = 1.0
    return obs_mean, obs_std, action_mean, action_std


@torch.inference_mode()
def _evaluate(model: BehaviorCloningPolicy, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    normalized_error = 0.0
    raw_error = 0.0
    elements = 0
    for obs, actions in loader:
        obs, actions = obs.to(device), actions.to(device)
        prediction = model.forward_normalized(obs)
        target = model.normalize_actions(actions)
        normalized_error += F.mse_loss(prediction, target, reduction="sum").item()
        raw_error += F.mse_loss(model.denormalize_actions(prediction), actions, reduction="sum").item()
        elements += actions.numel()
    return normalized_error / elements, raw_error / elements


def _init_wandb(mode: str, output_dir: Path, config: dict[str, Any]):
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requested but wandb is not installed.") from exc
    return wandb.init(
        project="d2h-bc",
        name=f"rotate_object_bc_{output_dir.name}",
        mode=mode,
        config=config,
        dir=str(output_dir),
    )


def train(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_dir: str | Path | None = None,
    *,
    future_length: int = 4,
    hidden_dim: int = 256,
    num_layers: int = 2,
    normalization_eps: float = 1e-6,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 3e-4,
    val_ratio: float = 0.2,
    seed: int = 0,
    device: str | None = None,
    wandb_mode: str = "disabled",
    save_every_steps: int = 20_000,
) -> Path:
    if (
        epochs < 1
        or batch_size < 1
        or lr <= 0
        or normalization_eps <= 0
        or save_every_steps < 1
    ):
        raise ValueError(
            "epochs, batch_size, lr, normalization_eps, and save_every_steps must be positive"
        )

    dataset_path = Path(dataset_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir).expanduser() if output_dir is not None else Path("logs/bc") / dataset_path.name / timestamp
    if (out / "config.json").exists() or (out / "best.pt").exists():
        raise FileExistsError(f"Output directory already contains a BC run: {out}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = load_behavior_cloning_data(dataset_path, future_length, val_ratio, seed)
    config = BehaviorCloningConfig(
        state_dim=data.train_obs.shape[1],
        action_dim=data.train_actions.shape[2],
        future_length=future_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        normalization_eps=normalization_eps,
    )
    model = BehaviorCloningPolicy(config).to(torch_device)
    model.set_normalization(*_normalization(data, normalization_eps))

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(data.train_obs), torch.from_numpy(data.train_actions)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(data.val_obs), torch.from_numpy(data.val_actions)),
        batch_size=batch_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    out.mkdir(parents=True, exist_ok=True)
    training_config = {
        "dataset_dir": str(dataset_path),
        "output_dir": str(out),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "val_ratio": val_ratio,
        "seed": seed,
        "device": str(torch_device),
        "wandb_mode": wandb_mode,
        "save_every_steps": save_every_steps,
        "normalization": {"enabled": True, "eps": normalization_eps, "source": "train_split"},
    }
    config_payload = {
        "model_type": MODEL_TYPE,
        "config": asdict(config),
        "training": training_config,
        "data": {
            "observation_key": OBS_KEY,
            "action_key": ACTION_KEY,
            "train_episodes": data.train_episodes,
            "val_episodes": data.val_episodes,
            "train_windows": len(data.train_obs),
            "val_windows": len(data.val_obs),
        },
    }
    (out / "config.json").write_text(json.dumps(config_payload, indent=2))
    run = _init_wandb(wandb_mode, out, config_payload)
    best_path = out / "best.pt"
    best_val_loss = float("inf")
    global_step = 0

    print(
        f"device={torch_device} train_episodes={len(data.train_episodes)} "
        f"val_episodes={len(data.val_episodes)} train_windows={len(data.train_obs)} "
        f"val_windows={len(data.val_obs)}"
    )
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            elements = 0
            for obs, actions in train_loader:
                obs, actions = obs.to(torch_device), actions.to(torch_device)
                prediction = model.forward_normalized(obs)
                target = model.normalize_actions(actions)
                loss = F.mse_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                global_step += 1
                if global_step % save_every_steps == 0:
                    periodic_path = out / f"step_{global_step:09d}.pt"
                    torch.save(
                        {
                            "model_type": MODEL_TYPE,
                            "config": asdict(config),
                            "model_state_dict": model.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "observation_key": OBS_KEY,
                            "action_key": ACTION_KEY,
                        },
                        periodic_path,
                    )
                    if run is not None:
                        run.save(str(periodic_path))
                loss_sum += loss.item() * actions.numel()
                elements += actions.numel()

            train_loss = loss_sum / elements
            val_loss, val_raw_mse = _evaluate(model, val_loader, torch_device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_type": MODEL_TYPE,
                        "config": asdict(config),
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss,
                        "best_val_raw_mse": val_raw_mse,
                        "observation_key": OBS_KEY,
                        "action_key": ACTION_KEY,
                    },
                    best_path,
                )
            metrics = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/raw_mse": val_raw_mse,
                "best_val_loss": best_val_loss,
            }
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"val_raw_mse={val_raw_mse:.6f} best_val_loss={best_val_loss:.6f}"
            )
            if run is not None:
                run.log(metrics, step=epoch)
        if run is not None:
            run.save(str(best_path))
    finally:
        if run is not None:
            run.finish()
    return best_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=20_000)
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        wandb_mode=args.wandb_mode,
        save_every_steps=args.save_every_steps,
    )
