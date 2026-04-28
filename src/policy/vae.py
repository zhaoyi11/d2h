"""Conditional VAE for compressing collected trajectory chunks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


STATE_KEYS = ("observation.policy", "observation.perception")


@dataclass
class VAEConfig:
    state_dim: int
    action_dim: int
    past_length: int = 4
    future_length: int = 8
    latent_dim: int = 64
    condition_dim: int = 256
    hidden_dim: int = 512

    @property
    def context_dim(self) -> int:
        return self.past_length * (self.state_dim + self.action_dim)

    @property
    def encoder_input_dim(self) -> int:
        return (self.past_length + self.future_length) * (self.state_dim + self.action_dim)

    @property
    def target_length(self) -> int:
        return self.future_length


class TrajectoryChunkDataset(Dataset):
    """Loads one random fixed-length action window per eligible episode."""

    def __init__(
        self,
        dataset_dir: str | Path,
        past_length: int = 4,
        future_length: int = 8,
        state_keys: tuple[str, ...] = STATE_KEYS,
        min_chunks: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.past_length = int(past_length)
        self.future_length = int(future_length)
        self.required_length = self.past_length + self.future_length
        self.state_keys = tuple(state_keys)
        self.min_chunks = None if min_chunks is None else int(min_chunks)
        self._rng = np.random.default_rng(seed)

        if self.past_length < 1:
            raise ValueError("past_length must be at least 1")
        if self.future_length < 1:
            raise ValueError("future_length must be at least 1")
        if self.min_chunks is not None and self.min_chunks < 1:
            raise ValueError("min_chunks must be at least 1")
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        metadata_path = self.dataset_dir / "metadata.json"
        self.metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with metadata_path.open() as f:
                self.metadata = json.load(f)

        self._episodes: list[tuple[Path, int]] = []
        self._episode_indices: list[int] = []
        self.state_dim: int | None = None
        self.action_dim: int | None = None

        for episode_path in sorted(self.dataset_dir.glob("*.npz")):
            with np.load(episode_path) as episode:
                for key in (*self.state_keys, "action"):
                    if key not in episode:
                        raise KeyError(f"{episode_path} is missing required array {key!r}")

                length = int(episode["action"].shape[0])
                if length < self.required_length:
                    print(
                        f"[WARN] Skipping {episode_path}: length {length} < required_length {self.required_length}",
                        file=sys.stderr,
                    )
                    continue

                states = self._states_from_episode(episode)
                actions = self._actions_from_episode(episode)
                if states.shape[0] < length:
                    raise ValueError(
                        f"{episode_path} has {states.shape[0]} states for {length} actions; expected at least {length}"
                    )

                if self.state_dim is None:
                    self.state_dim = int(states.shape[-1])
                    self.action_dim = int(actions.shape[-1])
                elif self.state_dim != states.shape[-1] or self.action_dim != actions.shape[-1]:
                    raise ValueError(f"{episode_path} has dimensions inconsistent with earlier episodes")

                self._episodes.append((episode_path, length))

        if not self._episodes:
            raise RuntimeError(
                f"No eligible episodes found in {self.dataset_dir} for required_length {self.required_length}"
            )

        assert self.state_dim is not None
        assert self.action_dim is not None
        target_chunks = (
            len(self._episodes) if self.min_chunks is None else max(len(self._episodes), self.min_chunks)
        )
        self._episode_indices = [i % len(self._episodes) for i in range(target_chunks)]

    def __len__(self) -> int:
        return len(self._episode_indices)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        episode_path, length = self._episodes[self._episode_indices[idx]]
        max_start = length - self.required_length
        start = int(self._rng.integers(0, max_start + 1)) if max_start > 0 else 0
        end = start + self.required_length

        with np.load(episode_path) as episode:
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


class ConditionalTrajectoryVAE(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config.encoder_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, config.hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)

        self.context_encoder = nn.Sequential(
            nn.Linear(config.context_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, config.condition_dim),
            nn.LayerNorm(config.condition_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(config.condition_dim + config.latent_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, config.target_length * config.action_dim),
        )

    def encode(self, encoder_input: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(encoder_input)
        return self.mu(hidden), self.logvar(hidden)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, condition: Tensor, z: Tensor) -> Tensor:
        flat = self.decoder(torch.cat([condition, z], dim=-1))
        return flat.reshape(-1, self.config.target_length, self.config.action_dim)

    def forward(self, encoder_input: Tensor, context: Tensor) -> dict[str, Tensor]:
        mu, logvar = self.encode(encoder_input)
        z = self.reparameterize(mu, logvar)
        condition = self.context_encoder(context)
        reconstruction = self.decode(condition, z)
        return {"reconstruction": reconstruction, "mu": mu, "logvar": logvar, "z": z}

    def loss(self, output: dict[str, Tensor], target_actions: Tensor, beta: float) -> tuple[Tensor, dict[str, Tensor]]:
        reconstruction_loss = F.mse_loss(output["reconstruction"], target_actions)
        kl_loss = -0.5 * torch.mean(1 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp())
        loss = reconstruction_loss + beta * kl_loss
        return loss, {"reconstruction_loss": reconstruction_loss.detach(), "kl_loss": kl_loss.detach()}


def _move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_vae(
    dataset_dir: str | Path,
    output_dir: str | Path,
    past_length: int = 4,
    future_length: int = 8,
    latent_dim: int = 64,
    condition_dim: int = 256,
    hidden_dim: int = 512,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    beta: float = 1e-4,
    seed: int = 0,
    device: str = "cuda",
) -> ConditionalTrajectoryVAE:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TrajectoryChunkDataset(
        dataset_dir,
        past_length=past_length,
        future_length=future_length,
        min_chunks=batch_size,
        seed=seed,
    )
    config = VAEConfig(
        state_dim=int(dataset.state_dim),
        action_dim=int(dataset.action_dim),
        past_length=past_length,
        future_length=future_length,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
    )

    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ConditionalTrajectoryVAE(config).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    train_args = {
        "dataset_dir": str(Path(dataset_dir).expanduser()),
        "past_length": past_length,
        "future_length": future_length,
        "latent_dim": latent_dim,
        "condition_dim": condition_dim,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "beta": beta,
        "seed": seed,
        "device": str(torch_device),
    }
    config_payload = asdict(config) | {"training": train_args}
    with (output_path / "config.json").open("w") as f:
        json.dump(config_payload, f, indent=2)

    log_path = output_path / "train_log.jsonl"
    with log_path.open("w") as log_file:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_reconstruction = 0.0
            total_kl = 0.0
            total_batches = 0

            for batch in loader:
                batch = _move_batch(batch, torch_device)
                import ipdb; ipdb.set_trace()
                output = model(batch["encoder_input"], batch["context"])
                loss, metrics = model.loss(output, batch["target_actions"], beta=beta)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_reconstruction += float(metrics["reconstruction_loss"].cpu())
                total_kl += float(metrics["kl_loss"].cpu())
                total_batches += 1

            row = {
                "epoch": epoch,
                "loss": total_loss / total_batches,
                "reconstruction_loss": total_reconstruction / total_batches,
                "kl_loss": total_kl / total_batches,
            }
            print(json.dumps(row), file=log_file, flush=True)
            print(
                f"epoch {epoch:04d} loss={row['loss']:.6f} "
                f"recon={row['reconstruction_loss']:.6f} kl={row['kl_loss']:.6f}",
                flush=True,
            )

    torch.save({"config": asdict(config), "model_state_dict": model.state_dict()}, output_path / "model.pt")
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional trajectory VAE from collected episodes.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing metadata.json and episode .npz files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write config.json, model.pt, and logs.")
    parser.add_argument("--past-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_vae(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        past_length=args.past_length,
        future_length=args.future_length,
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
