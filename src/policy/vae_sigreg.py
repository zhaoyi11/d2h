"""SIGReg-regularized deterministic autoencoder for trajectory chunks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.policy.vae import TrajectoryChunkDataset, VAEConfig, _move_batch


MODEL_TYPE = "conditional_trajectory_sigreg_autoencoder"


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer from LeWorldModel."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        if knots < 2:
            raise ValueError("knots must be at least 2")
        if num_proj < 1:
            raise ValueError("num_proj must be at least 1")

        self.num_proj = int(num_proj)
        t = torch.linspace(0, 3, int(knots), dtype=torch.float32)
        dt = 3 / (int(knots) - 1)
        weights = torch.full((int(knots),), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Compute SIGReg for embeddings shaped (batch, latent_dim)."""
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (batch, latent_dim)")

        directions = torch.randn(
            embeddings.size(-1),
            self.num_proj,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        directions = directions.div_(directions.norm(p=2, dim=0).clamp_min(torch.finfo(embeddings.dtype).eps))

        t = self.t.to(device=embeddings.device, dtype=embeddings.dtype)
        phi = self.phi.to(device=embeddings.device, dtype=embeddings.dtype)
        weights = self.weights.to(device=embeddings.device, dtype=embeddings.dtype)
        projected = (embeddings @ directions).unsqueeze(-1) * t
        err = (projected.cos().mean(0) - phi).square() + projected.sin().mean(0).square()
        statistic = (err @ weights) * embeddings.size(0)
        return statistic.mean()


class ConditionalTrajectorySIGRegAutoencoder(nn.Module):
    def __init__(self, config: VAEConfig, sigreg_knots: int = 17, sigreg_num_proj: int = 1024) -> None:
        super().__init__()
        self.config = config
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

        self.encoder = nn.Sequential(
            nn.Linear(config.encoder_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

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

    def encode(self, encoder_input: Tensor) -> Tensor:
        return self.encoder(encoder_input)

    def decode(self, condition: Tensor, latent: Tensor) -> Tensor:
        flat = self.decoder(torch.cat([condition, latent], dim=-1))
        return flat.reshape(-1, self.config.target_length, self.config.action_dim)

    def forward(self, encoder_input: Tensor, context: Tensor) -> dict[str, Tensor]:
        latent = self.encode(encoder_input)
        print('latent std', latent.std(dim=0).mean().item(), 'latent mean', latent.mean().item())
        condition = self.context_encoder(context)
        reconstruction = self.decode(condition, latent)
        return {"reconstruction": reconstruction, "latent": latent}

    def loss(
        self,
        output: dict[str, Tensor],
        target_actions: Tensor,
        sigreg_weight: float,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        reconstruction_loss = F.mse_loss(output["reconstruction"], target_actions)
        sigreg_loss = self.sigreg(output["latent"])
        loss = reconstruction_loss + sigreg_weight * sigreg_loss
        return loss, {
            "reconstruction_loss": reconstruction_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
        }


def train_vae_sigreg(
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
    sigreg_weight: float = 0.09,
    sigreg_knots: int = 17,
    sigreg_num_proj: int = 1024,
    seed: int = 0,
    device: str = "cuda",
) -> ConditionalTrajectorySIGRegAutoencoder:
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
    model = ConditionalTrajectorySIGRegAutoencoder(
        config,
        sigreg_knots=sigreg_knots,
        sigreg_num_proj=sigreg_num_proj,
    ).to(torch_device)
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
        "sigreg_weight": sigreg_weight,
        "sigreg_knots": sigreg_knots,
        "sigreg_num_proj": sigreg_num_proj,
        "seed": seed,
        "device": str(torch_device),
    }
    config_payload = asdict(config) | {"model_type": MODEL_TYPE, "training": train_args}
    with (output_path / "config.json").open("w") as f:
        json.dump(config_payload, f, indent=2)

    log_path = output_path / "train_log.jsonl"
    with log_path.open("w") as log_file:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_reconstruction = 0.0
            total_sigreg = 0.0
            total_batches = 0

            for batch in loader:
                batch = _move_batch(batch, torch_device)
                output = model(batch["encoder_input"], batch["context"])
                loss, metrics = model.loss(output, batch["target_actions"], sigreg_weight=sigreg_weight)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_reconstruction += float(metrics["reconstruction_loss"].cpu())
                total_sigreg += float(metrics["sigreg_loss"].cpu())
                total_batches += 1

            row = {
                "epoch": epoch,
                "loss": total_loss / total_batches,
                "reconstruction_loss": total_reconstruction / total_batches,
                "sigreg_loss": total_sigreg / total_batches,
            }
            print(json.dumps(row), file=log_file, flush=True)
            print(
                f"epoch {epoch:04d} loss={row['loss']:.6f} "
                f"recon={row['reconstruction_loss']:.6f} sigreg={row['sigreg_loss']:.6f}",
                flush=True,
            )

    torch.save(
        {
            "config": asdict(config),
            "model_type": MODEL_TYPE,
            "sigreg": {"weight": sigreg_weight, "knots": sigreg_knots, "num_proj": sigreg_num_proj},
            "model_state_dict": model.state_dict(),
        },
        output_path / "model.pt",
    )
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SIGReg trajectory autoencoder from collected episodes.")
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
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_vae_sigreg(
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
        sigreg_weight=args.sigreg_weight,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
