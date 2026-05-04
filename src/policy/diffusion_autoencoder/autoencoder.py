"""DiT-based trajectory autoencoder with SIGReg latent regularization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.policy.diffusion_autoencoder.fm import FlowMatchingObjective
from src.policy.diffusion_autoencoder.transformer import SinusoidalPosEmb, modulate
from src.policy.vae import TrajectoryChunkDataset, _move_batch
from src.policy.vae_sigreg import SIGReg


MODEL_TYPE = "conditional_trajectory_diffusion_autoencoder"


@dataclass
class DiffusionAutoencoderConfig:
    state_dim: int
    action_dim: int
    past_length: int = 4
    future_length: int = 8
    latent_dim: int = 64
    condition_dim: int = 256
    hidden_dim: int = 512
    transformer_hidden_dim: int = 128
    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_dropout: float = 0.0
    timestep_embed_dim: int = 128
    objective: str = "flow_matching"
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    sigma_min: float = 1e-4
    num_diffusion_steps: int = 100
    num_sample_steps: int = 20

    @property
    def context_dim(self) -> int:
        return self.past_length * (self.state_dim + self.action_dim)

    @property
    def encoder_input_dim(self) -> int:
        return (self.past_length + self.future_length) * (self.state_dim + self.action_dim)

    @property
    def target_length(self) -> int:
        return self.future_length


class SequenceEncoder(nn.Module):
    def __init__(self, config: DiffusionAutoencoderConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
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

    def forward(self, encoder_input: Tensor) -> Tensor:
        return self.net(encoder_input)


class SeparateConditionTransformerBlock(nn.Module):
    """DiT block with separate AdaLN branches for time, context, and latent."""

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
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 6 * hidden_size, bias=True))
        self.context_modulation = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, 6 * hidden_size, bias=True))
        self.latent_modulation = nn.Sequential(nn.SiLU(), nn.Linear(latent_dim, 6 * hidden_size, bias=True))

    def forward(self, x: Tensor, time_features: Tensor, context_features: Tensor, latent: Tensor) -> Tensor:
        modulation = (
            self.time_modulation(time_features)
            + self.context_modulation(context_features)
            + self.latent_modulation(latent)
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        attn_input = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        attn_out, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + gate_msa.unsqueeze(1) * attn_out

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
            for branch in (block.time_modulation, block.context_modulation, block.latent_modulation):
                nn.init.constant_(branch[-1].weight, 0)
                nn.init.constant_(branch[-1].bias, 0)

    def forward(self, actions: Tensor, timestep: Tensor, context_features: Tensor, latent: Tensor) -> Tensor:
        _, seq_len, _ = actions.shape
        hidden = self.input_proj(actions)
        hidden = hidden + self.pos_embedding[:, :seq_len]
        time_features = self.time_mlp(timestep)

        for block in self.blocks:
            hidden = block(hidden, time_features, context_features, latent)
        return self.output_proj(hidden)


class ConditionalTrajectoryDiffusionAutoencoder(nn.Module):
    def __init__(self, config: DiffusionAutoencoderConfig) -> None:
        super().__init__()
        if config.objective not in {"diffusion", "flow_matching"}:
            raise ValueError("objective must be 'diffusion' or 'flow_matching'")
        self.config = config
        self.encoder = SequenceEncoder(config)
        self.context_encoder = nn.Sequential(
            nn.Linear(config.context_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, config.condition_dim),
            nn.LayerNorm(config.condition_dim),
            nn.ReLU(),
        )
        self.decoder = TrajectoryDiTDecoder(config)
        self.sigreg = SIGReg(knots=config.sigreg_knots, num_proj=config.sigreg_num_proj)
        self.latent_prior = nn.Parameter(torch.zeros(config.latent_dim))
        self.flow_matching = FlowMatchingObjective(config, config.action_dim, config.target_length)

        beta = torch.linspace(1e-4, 2e-2, config.num_diffusion_steps, dtype=torch.float32)
        alpha = 1.0 - beta
        self.register_buffer("sqrt_alpha_bar", torch.cumprod(alpha, dim=0).sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - torch.cumprod(alpha, dim=0)).sqrt())

    def encode(self, encoder_input: Tensor) -> Tensor:
        return self.encoder(encoder_input)

    def _flow_matching_forward(self, context_features: Tensor, latent: Tensor, target_actions: Tensor) -> dict[str, Tensor]:
        noised, timestep, target = self.flow_matching.prepare_training_sample(target_actions)
        prediction = self.decoder(noised, timestep, context_features, latent)
        return {"prediction": prediction, "target": target, "objective_loss_key": "flow_matching_loss"}

    def _diffusion_forward(self, context_features: Tensor, latent: Tensor, target_actions: Tensor) -> dict[str, Tensor]:
        batch_size = target_actions.shape[0]
        timesteps = torch.randint(
            0,
            self.config.num_diffusion_steps,
            (batch_size,),
            device=target_actions.device,
        )
        noise = torch.randn_like(target_actions)
        alpha = self.sqrt_alpha_bar[timesteps].to(dtype=target_actions.dtype).view(-1, 1, 1)
        sigma = self.sqrt_one_minus_alpha_bar[timesteps].to(dtype=target_actions.dtype).view(-1, 1, 1)
        noised = alpha * target_actions + sigma * noise
        prediction = self.decoder(noised, timesteps.to(dtype=target_actions.dtype), context_features, latent)
        return {"prediction": prediction, "target": noise, "objective_loss_key": "diffusion_loss"}

    def forward(self, encoder_input: Tensor, context: Tensor, target_actions: Tensor) -> dict[str, Tensor]:
        latent = self.encode(encoder_input)
        context_features = self.context_encoder(context)
        if self.config.objective == "flow_matching":
            output = self._flow_matching_forward(context_features, latent, target_actions)
        else:
            output = self._diffusion_forward(context_features, latent, target_actions)
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
    def sample(self, context: Tensor, latent: Tensor | None = None) -> Tensor:
        batch_size = context.shape[0]
        context_features = self.context_encoder(context)
        if latent is None:
            latent = self.latent_prior.unsqueeze(0).expand(batch_size, -1)

        sample = torch.randn(
            batch_size,
            self.config.target_length,
            self.config.action_dim,
            device=context.device,
            dtype=context.dtype,
        )

        if self.config.objective == "flow_matching":
            return self._sample_flow_matching(sample, context_features, latent)
        return self._sample_diffusion(sample, context_features, latent)

    def _sample_flow_matching(self, sample: Tensor, context_features: Tensor, latent: Tensor) -> Tensor:
        return self.flow_matching.integrate(
            lambda x, t: self.decoder(x, t, context_features, latent),
            sample,
        )

    def _sample_diffusion(self, sample: Tensor, context_features: Tensor, latent: Tensor) -> Tensor:
        steps = torch.linspace(
            self.config.num_diffusion_steps - 1,
            0,
            self.config.num_sample_steps,
            device=sample.device,
        ).long()
        x = sample
        for timestep in steps:
            t_batch = timestep.to(dtype=x.dtype).expand(x.shape[0])
            predicted_noise = self.decoder(x, t_batch, context_features, latent)
            alpha = self.sqrt_alpha_bar[timestep].to(dtype=x.dtype).clamp_min(torch.finfo(x.dtype).eps)
            sigma = self.sqrt_one_minus_alpha_bar[timestep].to(dtype=x.dtype)
            x = (x - sigma * predicted_noise) / alpha
        return x


def train_diffusion_autoencoder(
    dataset_dir: str | Path,
    output_dir: str | Path,
    past_length: int = 4,
    future_length: int = 8,
    latent_dim: int = 64,
    condition_dim: int = 256,
    hidden_dim: int = 512,
    transformer_hidden_dim: int = 128,
    transformer_layers: int = 4,
    transformer_heads: int = 4,
    transformer_dropout: float = 0.0,
    timestep_embed_dim: int = 128,
    objective: str = "flow_matching",
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    sigreg_weight: float = 0.09,
    sigreg_knots: int = 17,
    sigreg_num_proj: int = 1024,
    sigma_min: float = 1e-4,
    num_diffusion_steps: int = 100,
    num_sample_steps: int = 20,
    seed: int = 0,
    device: str = "cuda",
) -> ConditionalTrajectoryDiffusionAutoencoder:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TrajectoryChunkDataset(
        dataset_dir,
        past_length=past_length,
        future_length=future_length,
        min_chunks=batch_size,
        seed=seed,
    )
    config = DiffusionAutoencoderConfig(
        state_dim=int(dataset.state_dim),
        action_dim=int(dataset.action_dim),
        past_length=past_length,
        future_length=future_length,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        transformer_hidden_dim=transformer_hidden_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_dropout=transformer_dropout,
        timestep_embed_dim=timestep_embed_dim,
        objective=objective,
        sigreg_weight=sigreg_weight,
        sigreg_knots=sigreg_knots,
        sigreg_num_proj=sigreg_num_proj,
        sigma_min=sigma_min,
        num_diffusion_steps=num_diffusion_steps,
        num_sample_steps=num_sample_steps,
    )

    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ConditionalTrajectoryDiffusionAutoencoder(config).to(torch_device)
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
        "transformer_hidden_dim": transformer_hidden_dim,
        "transformer_layers": transformer_layers,
        "transformer_heads": transformer_heads,
        "transformer_dropout": transformer_dropout,
        "timestep_embed_dim": timestep_embed_dim,
        "objective": objective,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "sigreg_weight": sigreg_weight,
        "sigreg_knots": sigreg_knots,
        "sigreg_num_proj": sigreg_num_proj,
        "sigma_min": sigma_min,
        "num_diffusion_steps": num_diffusion_steps,
        "num_sample_steps": num_sample_steps,
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
            total_objective = 0.0
            total_sigreg = 0.0
            total_batches = 0
            objective_key = "flow_matching_loss" if objective == "flow_matching" else "diffusion_loss"

            for batch in loader:
                batch = _move_batch(batch, torch_device)
                output = model(batch["encoder_input"], batch["context"], batch["target_actions"])
                loss, metrics = model.loss(output, batch["target_actions"])

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_objective += float(metrics[objective_key].cpu())
                total_sigreg += float(metrics["sigreg_loss"].cpu())
                total_batches += 1

            row = {
                "epoch": epoch,
                "loss": total_loss / total_batches,
                objective_key: total_objective / total_batches,
                "sigreg_loss": total_sigreg / total_batches,
            }
            print(json.dumps(row), file=log_file, flush=True)
            print(
                f"epoch {epoch:04d} loss={row['loss']:.6f} "
                f"{objective_key}={row[objective_key]:.6f} sigreg={row['sigreg_loss']:.6f}",
                flush=True,
            )

    torch.save(
        {
            "config": asdict(config),
            "model_type": MODEL_TYPE,
            "model_state_dict": model.state_dict(),
        },
        output_path / "model.pt",
    )
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SIGReg DiT trajectory autoencoder.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing metadata.json and episode .npz files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write config.json, model.pt, and logs.")
    parser.add_argument("--past-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--transformer-hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--timestep-embed-dim", type=int, default=128)
    parser.add_argument("--objective", choices=("diffusion", "flow_matching"), default="flow_matching")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--num-diffusion-steps", type=int, default=100)
    parser.add_argument("--num-sample-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_diffusion_autoencoder(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        past_length=args.past_length,
        future_length=args.future_length,
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        hidden_dim=args.hidden_dim,
        transformer_hidden_dim=args.transformer_hidden_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dropout=args.transformer_dropout,
        timestep_embed_dim=args.timestep_embed_dim,
        objective=args.objective,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        sigreg_weight=args.sigreg_weight,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        sigma_min=args.sigma_min,
        num_diffusion_steps=args.num_diffusion_steps,
        num_sample_steps=args.num_sample_steps,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
