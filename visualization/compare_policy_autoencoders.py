"""Compare SIGReg and diffusion trajectory autoencoder checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.policy.diffusion_autoencoder.autoencoder import (
    ConditionalTrajectoryDiffusionAutoencoder,
    DiffusionAutoencoderConfig,
    _build_window_datasets,
    _torch_load_checkpoint,
)
from src.policy.vae import VAEConfig, _move_batch
from src.policy.vae_sigreg import ConditionalTrajectorySIGRegAutoencoder


def _load_sigreg_model(path: Path, device: torch.device) -> ConditionalTrajectorySIGRegAutoencoder:
    checkpoint = _torch_load_checkpoint(path, map_location=device)
    config = VAEConfig(**checkpoint["config"])
    sigreg = checkpoint.get("sigreg", {})
    model = ConditionalTrajectorySIGRegAutoencoder(
        config,
        sigreg_knots=int(sigreg.get("knots", 17)),
        sigreg_num_proj=int(sigreg.get("num_proj", 1024)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _normalize_diffusion_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    objective = normalized.pop("objective", None)
    if objective not in {None, "flow_matching"}:
        raise ValueError(f"Unsupported diffusion autoencoder checkpoint objective: {objective!r}")
    normalized.pop("num_diffusion_steps", None)
    return normalized


def _load_diffusion_model(path: Path, device: torch.device) -> ConditionalTrajectoryDiffusionAutoencoder:
    checkpoint = _torch_load_checkpoint(path, map_location=device)
    config = DiffusionAutoencoderConfig(**_normalize_diffusion_config(checkpoint["config"]))
    state_dict = checkpoint["model_state_dict"]
    encoder_hidden = state_dict.get("encoder.net.6.weight")
    if encoder_hidden is not None and int(encoder_hidden.shape[0]) != config.hidden_dim:
        config.hidden_dim = int(encoder_hidden.shape[0])

    model = ConditionalTrajectoryDiffusionAutoencoder(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _initial_sample(reference: Tensor, seed: int | None, batch_idx: int) -> Tensor | None:
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + batch_idx)
    sample = torch.randn(reference.shape, dtype=reference.dtype, generator=generator)
    return sample.to(device=reference.device)


@torch.no_grad()
def compare_checkpoints(
    dataset_dir: str | Path,
    sigreg_checkpoint: str | Path,
    diffusion_checkpoint: str | Path,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    val_windows: int = 10_000,
    seed: int = 0,
    initial_sample_seed: int | None = 0,
    num_workers: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    sigreg_model = _load_sigreg_model(Path(sigreg_checkpoint).expanduser(), torch_device)
    diffusion_model = _load_diffusion_model(Path(diffusion_checkpoint).expanduser(), torch_device)

    if sigreg_model.config.context_dim != diffusion_model.config.context_input_dim:
        raise ValueError("Checkpoint context dimensions do not match.")
    if sigreg_model.config.target_length != diffusion_model.config.target_length:
        raise ValueError("Checkpoint target lengths do not match.")
    if sigreg_model.config.action_dim != diffusion_model.config.action_dim:
        raise ValueError("Checkpoint action dimensions do not match.")

    training = _torch_load_checkpoint(Path(diffusion_checkpoint).expanduser(), map_location="cpu").get("training", {})
    state_keys = training.get("state_keys")
    _, val_dataset, info = _build_window_datasets(
        dataset_dir,
        past_length=diffusion_model.config.past_length,
        future_length=diffusion_model.config.future_length,
        val_ratio=val_ratio,
        train_windows_per_epoch=1,
        val_windows=val_windows,
        seed=seed,
        state_keys=None if state_keys is None else tuple(state_keys),
    )
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    sigreg_sum = 0.0
    diffusion_sum = 0.0
    elements = 0
    for batch_idx, batch in enumerate(loader):
        batch = _move_batch(batch, torch_device)
        target = batch["target_actions"]

        sigreg_output = sigreg_model(batch["encoder_input"], batch["context"])
        sigreg_sum += float(F.mse_loss(sigreg_output["reconstruction"], target, reduction="sum").cpu())

        diffusion_output = diffusion_model(batch["encoder_input"], batch["context"], target)
        diffusion_sample = diffusion_model.sample(
            batch["context"],
            latent=diffusion_output["latent"],
            initial_sample=_initial_sample(target, initial_sample_seed, batch_idx),
        )
        diffusion_sum += float(F.mse_loss(diffusion_sample, target, reduction="sum").cpu())
        elements += int(target.numel())

    return {
        "sigreg_reconstruction_mse": sigreg_sum / elements,
        "diffusion_sample_mse": diffusion_sum / elements,
        "val_windows": val_windows,
        "state_keys": info.state_keys,
        "device": str(torch_device),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--sigreg-checkpoint", required=True)
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-windows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-sample-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = compare_checkpoints(
        dataset_dir=args.dataset_dir,
        sigreg_checkpoint=args.sigreg_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        val_windows=args.val_windows,
        seed=args.seed,
        initial_sample_seed=args.initial_sample_seed,
        num_workers=args.num_workers,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
