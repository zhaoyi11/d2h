"""Visualize latent spaces learned by VAE and SIGReg trajectory autoencoders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde, kurtosis, skew, wasserstein_distance
from torch import Tensor
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.policy.vae import (  # noqa: E402
    ConditionalTrajectoryVAE,
    VAEConfig,
    _build_window_datasets,
    _move_batch,
    _torch_load_checkpoint,
)
from src.policy.vae_sigreg import ConditionalTrajectorySIGRegAutoencoder  # noqa: E402


DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "Reorient_Play-v0" / "test"
DEFAULT_VAE_CHECKPOINT = REPO_ROOT / "vae" / "best.pt"
DEFAULT_SIGREG_CHECKPOINT = REPO_ROOT / "vae_sigreg" / "best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "visualization" / "compare_sigreg_vae_latent_kde.png"
NORMAL_REFERENCE_SEED = 123
DEFAULT_KDE_MIN_DENSITY_FRACTION = 0.05


def _load_vae_model(path: Path, device: torch.device) -> tuple[ConditionalTrajectoryVAE, dict[str, Any]]:
    checkpoint = _torch_load_checkpoint(path, map_location=device)
    config = VAEConfig(**checkpoint["config"])
    model = ConditionalTrajectoryVAE(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _load_sigreg_model(
    path: Path,
    device: torch.device,
) -> tuple[ConditionalTrajectorySIGRegAutoencoder, dict[str, Any]]:
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
    return model, checkpoint


def _checkpoint_state_keys(checkpoint: dict[str, Any]) -> tuple[str, ...] | None:
    state_keys = checkpoint.get("training", {}).get("state_keys")
    return None if state_keys is None else tuple(state_keys)


def _validate_matching_configs(vae_config: VAEConfig, sigreg_config: VAEConfig) -> None:
    fields = ("state_dim", "action_dim", "past_length", "future_length", "latent_dim")
    mismatches = [name for name in fields if getattr(vae_config, name) != getattr(sigreg_config, name)]
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(f"VAE and SIGReg checkpoint configs differ for: {joined}")


@torch.no_grad()
def collect_latents(
    dataset_dir: str | Path,
    vae_checkpoint: str | Path,
    sigreg_checkpoint: str | Path,
    batch_size: int = 256,
    num_windows: int = 10_000,
    val_ratio: float = 0.1,
    seed: int = 0,
    num_workers: int = 0,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    vae_model, vae_payload = _load_vae_model(Path(vae_checkpoint).expanduser(), torch_device)
    sigreg_model, sigreg_payload = _load_sigreg_model(Path(sigreg_checkpoint).expanduser(), torch_device)
    _validate_matching_configs(vae_model.config, sigreg_model.config)

    state_keys = _checkpoint_state_keys(vae_payload) or _checkpoint_state_keys(sigreg_payload)
    _, val_dataset, info = _build_window_datasets(
        dataset_dir,
        past_length=vae_model.config.past_length,
        future_length=vae_model.config.future_length,
        val_ratio=val_ratio,
        train_windows_per_epoch=1,
        val_windows=num_windows,
        seed=seed,
        state_keys=state_keys,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch_device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    vae_latents = []
    sigreg_latents = []
    for batch in loader:
        batch = _move_batch(batch, torch_device)
        mu, logvar = vae_model.encode(batch["encoder_input"])
        vae_latents.append(vae_model.reparameterize(mu, logvar).cpu())
        sigreg_latents.append(sigreg_model.encode(batch["encoder_input"]).cpu())

    metadata = {
        "dataset_dir": str(Path(dataset_dir).expanduser()),
        "vae_checkpoint": str(Path(vae_checkpoint).expanduser()),
        "sigreg_checkpoint": str(Path(sigreg_checkpoint).expanduser()),
        "num_windows": num_windows,
        "state_keys": info.state_keys,
        "device": str(torch_device),
    }
    return torch.cat(vae_latents).numpy(), torch.cat(sigreg_latents).numpy(), metadata


def _as_latent_matrix(latents: np.ndarray, label: str) -> np.ndarray:
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim != 2 or latents.shape[1] < 2:
        raise ValueError(f"{label} latents must have shape (num_samples, latent_dim>=2)")
    if latents.shape[0] < 3:
        raise ValueError(f"{label} latents need at least 3 samples for KDE")
    return latents


def _validate_latent_dims(dims: tuple[int, int], latent_dim: int, label: str) -> tuple[int, int]:
    dim_0, dim_1 = (int(dims[0]), int(dims[1]))
    if dim_0 == dim_1:
        raise ValueError(f"{label} latent dims must be distinct")
    for dim in (dim_0, dim_1):
        if dim < 0 or dim >= latent_dim:
            raise ValueError(f"{label} latent dim {dim} is outside [0, {latent_dim})")
    return dim_0, dim_1


def _latent_dim_statistics(latents: np.ndarray) -> dict[str, np.ndarray]:
    reference = np.random.default_rng(NORMAL_REFERENCE_SEED).standard_normal(latents.shape[0])
    means = latents.mean(axis=0)
    stds = latents.std(axis=0)
    skews = np.nan_to_num(skew(latents, axis=0, bias=False), nan=0.0)
    excess_kurtosis = np.nan_to_num(kurtosis(latents, axis=0, fisher=True, bias=False), nan=0.0)
    wasserstein = np.array(
        [wasserstein_distance(latents[:, dim], reference) for dim in range(latents.shape[1])],
        dtype=np.float64,
    )
    scores = (
        1.5 * np.abs(skews)
        + np.abs(means)
        + np.abs(stds - 1.0)
        + 0.25 * np.abs(excess_kurtosis)
        + wasserstein
    )
    return {
        "mean": means,
        "std": stds,
        "skew": skews,
        "excess_kurtosis": excess_kurtosis,
        "wasserstein_distance": wasserstein,
        "score": scores,
    }


def _statistics_metadata(statistics: dict[str, np.ndarray]) -> list[dict[str, float]]:
    dim_count = len(statistics["score"])
    return [
        {
            "mean": float(statistics["mean"][dim]),
            "std": float(statistics["std"][dim]),
            "skew": float(statistics["skew"][dim]),
            "excess_kurtosis": float(statistics["excess_kurtosis"][dim]),
            "wasserstein_distance": float(statistics["wasserstein_distance"][dim]),
            "score": float(statistics["score"][dim]),
        }
        for dim in range(dim_count)
    ]


def select_latent_dims(
    latents: np.ndarray,
    label: str,
    dims: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], dict[str, Any]]:
    latents = _as_latent_matrix(latents, label)
    statistics = _latent_dim_statistics(latents)
    if dims is None:
        selected = tuple(int(dim) for dim in np.argsort(statistics["score"])[-2:][::-1])
    else:
        selected = _validate_latent_dims(dims, latents.shape[1], label)

    metadata = {
        "dims": [int(selected[0]), int(selected[1])],
        "score_formula": (
            "1.5*abs(skew) + abs(mean) + abs(std-1) + "
            "0.25*abs(excess_kurtosis) + wasserstein_distance_to_N(0,1)"
        ),
        "normal_reference_seed": NORMAL_REFERENCE_SEED,
        "per_dim": _statistics_metadata(statistics),
    }
    return selected, metadata


def _latent_pair(latents: np.ndarray, label: str, dims: tuple[int, int] | None) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    latents = _as_latent_matrix(latents, label)
    selected_dims, metadata = select_latent_dims(latents, label, dims=dims)
    return latents[:, selected_dims], selected_dims, metadata


def _axis_limits(*arrays: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.concatenate(arrays, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    padding = 0.08 * span
    return (float(mins[0] - padding[0]), float(maxs[0] + padding[0])), (
        float(mins[1] - padding[1]),
        float(maxs[1] + padding[1]),
    )


def _kde_levels(density: np.ndarray, num_levels: int, min_density_fraction: float) -> np.ndarray:
    if num_levels < 2:
        raise ValueError("num_levels must be at least 2")
    if min_density_fraction < 0.0 or min_density_fraction >= 1.0:
        raise ValueError("min_density_fraction must be in [0, 1)")

    max_density = float(np.max(density))
    if not np.isfinite(max_density) or max_density <= 0.0:
        raise ValueError("density must contain a positive finite value")
    min_density = max_density * float(min_density_fraction)
    return np.linspace(min_density, max_density, num_levels)


def _plot_panel(
    ax: plt.Axes,
    points: np.ndarray,
    title: str,
    dims: tuple[int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    kde_min_density_fraction: float,
) -> None:
    x_grid, y_grid = np.mgrid[xlim[0] : xlim[1] : 120j, ylim[0] : ylim[1] : 120j]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    density = gaussian_kde(points.T)(positions).reshape(x_grid.shape)
    filled_levels = _kde_levels(density, num_levels=16, min_density_fraction=kde_min_density_fraction)
    line_levels = _kde_levels(density, num_levels=8, min_density_fraction=kde_min_density_fraction)

    ax.contourf(x_grid, y_grid, density, levels=filled_levels, cmap="Blues", alpha=0.65)
    ax.contour(x_grid, y_grid, density, levels=line_levels, colors="#1f4e79", linewidths=0.75, alpha=0.8)
    ax.scatter(points[:, 0], points[:, 1], s=8, c="#202020", alpha=0.25, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel(f"latent dim {dims[0]}")
    ax.set_ylabel(f"latent dim {dims[1]}")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def plot_latent_kde(
    vae_latents: np.ndarray,
    sigreg_latents: np.ndarray,
    output: str | Path,
    vae_label: str = "Original VAE sampled z",
    sigreg_label: str = "SIGReg latent",
    vae_dims: tuple[int, int] | None = None,
    sigreg_dims: tuple[int, int] | None = None,
    kde_min_density_fraction: float = DEFAULT_KDE_MIN_DENSITY_FRACTION,
) -> dict[str, Any]:
    vae_points, selected_vae_dims, vae_stats = _latent_pair(vae_latents, "VAE", vae_dims)
    sigreg_points, selected_sigreg_dims, sigreg_stats = _latent_pair(sigreg_latents, "SIGReg", sigreg_dims)
    xlim, ylim = _axis_limits(vae_points, sigreg_points)

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _plot_panel(axes[0], vae_points, vae_label, selected_vae_dims, xlim, ylim, kde_min_density_fraction)
    _plot_panel(axes[1], sigreg_points, sigreg_label, selected_sigreg_dims, xlim, ylim, kde_min_density_fraction)
    fig.suptitle("Selected Non-Gaussian Latent Dimensions with KDE Fit")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "vae_latent_dims": [int(selected_vae_dims[0]), int(selected_vae_dims[1])],
        "sigreg_latent_dims": [int(selected_sigreg_dims[0]), int(selected_sigreg_dims[1])],
        "kde_min_density_fraction": float(kde_min_density_fraction),
        "vae_latent_dim_stats": vae_stats,
        "sigreg_latent_dim_stats": sigreg_stats,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--vae-checkpoint", default=str(DEFAULT_VAE_CHECKPOINT))
    parser.add_argument("--sigreg-checkpoint", default=str(DEFAULT_SIGREG_CHECKPOINT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-windows", type=int, default=10_000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-latent-dims", type=int, nargs=2, default=None, metavar=("I", "J"))
    parser.add_argument("--sigreg-latent-dims", type=int, nargs=2, default=None, metavar=("I", "J"))
    parser.add_argument("--kde-min-density-fraction", type=float, default=DEFAULT_KDE_MIN_DENSITY_FRACTION)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    vae_latents, sigreg_latents, metadata = collect_latents(
        dataset_dir=args.dataset_dir,
        vae_checkpoint=args.vae_checkpoint,
        sigreg_checkpoint=args.sigreg_checkpoint,
        batch_size=args.batch_size,
        num_windows=args.num_windows,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )
    plot_metadata = plot_latent_kde(
        vae_latents,
        sigreg_latents,
        args.output,
        vae_label=f"Original VAE sampled z\n{Path(args.vae_checkpoint).name}",
        sigreg_label=f"SIGReg latent\n{Path(args.sigreg_checkpoint).name}",
        vae_dims=None if args.vae_latent_dims is None else tuple(args.vae_latent_dims),
        sigreg_dims=None if args.sigreg_latent_dims is None else tuple(args.sigreg_latent_dims),
        kde_min_density_fraction=args.kde_min_density_fraction,
    )
    metadata["output"] = str(Path(args.output).expanduser())
    metadata.update(plot_metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
