"""Visualize how close VAE and SIGReg latents are to N(0, I)."""

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
from scipy.stats import chi2, kurtosis, norm, probplot, skew, wasserstein_distance
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.policy.vae import _build_window_datasets, _move_batch  # noqa: E402
from visualization.compare_sigreg_vae import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    DEFAULT_SIGREG_CHECKPOINT,
    DEFAULT_VAE_CHECKPOINT,
    _checkpoint_state_keys,
    _load_sigreg_model,
    _load_vae_model,
    _validate_matching_configs,
)


DEFAULT_OUTPUT = REPO_ROOT / "visualization" / "sigreg_gaussianity.png"
DEFAULT_METRICS_OUTPUT = REPO_ROOT / "visualization" / "sigreg_gaussianity_metrics.json"


def _as_latent_matrix(latents: np.ndarray, label: str = "latents") -> np.ndarray:
    latents = np.asarray(latents, dtype=np.float64)
    if latents.ndim != 2:
        raise ValueError(f"{label} must have shape (num_samples, latent_dim)")
    if latents.shape[0] < 3:
        raise ValueError(f"{label} must contain at least 3 samples")
    if latents.shape[1] < 1:
        raise ValueError(f"{label} must contain at least 1 latent dimension")
    return latents


def _per_dim_stats(latents: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": latents.mean(axis=0),
        "std": latents.std(axis=0),
        "skew": np.nan_to_num(skew(latents, axis=0, bias=False), nan=0.0),
        "excess_kurtosis": np.nan_to_num(kurtosis(latents, axis=0, fisher=True, bias=False), nan=0.0),
    }


def _covariance(latents: np.ndarray) -> np.ndarray:
    if latents.shape[1] == 1:
        return np.array([[float(np.var(latents[:, 0], ddof=1))]], dtype=np.float64)
    return np.asarray(np.cov(latents, rowvar=False), dtype=np.float64)


def _random_projection_wasserstein(
    latents: np.ndarray,
    num_random_projections: int,
    seed: int,
) -> np.ndarray:
    if num_random_projections < 1:
        raise ValueError("num_random_projections must be at least 1")
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((latents.shape[1], int(num_random_projections)))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True).clip(min=1e-12)
    projections = latents @ directions
    distances = []
    for idx in range(projections.shape[1]):
        reference = rng.standard_normal(latents.shape[0])
        distances.append(wasserstein_distance(projections[:, idx], reference))
    return np.asarray(distances, dtype=np.float64)


def compute_gaussianity_metrics(
    latents: np.ndarray,
    num_random_projections: int = 256,
    seed: int = 0,
) -> dict[str, Any]:
    latents = _as_latent_matrix(latents)
    stats = _per_dim_stats(latents)
    covariance = _covariance(latents)
    identity = np.eye(latents.shape[1], dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(covariance)

    rng = np.random.default_rng(seed + 17)
    norm_sq = np.sum(latents * latents, axis=1)
    chi_square_reference = rng.chisquare(df=latents.shape[1], size=latents.shape[0])
    projection_distances = _random_projection_wasserstein(latents, num_random_projections, seed + 29)

    return {
        "num_samples": int(latents.shape[0]),
        "latent_dim": int(latents.shape[1]),
        "mean_abs": float(np.mean(np.abs(stats["mean"]))),
        "std_abs_error": float(np.mean(np.abs(stats["std"] - 1.0))),
        "skew_abs_mean": float(np.mean(np.abs(stats["skew"]))),
        "excess_kurtosis_abs_mean": float(np.mean(np.abs(stats["excess_kurtosis"]))),
        "covariance_frobenius_to_identity": float(np.linalg.norm(covariance - identity, ord="fro") / np.sqrt(latents.shape[1])),
        "covariance_eigenvalue_min": float(np.min(eigenvalues)),
        "covariance_eigenvalue_median": float(np.median(eigenvalues)),
        "covariance_eigenvalue_max": float(np.max(eigenvalues)),
        "squared_norm_wasserstein_to_chi_square": float(wasserstein_distance(norm_sq, chi_square_reference)),
        "random_projection_wasserstein_mean": float(np.mean(projection_distances)),
        "random_projection_wasserstein_max": float(np.max(projection_distances)),
    }


def compute_all_metrics(
    latent_sets: dict[str, np.ndarray],
    num_random_projections: int = 256,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    return {
        name: compute_gaussianity_metrics(latents, num_random_projections=num_random_projections, seed=seed)
        for name, latents in latent_sets.items()
    }


def save_metrics_json(
    latent_sets: dict[str, np.ndarray],
    output: str | Path,
    num_random_projections: int = 256,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    metrics = compute_all_metrics(latent_sets, num_random_projections=num_random_projections, seed=seed)
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _selected_dims(latents: np.ndarray, num_example_dims: int) -> np.ndarray:
    stats = _per_dim_stats(latents)
    score = np.abs(stats["mean"]) + np.abs(stats["std"] - 1.0) + np.abs(stats["skew"]) + 0.25 * np.abs(stats["excess_kurtosis"])
    count = min(max(int(num_example_dims), 1), latents.shape[1])
    return np.argsort(score)[-count:][::-1]


def _plot_marginals(ax: plt.Axes, latents: np.ndarray, dims: np.ndarray, title: str) -> None:
    xs = np.linspace(-4.0, 4.0, 300)
    ax.plot(xs, norm.pdf(xs), color="black", linewidth=1.5, label="N(0,1)")
    for dim in dims:
        ax.hist(latents[:, dim], bins=35, density=True, alpha=0.25, label=f"dim {int(dim)}")
    ax.set_title(title)
    ax.set_xlim(-4.0, 4.0)
    ax.legend(fontsize=7)


def _plot_qq(ax: plt.Axes, latents: np.ndarray, dim: int, title: str) -> None:
    theoretical, ordered = probplot(latents[:, dim], dist="norm", fit=False)
    ax.scatter(theoretical, ordered, s=7, alpha=0.35, linewidths=0)
    low = min(float(np.min(theoretical)), float(np.min(ordered)))
    high = max(float(np.max(theoretical)), float(np.max(ordered)))
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set_title(f"{title} Q-Q dim {dim}")
    ax.set_xlabel("N(0,1) quantile")
    ax.set_ylabel("empirical quantile")


def _plot_mean_std(ax: plt.Axes, latents: np.ndarray, title: str) -> None:
    stats = _per_dim_stats(latents)
    dims = np.arange(latents.shape[1])
    ax.scatter(dims, stats["mean"], s=7, alpha=0.6, label="mean")
    ax.scatter(dims, stats["std"], s=7, alpha=0.6, label="std")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{title} per-dim mean/std")
    ax.set_xlabel("latent dim")
    ax.legend(fontsize=7)


def _plot_correlation(ax: plt.Axes, latents: np.ndarray, title: str) -> None:
    corr = np.corrcoef(latents, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="nearest")
    ax.set_title(f"{title} correlation")
    ax.set_xlabel("latent dim")
    ax.set_ylabel("latent dim")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_eigenvalues(ax: plt.Axes, latents: np.ndarray, title: str) -> None:
    eigenvalues = np.linalg.eigvalsh(_covariance(latents))
    ax.plot(np.sort(eigenvalues), marker=".", linewidth=1)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{title} covariance eigenvalues")
    ax.set_xlabel("sorted index")
    ax.set_ylabel("eigenvalue")


def _plot_norm_sq(ax: plt.Axes, latents: np.ndarray, title: str) -> None:
    norm_sq = np.sum(latents * latents, axis=1)
    upper = max(float(np.quantile(norm_sq, 0.995)), float(chi2.ppf(0.995, df=latents.shape[1])))
    xs = np.linspace(0.0, upper, 300)
    ax.hist(norm_sq, bins=40, density=True, alpha=0.45, label="empirical")
    ax.plot(xs, chi2.pdf(xs, df=latents.shape[1]), color="black", linewidth=1.5, label=f"chi2({latents.shape[1]})")
    ax.set_title(f"{title} squared norm")
    ax.legend(fontsize=7)


def _plot_projection_summary(
    ax: plt.Axes,
    latent_sets: dict[str, np.ndarray],
    num_random_projections: int,
    seed: int,
) -> None:
    names = list(latent_sets)
    means = []
    maxes = []
    for index, name in enumerate(names):
        distances = _random_projection_wasserstein(_as_latent_matrix(latent_sets[name], name), num_random_projections, seed + index)
        means.append(float(np.mean(distances)))
        maxes.append(float(np.max(distances)))
    x = np.arange(len(names))
    ax.bar(x - 0.18, means, width=0.36, label="mean")
    ax.bar(x + 0.18, maxes, width=0.36, label="max")
    ax.set_xticks(x, names, rotation=20)
    ax.set_title("Random projection Wasserstein to N(0,1)")
    ax.legend(fontsize=7)


def plot_gaussianity_dashboard(
    latent_sets: dict[str, np.ndarray],
    output: str | Path,
    num_example_dims: int = 6,
    num_random_projections: int = 256,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    if not latent_sets:
        raise ValueError("latent_sets must not be empty")
    prepared = {name: _as_latent_matrix(latents, name) for name, latents in latent_sets.items()}
    names = list(prepared)
    metrics = compute_all_metrics(prepared, num_random_projections=num_random_projections, seed=seed)

    rows = 7
    cols = len(names)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.2 * rows), squeeze=False, constrained_layout=True)
    for col, name in enumerate(names):
        latents = prepared[name]
        dims = _selected_dims(latents, num_example_dims)
        _plot_marginals(axes[0, col], latents, dims, f"{name} marginals")
        _plot_qq(axes[1, col], latents, int(dims[0]), name)
        _plot_mean_std(axes[2, col], latents, name)
        _plot_correlation(axes[3, col], latents, name)
        _plot_eigenvalues(axes[4, col], latents, name)
        _plot_norm_sq(axes[5, col], latents, name)
    _plot_projection_summary(axes[6, 0], prepared, num_random_projections, seed)
    for col in range(1, cols):
        axes[6, col].axis("off")
    fig.suptitle("Latent Gaussianity Diagnostics")

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return metrics


@torch.no_grad()
def collect_latent_sets(
    dataset_dir: str | Path,
    vae_checkpoint: str | Path,
    sigreg_checkpoint: str | Path,
    batch_size: int = 256,
    num_windows: int = 10_000,
    val_ratio: float = 0.1,
    seed: int = 0,
    num_workers: int = 0,
    device: str = "cuda",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
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

    vae_z = []
    vae_mu = []
    sigreg = []
    for batch in loader:
        batch = _move_batch(batch, torch_device)
        mu, logvar = vae_model.encode(batch["encoder_input"])
        vae_mu.append(mu.cpu())
        vae_z.append(vae_model.reparameterize(mu, logvar).cpu())
        sigreg.append(sigreg_model.encode(batch["encoder_input"]).cpu())

    metadata = {
        "dataset_dir": str(Path(dataset_dir).expanduser()),
        "vae_checkpoint": str(Path(vae_checkpoint).expanduser()),
        "sigreg_checkpoint": str(Path(sigreg_checkpoint).expanduser()),
        "num_windows": num_windows,
        "state_keys": info.state_keys,
        "device": str(torch_device),
    }
    return {
        "vae_z": torch.cat(vae_z).numpy(),
        "vae_mu": torch.cat(vae_mu).numpy(),
        "sigreg": torch.cat(sigreg).numpy(),
    }, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--vae-checkpoint", default=str(DEFAULT_VAE_CHECKPOINT))
    parser.add_argument("--sigreg-checkpoint", default=str(DEFAULT_SIGREG_CHECKPOINT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metrics-output", default=str(DEFAULT_METRICS_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-windows", type=int, default=10_000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-random-projections", type=int, default=256)
    parser.add_argument("--num-example-dims", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    latent_sets, metadata = collect_latent_sets(
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
    metrics = plot_gaussianity_dashboard(
        latent_sets,
        args.output,
        num_example_dims=args.num_example_dims,
        num_random_projections=args.num_random_projections,
        seed=args.seed,
    )
    metrics_output = Path(args.metrics_output).expanduser()
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w") as f:
        json.dump(metrics, f, indent=2)
    metadata.update({"output": str(Path(args.output).expanduser()), "metrics_output": str(metrics_output)})
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
