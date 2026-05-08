"""Visualize VAE and SIGReg latent spaces with t-SNE and UMAP."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualization.compare_sigreg_vae import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    DEFAULT_SIGREG_CHECKPOINT,
    DEFAULT_VAE_CHECKPOINT,
    collect_latents,
)


DEFAULT_OUTPUT = REPO_ROOT / "visualization" / "compare_sigreg_vae_tsne_umap.png"


def _as_latent_matrix(latents: np.ndarray, label: str) -> np.ndarray:
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim != 2:
        raise ValueError(f"{label} latents must have shape (num_samples, latent_dim)")
    if latents.shape[0] < 3:
        raise ValueError(f"{label} latents need at least 3 samples")
    return latents


def _validate_perplexity(num_samples: int, perplexity: float) -> None:
    if perplexity <= 0:
        raise ValueError("t-SNE perplexity must be positive")
    if perplexity >= num_samples:
        raise ValueError("t-SNE perplexity must be smaller than the number of samples")


def reduce_tsne(
    latents: np.ndarray,
    perplexity: float = 30.0,
    seed: int = 0,
    max_iter: int = 1_000,
) -> np.ndarray:
    latents = _as_latent_matrix(latents, "t-SNE")
    _validate_perplexity(latents.shape[0], perplexity)
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        max_iter=max_iter,
    )
    return reducer.fit_transform(latents)


def _load_umap() -> Any:
    try:
        return importlib.import_module("umap")
    except ImportError as exc:
        raise ImportError("UMAP visualization requires umap-learn. Install it with: pip install umap-learn") from exc


def reduce_umap(
    latents: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    latents = _as_latent_matrix(latents, "UMAP")
    if n_neighbors < 2:
        raise ValueError("UMAP n_neighbors must be at least 2")
    if n_neighbors >= latents.shape[0]:
        raise ValueError("UMAP n_neighbors must be smaller than the number of samples")
    umap_module = _load_umap()
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(latents)


def reduce_latents(
    vae_latents: np.ndarray,
    sigreg_latents: np.ndarray,
    tsne_perplexity: float = 30.0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    return {
        "vae_tsne": reduce_tsne(vae_latents, perplexity=tsne_perplexity, seed=seed),
        "sigreg_tsne": reduce_tsne(sigreg_latents, perplexity=tsne_perplexity, seed=seed),
        "vae_umap": reduce_umap(
            vae_latents,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            seed=seed,
        ),
        "sigreg_umap": reduce_umap(
            sigreg_latents,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            seed=seed,
        ),
    }


def _set_axis_limits(ax: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    padding = 0.08 * span
    ax.set_xlim(float(mins[0] - padding[0]), float(maxs[0] + padding[0]))
    ax.set_ylim(float(mins[1] - padding[1]), float(maxs[1] + padding[1]))


def _plot_embedding(ax: plt.Axes, points: np.ndarray, title: str) -> None:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{title} embedding must have shape (num_samples, 2)")
    ax.scatter(points[:, 0], points[:, 1], s=8, c="#202020", alpha=0.35, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("component 0")
    ax.set_ylabel("component 1")
    _set_axis_limits(ax, points)


def plot_reduced_latents(
    embeddings: dict[str, np.ndarray],
    output: str | Path,
    vae_label: str = "Original VAE sampled z",
    sigreg_label: str = "SIGReg latent",
) -> None:
    required = ("vae_tsne", "sigreg_tsne", "vae_umap", "sigreg_umap")
    missing = [key for key in required if key not in embeddings]
    if missing:
        raise KeyError(f"Missing reduced embeddings: {', '.join(missing)}")

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    _plot_embedding(axes[0, 0], embeddings["vae_tsne"], f"{vae_label}\nt-SNE")
    _plot_embedding(axes[0, 1], embeddings["sigreg_tsne"], f"{sigreg_label}\nt-SNE")
    _plot_embedding(axes[1, 0], embeddings["vae_umap"], f"{vae_label}\nUMAP")
    _plot_embedding(axes[1, 1], embeddings["sigreg_umap"], f"{sigreg_label}\nUMAP")
    fig.suptitle("Full Latent Space Reduced to 2D")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-max-iter", type=int, default=1_000)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _load_umap()
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
    embeddings = {
        "vae_tsne": reduce_tsne(
            vae_latents,
            perplexity=args.tsne_perplexity,
            seed=args.seed,
            max_iter=args.tsne_max_iter,
        ),
        "sigreg_tsne": reduce_tsne(
            sigreg_latents,
            perplexity=args.tsne_perplexity,
            seed=args.seed,
            max_iter=args.tsne_max_iter,
        ),
        "vae_umap": reduce_umap(
            vae_latents,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            seed=args.seed,
        ),
        "sigreg_umap": reduce_umap(
            sigreg_latents,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            seed=args.seed,
        ),
    }
    plot_reduced_latents(
        embeddings,
        args.output,
        vae_label=f"Original VAE sampled z\n{Path(args.vae_checkpoint).name}",
        sigreg_label=f"SIGReg latent\n{Path(args.sigreg_checkpoint).name}",
    )
    metadata.update(
        {
            "output": str(Path(args.output).expanduser()),
            "tsne_perplexity": args.tsne_perplexity,
            "umap_n_neighbors": args.umap_n_neighbors,
            "umap_min_dist": args.umap_min_dist,
        }
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
