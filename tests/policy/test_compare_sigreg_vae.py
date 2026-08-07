import tempfile
import unittest
from pathlib import Path

import numpy as np

from visualization.compare_sigreg_vae import _kde_levels, plot_latent_kde, select_latent_dims


class CompareSigregVaeVisualizationTest(unittest.TestCase):
    def test_kde_levels_omit_low_density_tail(self):
        density = np.array([[0.0, 0.1], [1.0, 10.0]], dtype=np.float32)

        levels = _kde_levels(density, num_levels=5, min_density_fraction=0.2)

        self.assertEqual(len(levels), 5)
        self.assertGreaterEqual(levels[0], 2.0)
        self.assertEqual(levels[-1], 10.0)
        self.assertTrue(np.all(np.diff(levels) > 0))

    def test_select_latent_dims_prefers_skewed_non_gaussian_axes(self):
        rng = np.random.default_rng(0)
        latents = rng.normal(size=(256, 5)).astype(np.float32)
        latents[:, 2] = rng.exponential(scale=1.0, size=256).astype(np.float32)
        latents[:, 4] = rng.normal(loc=3.0, scale=0.2, size=256).astype(np.float32)

        dims, stats = select_latent_dims(latents, "synthetic")

        self.assertEqual(dims, (2, 4))
        self.assertEqual(stats["dims"], [2, 4])
        self.assertGreater(stats["per_dim"][2]["skew"], 1.0)
        self.assertGreater(abs(stats["per_dim"][4]["mean"]), 2.0)

    def test_select_latent_dims_uses_manual_dims(self):
        latents = np.zeros((8, 4), dtype=np.float32)

        dims, stats = select_latent_dims(latents, "manual", dims=(3, 1))

        self.assertEqual(dims, (3, 1))
        self.assertEqual(stats["dims"], [3, 1])

    def test_select_latent_dims_rejects_duplicate_manual_dims(self):
        latents = np.zeros((8, 4), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "distinct"):
            select_latent_dims(latents, "manual", dims=(1, 1))

    def test_plot_latent_kde_writes_nonempty_png(self):
        rng = np.random.default_rng(0)
        vae_latents = rng.normal(size=(64, 4)).astype(np.float32)
        sigreg_latents = rng.normal(loc=0.5, size=(64, 4)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "latent_kde.png"

            metadata = plot_latent_kde(
                vae_latents,
                sigreg_latents,
                output,
                vae_label="VAE",
                sigreg_label="SIGReg",
                vae_dims=(2, 3),
                sigreg_dims=(1, 2),
                kde_min_density_fraction=0.2,
            )

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(metadata["vae_latent_dims"], [2, 3])
            self.assertEqual(metadata["sigreg_latent_dims"], [1, 2])
            self.assertEqual(metadata["kde_min_density_fraction"], 0.2)


if __name__ == "__main__":
    unittest.main()
