import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from visualization.visualization_sigreg import (
    compute_gaussianity_metrics,
    plot_gaussianity_dashboard,
    save_metrics_json,
)


class SigregGaussianityVisualizationTest(unittest.TestCase):
    def test_standard_normal_has_better_metrics_than_shifted_scaled_latents(self):
        rng = np.random.default_rng(0)
        standard = rng.standard_normal((2048, 4)).astype(np.float32)
        shifted_scaled = (2.0 * rng.standard_normal((2048, 4)) + 3.0).astype(np.float32)

        standard_metrics = compute_gaussianity_metrics(standard, num_random_projections=32, seed=0)
        shifted_metrics = compute_gaussianity_metrics(shifted_scaled, num_random_projections=32, seed=0)

        self.assertLess(standard_metrics["mean_abs"], shifted_metrics["mean_abs"])
        self.assertLess(standard_metrics["std_abs_error"], shifted_metrics["std_abs_error"])
        self.assertLess(
            standard_metrics["covariance_frobenius_to_identity"],
            shifted_metrics["covariance_frobenius_to_identity"],
        )

    def test_plot_gaussianity_dashboard_writes_nonempty_png(self):
        rng = np.random.default_rng(1)
        latent_sets = {
            "vae_z": rng.standard_normal((512, 4)).astype(np.float32),
            "vae_mu": (0.5 * rng.standard_normal((512, 4))).astype(np.float32),
            "sigreg": rng.standard_normal((512, 4)).astype(np.float32),
        }

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "gaussianity.png"

            plot_gaussianity_dashboard(latent_sets, output, num_example_dims=2, num_random_projections=16, seed=0)

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_save_metrics_json_writes_expected_top_level_keys(self):
        rng = np.random.default_rng(2)
        latent_sets = {
            "vae_z": rng.standard_normal((256, 3)).astype(np.float32),
            "vae_mu": rng.standard_normal((256, 3)).astype(np.float32),
            "sigreg": rng.standard_normal((256, 3)).astype(np.float32),
        }

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "metrics.json"

            save_metrics_json(latent_sets, output, num_random_projections=8, seed=0)

            payload = json.loads(output.read_text())
            self.assertEqual(set(payload), {"vae_z", "vae_mu", "sigreg"})
            self.assertIn("random_projection_wasserstein_mean", payload["sigreg"])


if __name__ == "__main__":
    unittest.main()
