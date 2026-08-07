import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from visualization.compare_sigreg_vae_umap import (
    _load_umap,
    plot_reduced_latents,
    reduce_tsne,
)


class CompareSigregVaeUmapVisualizationTest(unittest.TestCase):
    def test_reduce_tsne_returns_two_dimensional_embedding(self):
        rng = np.random.default_rng(0)
        latents = rng.normal(size=(24, 8)).astype(np.float32)

        reduced = reduce_tsne(latents, perplexity=5, seed=0, max_iter=250)

        self.assertEqual(reduced.shape, (24, 2))
        self.assertTrue(np.isfinite(reduced).all())

    def test_plot_reduced_latents_writes_nonempty_png(self):
        rng = np.random.default_rng(1)
        embeddings = {
            "vae_tsne": rng.normal(size=(32, 2)),
            "sigreg_tsne": rng.normal(loc=0.3, size=(32, 2)),
            "vae_umap": rng.normal(loc=0.6, size=(32, 2)),
            "sigreg_umap": rng.normal(loc=0.9, size=(32, 2)),
        }

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "tsne_umap.png"

            plot_reduced_latents(embeddings, output)

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_load_umap_reports_missing_dependency(self):
        original_import_module = importlib.import_module

        def fake_import_module(name):
            if name == "umap":
                raise ImportError("missing umap")
            return original_import_module(name)

        with patch("importlib.import_module", side_effect=fake_import_module):
            with self.assertRaisesRegex(ImportError, "pip install umap-learn"):
                _load_umap()


if __name__ == "__main__":
    unittest.main()
