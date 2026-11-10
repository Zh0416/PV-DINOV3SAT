import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from run_experiment import (
    aggregate_metrics,
    binary_counts,
    cosine_difference,
    frequency_prototypes,
    local_fft_descriptors,
    patch_mask_fractions,
    save_fusion_cam,
    window_coverage,
)


class PatchTokenFrequencyTests(unittest.TestCase):
    def test_fft_stays_on_spatial_window_axes_and_returns_one_map_vector_per_token(self):
        torch.manual_seed(7)
        features = torch.randn(12, 9, 11)
        descriptors = local_fft_descriptors(features, window_size=4, chunk_rows=2)
        self.assertEqual(tuple(descriptors.shape), (9, 11, 32))
        self.assertTrue(torch.isfinite(descriptors).all())
        norms = descriptors.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
        positive = torch.nn.functional.normalize(torch.randn(32), dim=0)
        negative = torch.nn.functional.normalize(torch.randn(32), dim=0)
        scores = cosine_difference(descriptors, positive, negative)
        self.assertEqual(tuple(scores.shape), (9, 11))

    def test_window_coverage_uses_reflected_spatial_neighborhoods(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 4:12] = True
        fractions = patch_mask_fractions(mask, (4, 4))
        coverage = window_coverage(fractions, 2)
        self.assertEqual(tuple(coverage.shape), (4, 4))
        self.assertGreater(float(coverage[1, 1]), float(coverage[0, 0]))
        self.assertLessEqual(float(coverage.max()), 1.0)

    def test_support_prototypes_ignore_mixed_coverage_windows(self):
        torch.manual_seed(9)
        descriptors = torch.nn.functional.normalize(torch.randn(5, 5, 18), dim=-1)
        coverage = torch.full((5, 5), 0.3)
        coverage[:2, :2] = 0.9
        coverage[2:, 2:] = 0.0
        positive, negative, stats = frequency_prototypes(
            descriptors, coverage, foreground_coverage=0.6, background_coverage=0.05)
        self.assertEqual(tuple(positive.shape), (18,))
        self.assertEqual(tuple(negative.shape), (18,))
        self.assertEqual(stats["positive_windows"], 4)
        self.assertEqual(stats["background_candidates"], 9)
        self.assertEqual(stats["ignored_mixed_windows"], 12)

    def test_spatial_cosine_cam_accepts_channel_first_dino_features(self):
        torch.manual_seed(11)
        features = torch.randn(8, 9, 11)
        positive = torch.nn.functional.normalize(torch.randn(8), dim=0)
        negative = torch.nn.functional.normalize(torch.randn(8), dim=0)
        score = cosine_difference(features, positive, negative, channels_first=True)
        unit = torch.nn.functional.normalize(features, dim=0)
        expected = (unit * positive[:, None, None]).sum(0) - (unit * negative[:, None, None]).sum(0)
        self.assertEqual(tuple(score.shape), (9, 11))
        self.assertTrue(torch.allclose(score, expected))

    def test_fusion_cam_saves_full_precision_array_and_viewable_images(self):
        probability = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "query.bmp"
            from PIL import Image
            Image.new("RGB", (8, 8), (100, 120, 140)).save(image_path)
            files = save_fusion_cam(probability, image_path, root / "fusion_cam")
            restored = np.load(root / "fusion_cam" / files["array"], allow_pickle=False)
            self.assertEqual(restored.dtype, np.float32)
            self.assertTrue(np.array_equal(restored, probability))
            for key in ("heatmap", "overlay"):
                with Image.open(root / "fusion_cam" / files[key]) as visualization:
                    self.assertEqual(visualization.size, (8, 8))
                    self.assertEqual(visualization.mode, "RGB")

    def test_fixed_threshold_metrics_aggregate_counts(self):
        truth = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
        score = np.array([[0.8, 0.2, 0.9], [0.1, 0.1, 0.1]], dtype=np.float32)
        row = binary_counts(truth, score)
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["tn"]), (1, 1, 1, 3))
        aggregate = aggregate_metrics([row, row])
        self.assertEqual(aggregate["image_count"], 2)
        self.assertEqual(aggregate["micro"]["iou"], 1 / 3)
        self.assertEqual(aggregate["macro_per_image"]["dice"], 0.5)


if __name__ == "__main__":
    unittest.main()
