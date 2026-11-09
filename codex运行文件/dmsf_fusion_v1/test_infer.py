"""Pinned checkpoint and three-CAM inference checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from infer import (letterbox_geometry, letterbox_support_mask, load_feature_input,
                   load_model, predict, predict_original_size, restore_map_to_original,
                   source_sha256, validate_inputs, version)


HERE = Path(__file__).resolve().parent
PROBE = (HERE.parent / 'dmsf_stage2' / 'kaggle_output' / 'version1' /
         'dmsf_stage2' / 'verification_probe' / 'inputs.npz')


class FusionVersionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = load_model(torch.device('cpu'))

    def test_pinned_model_routes_three_cams(self):
        self.assertEqual(self.model.identity()['branch'], 'fusion')
        self.assertEqual(self.model.identity()['text_kind'], 'real')
        self.assertEqual(self.model.head[0].in_channels, 257)
        self.assertTrue(bool(self.model.dmsf.semantic_refiner.net[2].weight.abs().sum() > 0))
        self.assertTrue(bool(self.model.dmsf.spatial_refiner.net[2].weight.abs().sum() > 0))

    def test_source_hash_is_checkout_line_ending_independent(self):
        source = HERE.parent / 'dmsf_stage3' / 'stage3_model.py'
        expected = version()['source_sha256']['dmsf_stage3/stage3_model.py']
        self.assertEqual(source_sha256(source), expected)
        with tempfile.TemporaryDirectory() as directory:
            converted = Path(directory) / source.name
            converted.write_bytes(source.read_bytes().replace(b'\n', b'\r\n'))
            self.assertEqual(source_sha256(converted), expected)

    def test_synthetic_forward_without_local_probe(self):
        support = torch.ones(1, 1024, 8, 8)
        query = torch.randn(1, 1024, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, :2, :2] = 1
        text = torch.nn.functional.normalize(torch.ones(1, 512), dim=-1)
        with torch.inference_mode():
            maps = self.model(support, query, mask, text)
        self.assertEqual(maps['spatial_cam'].shape, (1, 1, 8, 8))
        self.assertEqual(maps['semantic_cam'].shape, (1, 1, 8, 8))
        torch.testing.assert_close(
            maps['selected_cam'], maps['spatial_weight'] * maps['spatial_cam'] +
            (1 - maps['spatial_weight']) * maps['semantic_cam'])

    def test_variable_size_letterbox_and_output_restore(self):
        geometry = letterbox_geometry(720, 1280)
        self.assertEqual((geometry['resized_height'], geometry['resized_width']), (576, 1024))
        values = torch.zeros(1, 1, 64, 64)
        restored = restore_map_to_original(values, geometry)
        self.assertEqual(restored.shape, (720, 1280))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mask.png'
            raw = np.zeros((720, 1280), dtype=np.uint8)
            raw[100:400, 200:800] = 255
            Image.fromarray(raw, mode='L').save(path)
            mask = letterbox_support_mask(path, geometry)
            self.assertEqual(mask.shape, (1, 1, 64, 64))
            self.assertGreater(float(mask.sum()), 0)
        support = torch.randn(1, 1024, 8, 8)
        query = torch.randn(1, 1024, 8, 8)
        support_mask = torch.zeros(1, 1, 8, 8)
        support_mask[:, :, :2, :2] = 1
        text = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        restored = predict_original_size(self.model, (support, query, support_mask, text), geometry)
        self.assertEqual(restored['fused_cam'].shape, (720, 1280))
        self.assertEqual(restored['mask_full'].shape, (720, 1280))

    def test_input_rejects_empty_support(self):
        arrays = (np.zeros((1, 1024, 64, 64), np.float32),
                  np.zeros((1, 1024, 64, 64), np.float32),
                  np.zeros((1, 1, 64, 64), np.float32),
                  np.ones((1, 512), np.float32) / np.sqrt(512))
        with self.assertRaisesRegex(ValueError, 'nonempty binary'):
            validate_inputs(arrays, torch.device('cpu'))

    @unittest.skipUnless(PROBE.exists(), 'Local frozen validation probe is not present')
    def test_real_frozen_feature_probe(self):
        result = predict(self.model, load_feature_input(PROBE, torch.device('cpu')))
        self.assertEqual(result['spatial_cam'].shape, (64, 64))
        self.assertEqual(result['semantic_cam'].shape, (64, 64))
        self.assertEqual(result['fused_cam'].shape, (64, 64))
        self.assertEqual(result['mask_full'].shape, (1024, 1024))
        self.assertAlmostEqual(float(result['spatial_weight']), 0.6602444649, places=5)
        self.assertAlmostEqual(float(result['spatial_cam'].mean()), 0.0578892902, places=4)
        self.assertAlmostEqual(float(result['semantic_cam'].mean()), 0.3735718131, places=4)
        self.assertAlmostEqual(float(result['logits_patch'].mean()), -12.9190216064, places=3)
        np.testing.assert_allclose(
            result['fused_cam'],
            result['spatial_weight'] * result['spatial_cam'] +
            (1 - result['spatial_weight']) * result['semantic_cam'],
            rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(result['mask_full'], result['logits_full'] >= 0)


if __name__ == '__main__':
    unittest.main()
