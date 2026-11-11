"""CPU checks for controlled routing, leakage and source calibration."""
import math
import unittest

import torch
from torch.nn import functional as F
from stage3_model import (DMSFConfig, Segmenter, SourceCalibrator,
                          gradient_norms, leave_block_out_scores,
                          normalize_cam, supervised_loss)


torch.set_num_threads(2)


class ControlledFusionTests(unittest.TestCase):
    def setUp(self):
        self.config = DMSFConfig(visual_dim=8, text_dim=6, hidden_dim=4,
                                 text_mlp_width=8, refinement_width=2)
        self.inputs = (torch.randn(2, 8, 8, 8), torch.randn(2, 8, 8, 8),
                       torch.tensor([[[[1., 1., 0., 0., 0., 0., 0., 0.]]]]).expand(2, 1, 8, 8).clone(),
                       torch.randn(2, 6))
        self.target = (torch.rand(2, 1, 8, 8) > .6).float()

    def model(self, branch='fusion', normalization='minmax', text_kind='real', beta_init=.7,
              fusion_mode='equal'):
        torch.manual_seed(42)
        return Segmenter(branch, normalization, text_kind, beta_init, self.config,
                         fusion_mode=fusion_mode)

    def test_head_only_receives_visual_features_and_selected_cam(self):
        spatial = self.model('spatial')
        fusion = self.model()
        self.assertEqual(spatial.head[0].in_channels, self.config.hidden_dim + 1)
        self.assertEqual(fusion.head[0].in_channels, self.config.hidden_dim + 1)
        torch.testing.assert_close(spatial.head[0].weight, fusion.head[0].weight)
        a, b = spatial(*self.inputs), fusion(*self.inputs)
        torch.testing.assert_close(a['spatial_cam'], b['spatial_cam'])
        torch.testing.assert_close(a['selected_cam'], a['spatial_cam'])
        torch.testing.assert_close(b['selected_cam'], .5 * (b['spatial_cam'] + b['semantic_cam']))
        self.assertEqual(float(b['spatial_weight']), .5)
        self.assertFalse(fusion.dmsf.fusion_logit.requires_grad)

    def test_legacy_learned_fusion_remains_explicit_for_reproduction(self):
        model = self.model(fusion_mode='learned')
        maps = model(*self.inputs)
        loss, _ = supervised_loss(maps, self.target)
        loss.backward()
        self.assertGreater(gradient_norms(model)['fusion_logit'], 0)
        with torch.no_grad():
            initial = maps['logits'].clone()
            model.dmsf.fusion_logit.add_(2)
            changed = model(*self.inputs)['logits']
        self.assertFalse(torch.equal(initial, changed))

    def test_random_text_fixed_and_legacy_weight_085(self):
        model = self.model(text_kind='random', beta_init=math.log(.85 / .15),
                           fusion_mode='learned')
        changed = list(self.inputs)
        changed[-1] = torch.randn_like(changed[-1])
        torch.testing.assert_close(model(*self.inputs)['logits'], model(*changed)['logits'], rtol=0, atol=0)
        self.assertAlmostEqual(float(model(*self.inputs)['spatial_weight'].detach()), .85, places=6)

    def test_query_target_cannot_enter_forward_or_support_calibration(self):
        model = self.model(normalization='calibrated')
        maps = model(*self.inputs)
        target_change = 1 - self.target
        a, _ = supervised_loss(maps, self.target)
        b, _ = supervised_loss(maps, target_change)
        self.assertNotEqual(float(a.detach()), float(b.detach()))
        torch.testing.assert_close(maps['logits'], model(*self.inputs)['logits'], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            supervised_loss(maps, self.target + 2)

    def test_calibrated_branch_parameters_receive_training_gradients(self):
        model = self.model(normalization='calibrated')
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        loss, _ = supervised_loss(model(*self.inputs), self.target)
        loss.backward()
        norms = gradient_norms(model)
        self.assertGreater(norms['spatial_calibrator'], 0)
        self.assertGreater(norms['semantic_calibrator'], 0)
        optimizer.step()
        self.assertFalse(torch.equal(before['spatial_calibrator.source_threshold'],
                                     model.spatial_calibrator.source_threshold))
        self.assertFalse(torch.equal(before['semantic_calibrator.log_temperature'],
                                     model.semantic_calibrator.log_temperature))


class CalibrationTests(unittest.TestCase):
    def test_leave_block_out_removes_self_evaluation(self):
        feature = torch.zeros(1, 2, 4, 4)
        feature[:, 0] = 1
        feature[:, 0, :2, :2] = 0
        feature[:, 1, :2, :2] = 1
        mask = torch.zeros(1, 1, 4, 4)
        mask[:, :, :2, :2] = 1
        mask[:, :, 2:, 2:] = 1
        score, valid = leave_block_out_scores(feature, mask, blocks=2)
        self.assertTrue(bool(valid.all()))
        self.assertLess(float(score[0, 0, 0, 0]), .01)
        naive = (F.normalize(feature, dim=1) * F.normalize((feature * mask).sum((2, 3)), dim=1)[:, :, None, None]).sum(1)
        self.assertGreater(float(naive[0, 0, 0]), .5)

    def test_single_foreground_block_has_no_self_score(self):
        feature = torch.randn(1, 2, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)
        mask[:, :, :2, :2] = 1
        _, valid = leave_block_out_scores(feature, mask, blocks=2)
        self.assertFalse(bool(valid[0, 0, 0, 0]))

    def test_support_only_shrink_and_empty_query_diagnostic(self):
        calibrator = SourceCalibrator(shrink=.25)
        with torch.no_grad():
            calibrator.source_threshold.fill_(.5)
            calibrator.log_temperature.fill_(math.log(.1))
        support = torch.tensor([[[[.2, .8]]]])
        mask = torch.tensor([[[[0., 1.]]]])
        query = torch.full((1, 1, 2, 2), .1)
        cam, info = calibrator(query, support, mask, torch.ones_like(mask, dtype=torch.bool))
        self.assertTrue(bool(info['support_calibration_used']))
        self.assertLess(float(cam.detach().max()), .02)
        self.assertEqual(float(normalize_cam(query, 1e-8).max()), 0.)
        varying_empty = torch.tensor([[[[.1, .15], [.2, .25]]]])
        self.assertAlmostEqual(float(normalize_cam(varying_empty, 1e-8).max()), 1., places=6)
        calibrated, _ = calibrator(varying_empty, support, mask, torch.ones_like(mask, dtype=torch.bool))
        self.assertLess(float(calibrated.detach().max()), .08)


if __name__ == '__main__':
    unittest.main()
