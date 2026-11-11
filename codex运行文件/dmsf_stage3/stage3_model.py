"""Controlled one-CAM segmentation and support-only source calibration."""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'dmsf_v1'))
from dmsf import DMSF, DMSFConfig, normalize_cam
import torch
from torch import nn
from torch.nn import functional as F


def leave_block_out_scores(features, mask, blocks=4, eps=1e-8):
    """Score support patches against a prototype excluding their own block."""
    batch, channels, height, width = features.shape
    if height % blocks or width % blocks:
        raise ValueError('Support grid must divide into equal calibration blocks.')
    if mask.shape != (batch, 1, height, width):
        raise ValueError('Support mask/grid mismatch.')
    cells = F.unfold(features * mask, (height // blocks, width // blocks),
                     stride=(height // blocks, width // blocks))
    cells = cells.reshape(batch, channels, height // blocks * width // blocks, blocks * blocks).sum(2)
    counts = F.avg_pool2d(mask, (height // blocks, width // blocks)).flatten(2) * (height * width // blocks ** 2)
    remaining = counts.sum(2, keepdim=True) - counts
    prototype = (cells.sum(2, keepdim=True) - cells) / remaining.clamp_min(1)
    prototype = F.normalize(prototype, dim=1, eps=eps)
    patch_unit = F.normalize(features, dim=1, eps=eps)
    patch_blocks = torch.arange(blocks * blocks, device=features.device).reshape(blocks, blocks)
    patch_blocks = patch_blocks.repeat_interleave(height // blocks, 0).repeat_interleave(width // blocks, 1)
    selected = prototype[:, :, patch_blocks]
    score = (patch_unit * selected).sum(1, keepdim=True)
    valid = (remaining[:, :, patch_blocks] > 0).reshape(batch, 1, height, width)
    return score, valid


def support_midpoint(scores, mask, valid):
    """Detached class-center midpoint; no query label can enter this update."""
    foreground = (mask > 0) & valid
    background = (mask == 0) & valid
    fg_count = foreground.sum((1, 2, 3))
    bg_count = background.sum((1, 2, 3))
    fg_mean = (scores * foreground).sum((1, 2, 3)) / fg_count.clamp_min(1)
    bg_mean = (scores * background).sum((1, 2, 3)) / bg_count.clamp_min(1)
    midpoint = (fg_mean + bg_mean) / 2
    usable = (fg_count > 0) & (bg_count > 0)
    return midpoint.detach().reshape(-1, 1, 1, 1), usable.reshape(-1, 1, 1, 1)


class SourceCalibrator(nn.Module):
    def __init__(self, shrink=0.25):
        super().__init__()
        self.source_threshold = nn.Parameter(torch.tensor(0.0))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.1)))
        self.shrink = shrink

    def forward(self, query_scores, support_scores, support_mask, valid):
        midpoint, usable = support_midpoint(support_scores, support_mask, valid)
        offset = torch.where(usable, self.shrink * (midpoint - self.source_threshold),
                             torch.zeros_like(midpoint))
        threshold = self.source_threshold + offset
        temperature = self.log_temperature.exp().clamp_min(1e-4)
        return torch.sigmoid((query_scores - threshold) / temperature), {
            'source_threshold': self.source_threshold.detach(),
            'temperature': temperature.detach(),
            'support_midpoint': midpoint,
            'support_calibration_used': usable,
        }


class Segmenter(nn.Module):
    def __init__(self, branch, normalization, text_kind='real', beta_init=0.7,
                 config=None, random_text_seed=4242, calibration_shrink=0.25,
                 fusion_mode='equal'):
        super().__init__()
        if branch not in ('spatial', 'fusion') or normalization not in ('minmax', 'calibrated'):
            raise ValueError('Unknown branch or normalization.')
        if text_kind not in ('real', 'random') or not math.isfinite(beta_init):
            raise ValueError('Unknown text kind or beta.')
        if fusion_mode not in ('equal', 'learned'):
            raise ValueError('fusion_mode must be equal or learned.')
        self.branch, self.normalization, self.text_kind = branch, normalization, text_kind
        self.fusion_mode = fusion_mode
        c = replace(config, fusion_logit_init=beta_init) if config else DMSFConfig(fusion_logit_init=beta_init)
        self.dmsf = DMSF(c)
        self.head = nn.Sequential(nn.Conv2d(c.hidden_dim + 1, 64, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.spatial_calibrator = SourceCalibrator(calibration_shrink) if normalization == 'calibrated' else None
        self.semantic_calibrator = SourceCalibrator(calibration_shrink) if normalization == 'calibrated' and branch == 'fusion' else None
        gen = torch.Generator().manual_seed(random_text_seed)
        self.register_buffer('random_text', F.normalize(torch.randn(1, c.text_dim, generator=gen), dim=-1))
        if branch == 'spatial':
            for module in (self.dmsf.text_to_visual, self.dmsf.text_projection, self.dmsf.semantic_refiner):
                module.requires_grad_(False)
            self.dmsf.fusion_logit.requires_grad_(False)
        elif fusion_mode == 'equal':
            self.dmsf.fusion_logit.requires_grad_(False)

    def identity(self):
        return {'branch': self.branch, 'normalization': self.normalization,
                'text_kind': self.text_kind, 'beta_init': self.dmsf.config.fusion_logit_init,
                'calibration_shrink': self.spatial_calibrator.shrink if self.spatial_calibrator else None,
                'fusion_mode': self.fusion_mode}

    def forward(self, support, query, mask, text):
        c = self.dmsf.config
        if support.ndim != 4 or query.ndim != 4 or support.shape[0] != query.shape[0] or support.shape[1] != c.visual_dim or query.shape[1] != c.visual_dim:
            raise ValueError('Invalid frozen visual feature shape.')
        if any(x.requires_grad for x in (support, query, mask, text)):
            raise ValueError('Frozen encoder inputs must be detached.')
        if mask.shape != (support.shape[0], 1, *support.shape[-2:]) or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('Aligned binary support mask required.')
        if text.shape != (support.shape[0], c.text_dim):
            raise ValueError('One text embedding per episode required.')
        fs, fq = self.dmsf.project_visual(support), self.dmsf.project_visual(query)
        count = mask.sum((-2, -1))
        if (count <= 0).any():
            raise ValueError('Empty support foreground.')
        prototype = (fs * mask).sum((-2, -1)) / count
        query_unit = F.normalize(fq, dim=1, eps=c.eps)
        spatial_raw = (query_unit * F.normalize(prototype, dim=1, eps=c.eps)[:, :, None, None]).sum(1, keepdim=True)
        if self.normalization == 'minmax':
            spatial_input = normalize_cam(spatial_raw, c.eps)
            spatial_info = {}
        else:
            support_spatial, valid = leave_block_out_scores(fs, mask, eps=c.eps)
            spatial_input, spatial_info = self.spatial_calibrator(spatial_raw, support_spatial, mask, valid)
        spatial = self.dmsf.spatial_refiner(spatial_input)
        semantic = None
        semantic_raw = None
        semantic_info = {}
        if self.branch == 'fusion':
            if self.text_kind == 'random':
                text = self.random_text.expand(query.shape[0], -1)
            text_unit = F.normalize(text, dim=1, eps=c.eps)
            t = self.dmsf.text_projection(self.dmsf.text_to_visual(text_unit))
            t_unit = F.normalize(t, dim=1, eps=c.eps)[:, :, None, None]
            semantic_raw = (query_unit * t_unit).sum(1, keepdim=True)
            if self.normalization == 'minmax':
                semantic_input = normalize_cam(semantic_raw, c.eps)
            else:
                support_semantic = (F.normalize(fs, dim=1, eps=c.eps) * t_unit).sum(1, keepdim=True)
                semantic_input, semantic_info = self.semantic_calibrator(
                    semantic_raw, support_semantic, mask, torch.ones_like(mask, dtype=torch.bool))
            semantic = self.dmsf.semantic_refiner(semantic_input)
            if self.fusion_mode == 'equal':
                weight = spatial.new_tensor(0.5)
                selected = 0.5 * (spatial + semantic)
            else:
                weight = self.dmsf.fusion_logit.sigmoid()
                selected = weight * spatial + (1 - weight) * semantic
        else:
            weight = spatial.new_tensor(1.)
            selected = spatial
        logits = self.head(torch.cat((fq, selected), dim=1))
        return {'logits': logits, 'spatial_cam': spatial, 'semantic_cam': semantic,
                'selected_cam': selected, 'query_features': fq, 'spatial_raw': spatial_raw,
                'semantic_raw': semantic_raw,
                'spatial_weight': weight, 'spatial_calibration': spatial_info,
                'semantic_calibration': semantic_info}


def dice_loss(probabilities, target):
    dims = tuple(range(1, target.ndim))
    return (1 - (2 * (probabilities * target).sum(dims) + 1) /
            (probabilities.sum(dims) + target.sum(dims) + 1)).mean()


def supervised_loss(maps, target):
    if target.shape != maps['logits'].shape or not torch.all((target == 0) | (target == 1)):
        raise ValueError('Matching binary query labels required for training loss.')
    bce = F.binary_cross_entropy_with_logits(maps['logits'], target)
    dice = dice_loss(maps['logits'].sigmoid(), target)
    spatial = dice_loss(maps['spatial_cam'], target)
    return bce + dice + .2 * spatial, {'bce': bce, 'dice': dice, 'spatial_dice': spatial}


def gradient_norms(model):
    groups = {'visual_projection': model.dmsf.visual_projection,
              'text_to_visual': model.dmsf.text_to_visual,
              'text_projection': model.dmsf.text_projection,
              'spatial_refiner': model.dmsf.spatial_refiner,
              'semantic_refiner': model.dmsf.semantic_refiner,
              'head': model.head,
              'spatial_calibrator': model.spatial_calibrator,
              'semantic_calibrator': model.semantic_calibrator}
    result = {}
    for name, module in groups.items():
        params = [p for p in module.parameters() if p.requires_grad] if module is not None else []
        if not params:
            result[name] = None
            continue
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError('Missing/nonfinite gradient: ' + name)
        result[name] = float(sum(p.grad.float().square().sum() for p in params).sqrt().detach().cpu())
    p = model.dmsf.fusion_logit
    result['fusion_logit'] = float(p.grad.abs().detach().cpu()) if p.requires_grad and p.grad is not None else None
    if p.requires_grad and (result['fusion_logit'] is None or not torch.isfinite(p.grad)):
        raise RuntimeError('Missing/nonfinite fusion gradient.')
    return result
