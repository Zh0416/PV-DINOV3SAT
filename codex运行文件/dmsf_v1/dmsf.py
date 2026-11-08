"""SSPNet III-C priors with frozen DINOv3-SAT and GeoRSCLIP inputs.

This module contains only trainable projections, CAM refiners and fusion.
It consumes encoder features, never GeoRSCLIP image features or window CAMs.
No alignment loss or segmentation decoder is invented here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DMSFConfig:
    visual_dim: int = 1024
    text_dim: int = 512
    hidden_dim: int = 256
    text_mlp_width: int = 1024
    refinement_width: int = 16
    residual_scale: float = 0.1
    fusion_logit_init: float = 0.7
    eps: float = 1e-8

    def __post_init__(self):
        if any(not isinstance(v, int) or v <= 0 for v in
               (self.visual_dim, self.text_dim, self.hidden_dim, self.text_mlp_width, self.refinement_width)):
            raise ValueError('Dimensions must be positive integers.')
        if not (math.isfinite(self.eps) and self.eps > 0):
            raise ValueError('eps must be finite and positive.')
        if not (math.isfinite(self.residual_scale) and 0 <= self.residual_scale <= 1):
            raise ValueError('Residual scale must be in [0,1].')
        if not math.isfinite(self.fusion_logit_init):
            raise ValueError('Fusion logit must be finite.')


def normalize_cam(scores: Tensor, eps: float) -> Tensor:
    """Per-episode spatial min-max, including finite zero for constant maps."""
    low = scores.amin(dim=(-2, -1), keepdim=True)
    span = scores.amax(dim=(-2, -1), keepdim=True) - low
    return torch.where(span > eps, (scores - low) / (span + eps), torch.zeros_like(scores))


class CAMRefiner(nn.Module):
    def __init__(self, width: int, scale: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(width, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = scale

    def forward(self, cam: Tensor) -> Tensor:
        return (cam + self.scale * torch.tanh(self.net(cam))).clamp(0.0, 1.0)


class DMSF(nn.Module):
    def __init__(self, config: DMSFConfig | None = None) -> None:
        super().__init__()
        self.config = config or DMSFConfig()
        c = self.config
        # phi is shared between support and query, with LN over channels.
        self.visual_projection = nn.Sequential(
            nn.Linear(c.visual_dim, c.hidden_dim), nn.LayerNorm(c.hidden_dim), nn.GELU())
        # Three linear layers interleaved with LN/GELU (paper Eq. 4).
        self.text_to_visual = nn.Sequential(
            nn.Linear(c.text_dim, c.text_mlp_width),
            nn.LayerNorm(c.text_mlp_width), nn.GELU(),
            nn.Linear(c.text_mlp_width, c.text_mlp_width),
            nn.LayerNorm(c.text_mlp_width), nn.GELU(),
            nn.Linear(c.text_mlp_width, c.visual_dim))
        # psi is distinct from the shared support/query projection phi.
        self.text_projection = nn.Sequential(
            nn.Linear(c.visual_dim, c.hidden_dim), nn.LayerNorm(c.hidden_dim), nn.GELU())
        self.spatial_refiner = CAMRefiner(c.refinement_width, c.residual_scale)
        self.semantic_refiner = CAMRefiner(c.refinement_width, c.residual_scale)
        self.fusion_logit = nn.Parameter(torch.tensor(c.fusion_logit_init))

    def config_dict(self) -> dict:
        return asdict(self.config)

    def project_visual(self, features: Tensor) -> Tensor:
        return self.visual_projection(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, support: Tensor, query: Tensor,
                support_mask: Tensor, text: Tensor) -> dict[str, Tensor]:
        """One-shot episodes: Fs/Fq [B,C,H,W], Ms [B,1,Hs,Ws], t [B,D].

        Fs/Fq are frozen final-LayerNorm DINO patch features BEFORE L2
        normalization. Ms is a binary foreground mask, resized with nearest
        sampling to implement Eq. 1. Query masks are never a forward input.
        """
        c = self.config
        if support.ndim != 4 or query.ndim != 4:
            raise ValueError("Support/query features must be BCHW tensors.")
        if any(min(x.shape) <= 0 for x in (support, query, support_mask, text)):
            raise ValueError('Inputs must have nonempty dimensions.')
        batch = query.shape[0]
        if support.shape[0] != batch or support.shape[1] != c.visual_dim or query.shape[1] != c.visual_dim:
            raise ValueError("Support/query batch or encoder dimension mismatch.")
        if support_mask.ndim != 4 or support_mask.shape[:2] != (batch, 1):
            raise ValueError("Support mask must have shape [B,1,H,W].")
        if text.shape != (batch, c.text_dim):
            raise ValueError("Provide one text embedding per episode: [B,text_dim].")
        if any(x.requires_grad for x in (support, query, text, support_mask)):
            raise ValueError('Encoder features/text and masks must be detached frozen inputs.')
        if any(not x.is_floating_point() for x in (support, query, text)):
            raise ValueError('Encoder features and text must be floating point.')
        if any(x.device != self.fusion_logit.device for x in (support, query, text, support_mask)):
            raise ValueError('Inputs and DMSF parameters must share a device.')
        if any(x.dtype != self.fusion_logit.dtype for x in (support, query, text)):
            raise ValueError('Features/text and DMSF parameters must share a dtype (float32 recommended).')
        if not all(torch.isfinite(x).all() for x in (support, query, text, support_mask)):
            raise ValueError("DMSF inputs must be finite.")
        if not torch.all((support_mask == 0) | (support_mask == 1)):
            raise ValueError("Support masks must be binary 0/1, not dataset class IDs.")
        if (text.norm(dim=-1) <= c.eps).any():
            raise ValueError('Text embeddings must have nonzero norm.')

        fs = self.project_visual(support)
        fq = self.project_visual(query)
        mask = F.interpolate(support_mask.to(dtype=fs.dtype), size=fs.shape[-2:], mode="nearest")
        count = mask.sum(dim=(-2, -1))
        if (count <= 0).any():
            raise ValueError("Support foreground is empty at the patch resolution; resample the episode.")
        # Pool projected features first; normalize the resulting prototype for cosine.
        prototype = (fs * mask).sum(dim=(-2, -1)) / count
        text_hidden = self.text_projection(self.text_to_visual(F.normalize(text, dim=-1, eps=c.eps)))
        if any(not torch.isfinite(x).all() for x in (fs, fq, prototype, text_hidden)):
            raise ValueError('Non-finite projected features; inspect inputs/parameters.')
        if (prototype.norm(dim=-1) <= c.eps).any() or (text_hidden.norm(dim=-1) <= c.eps).any() or (fq.norm(dim=1) <= c.eps).any():
            raise ValueError('Projected cosine vectors must have nonzero norm.')
        query_unit = F.normalize(fq, dim=1, eps=c.eps)
        spatial_raw = (query_unit * F.normalize(prototype, dim=-1, eps=c.eps)[:, :, None, None]).sum(1, keepdim=True)
        semantic_raw = (query_unit * F.normalize(text_hidden, dim=-1, eps=c.eps)[:, :, None, None]).sum(1, keepdim=True)
        spatial_normalized = normalize_cam(spatial_raw, c.eps)
        semantic_normalized = normalize_cam(semantic_raw, c.eps)
        spatial = self.spatial_refiner(spatial_normalized)
        semantic = self.semantic_refiner(semantic_normalized)
        gamma = self.fusion_logit.sigmoid()
        return {
            "query_features": fq, "support_features": fs,
            "foreground_prototype": prototype, "text_hidden": text_hidden,
            "spatial_raw": spatial_raw, "semantic_raw": semantic_raw,
            "spatial_normalized": spatial_normalized, "semantic_normalized": semantic_normalized,
            "spatial_cam": spatial, "semantic_cam": semantic,
            "fused_cam": gamma * spatial + (1.0 - gamma) * semantic,
            "spatial_weight": gamma,
        }


def gradient_audit(model: DMSF, support: Tensor, query: Tensor,
                   support_mask: Tensor, text: Tensor) -> dict:
    """Engineering VJP check only: no optimizer, labels or parameter updates."""
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.zero_grad(set_to_none=True)
    output = model(support, query, support_mask, text)
    for name in ('spatial_cam', 'semantic_cam', 'fused_cam'):
        value = output[name]
        if value.shape != (query.shape[0], 1, *query.shape[-2:]) or not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise RuntimeError('CAM shape/range/finite check failed: ' + name)
    for name in ('spatial_raw', 'semantic_raw'):
        if output[name].abs().max() > 1.00001:
            raise RuntimeError('Cosine outside [-1,1]: ' + name)
    weights = torch.linspace(0.1, 1.0, output["fused_cam"].numel(),
                             device=query.device).reshape_as(output["fused_cam"])
    (output["fused_cam"] * weights).mean().backward()
    modules = {
        "visual_projection": model.visual_projection,
        "text_to_visual": model.text_to_visual,
        "text_projection": model.text_projection,
        "spatial_refiner_output": model.spatial_refiner.net[-1],
        "semantic_refiner_output": model.semantic_refiner.net[-1],
    }
    norms = {}
    for name, module in modules.items():
        grads = [p.grad for p in module.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise RuntimeError("Missing/non-finite gradient: " + name)
        norms[name] = float(sum(g.float().square().sum() for g in grads).sqrt().detach().cpu())
        if norms[name] == 0:
            raise RuntimeError("Zero gradient: " + name)
    grad = model.fusion_logit.grad
    if grad is None or not torch.isfinite(grad) or grad.abs() == 0:
        raise RuntimeError("Fusion scalar has no finite nonzero gradient.")
    norms["fusion_logit"] = float(grad.abs().detach().cpu())
    preceding = {}
    for name in ('spatial_refiner', 'semantic_refiner'):
        conv = getattr(model, name).net[0]
        grads = [p.grad for p in conv.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Missing/non-finite preceding-convolution gradient.')
        preceding[name] = float(sum(g.float().square().sum() for g in grads).sqrt().detach().cpu())
        if preceding[name] != 0:
            raise RuntimeError('Expected zero initial preceding-convolution gradient.')
    unchanged = all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    if not unchanged:
        raise RuntimeError('Engineering audit changed model parameters/buffers.')
    report = {
        "status": "passed", "alignment_trained": False, "optimizer_steps": 0,
        "query_labels_used": False, "probe": "synthetic vector-Jacobian product, not a training loss",
        "shapes": {key: list(value.shape) for key, value in output.items()},
        "gradient_norms": norms, "initial_spatial_weight": float(output["spatial_weight"].detach().cpu()),
        'preceding_refiner_gradient_norms_expected_zero': preceding,
        'dmsf_state_unchanged': unchanged,
        'output_ranges': {k: [float(v.detach().min().cpu()), float(v.detach().max().cpu())]
                          for k, v in output.items() if k.endswith('_cam') or k.endswith('_raw')},
        "note": "Zero-initialized refiner output layers initially block gradients to their preceding convolutions; this is expected.",
    }
    model.zero_grad(set_to_none=True)
    return report
