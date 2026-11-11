"""S0 spatial-prototype baseline and the decoder shared by S0-S3."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class UpsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.layers(x)


class SharedDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(257, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(ResidualBlock(), ResidualBlock())
        self.upsample = nn.Sequential(
            UpsampleBlock(64, 64), UpsampleBlock(64, 32),
            UpsampleBlock(32, 16), UpsampleBlock(16, 8),
        )
        self.output = nn.Conv2d(8, 1, 1)

    def forward(self, projected_query: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
        return self.output(self.upsample(self.residual(
            self.stem(torch.cat((projected_query, cam), dim=1)))))


class S0Segmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual_projection = nn.Sequential(
            nn.Conv2d(1024, 256, 1), nn.GroupNorm(32, 256), nn.ReLU(inplace=True),
        )
        self.decoder = SharedDecoder()

    def forward(self, support: torch.Tensor, query: torch.Tensor,
                support_mask: torch.Tensor, include_logits: bool = True) -> dict[str, torch.Tensor]:
        if support.ndim != 4 or support.shape[1:] != (1024, 32, 32):
            raise ValueError("Support token shape must be [B,1024,32,32].")
        if query.shape != support.shape or support_mask.shape != (support.shape[0], 1, 512, 512):
            raise ValueError("Query tokens or full-resolution support mask have the wrong shape.")
        if any(x.requires_grad for x in (support, query, support_mask)):
            raise ValueError("Frozen tokens and support mask must be detached inputs.")
        if not torch.all((support_mask == 0) | (support_mask == 1)):
            raise ValueError("Support mask must contain only 0 and 1.")
        projected_support = self.visual_projection(support)
        projected_query = self.visual_projection(query)
        fractions = F.avg_pool2d(support_mask.float(), kernel_size=16, stride=16)
        foreground_count = fractions.sum((-2, -1))
        background = 1 - fractions
        background_count = background.sum((-2, -1))
        if (foreground_count <= 0).any() or (background_count <= 0).any():
            raise ValueError("Support must contain foreground and background at token scale.")
        support_float = projected_support.float()
        query_float = projected_query.float()
        foreground_proto = (support_float * fractions).sum((-2, -1)) / foreground_count
        background_proto = (support_float * background).sum((-2, -1)) / background_count
        query_unit = F.normalize(query_float, dim=1, eps=1e-6)
        foreground_score = (query_unit * F.normalize(foreground_proto, dim=1, eps=1e-6)[:, :, None, None]).sum(1, keepdim=True)
        background_score = (query_unit * F.normalize(background_proto, dim=1, eps=1e-6)[:, :, None, None]).sum(1, keepdim=True)
        cam = foreground_score - background_score
        return {"logits": self.decoder(projected_query, cam) if include_logits else None, "cam": cam}


def s0_loss(output: dict[str, torch.Tensor], target: torch.Tensor,
            hard_negative: bool, positive_weight: float = 2.0,
            cam_weight: float = 0.1, phase: str = "joint") -> tuple[torch.Tensor, dict[str, float]]:
    if phase not in ("warmup", "joint"):
        raise ValueError("Unknown S0 training phase.")
    target = target.float()
    if target.shape != (output["cam"].shape[0], 1, 512, 512) or not torch.all((target == 0) | (target == 1)):
        raise ValueError("Target must be a matching binary 512x512 mask.")
    if hard_negative and target.any():
        raise ValueError("Hard-negative query must have an empty mask.")
    if not hard_negative and not target.any():
        raise ValueError("Positive query must contain foreground.")
    # Fixed temperature 1.0: supervise the raw cosine difference directly.
    cam_logits = F.interpolate(output["cam"].float(), size=(512, 512), mode="bilinear", align_corners=False)
    cam_loss = F.binary_cross_entropy_with_logits(cam_logits, target, pos_weight=target.new_tensor(positive_weight))
    if phase == "warmup":
        return cam_loss, {"bce": 0.0, "dice": 0.0, "cam": float(cam_loss.detach())}
    logits = output["logits"].float()
    if logits.shape != target.shape:
        raise ValueError("Decoder and target shapes differ.")
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=logits.new_tensor(positive_weight))
    if hard_negative:
        return bce + cam_weight * cam_loss, {"bce": float(bce.detach()), "dice": 0.0, "cam": float(cam_loss.detach())}
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
    return bce + dice + cam_weight * cam_loss, {
        "bce": float(bce.detach()), "dice": float(dice.detach()), "cam": float(cam_loss.detach())}
