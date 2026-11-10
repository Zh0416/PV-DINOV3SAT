#!/usr/bin/env python3
"""Support-guided local 2-D FFT CAMs over frozen DINOv3-SAT patch tokens."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DMSF_ROOT = REPO_ROOT / "codex运行文件" / "dmsf_v1"
sys.path.insert(0, str(DMSF_ROOT))
from feature_cache import (  # noqa: E402
    DINOV3_COMMIT,
    SAT_MEAN,
    SAT_STD,
    WEIGHT_NAME,
    encode_dino,
    load_dino,
)


EPS = 1e-8
DEFAULT_DATA_ROOT = Path("/kaggle/input/datasets/zh0416/pv-dinov3sat-experiment-a")
DEFAULT_SCALES = (4, 6, 8)
CAM_TEMPERATURE = 5.0
CAM_THRESHOLD = 0.5


@dataclass(frozen=True)
class ImagePair:
    name: str
    image: Path
    mask: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dino-repo", type=Path, default=Path("/kaggle/working/dinov3"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/dinov3_patch_token_frequency_cam"))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--window-sizes", nargs="+", type=int, default=list(DEFAULT_SCALES))
    parser.add_argument("--foreground-coverage", type=float, default=0.60)
    parser.add_argument("--background-coverage", type=float, default=0.05)
    parser.add_argument("--hard-negatives", type=int, default=0,
                        help="Optional support-only top-k confusing background windows per feature space.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_dataset_layout(path: Path) -> bool:
    return (path / "images").is_dir() and (path / "mask").is_dir()


def resolve_dataset_root(requested: Path) -> Path:
    candidates = [
        requested,
        Path("/kaggle/input/datasets/zh0416/pv-dinov3sat-experiment-a"),
        Path("/kaggle/input/pv-dinov3sat-experiment-a"),
    ]
    for candidate in candidates:
        if has_dataset_layout(candidate):
            return candidate.resolve()
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for benchmark in kaggle_input.rglob("spatial_cam_code/benchmark_v1.json"):
            candidate = benchmark.parent.parent
            if has_dataset_layout(candidate):
                return candidate.resolve()
    locations = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate the Experiment A images/mask folders. Checked: {locations}")


def find_benchmark(root: Path) -> Path:
    preferred = [root / "spatial_cam_code" / "benchmark_v1.json",
                 root / "layer_selection_code" / "benchmark_v1.json",
                 root / "benchmark_v1.json"]
    for path in preferred:
        if path.is_file():
            return path
    found = sorted(root.rglob("benchmark_v1.json"))
    if found:
        return found[0]
    raise FileNotFoundError(f"benchmark_v1.json was not found under {root}")


def read_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        raw = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    return raw > 0


def discover_pairs(root: Path) -> dict[str, ImagePair]:
    image_paths = sorted(path for path in (root / "images").rglob("*")
                         if path.is_file() and path.suffix.lower() in {".bmp", ".png", ".tif", ".tiff"})
    mask_paths = sorted(path for path in (root / "mask").rglob("*")
                        if path.is_file() and path.suffix.lower() in {".bmp", ".png", ".tif", ".tiff"})
    masks: dict[str, list[Path]] = {}
    for path in mask_paths:
        stem = path.stem[:-6] if path.stem.endswith("_label") else path.stem
        masks.setdefault(stem, []).append(path)
    pairs: dict[str, ImagePair] = {}
    missing = []
    for image in image_paths:
        found = masks.get(image.stem, [])
        if len(found) != 1:
            if not found:
                missing.append(image.name)
            else:
                raise ValueError(f"Ambiguous masks for {image.name}: {found}")
            continue
        if image.name in pairs:
            raise ValueError(f"Duplicate image basename: {image.name}")
        pairs[image.name] = ImagePair(image.name, image, found[0])
    if missing:
        raise FileNotFoundError("Images without one matching mask: " + ", ".join(missing[:10]))
    if not pairs or len(pairs) != len(image_paths):
        raise ValueError("No complete image/mask pairs were found.")
    return pairs


def read_split(root: Path, pairs: dict[str, ImagePair]) -> tuple[dict[str, Any], str, list[str], Path]:
    manifest_path = find_benchmark(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "support_image" not in manifest or "query_images" not in manifest:
        raise ValueError("benchmark_v1.json must define support_image and query_images.")
    support = Path(manifest["support_image"]).name
    queries = [Path(name).name for name in manifest["query_images"]]
    if len(set(queries)) != len(queries) or support in queries:
        raise ValueError("Benchmark support and query filenames must be unique and disjoint.")
    absent = sorted(({support, *queries}) - set(pairs))
    if absent:
        raise FileNotFoundError("Benchmark files are missing from the mounted image/mask pairs: " + ", ".join(absent))
    return manifest, support, queries, manifest_path


def audit_pairs(pairs: dict[str, ImagePair], selected_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for name in selected_names:
        pair = pairs[name]
        with Image.open(pair.image) as image:
            image_size = list(image.size)
        with Image.open(pair.mask) as mask_image:
            mask_size = list(mask_image.size)
            mask_values = sorted(int(value) for value in np.unique(np.asarray(mask_image.convert("L"))))
        if image_size != [1024, 1024] or mask_size != image_size:
            raise ValueError(f"Expected paired 1024x1024 files for {name}: {image_size} and {mask_size}.")
        rows.append({"image": name, "mask": pair.mask.name, "image_hw": [1024, 1024],
                     "mask_values": mask_values, "image_sha256": sha256_file(pair.image),
                     "mask_sha256": sha256_file(pair.mask)})
    return rows


def coordinate_from_name(name: str) -> tuple[int, int] | None:
    match = re.search(r"PV08_(\d+)_(\d+)", name)
    return (int(match.group(1)), int(match.group(2))) if match else None


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def resolve_weights(requested: Path | None) -> Path:
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(f"DINOv3-SAT checkpoint not found: {requested}")
        return requested.resolve()
    env_path = os.environ.get("DMSF_DINO_WEIGHTS")
    if env_path and Path(env_path).is_file():
        return Path(env_path).resolve()
    roots = [Path("/kaggle/input"), Path.cwd()]
    matches = []
    for root in roots:
        if root.is_dir():
            matches.extend(root.rglob(WEIGHT_NAME))
    matches = sorted({path.resolve() for path in matches})
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one mounted {WEIGHT_NAME}, found {len(matches)}. Pass --weights explicitly.")
    return matches[0]


def ensure_dinov3_repo(repo: Path) -> Path:
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/facebookresearch/dinov3.git", str(repo)], check=True)
    if not (repo / ".git").is_dir():
        raise ValueError(f"DINOv3 path exists but is not a Git checkout: {repo}")
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", DINOV3_COMMIT], check=True)
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != DINOV3_COMMIT:
        raise ValueError("DINOv3 source checkout differs from the repository's pinned encoder commit.")
    return repo


def patch_mask_fractions(mask: np.ndarray, grid_hw: tuple[int, int]) -> torch.Tensor:
    height, width = mask.shape
    grid_h, grid_w = grid_hw
    if height % grid_h or width % grid_w:
        raise ValueError("Mask dimensions must be divisible by the DINO patch grid.")
    fractions = F.adaptive_avg_pool2d(torch.from_numpy(mask.astype(np.float32))[None, None], grid_hw)[0, 0]
    return fractions


def _reflect_pad(tensor: torch.Tensor, before: int, after: int) -> torch.Tensor:
    if before == 0 and after == 0:
        return tensor
    return F.pad(tensor, (before, after, before, after), mode="reflect")


def window_coverage(patch_fraction: torch.Tensor, window_size: int) -> torch.Tensor:
    if patch_fraction.ndim != 2:
        raise ValueError("Patch coverage must be a 2-D token grid.")
    before = (window_size - 1) // 2
    after = window_size - 1 - before
    padded = _reflect_pad(patch_fraction[None, None].float(), before, after)
    return F.avg_pool2d(padded, kernel_size=window_size, stride=1)[0, 0]


def local_fft_descriptors(features: torch.Tensor, window_size: int, chunk_rows: int = 8) -> torch.Tensor:
    """Return HxWxD descriptors from per-channel spatial FFTs of local windows.

    Every feature channel is windowed, centered, and transformed over its two
    spatial window dimensions. Per-channel normalized log-power is summarized
    by channel mean and standard deviation at each 2-D frequency bin.
    """
    if features.ndim != 3:
        raise ValueError("Features must be CxHxW spatial patch tokens.")
    channels, height, width = features.shape
    if window_size < 4:
        raise ValueError("Use a spatial window of at least four tokens.")
    before = (window_size - 1) // 2
    after = window_size - 1 - before
    padded = _reflect_pad(features[None].float().contiguous(), before, after)
    hann = torch.hann_window(window_size, periodic=False, dtype=torch.float32, device=features.device)
    taper = hann[:, None] * hann[None, :]
    chunks = []
    for row in range(0, height, chunk_rows):
        rows = min(chunk_rows, height - row)
        stripe = padded[:, :, row:row + rows + window_size - 1, :]
        patches = F.unfold(stripe, kernel_size=window_size, stride=1)
        windows = patches[0].transpose(0, 1).reshape(rows * width, channels, window_size, window_size)
        centered = windows - windows.mean(dim=(-2, -1), keepdim=True)
        spectrum = torch.fft.fft2(centered * taper, dim=(-2, -1), norm="ortho")
        power = spectrum.abs().square()
        power = power / power.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
        log_power = power.clamp_min(EPS).log()
        channel_mean = log_power.mean(dim=1)
        channel_std = log_power.std(dim=1, unbiased=False)
        descriptor = torch.cat((channel_mean.flatten(1), channel_std.flatten(1)), dim=1)
        chunks.append(F.normalize(descriptor, dim=1, eps=EPS))
    return torch.cat(chunks, dim=0).reshape(height, width, 2 * window_size * window_size)


def spatial_prototypes(features: torch.Tensor, patch_fraction: torch.Tensor,
                       hard_negatives: int = 0) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    unit = F.normalize(features.float(), dim=0, eps=EPS)
    weights = patch_fraction.to(unit.device)
    if weights.shape != unit.shape[-2:]:
        raise ValueError("Support pixel mask and DINO token grid differ.")
    fg_weight, bg_weight = weights.flatten(), (1.0 - weights).flatten()
    vectors = unit.permute(1, 2, 0).reshape(-1, unit.shape[0])
    if fg_weight.sum() <= 0 or bg_weight.sum() <= 0:
        raise ValueError("Support mask must include PV foreground and background.")
    positive = F.normalize((vectors * fg_weight[:, None]).sum(0), dim=0, eps=EPS)
    negative_candidates = vectors[bg_weight > 0.5]
    negative_count = len(negative_candidates)
    if hard_negatives > 0 and negative_count > hard_negatives:
        keep = torch.topk(negative_candidates @ positive, k=hard_negatives, largest=True, sorted=True).indices
        negative_candidates = negative_candidates[keep]
    negative = F.normalize(negative_candidates.mean(0), dim=0, eps=EPS)
    return positive, negative, {"positive_weight": int((fg_weight > 0).sum()),
                                "background_candidates": negative_count,
                                "background_prototype_count": int(len(negative_candidates))}


def frequency_prototypes(descriptors: torch.Tensor, coverage: torch.Tensor,
                          foreground_coverage: float, background_coverage: float,
                          hard_negatives: int = 0) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    descriptor = descriptors.reshape(-1, descriptors.shape[-1]).float()
    fractions = coverage.flatten().to(descriptor.device)
    positive_mask = fractions >= foreground_coverage
    negative_mask = fractions <= background_coverage
    positive_samples = descriptor[positive_mask]
    negative_samples = descriptor[negative_mask]
    if len(positive_samples) == 0 or len(negative_samples) == 0:
        raise ValueError(
            f"Window {coverage.shape} has {len(positive_samples)} PV and "
            f"{len(negative_samples)} background centers; adjust mask coverage criteria."
        )
    positive = F.normalize(positive_samples.mean(0), dim=0, eps=EPS)
    candidate_count = len(negative_samples)
    if hard_negatives > 0 and candidate_count > hard_negatives:
        keep = torch.topk(negative_samples @ positive, k=hard_negatives, largest=True, sorted=True).indices
        negative_samples = negative_samples[keep]
    negative = F.normalize(negative_samples.mean(0), dim=0, eps=EPS)
    diagnostics = {
        "positive_windows": int(len(positive_samples)),
        "background_candidates": int(candidate_count),
        "background_prototype_windows": int(len(negative_samples)),
        "ignored_mixed_windows": int((~positive_mask & ~negative_mask).sum()),
        "mean_pv_coverage": float(fractions[positive_mask].mean()),
        "mean_background_coverage": float(fractions[negative_mask].mean()),
    }
    return positive, negative, diagnostics


def cosine_difference(features: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                      *, channels_first: bool = False) -> torch.Tensor:
    if positive.ndim != 1 or negative.shape != positive.shape:
        raise ValueError("Positive and negative prototypes must be same-sized vectors.")
    if features.ndim != 3:
        raise ValueError("Cosine CAM features must be a three-dimensional grid.")
    if channels_first:
        if features.shape[0] != positive.numel():
            raise ValueError("Channel-first feature dimension and prototype size differ.")
        unit = F.normalize(features.float(), dim=0, eps=EPS)
        return (unit * positive[:, None, None]).sum(0) - (unit * negative[:, None, None]).sum(0)
    if features.shape[-1] != positive.numel():
        raise ValueError("Channel-last descriptor dimension and prototype size differ.")
    flattened = features.reshape(-1, features.shape[-1]).float()
    scores = flattened @ positive.float() - flattened @ negative.float()
    return scores.reshape(features.shape[:2])


def score_to_probability(score: torch.Tensor, output_hw: tuple[int, int]) -> np.ndarray:
    probability = torch.sigmoid(score.float() * CAM_TEMPERATURE)[None, None]
    upsampled = F.interpolate(probability, size=output_hw, mode="bilinear", align_corners=False)
    return upsampled[0, 0].cpu().numpy().astype(np.float32)


def binary_counts(target: np.ndarray, probability: np.ndarray) -> dict[str, int | float]:
    truth = np.asarray(target, dtype=bool)
    predicted = probability >= CAM_THRESHOLD
    tp = int(np.logical_and(truth, predicted).sum())
    fp = int(np.logical_and(~truth, predicted).sum())
    fn = int(np.logical_and(truth, ~predicted).sum())
    tn = int(np.logical_and(~truth, ~predicted).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": tp / max(1, tp + fp + fn),
            "dice": 2 * tp / max(1, 2 * tp + fp + fn),
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn)}


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "tn")}
    micro = binary_counts_from_counts(counts)
    metric_names = ("iou", "dice", "precision", "recall")
    macro = {name: float(np.mean([row[name] for row in rows])) for name in metric_names}
    return {"micro": {**counts, **micro}, "macro_per_image": macro,
            "image_count": len(rows)}


def binary_counts_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {"iou": tp / max(1, tp + fp + fn),
            "dice": 2 * tp / max(1, 2 * tp + fp + fn),
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn)}


def _cam_color(values: np.ndarray) -> np.ndarray:
    value = np.clip(values, 0.0, 1.0)[..., None]
    red = np.clip(2.0 * value, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * value - 1.0) * 1.5, 0.0, 1.0)
    blue = np.clip(1.0 - 2.0 * value, 0.0, 1.0)
    return (np.concatenate((red, green, blue), axis=-1) * 255).astype(np.uint8)


def _error_overlay(rgb: np.ndarray, truth: np.ndarray, probability: np.ndarray) -> np.ndarray:
    predicted = probability >= CAM_THRESHOLD
    colors = np.zeros_like(rgb)
    colors[truth & predicted] = (0, 210, 70)
    colors[truth & ~predicted] = (35, 95, 255)
    colors[~truth & predicted] = (255, 45, 35)
    active = truth | predicted
    output = rgb.copy()
    output[active] = (0.25 * rgb[active] + 0.75 * colors[active]).astype(np.uint8)
    return output


def save_diagnostics(image_path: Path, truth: np.ndarray, probabilities: dict[str, np.ndarray],
                     output_path: Path) -> None:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.shape[:2] != truth.shape:
        raise ValueError("Query image and mask dimensions differ during visualization.")
    tiles = [("Image", rgb), ("PV mask", np.repeat(truth[..., None] * 255, 3, axis=2).astype(np.uint8)),
             ("DINO spatial CAM", _cam_color(probabilities["dino_spatial"])),
             ("Token FFT CAM", _cam_color(probabilities["token_fft"])),
             ("Equal-weight fusion", _cam_color(probabilities["fusion"])),
             ("Fusion errors: TP green, FN blue, FP red",
              _error_overlay(rgb, truth, probabilities["fusion"]))]
    tile_size = 384
    label_height = 30
    canvas = Image.new("RGB", (tile_size * 3, (tile_size + label_height) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, array) in enumerate(tiles):
        tile = Image.fromarray(array).resize((tile_size, tile_size), Image.Resampling.BILINEAR)
        x = (index % 3) * tile_size
        y = (index // 3) * (tile_size + label_height)
        draw.text((x + 5, y + 5), label, fill="black", font=font)
        canvas.paste(tile, (x, y + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=88, optimize=True)


def support_spectral_profiles(descriptors: torch.Tensor, coverage: torch.Tensor,
                              window_size: int, foreground_coverage: float,
                              background_coverage: float) -> dict[str, Any]:
    fractions = coverage.flatten().to(descriptors.device)
    descriptor = descriptors.reshape(-1, descriptors.shape[-1])
    positive = descriptor[fractions >= foreground_coverage]
    negative = descriptor[fractions <= background_coverage]
    frequency_count = window_size * window_size
    return {
        "window_size_tokens": window_size,
        "fft_axes": ["local_token_rows", "local_token_columns"],
        "frequency_layout": "standard fft2 ordering; DC at [0,0]",
        "channel_summary": "mean and population standard deviation across per-channel normalized log-power",
        "pv_mean_log_power_by_flattened_frequency_bin": positive[:, :frequency_count].mean(0).tolist(),
        "background_mean_log_power_by_flattened_frequency_bin": negative[:, :frequency_count].mean(0).tolist(),
        "frequency_bins_per_channel": frequency_count,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not (0.0 < args.background_coverage < args.foreground_coverage <= 1.0):
        raise ValueError("Require 0 < background coverage < foreground coverage <= 1.")
    if args.hard_negatives < 0 or any(size < 4 for size in args.window_sizes):
        raise ValueError("Hard-negative count must be nonnegative and FFT windows at least four tokens.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    root = resolve_dataset_root(args.dataset_root)
    pairs = discover_pairs(root)
    manifest, support_name, query_names, manifest_path = read_split(root, pairs)
    split_names = [support_name, *query_names]
    pair_audit = audit_pairs(pairs, split_names)
    device = resolve_device(args.device)
    weight_path = resolve_weights(args.weights)
    dino_repo = ensure_dinov3_repo(args.dino_repo)
    model, encoder_info = load_dino(dino_repo, weight_path, device)

    support_pair = pairs[support_name]
    support_mask = read_binary_mask(support_pair.mask)
    with Image.open(support_pair.image) as image:
        if image.mode != "RGB" or image.size != (1024, 1024):
            raise ValueError("DINOv3 encoder contract expects RGB 1024x1024 Experiment A images.")
    support_features = encode_dino(model, support_pair.image, device).to(device=device, dtype=torch.float32)
    if support_features.shape != (1024, 64, 64):
        raise ValueError(f"Unexpected DINO feature shape: {tuple(support_features.shape)}")
    patch_fraction = patch_mask_fractions(support_mask, (64, 64))
    spatial_positive, spatial_negative, spatial_support_stats = spatial_prototypes(
        support_features, patch_fraction, args.hard_negatives)

    scale_models: dict[int, dict[str, Any]] = {}
    support_profiles: dict[str, Any] = {}
    for scale in args.window_sizes:
        descriptor = local_fft_descriptors(support_features, scale)
        coverage = window_coverage(patch_fraction, scale)
        positive, negative, stats = frequency_prototypes(
            descriptor, coverage, args.foreground_coverage,
            args.background_coverage, args.hard_negatives)
        scale_models[scale] = {"positive": positive, "negative": negative, "stats": stats}
        support_profiles[str(scale)] = support_spectral_profiles(
            descriptor, coverage, scale, args.foreground_coverage, args.background_coverage)
        del descriptor
        print(f"Support FFT windows {scale}: PV={stats['positive_windows']} "
              f"background={stats['background_candidates']} ignored={stats['ignored_mixed_windows']}", flush=True)

    all_rows: dict[str, list[dict[str, Any]]] = {
        "dino_spatial": [], "token_fft": [], "fusion": []}
    per_image: dict[str, Any] = {}
    visuals_dir = args.output_dir / "diagnostics"
    for index, query_name in enumerate(query_names, start=1):
        query_pair = pairs[query_name]
        with Image.open(query_pair.image) as image:
            if image.mode != "RGB" or image.size != (1024, 1024):
                raise ValueError("DINOv3 encoder contract expects RGB 1024x1024 Experiment A images.")
        query_features = encode_dino(model, query_pair.image, device).to(device=device, dtype=torch.float32)
        spatial_score = cosine_difference(query_features, spatial_positive, spatial_negative,
                                           channels_first=True)
        spatial_probability = score_to_probability(spatial_score, (1024, 1024))
        scale_probabilities = {}
        for scale, prototypes in scale_models.items():
            descriptors = local_fft_descriptors(query_features, scale)
            score = cosine_difference(descriptors, prototypes["positive"], prototypes["negative"])
            scale_probabilities[scale] = score_to_probability(score, (1024, 1024))
            del descriptors
        fft_probability = np.mean([scale_probabilities[scale] for scale in args.window_sizes], axis=0).astype(np.float32)
        fusion_probability = (0.5 * spatial_probability + 0.5 * fft_probability).astype(np.float32)
        probabilities = {"dino_spatial": spatial_probability,
                         "token_fft": fft_probability,
                         "fusion": fusion_probability}

        # Query masks enter only after all three prediction maps have been computed.
        truth = read_binary_mask(query_pair.mask)
        method_metrics = {name: binary_counts(truth, probability)
                          for name, probability in probabilities.items()}
        for name in all_rows:
            all_rows[name].append({"image": query_name, **method_metrics[name]})
        per_image[query_name] = {
            "metrics": method_metrics,
            "predicted_pv_pixels": {name: int((probability >= CAM_THRESHOLD).sum())
                                    for name, probability in probabilities.items()},
            "true_pv_pixels": int(truth.sum()),
        }
        save_diagnostics(query_pair.image, truth, probabilities,
                         visuals_dir / (Path(query_name).stem + "_diagnostics.jpg"))
        print(f"[{index}/{len(query_names)}] {query_name}: "
              f"spatial IoU={method_metrics['dino_spatial']['iou']:.4f}, "
              f"FFT IoU={method_metrics['token_fft']['iou']:.4f}, "
              f"fusion IoU={method_metrics['fusion']['iou']:.4f}", flush=True)

    aggregate = {name: aggregate_metrics(rows) for name, rows in all_rows.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["image", "method", "tp", "fp", "fn", "tn", "iou", "dice", "precision", "recall"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in all_rows.items():
            for row in rows:
                writer.writerow({"method": method, **row})

    support_coord = coordinate_from_name(support_name)
    query_distances = {}
    if support_coord is not None:
        for name in query_names:
            coordinate = coordinate_from_name(name)
            if coordinate is not None:
                query_distances[name] = float(math.dist(support_coord, coordinate))
    split_record = {
        "source_manifest": str(manifest_path.relative_to(root)),
        "source_manifest_sha256": sha256_file(manifest_path),
        "support_image": support_name,
        "development_query_images": query_names,
        "paired_image_count_in_mounted_dataset": len(pairs),
        "evaluated_pair_count": len(pair_audit),
        "support_query_disjoint": support_name not in query_names,
        "spatial_buffer_coordinate_units": manifest.get("spatial_buffer_coordinate_units", "filename coordinate units; physical units not independently verified"),
        "nearest_support_query_distance": min(query_distances.values()) if query_distances else None,
        "support_query_distances": query_distances,
        "pair_files": pair_audit,
        "source_manifest_contents": manifest,
        "split_interpretation": "Existing benchmark_v1 1-support/20-query whole-image split; development comparison, not an independent test.",
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(split_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "support_spectral_profiles.json").write_text(
        json.dumps(support_profiles, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "experiment": "frozen_dinov3_sat_patch_token_local_fft_cam",
        "status": "completed",
        "evaluation_label": "development benchmark; query masks were not used to fit prototypes or select thresholds",
        "dataset_root": str(root),
        "dataset_image_mask_pairs": len(pairs),
        "support_image": support_name,
        "development_queries": len(query_names),
        "split_manifest": "split_manifest.json",
        "encoder": {**encoder_info, "source_commit": DINOV3_COMMIT,
                    "weights_path": str(weight_path), "device": str(device),
                    "backbone_frozen": True, "query_mask_used_for_encoder_or_prototypes": False,
                    "special_tokens": "get_intermediate_layers(..., reshape=True) returns the spatial patch grid; CLS/register tokens are excluded"},
        "input_geometry": {"image_hw": [1024, 1024], "patch_size": 16,
                           "crop_hw": [512, 512], "crop_grid_hw": [32, 32],
                           "stitched_token_grid_hw": [64, 64],
                           "preprocessing_mean": SAT_MEAN, "preprocessing_std": SAT_STD,
                           "spatial_geometry": "four independent crops with reset attention/RoPE coordinates, stitched as in dmsf_v1/feature_cache.py"},
        "frequency_method": {
            "window_sizes_tokens": args.window_sizes,
            "fft_axes": ["local token rows", "local token columns"],
            "channel_axis_fft": False,
            "token_sequence_flattened_for_fft": False,
            "preprocessing": "per-channel local mean removal, 2-D Hann window, fft2 over the two spatial window axes, power normalized per channel, log-power, then channel mean/population-standard-deviation summary per frequency bin",
            "descriptor_dimensions_per_scale": {str(size): 2 * size * size for size in args.window_sizes},
            "prototype": "cosine similarity difference from support PV windows minus support background windows; mixed-coverage windows ignored",
            "positive_coverage_at_least": args.foreground_coverage,
            "background_coverage_at_most": args.background_coverage,
            "hard_negatives_per_class_prototype": args.hard_negatives,
            "support_windows": {str(scale): model["stats"] for scale, model in scale_models.items()},
            "spectrum_diagnostics": "support_spectral_profiles.json",
        },
        "prediction": {"cam_conversion": f"sigmoid({CAM_TEMPERATURE} * cosine_difference)",
                       "threshold": CAM_THRESHOLD, "threshold_selection": "fixed before query evaluation; no query-mask threshold tuning",
                       "fusion": "equal-weight mean of DINO spatial and multi-scale token-FFT probabilities"},
        "seed": args.seed,
        "metrics": aggregate,
        "per_image": per_image,
        "visualizations": "diagnostics/<image>_diagnostics.jpg; green=TP, blue=FN, red=FP in fusion error panels",
        "background_semantics": "All mask-zero pixels are one background class. Bare-soil/cropland subtypes are not labeled, so category-specific effects are not separately measurable.",
        "dino_baseline_provenance": "DINO spatial CAM was recomputed from the mounted support image and frozen patch tokens in this run; no pre-existing notebook CAM artifact was assumed.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "support": support_name,
                      "queries": len(query_names), "metrics": aggregate}, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
