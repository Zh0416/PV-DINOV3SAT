from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)
COORDINATE_PATTERN = re.compile(r"^PV\d+_(\d+)_(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ImagePair:
    name: str
    image_path: Path
    mask_path: Path
    x: int | None
    y: int | None
    foreground_fraction: float


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Experiment A: frozen DINOv3-SAT foreground/background prototype matching."
    )
    parser.add_argument("--data-root", type=Path, default=project_root / "datasets")
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--masks-dir", type=Path, default=None)
    parser.add_argument("--dinov3-repo", type=Path, default=Path(__file__).resolve().parents[1] / "dinov3")
    parser.add_argument("--weights", type=Path, required=False)
    parser.add_argument("--support", default="auto", help="Image basename or 'auto'.")
    parser.add_argument(
        "--spatial-buffer",
        type=float,
        default=1000.0,
        help="Exclude query tiles within this filename-coordinate distance of support; use 0 to disable.",
    )
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--limit-query", type=int, default=0, help="Limit queries for a short smoke run.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Audit image/mask pairs without loading DINOv3.")
    return parser.parse_args()


def read_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    return mask > 0


def find_pairs(images_dir: Path, masks_dir: Path) -> list[ImagePair]:
    if not images_dir.is_dir():
        raise FileNotFoundError("Images directory does not exist: {}".format(images_dir))
    if not masks_dir.is_dir():
        raise FileNotFoundError("Masks directory does not exist: {}".format(masks_dir))

    masks = {
        path.name.removesuffix("_label.bmp") + ".bmp": path
        for path in masks_dir.glob("*_label.bmp")
    }
    pairs: list[ImagePair] = []
    missing_masks = []
    for image_path in sorted(images_dir.glob("*.bmp")):
        mask_path = masks.get(image_path.name)
        if mask_path is None:
            missing_masks.append(image_path.name)
            continue
        match = COORDINATE_PATTERN.match(image_path.stem)
        x, y = (int(match.group(1)), int(match.group(2))) if match else (None, None)
        binary_mask = read_binary_mask(mask_path)
        pairs.append(
            ImagePair(
                name=image_path.name,
                image_path=image_path,
                mask_path=mask_path,
                x=x,
                y=y,
                foreground_fraction=float(binary_mask.mean()),
            )
        )

    if missing_masks:
        raise ValueError("Images without matching *_label.bmp masks: {}".format(", ".join(missing_masks[:10])))
    if not pairs:
        raise ValueError("No BMP image/mask pairs found.")
    return pairs


def choose_support(pairs: list[ImagePair], requested: str) -> ImagePair:
    if requested.lower() != "auto":
        target = Path(requested).name
        for pair in pairs:
            if pair.name == target or Path(pair.name).stem == Path(target).stem:
                return pair
        raise ValueError("Support image is not in the paired dataset: {}".format(requested))

    coordinate_pairs = [pair for pair in pairs if pair.x is not None and pair.y is not None]
    candidates = coordinate_pairs or pairs
    median_fraction = float(np.median([pair.foreground_fraction for pair in candidates]))
    if coordinate_pairs:
        median_x = float(np.median([pair.x for pair in coordinate_pairs]))
        median_y = float(np.median([pair.y for pair in coordinate_pairs]))
        span_x = max(1.0, max(pair.x for pair in coordinate_pairs) - min(pair.x for pair in coordinate_pairs))
        span_y = max(1.0, max(pair.y for pair in coordinate_pairs) - min(pair.y for pair in coordinate_pairs))

        def score(pair: ImagePair) -> float:
            spatial = math.hypot((pair.x - median_x) / span_x, (pair.y - median_y) / span_y)
            return spatial + abs(pair.foreground_fraction - median_fraction)

        return min(coordinate_pairs, key=score)
    return min(candidates, key=lambda pair: abs(pair.foreground_fraction - median_fraction))


def select_queries(pairs: list[ImagePair], support: ImagePair, buffer_distance: float) -> tuple[list[ImagePair], int]:
    queries = []
    excluded = 0
    for pair in pairs:
        if pair.name == support.name:
            continue
        if buffer_distance > 0 and None not in (pair.x, pair.y, support.x, support.y):
            distance = math.hypot(pair.x - support.x, pair.y - support.y)
            if distance <= buffer_distance:
                excluded += 1
                continue
        queries.append(pair)
    return queries, excluded


def audit_data(pairs: list[ImagePair], support: ImagePair, queries: list[ImagePair], excluded: int) -> dict[str, Any]:
    image_sizes = set()
    for pair in pairs:
        with Image.open(pair.image_path) as image:
            image_sizes.add(image.size)
    return {
        "paired_images": len(pairs),
        "positive_masks": sum(pair.foreground_fraction > 0 for pair in pairs),
        "foreground_fraction_mean": float(np.mean([pair.foreground_fraction for pair in pairs])),
        "foreground_fraction_median": float(np.median([pair.foreground_fraction for pair in pairs])),
        "image_sizes": sorted([list(size) for size in image_sizes]),
        "support": support.name,
        "support_foreground_fraction": support.foreground_fraction,
        "query_images": len(queries),
        "queries_excluded_by_spatial_buffer": excluded,
    }


def make_model(repo_path: Path, weights_path: Path, device: torch.device) -> torch.nn.Module:
    if not repo_path.is_dir():
        raise FileNotFoundError("DINOv3 repository does not exist: {}".format(repo_path))
    if not weights_path.is_file():
        raise FileNotFoundError(
            "SAT-493M weights were not found at {}. Request official access and pass the downloaded ViT-L/16 checkpoint with --weights.".format(
                weights_path
            )
        )
    if not weights_path.name.endswith("-eadcf0ff.pth"):
        raise ValueError(
            "Expected the official ViT-L/16 SAT-493M checkpoint ending in '-eadcf0ff.pth'; got {}".format(
                weights_path.name
            )
        )

    sys.path.insert(0, str(repo_path.resolve()))
    from dinov3.hub.backbones import dinov3_vitl16

    model = dinov3_vitl16(pretrained=True, weights=str(weights_path.resolve()))
    model.eval().requires_grad_(False)
    return model.to(device)


def image_tensor(path: Path, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    height, width = array.shape[:2]
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device=device, dtype=torch.float32).div_(255.0)
    mean = torch.tensor(SAT_MEAN, device=device, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(SAT_STD, device=device, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std, (height, width)


@torch.inference_mode()
def extract_image_features(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    crop_size: int,
    patch_size: int,
) -> torch.Tensor:
    image, (height, width) = image_tensor(image_path, device)
    if height % crop_size or width % crop_size:
        raise ValueError(
            "{} has size {}x{}; dimensions must be divisible by crop-size {}.".format(
                image_path.name, width, height, crop_size
            )
        )
    if crop_size % patch_size:
        raise ValueError("crop-size must be divisible by patch-size.")

    rows = []
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.float16
    for top in range(0, height, crop_size):
        row = []
        for left in range(0, width, crop_size):
            crop = image[:, top : top + crop_size, left : left + crop_size].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                layers = model.get_intermediate_layers(crop, n=4, reshape=True, norm=True)
            normalized_layers = [F.normalize(layer.float(), dim=1) for layer in layers]
            features = F.normalize(torch.stack(normalized_layers).mean(dim=0), dim=1)
            row.append(features[0].cpu())
        rows.append(torch.cat(row, dim=2))
    return torch.cat(rows, dim=1)


def make_prototypes(features: torch.Tensor, mask_path: Path, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    foreground = torch.from_numpy(read_binary_mask(mask_path)).float()[None, None]
    height, width = foreground.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError("Mask dimensions must be divisible by patch-size {}.".format(patch_size))
    fractions = F.avg_pool2d(foreground, kernel_size=patch_size, stride=patch_size)[0, 0]
    if tuple(fractions.shape) != tuple(features.shape[-2:]):
        raise ValueError("Support mask and extracted DINO feature grids do not align.")
    fg_weights = fractions.reshape(-1)
    bg_weights = (1.0 - fractions).reshape(-1)
    flat_features = features.permute(1, 2, 0).reshape(-1, features.shape[0])
    if fg_weights.sum() == 0 or bg_weights.sum() == 0:
        raise ValueError("Support mask must contain both foreground and background pixels.")
    foreground_prototype = F.normalize((flat_features * fg_weights[:, None]).sum(0), dim=0)
    background_prototype = F.normalize((flat_features * bg_weights[:, None]).sum(0), dim=0)
    return foreground_prototype, background_prototype


def predict_score(
    features: torch.Tensor,
    foreground_prototype: torch.Tensor,
    background_prototype: torch.Tensor,
    output_size: tuple[int, int],
) -> np.ndarray:
    flat_features = F.normalize(features, dim=0)
    score = (flat_features * foreground_prototype[:, None, None]).sum(0)
    score -= (flat_features * background_prototype[:, None, None]).sum(0)
    full_resolution = F.interpolate(
        score[None, None], size=output_size, mode="bilinear", align_corners=False
    )[0, 0]
    return full_resolution.float().cpu().numpy()


def metric_row(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    truth = target.astype(bool, copy=False)
    pred = prediction.astype(bool, copy=False)
    tp = int(np.logical_and(truth, pred).sum())
    fp = int(np.logical_and(~truth, pred).sum())
    fn = int(np.logical_and(truth, ~pred).sum())
    tn = int(np.logical_and(~truth, ~pred).sum())
    fg_iou = tp / max(1, tp + fp + fn)
    bg_iou = tn / max(1, tn + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou_foreground": fg_iou,
        "iou_background": bg_iou,
        "miou": (fg_iou + bg_iou) / 2.0,
        "dice": 2.0 * tp / max(1, 2 * tp + fp + fn),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def aggregate_metrics(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    totals = {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
    fg_iou = tp / max(1, tp + fp + fn)
    bg_iou = tn / max(1, tn + fp + fn)
    return {
        "query_count": len(rows),
        "micro_iou_foreground": fg_iou,
        "micro_iou_background": bg_iou,
        "micro_miou": (fg_iou + bg_iou) / 2.0,
        "micro_dice": 2.0 * tp / max(1, 2 * tp + fp + fn),
        "micro_precision": tp / max(1, tp + fp),
        "micro_recall": tp / max(1, tp + fn),
        "macro_miou": float(np.mean([float(row["miou"]) for row in rows])),
        "macro_dice": float(np.mean([float(row["dice"]) for row in rows])),
        **totals,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    images_dir = (args.images_dir or args.data_root / "images").resolve()
    masks_dir = (args.masks_dir or args.data_root / "mask").resolve()
    repo_path = args.dinov3_repo if args.dinov3_repo.is_absolute() else project_root / args.dinov3_repo
    pairs = find_pairs(images_dir, masks_dir)
    support = choose_support(pairs, args.support)
    queries, excluded = select_queries(pairs, support, args.spatial_buffer)
    if args.limit_query > 0:
        queries = queries[: args.limit_query]
    audit = audit_data(pairs, support, queries, excluded)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.validate_only:
        return
    if not queries:
        raise ValueError("No query images remain; reduce --spatial-buffer or provide more data.")
    if args.weights is None:
        raise ValueError("Pass the official SAT-493M ViT-L/16 checkpoint with --weights.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment.")
    model = make_model(repo_path.resolve(), args.weights.resolve(), device)
    support_features = extract_image_features(model, support.image_path, device, args.crop_size, args.patch_size)
    foreground_prototype, background_prototype = make_prototypes(
        support_features, support.mask_path, args.patch_size
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir = output_dir / "predictions"
    if not args.no_save_predictions:
        predictions_dir.mkdir()

    rows: list[dict[str, float | int | str]] = []
    for index, query in enumerate(queries, start=1):
        features = extract_image_features(model, query.image_path, device, args.crop_size, args.patch_size)
        with Image.open(query.image_path) as image:
            output_size = (image.height, image.width)
        scores = predict_score(features, foreground_prototype, background_prototype, output_size)
        prediction = scores >= args.threshold
        target = read_binary_mask(query.mask_path)
        values = metric_row(target, prediction)
        rows.append({"image": query.name, "foreground_fraction": query.foreground_fraction, **values})
        if not args.no_save_predictions:
            output_name = Path(query.name).with_suffix(".png").name
            Image.fromarray(prediction.astype(np.uint8) * 255, mode="L").save(predictions_dir / output_name)
        if index == 1 or index % 10 == 0 or index == len(queries):
            print("[{}/{}] {} mIoU={:.4f} Dice={:.4f}".format(index, len(queries), query.name, values["miou"], values["dice"]))

    metrics_path = output_dir / "per_image_metrics.csv"
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "A_frozen_dinov3_sat_foreground_background_prototypes",
        "support": support.name,
        "support_foreground_fraction": support.foreground_fraction,
        "spatial_buffer_coordinate_units": args.spatial_buffer,
        "queries_excluded_by_spatial_buffer": excluded,
        "query_count": len(rows),
        "backbone": "DINOv3 ViT-L/16 SAT-493M (frozen)",
        "checkpoint": str(args.weights.resolve()),
        "device": str(device),
        "crop_size": args.crop_size,
        "patch_size": args.patch_size,
        "feature_layers": "last 4 normalized patch feature maps, averaged",
        "prototype": "mask-fraction weighted foreground and background means",
        "score": "cosine(foreground prototype) - cosine(background prototype)",
        "threshold": args.threshold,
        "metrics": aggregate_metrics(rows),
        "data_audit": audit,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Summary: {}".format(output_dir / "summary.json"))
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
