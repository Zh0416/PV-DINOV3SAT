from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from run_experiment_a import (
    aggregate_metrics,
    image_tensor,
    make_model,
    metric_row,
    read_binary_mask,
)


LAYER_NUMBERS = (6, 12, 18, 24)
BLOCK_INDICES = tuple(number - 1 for number in LAYER_NUMBERS)
SAT_PATCH_SIZE = 16


@dataclass(frozen=True)
class ImagePair:
    name: str
    image_path: Path
    mask_path: Path
    x: int
    y: int


def parse_args() -> argparse.Namespace:
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="DINOv3-SAT intermediate-layer selection experiment.")
    parser.add_argument("--data-root", type=Path, default=experiment_dir / "benchmark_v1_data")
    parser.add_argument("--dinov3-repo", type=Path, default=Path(__file__).resolve().parents[1] / "dinov3")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, default=Path(__file__).resolve().parent / "benchmark_v1.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--spatial-buffer", type=float, default=1000.0)
    parser.add_argument("--train-episodes", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--score-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_pairs(data_root: Path, expected_names: list[str]) -> list[ImagePair]:
    images_dir = data_root / "images"
    masks_dir = data_root / "mask"
    image_by_name = {path.name: path for path in images_dir.glob("*.bmp")}
    mask_by_image = {
        path.name.removesuffix("_label.bmp") + ".bmp": path
        for path in masks_dir.glob("*_label.bmp")
    }
    expected = set(expected_names)
    if set(image_by_name) != expected or set(mask_by_image) != expected:
        raise ValueError(
            "Quick-test data must contain exactly the 21 images and masks in benchmark_v1.json."
        )
    pairs = []
    for name in expected_names:
        image_path = image_by_name[name]
        mask_path = mask_by_image[name]
        parts = name.removesuffix(".bmp").split("_")
        if len(parts) < 3:
            raise ValueError("Expected x/y coordinates in filename: {}".format(name))
        pairs.append(
            ImagePair(
                name=name,
                image_path=image_path,
                mask_path=mask_path,
                x=int(parts[-2]),
                y=int(parts[-1]),
            )
        )
    return pairs


def spatial_distance(left: ImagePair, right: ImagePair) -> float:
    return float(np.hypot(left.x - right.x, left.y - right.y))


@torch.no_grad()
def extract_layer_features(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    crop_size: int,
) -> torch.Tensor:
    image, (height, width) = image_tensor(image_path, device)
    if height % crop_size or width % crop_size or crop_size % SAT_PATCH_SIZE:
        raise ValueError("Image dimensions must divide evenly into crop-size and 16-pixel patches.")

    rows: list[list[list[torch.Tensor]]] = [
        [[] for _ in LAYER_NUMBERS] for _ in range(height // crop_size)
    ]
    for row_index, top in enumerate(range(0, height, crop_size)):
        for left in range(0, width, crop_size):
            crop = image[:, top : top + crop_size, left : left + crop_size].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                layers = model.get_intermediate_layers(
                    crop, n=BLOCK_INDICES, reshape=True, norm=True
                )
            if len(layers) != len(LAYER_NUMBERS):
                raise RuntimeError("DINOv3 did not return all requested intermediate blocks.")
            for layer_index, layer in enumerate(layers):
                normalized = F.normalize(layer.float(), dim=1)[0].to(device="cpu", dtype=torch.float16)
                rows[row_index][layer_index].append(normalized)

    complete_layers = []
    for layer_index in range(len(LAYER_NUMBERS)):
        feature_rows = [torch.cat(row[layer_index], dim=2) for row in rows]
        complete_layers.append(torch.cat(feature_rows, dim=1))
    return torch.stack(complete_layers)


def mask_fraction(mask_path: Path, device: torch.device) -> torch.Tensor:
    mask = torch.from_numpy(read_binary_mask(mask_path).astype(np.float32))[None, None]
    return F.avg_pool2d(mask, kernel_size=SAT_PATCH_SIZE, stride=SAT_PATCH_SIZE)[0, 0].to(device)


def fuse_layers(features: torch.Tensor, weights: torch.Tensor, device: torch.device) -> torch.Tensor:
    layer_features = features.to(device=device, dtype=torch.float32)
    fused = torch.sum(layer_features * weights[:, None, None, None], dim=0)
    return F.normalize(fused, dim=0)


def make_prototypes(features: torch.Tensor, fractions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = features.permute(1, 2, 0).reshape(-1, features.shape[0])
    fg_weights = fractions.reshape(-1)
    bg_weights = 1.0 - fg_weights
    if fg_weights.sum() <= 0 or bg_weights.sum() <= 0:
        raise ValueError("Episode support mask must contain foreground and background.")
    fg = F.normalize((flat * fg_weights[:, None]).sum(0), dim=0)
    bg = F.normalize((flat * bg_weights[:, None]).sum(0), dim=0)
    return fg, bg


def score_map(query_features: torch.Tensor, prototypes: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    query_features = F.normalize(query_features, dim=0)
    fg, bg = prototypes
    return torch.sum(query_features * fg[:, None, None], dim=0) - torch.sum(
        query_features * bg[:, None, None], dim=0
    )


def episode_loss(
    support_features: torch.Tensor,
    support_mask: torch.Tensor,
    query_features: torch.Tensor,
    query_mask: torch.Tensor,
    weights: torch.Tensor,
    score_scale: float,
) -> torch.Tensor:
    support_fused = fuse_layers(support_features, weights, weights.device)
    query_fused = fuse_layers(query_features, weights, weights.device)
    prototypes = make_prototypes(support_fused, support_mask)
    logits = score_map(query_fused, prototypes) * score_scale
    bce = F.binary_cross_entropy_with_logits(logits, query_mask)
    probability = torch.sigmoid(logits)
    intersection = torch.sum(probability * query_mask)
    dice = 1.0 - (2.0 * intersection + 1.0) / (probability.sum() + query_mask.sum() + 1.0)
    return 0.5 * bce + 0.5 * dice


def cached_episode_loss(
    support_features: torch.Tensor,
    support_mask: torch.Tensor,
    query_features: torch.Tensor,
    query_mask: torch.Tensor,
    weights: torch.Tensor,
    score_scale: float,
) -> float:
    with torch.no_grad():
        loss = episode_loss(
            support_features, support_mask,
            query_features, query_mask, weights, score_scale,
        )
    return float(loss.item())


def train_layer_weights(
    support_features: torch.Tensor,
    support_fractions: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if support_features.shape[-2:] != (64, 64) or support_fractions.shape != (64, 64):
        raise ValueError("Support image and mask must form a 64x64 patch grid at 512px crop size.")
    half = support_features.shape[-1] // 2

    def region(row: int, col: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = slice(row * half, (row + 1) * half)
        cols = slice(col * half, (col + 1) * half)
        return support_features[..., rows, cols], support_fractions[rows, cols]

    train_episodes = [(region(0, 0), region(0, 1)), (region(0, 1), region(0, 0))]
    validation_episodes = [(region(1, 0), region(1, 1)), (region(1, 1), region(1, 0))]
    device = support_features.device
    logits = torch.nn.Parameter(torch.zeros(len(LAYER_NUMBERS), device=device))
    optimizer = torch.optim.Adam([logits], lr=args.learning_rate)
    equal_weights = torch.full((len(LAYER_NUMBERS),), 1.0 / len(LAYER_NUMBERS), device=device)
    initial_validation_loss = float(np.mean([
        cached_episode_loss(s_feat, s_mask, q_feat, q_mask, equal_weights, args.score_scale)
        for (s_feat, s_mask), (q_feat, q_mask) in validation_episodes
    ]))
    history = []

    for episode_index in range(1, args.train_episodes + 1):
        (episode_support, episode_support_mask), (episode_query, episode_query_mask) = train_episodes[
            (episode_index - 1) % len(train_episodes)
        ]
        weights = torch.softmax(logits, dim=0)
        loss = episode_loss(
            episode_support, episode_support_mask,
            episode_query, episode_query_mask, weights, args.score_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({
            "episode": episode_index,
            "loss": float(loss.detach().item()),
            **{"weight_block_{}".format(layer): float(value) for layer, value in zip(LAYER_NUMBERS, torch.softmax(logits.detach(), dim=0))},
        })
        if episode_index == 1 or episode_index % 25 == 0 or episode_index == args.train_episodes:
            values = torch.softmax(logits.detach(), dim=0).cpu().tolist()
            print("[train {}/{}] loss={:.4f} weights={}".format(
                episode_index, args.train_episodes, float(loss.detach().item()),
                {layer: round(value, 4) for layer, value in zip(LAYER_NUMBERS, values)},
            ), flush=True)

    learned_weights = torch.softmax(logits.detach(), dim=0)
    final_validation_loss = float(np.mean([
        cached_episode_loss(s_feat, s_mask, q_feat, q_mask, learned_weights, args.score_scale)
        for (s_feat, s_mask), (q_feat, q_mask) in validation_episodes
    ]))
    return learned_weights, {
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "train_episodes": args.train_episodes,
        "validation_episodes": len(validation_episodes),
        "support_image_only_training": True,
        "training_regions": ["top-left 512x512", "top-right 512x512"],
        "validation_regions": ["bottom-left 512x512", "bottom-right 512x512"],
        "episode_sampling": "alternate directions between the two disjoint training quadrants",
        "initial_equal_weight_validation_loss": initial_validation_loss,
        "learned_weight_validation_loss": final_validation_loss,
        "train_loss_last_20_mean": float(np.mean([row["loss"] for row in history[-20:]])),
        "history": history,
    }


def evaluate_methods(
    model: torch.nn.Module,
    support: ImagePair,
    queries: list[ImagePair],
    learned_weights: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark_pairs = [support, *queries]
    feature_cache = {
        pair.name: extract_layer_features(model, pair.image_path, device, args.crop_size)
        for pair in benchmark_pairs
    }
    support_mask = mask_fraction(support.mask_path, device)
    learned_weights = learned_weights.to(device)
    methods: dict[str, torch.Tensor] = {
        "block_6_only": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        "block_12_only": torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        "block_18_only": torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
        "block_24_only": torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
        "equal_6_12_18_24": torch.full((4,), 0.25, device=device),
        "learned_6_12_18_24": learned_weights,
    }
    per_image_rows = []
    method_rows = []
    for method_name, weights in methods.items():
        support_fused = fuse_layers(feature_cache[support.name], weights, device)
        prototypes = make_prototypes(support_fused, support_mask)
        predictions_dir = output_dir / "predictions" / method_name
        predictions_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for query in queries:
            query_features = fuse_layers(feature_cache[query.name], weights, device)
            patch_scores = score_map(query_features, prototypes)
            with Image.open(query.image_path) as image:
                output_size = (image.height, image.width)
            scores = F.interpolate(
                patch_scores[None, None], size=output_size, mode="bilinear", align_corners=False
            )[0, 0]
            prediction = scores >= 0.0
            target = read_binary_mask(query.mask_path)
            values = metric_row(target, prediction.cpu().numpy())
            row = {"method": method_name, "image": query.name, **values}
            per_image_rows.append(row)
            rows.append(values)
            Image.fromarray(prediction.to(torch.uint8).cpu().numpy() * 255, mode="L").save(
                predictions_dir / Path(query.name).with_suffix(".png").name
            )
        metrics = aggregate_metrics(rows)
        method_rows.append({
            "method": method_name,
            "weights": json.dumps([float(value) for value in weights.detach().cpu().tolist()]),
            **metrics,
        })
        print("[eval] {} micro_mIoU={:.4f} macro_mIoU={:.4f} Dice={:.4f}".format(
            method_name, metrics["micro_miou"], metrics["macro_miou"], metrics["micro_dice"]
        ), flush=True)
    return method_rows, per_image_rows


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_root = args.data_root.resolve()
    repo_path = args.dinov3_repo if args.dinov3_repo.is_absolute() else Path(__file__).resolve().parents[1] / args.dinov3_repo
    manifest = json.loads(args.test_manifest.resolve().read_text(encoding="utf-8"))
    support_name = Path(manifest["support_image"]).name
    query_names = [Path(name).name for name in manifest["query_images"]]
    if len(query_names) != 20 or len(set(query_names)) != 20 or support_name in query_names:
        raise ValueError("Benchmark manifest must contain one support and 20 unique, separate queries.")
    manifest_buffer = float(manifest.get("spatial_buffer_coordinate_units", args.spatial_buffer))
    if manifest_buffer != args.spatial_buffer:
        raise ValueError("--spatial-buffer must match the fixed benchmark manifest.")

    pairs = load_pairs(data_root, [support_name, *query_names])
    by_name = {pair.name: pair for pair in pairs}
    if support_name not in by_name or any(name not in by_name for name in query_names):
        raise FileNotFoundError("Fixed benchmark images are missing from the local dataset.")
    support = by_name[support_name]
    queries = [by_name[name] for name in query_names]
    if args.crop_size != 512:
        raise ValueError("Support-image-only layer training currently uses four 512x512 quadrants.")
    for query in queries:
        if spatial_distance(support, query) <= args.spatial_buffer:
            raise ValueError("Benchmark query lies inside support buffer: {}".format(query.name))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    model = make_model(repo_path.resolve(), args.weights.resolve(), device)
    print("GPU/device: {}".format(torch.cuda.get_device_name(0) if device.type == "cuda" else device), flush=True)

    support_features = extract_layer_features(model, support.image_path, device, args.crop_size).to(device)
    support_fractions = mask_fraction(support.mask_path, device)
    learned_weights, training_info = train_layer_weights(support_features, support_fractions, args)
    output_dir = args.output_dir or Path(__file__).resolve().parent / "runs" / (
        "layer_selection_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    method_rows, per_image_rows = evaluate_methods(
        model, support, queries, learned_weights, device, args, output_dir
    )
    learned_list = [float(value) for value in learned_weights.cpu().tolist()]
    training_info["learned_weights_by_block"] = {
        str(layer): value for layer, value in zip(LAYER_NUMBERS, learned_list)
    }
    summary = {
        "experiment": "DINOv3-SAT intermediate layer ablation and learnable scalar fusion",
        "backbone": "DINOv3 ViT-L/16 SAT-493M frozen",
        "blocks_1_indexed": list(LAYER_NUMBERS),
        "support_image": support_name,
        "query_images": query_names,
        "test_manifest_version": manifest["version"],
        "spatial_buffer_coordinate_units": args.spatial_buffer,
        "training_protocol": "weights trained only from the fixed support image and its mask; top-row 512x512 quadrants provide alternating support/query episodes, bottom-row quadrants provide an internal spatial validation; BCE plus soft Dice at 16-pixel patch grid; backbone frozen; four softmax scalar layer weights trained by backpropagation",
        "training_images": [support_name],
        "training_mask_files": [Path(support.mask_path).name],
        "query_masks_read_only_after_weight_training": True,
        "training_split": {
            "all_benchmark_query_images_excluded_from_weight_training": True,
            "support_image_quadrant_partition": "top row for weight training; bottom row for internal validation; non-overlapping 512x512 crops",
        },
        "validation_and_training": training_info,
        "score_scale_for_training_loss": args.score_scale,
        "prediction_threshold": 0.0,
        "device": str(device),
        "crop_size": args.crop_size,
        "metrics_by_method": method_rows,
    }
    with (output_dir / "per_image_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)
    with (output_dir / "metrics_by_method.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(method_rows[0].keys()))
        writer.writeheader()
        writer.writerows(method_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Summary: {}".format(output_dir / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
