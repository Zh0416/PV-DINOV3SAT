from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


SUPPORT_NAME = "PV08_331824_1195530.bmp"
QUERY_MASK = "*_label.bmp"
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)
BLOCK_INDEX = 23
PATCH_SIZE = 16
CROP_SIZE = 512
EPSILON = 1e-8
EXPECTED_MASK_VALUES = {0, 12}


def parse_args() -> argparse.Namespace:
    project_files_dir = Path(__file__).resolve().parents[1]
    experiment_dir = project_files_dir / "experiment_a"
    parser = argparse.ArgumentParser(description="Generate support-guided DINOv3-SAT photovoltaic spatial CAMs.")
    parser.add_argument("--data-root", type=Path, default=experiment_dir / "benchmark_v1_data")
    parser.add_argument("--manifest", type=Path, default=experiment_dir / "benchmark_v1.json")
    parser.add_argument("--dinov3-repo", type=Path, default=project_files_dir / "dinov3")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "benchmark_v1_results")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    return parser.parse_args()


def sha256_prefix(path: Path, length: int = 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:length]


def manifest_names(manifest_path: Path) -> tuple[dict[str, Any], str, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    support_name = Path(manifest["support_image"]).name
    query_names = [Path(name).name for name in manifest["query_images"]]
    if support_name != SUPPORT_NAME:
        raise ValueError("benchmark_v1 support image changed: {}".format(support_name))
    if len(query_names) != 20 or len(set(query_names)) != 20 or support_name in query_names:
        raise ValueError("benchmark_v1 must contain one support image and 20 unique query images.")
    return manifest, support_name, query_names


def validate_benchmark_files(data_root: Path, support_name: str, query_names: list[str]) -> tuple[Path, Path, Path]:
    images_dir = data_root / "images"
    masks_dir = data_root / "mask"
    expected = {support_name, *query_names}
    image_files = {path.name for path in images_dir.glob("*.bmp")}
    mask_files = {path.name.removesuffix("_label.bmp") + ".bmp" for path in masks_dir.glob(QUERY_MASK)}
    if image_files != expected or mask_files != expected:
        raise ValueError("benchmark_v1_data must contain exactly the 21 manifest image/mask pairs.")
    support_mask = masks_dir / (Path(support_name).stem + "_label.bmp")
    if not (images_dir / support_name).is_file() or not support_mask.is_file():
        raise FileNotFoundError("The support image or support mask is missing.")
    return images_dir, masks_dir, support_mask


def load_model(repo_path: Path, weights_path: Path, device: torch.device) -> torch.nn.Module:
    if not repo_path.is_dir():
        raise FileNotFoundError("DINOv3 repository does not exist: {}".format(repo_path))
    if not weights_path.is_file():
        raise FileNotFoundError("SAT-493M checkpoint does not exist: {}".format(weights_path))
    if weights_path.name != "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth":
        raise ValueError("Expected the official SAT ViT-L/16 checkpoint; got {}".format(weights_path.name))
    sys.path.insert(0, str(repo_path.resolve()))
    from dinov3.hub.backbones import dinov3_vitl16

    model = dinov3_vitl16(pretrained=True, weights=str(weights_path.resolve()))
    model.eval().requires_grad_(False)
    model.to(device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("The DINOv3-SAT backbone must remain frozen and in eval mode.")
    return model


def normalized_image(path: Path, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.shape[:2] != (1024, 1024):
        raise ValueError("Expected 1024x1024 image {}, got {}.".format(path.name, rgb.shape[:2]))
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device=device, dtype=torch.float32).div_(255.0)
    mean = torch.tensor(SAT_MEAN, device=device, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(SAT_STD, device=device, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std, rgb


def stitch_tiles(tiles: list[torch.Tensor]) -> torch.Tensor:
    if len(tiles) != 4 or any(tuple(tile.shape[-2:]) != (32, 32) for tile in tiles):
        raise ValueError("Expected four position-ordered 32x32 feature tiles.")
    top = torch.cat((tiles[0], tiles[1]), dim=2)
    bottom = torch.cat((tiles[2], tiles[3]), dim=2)
    return torch.cat((top, bottom), dim=1)


@torch.inference_mode()
def extract_block24_features(model: torch.nn.Module, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    height, width = image.shape[-2:]
    if (height, width) != (1024, 1024):
        raise ValueError("Expected 1024x1024 image tensor, got {}x{}.".format(height, width))
    tiles = []
    autocast_enabled = device.type == "cuda"
    for top in (0, CROP_SIZE):
        for left in (0, CROP_SIZE):
            crop = image[:, top : top + CROP_SIZE, left : left + CROP_SIZE].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                outputs = model.get_intermediate_layers(crop, n=(BLOCK_INDEX,), reshape=True, norm=True)
            if len(outputs) != 1:
                raise RuntimeError("Expected exactly one transformer block output (block index 23).")
            feature_map = outputs[0].float()
            if tuple(feature_map.shape[-2:]) != (32, 32):
                raise RuntimeError("A 512x512 crop must produce a 32x32 patch feature grid.")
            tiles.append(F.normalize(feature_map, dim=1)[0].to(device="cpu", dtype=torch.float32))
    complete = stitch_tiles(tiles)
    if tuple(complete.shape[-2:]) != (64, 64):
        raise RuntimeError("Stitched block-24 features must form a 64x64 patch grid.")
    return complete


def support_fractions(mask_path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    with Image.open(mask_path) as image:
        raw = np.asarray(image.convert("L"), dtype=np.uint8)
    values, counts = np.unique(raw, return_counts=True)
    observed = set(int(value) for value in values)
    if not observed.issubset(EXPECTED_MASK_VALUES):
        raise ValueError("Unexpected support mask pixel values: {}".format(sorted(observed)))
    if raw.shape != (1024, 1024):
        raise ValueError("Support mask must be 1024x1024, got {}.".format(raw.shape))
    binary = torch.from_numpy((raw == 12).astype(np.float32))[None, None]
    fractions = F.avg_pool2d(binary, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)[0, 0]
    if fractions.sum().item() <= 0:
        raise ValueError("Support mask contains no photovoltaic foreground pixels; cannot build p_fg.")
    return fractions, {
        "observed_values": {str(int(value)): int(count) for value, count in zip(values, counts)},
        "pv_pixel_count": int((raw == 12).sum()),
        "background_pixel_count": int((raw == 0).sum()),
        "pv_pixel_fraction": float((raw == 12).mean()),
    }


def foreground_prototype(features: torch.Tensor, fractions: torch.Tensor) -> torch.Tensor:
    if tuple(features.shape[-2:]) != tuple(fractions.shape) or tuple(fractions.shape) != (64, 64):
        raise ValueError("Support feature and 16x16-pooled mask grids must both be 64x64.")
    flat = features.permute(1, 2, 0).reshape(-1, features.shape[0])
    flat = F.normalize(flat, dim=1)
    weights = fractions.reshape(-1).to(dtype=torch.float32)
    if not torch.isfinite(weights).all() or weights.sum().item() <= 0:
        raise ValueError("Support mask has no valid photovoltaic area for the foreground prototype.")
    prototype = F.normalize((flat * weights[:, None]).sum(dim=0), dim=0)
    if not torch.isfinite(prototype).all() or prototype.norm().item() <= 0:
        raise ValueError("Foreground prototype is invalid or has zero norm.")
    return prototype


def spatial_cam(features: torch.Tensor, prototype: torch.Tensor) -> dict[str, np.ndarray]:
    if tuple(features.shape[-2:]) != (64, 64):
        raise ValueError("Query features must be a 64x64 patch grid.")
    tokens = F.normalize(features, dim=0)
    score_patch = (tokens * prototype[:, None, None]).sum(dim=0).float()
    score_patch = torch.clamp(score_patch, -1.0, 1.0)
    score_full = F.interpolate(
        score_patch[None, None], size=(1024, 1024), mode="bilinear", align_corners=False
    )[0, 0]
    score_min = score_patch.min()
    score_max = score_patch.max()
    span = score_max - score_min
    if span.item() <= EPSILON:
        cam_patch = torch.zeros_like(score_patch)
    else:
        cam_patch = (score_patch - score_min) / (span + EPSILON)
    cam_full = F.interpolate(
        cam_patch[None, None], size=(1024, 1024), mode="bilinear", align_corners=False
    )[0, 0].clamp_(0.0, 1.0)
    arrays = {
        "S_fg_patch": score_patch.cpu().numpy().astype(np.float32, copy=False),
        "S_fg_upsampled": score_full.cpu().numpy().astype(np.float32, copy=False),
        "A_fg_patch": cam_patch.cpu().numpy().astype(np.float32, copy=False),
        "A_fg_upsampled": cam_full.cpu().numpy().astype(np.float32, copy=False),
    }
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise FloatingPointError("Non-finite value found in spatial CAM output.")
    if arrays["A_fg_upsampled"].min() < 0 or arrays["A_fg_upsampled"].max() > 1:
        raise RuntimeError("Normalized CAM must stay inside [0, 1].")
    return arrays


def colorize(array: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    low, high = value_range
    normalized = np.clip((array.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps

    rgb = colormaps["turbo"](normalized)[..., :3]
    return np.round(rgb * 255.0).astype(np.uint8)


def save_image_outputs(
    output_dir: Path,
    rgb: np.ndarray,
    arrays: dict[str, np.ndarray],
    query_name: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    normalized = arrays["A_fg_upsampled"]
    raw = arrays["S_fg_upsampled"]
    heat_cam = colorize(normalized, (0.0, 1.0))
    heat_raw = colorize(raw, (-1.0, 1.0))
    overlay = np.round(rgb.astype(np.float32) * 0.55 + heat_cam.astype(np.float32) * 0.45).astype(np.uint8)
    Image.fromarray(np.round(normalized * 255.0).astype(np.uint8), mode="L").save(output_dir / "cam_gray.png")
    Image.fromarray(heat_cam, mode="RGB").save(output_dir / "cam_color.png")
    Image.fromarray(heat_raw, mode="RGB").save(output_dir / "cosine_color.png")
    Image.fromarray(overlay, mode="RGB").save(output_dir / "overlay.png")
    np.savez_compressed(
        output_dir / "scores.npz",
        **{name: array.astype(np.float32, copy=False) for name, array in arrays.items()},
    )

    label_height = 44
    panel = Image.new("RGB", (1024 * 3, 1024 + label_height), color=(250, 250, 250))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    panel.paste(Image.fromarray(rgb, mode="RGB"), (0, label_height))
    panel.paste(Image.fromarray(heat_raw, mode="RGB"), (1024, label_height))
    panel.paste(Image.fromarray(overlay, mode="RGB"), (2048, label_height))
    draw.text((16, 14), "RGB | " + query_name, fill=(20, 20, 20), font=font)
    draw.text((1040, 14), "S_fg cosine [-1, 1]", fill=(20, 20, 20), font=font)
    draw.text((2064, 14), "A_fg min-max [0, 1] overlay", fill=(20, 20, 20), font=font)
    panel_path = output_dir / "panel.jpg"
    panel.save(panel_path, quality=92, subsampling=0)
    return {
        "directory": "images/" + output_dir.name,
        "scores": "scores.npz",
        "cam_gray": "cam_gray.png",
        "cam_color": "cam_color.png",
        "cosine_color": "cosine_color.png",
        "overlay": "overlay.png",
        "panel": "panel.jpg",
    }


def make_overview(output_dir: Path, records: list[dict[str, Any]]) -> None:
    cell_width, cell_height = 620, 240
    columns = 4
    rows = (len(records) + columns - 1) // columns
    overview = Image.new("RGB", (columns * cell_width, rows * cell_height), color=(242, 242, 242))
    draw = ImageDraw.Draw(overview)
    font = ImageFont.load_default()
    for index, record in enumerate(records):
        panel_path = output_dir / record["files"]["directory"] / "panel.jpg"
        with Image.open(panel_path) as image:
            image.thumbnail((cell_width, cell_height - 24), Image.Resampling.LANCZOS)
            x = (index % columns) * cell_width
            y = (index // columns) * cell_height + 24
            overview.paste(image, (x, y))
        draw.text(
            ((index % columns) * cell_width + 8, (index // columns) * cell_height + 5),
            "{:02d} {}".format(index + 1, record["query_image"]),
            fill=(25, 25, 25),
            font=font,
        )
    overview.save(output_dir / "overview.jpg", quality=92, subsampling=0)


def write_html_index(output_dir: Path, records: list[dict[str, Any]]) -> None:
    sections = []
    for record in records:
        directory = html.escape(record["files"]["directory"])
        query = html.escape(record["query_image"])
        minimum = record["raw_score_range"]["min"]
        maximum = record["raw_score_range"]["max"]
        sections.append(
            '<section><h2>{}</h2><p>Raw cosine range: {:.5f} to {:.5f}</p>'
            '<a href="{}/panel.jpg"><img src="{}/panel.jpg" alt="{}" loading="lazy"></a>'
            '<p><a href="{}/scores.npz">float32 scores</a> | '
            '<a href="{}/cam_gray.png">grayscale CAM</a> | '
            '<a href="{}/overlay.png">overlay</a></p></section>'.format(
                query, minimum, maximum, directory, directory, query,
                directory, directory, directory,
            )
        )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DINOv3-SAT PV Spatial CAM v1</title><style>
body{{font:15px sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}}section{{border-top:1px solid #bbb;padding:16px 0}}img{{max-width:100%;height:auto}}h2{{font-size:18px}}
</style></head><body><h1>DINOv3-SAT PV Spatial CAM v1</h1>
<p>Per-image min-max CAM is a relative response, not a PV probability. Raw cosine maps use the shared [-1,1] display range.</p>
{} </body></html>""".format("\n".join(sections))
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    repo_path = args.dinov3_repo.resolve()
    weights_path = args.weights.resolve()
    manifest, support_name, query_names = manifest_names(manifest_path)
    images_dir, _, support_mask_path = validate_benchmark_files(data_root, support_name, query_names)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    model = load_model(repo_path, weights_path, device)
    print("Device: {}".format(torch.cuda.get_device_name(0) if device.type == "cuda" else device), flush=True)

    support_tensor, _ = normalized_image(images_dir / support_name, device)
    support_features = extract_block24_features(model, support_tensor, device)
    fractions, support_mask_info = support_fractions(support_mask_path)
    prototype = foreground_prototype(support_features, fractions)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite existing CAM run: {}".format(output_dir))
    output_dir.mkdir(parents=True)
    records = []

    for index, query_name in enumerate(query_names, start=1):
        query_tensor, rgb = normalized_image(images_dir / query_name, device)
        query_features = extract_block24_features(model, query_tensor, device)
        arrays = spatial_cam(query_features, prototype)
        stem = Path(query_name).stem
        files = save_image_outputs(output_dir / "images" / stem, rgb, arrays, query_name)
        score_patch = arrays["S_fg_patch"]
        cam_patch = arrays["A_fg_patch"]
        record = {
            "query_image": query_name,
            "raw_score_range": {
                "min": float(score_patch.min()),
                "max": float(score_patch.max()),
                "mean": float(score_patch.mean()),
            },
            "normalized_cam_range": {
                "min": float(cam_patch.min()),
                "max": float(cam_patch.max()),
            },
            "patch_grid": list(score_patch.shape),
            "full_resolution": list(arrays["S_fg_upsampled"].shape),
            "files": files,
        }
        records.append(record)
        print("[CAM {}/20] {} S_fg=[{:.5f}, {:.5f}]".format(
            index, query_name, record["raw_score_range"]["min"], record["raw_score_range"]["max"]
        ), flush=True)

    make_overview(output_dir, records)
    write_html_index(output_dir, records)
    checkpoint_prefix = sha256_prefix(weights_path)
    metadata = {
        "experiment": "DINOv3-SAT support-guided photovoltaic spatial CAM",
        "method_scope": "Frozen pretrained feature prototype matching; not a full SSPNet reproduction",
        "backbone": "DINOv3 ViT-L/16 SAT-493M",
        "checkpoint": weights_path.name,
        "checkpoint_sha256_prefix": checkpoint_prefix,
        "backbone_frozen": True,
        "backbone_eval_mode": True,
        "transformer_block_1_indexed": 24,
        "transformer_block_zero_indexed": 23,
        "requested_intermediate_blocks": [23],
        "support_image": support_name,
        "query_images": query_names,
        "benchmark_manifest": str(manifest_path),
        "benchmark_version": manifest.get("version"),
        "query_masks_used": False,
        "training_or_parameter_updates": False,
        "normalization": {
            "color_order": "RGB",
            "input_scale": "uint8 / 255",
            "mean": list(SAT_MEAN),
            "std": list(SAT_STD),
            "token_normalization": "L2 per patch token after DINOv3 final LayerNorm output",
        },
        "tiling": {
            "image_size": [1024, 1024],
            "crop_size": CROP_SIZE,
            "crop_count": 4,
            "tile_order": ["top-left", "top-right", "bottom-left", "bottom-right"],
            "patch_size": PATCH_SIZE,
            "patch_grid_per_crop": [32, 32],
            "stitched_patch_grid": [64, 64],
            "description": "Four non-overlapping 512x512 image crops are independently encoded and stitched into their original spatial positions; this is not one global 1024x1024 encoder pass.",
        },
        "support_mask": {
            "path": str(support_mask_path),
            "pixel_values": {"background": 0, "photovoltaic": 12},
            "observed": support_mask_info,
            "patch_fraction": "16x16 average pooling of the binary PV mask",
        },
        "foreground_prototype": "L2-normalize support patch tokens; compute their PV-fraction-weighted sum; L2-normalize the sum",
        "score_formula": "S_fg(u,v) = cosine(F_query(u,v), p_fg)",
        "cam_formula": "A_fg = (S_fg - min(S_fg)) / (max(S_fg) - min(S_fg) + 1e-8); if score range <= 1e-8, A_fg is all zero; bilinear upsampling to 1024x1024 with align_corners=False",
        "display_ranges": {"raw_cosine": [-1.0, 1.0], "normalized_cam": [0.0, 1.0]},
        "relative_response_notice": "Per-image min-max CAM is relative response, not PV probability; a bright area alone does not establish that the query contains PV.",
        "device": str(device),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_results": records,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps({
        "experiment": metadata["experiment"],
        "support_image": support_name,
        "query_count": len(records),
        "query_masks_used": False,
        "checkpoint_sha256_prefix": checkpoint_prefix,
        "transformer_block_1_indexed": 24,
        "device": str(device),
        "overview": "overview.jpg",
        "index": "index.html",
        "query_raw_score_ranges": [
            {"query_image": row["query_image"], **row["raw_score_range"]} for row in records
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Outputs: {}".format(output_dir), flush=True)


if __name__ == "__main__":
    main()
