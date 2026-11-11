"""Evaluate equal spatial/semantic CAM fusion on Experiment A benchmark v1."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CODE = REPO / "codex运行文件"
MANIFEST_PATH = CODE / "experiment_a" / "benchmark_v1.json"
EXPERIMENT_A_ROOT = Path("/kaggle/input/datasets/zh0416/pv-dinov3sat-experiment-a")
INPUT_ROOT = Path("/kaggle/input")
WEIGHT_ROOT = Path("/kaggle/input")
DINOV3_REPO = Path("/kaggle/working/dinov3")
OUTPUT_DIR = Path("/kaggle/working/experiment_a_equal_fusion")
CAM_THRESHOLD = 0.5
MASK_VALUES = {0, 12}


def find_unique(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {name!r} below {root}; found {len(matches)}: {matches[:5]}"
        )
    return matches[0]


def find_data_root(root: Path, expected_names: set[str]) -> Path:
    candidates = []
    for images_dir in root.rglob("images"):
        if not images_dir.is_dir():
            continue
        masks_dir = images_dir.parent / "mask"
        if not masks_dir.is_dir():
            continue
        image_names = {path.name for path in images_dir.glob("*.bmp")}
        mask_names = {path.name.removesuffix("_label.bmp") + ".bmp"
                      for path in masks_dir.glob("*_label.bmp")}
        if expected_names <= image_names and expected_names <= mask_names:
            candidates.append((len(image_names - expected_names), str(images_dir), images_dir.parent))
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate the Experiment A images/mask directories under {root}."
        )
    candidates.sort()
    return candidates[0][2]


def ensure_dinov3(repo: Path) -> None:
    commit = "6876159a11b4df116f30f667f8c9888617df0751"
    if not repo.is_dir():
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/dinov3.git", str(repo)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--detach", commit],
        check=True,
    )
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise ValueError(f"Unexpected DINOv3 source commit: {actual}")


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    if mask.shape != (1024, 1024) or not set(np.unique(mask)) <= MASK_VALUES:
        raise ValueError(f"Expected a 1024x1024 0/12 Experiment A label: {path}")
    return mask == 12


def upsample_cam(values: torch.Tensor) -> np.ndarray:
    return F.interpolate(
        values, size=(1024, 1024), mode="bilinear", align_corners=False
    )[0, 0].float().cpu().numpy()


def counts(target: np.ndarray, prediction: np.ndarray) -> dict[str, int | float]:
    truth = target.astype(bool, copy=False)
    pred = prediction.astype(bool, copy=False)
    tp = int(np.logical_and(truth, pred).sum())
    fp = int(np.logical_and(~truth, pred).sum())
    fn = int(np.logical_and(truth, ~pred).sum())
    tn = int(np.logical_and(~truth, ~pred).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": tp / max(1, tp + fp + fn),
        "dice": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def save_panel(
    image_path: Path,
    target: np.ndarray,
    spatial: np.ndarray,
    semantic: np.ndarray,
    fused: np.ndarray,
    prediction: np.ndarray,
    row: dict,
    output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    tp = target & prediction
    fn = target & ~prediction
    fp = ~target & prediction
    errors = np.zeros((*target.shape, 4), dtype=np.float32)
    errors[tp] = (0.1, 0.78, 0.28, 0.8)
    errors[fn] = (0.15, 0.4, 1.0, 0.9)
    errors[fp] = (1.0, 0.18, 0.14, 0.8)

    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Query image")
    axes[0, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("PV reference mask")
    axes[0, 2].imshow(spatial, cmap="turbo", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Spatial CAM | IoU {row['spatial_iou']:.3f}")
    axes[1, 0].imshow(semantic, cmap="turbo", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Semantic CAM | IoU {row['semantic_iou']:.3f}")
    axes[1, 1].imshow(fused, cmap="turbo", vmin=0, vmax=1)
    axes[1, 1].set_title(
        f"Equal fusion 0.5/0.5 | IoU {row['equal_fusion_iou']:.3f}"
    )
    axes[1, 2].imshow(rgb)
    axes[1, 2].imshow(errors)
    axes[1, 2].set_title("Fusion errors: TP green, FN blue, FP red")
    for axis in axes.flat:
        axis.set_axis_off()
    figure.suptitle(
        f"{image_path.name} | spatial IoU={row['spatial_iou']:.3f} | "
        f"equal-fusion IoU={row['equal_fusion_iou']:.3f}",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def save_overview(records: list[dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(5, 4, figsize=(16, 19), constrained_layout=True)
    for axis, record in zip(axes.flat, records):
        with Image.open(record["image_path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        axis.imshow(rgb)
        axis.imshow(record["fused_cam"], cmap="turbo", vmin=0, vmax=1, alpha=0.52)
        axis.set_title(
            f"{Path(record['query_image']).name}\n"
            f"equal-fusion IoU={record['equal_fusion_iou']:.3f}",
            fontsize=8,
        )
        axis.set_axis_off()
    for axis in axes.flat[len(records):]:
        axis.set_axis_off()
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator in the Kaggle Notebook.")
    device = torch.device("cuda")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    support_name = Path(manifest["support_image"]).name
    query_names = [Path(name).name for name in manifest["query_images"]]
    if len(query_names) != 20 or len(set(query_names)) != 20 or support_name in query_names:
        raise ValueError("Expected the fixed Experiment A benchmark v1 split.")
    expected = {support_name, *query_names}

    data_root_override = os.environ.get("DMSF_DATA_ROOT")
    search_root = Path(data_root_override) if data_root_override else (
        EXPERIMENT_A_ROOT if EXPERIMENT_A_ROOT.is_dir() else INPUT_ROOT
    )
    mounted_manifest_path = Path(os.environ.get(
        "DMSF_BENCHMARK_MANIFEST",
        str(search_root / "spatial_cam_code" / "benchmark_v1.json"),
    ))
    if mounted_manifest_path.is_file():
        mounted_manifest = json.loads(mounted_manifest_path.read_text(encoding="utf-8"))
        mounted_support = Path(mounted_manifest["support_image"]).name
        mounted_queries = [Path(name).name for name in mounted_manifest["query_images"]]
        if mounted_support != support_name or mounted_queries != query_names:
            raise ValueError(
                "Mounted Experiment A benchmark differs from the repository benchmark_v1 split."
            )
        manifest = mounted_manifest
    data_root = find_data_root(search_root, expected)
    images_dir, masks_dir = data_root / "images", data_root / "mask"
    for name in expected:
        if not (images_dir / name).is_file() or not (
            masks_dir / f"{Path(name).stem}_label.bmp"
        ).is_file():
            raise FileNotFoundError(f"Missing Experiment A image/mask pair: {name}")

    dino_weights = Path(os.environ["DMSF_DINO_WEIGHTS"]) if os.environ.get(
        "DMSF_DINO_WEIGHTS"
    ) else find_unique(WEIGHT_ROOT, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    geo_weights = Path(os.environ["DMSF_GEO_WEIGHTS"]) if os.environ.get(
        "DMSF_GEO_WEIGHTS"
    ) else find_unique(WEIGHT_ROOT, "RS5M_ViT-B-32.pt")
    dino_repo = Path(os.environ.get("DMSF_DINO_REPO", str(DINOV3_REPO)))
    output_dir = Path(os.environ.get("DMSF_OUTPUT_DIR", str(OUTPUT_DIR)))
    threshold = float(os.environ.get("DMSF_CAM_THRESHOLD", str(CAM_THRESHOLD)))
    if not 0 <= threshold <= 1:
        raise ValueError("DMSF_CAM_THRESHOLD must be within [0, 1].")

    sys.path.insert(0, str(CODE / "dmsf_v1"))
    sys.path.insert(0, str(CODE / "georsclip_semantic_cam_e1"))
    sys.path.insert(0, str(CODE / "dmsf_fusion_v1"))
    from feature_cache import (
        aligned_support_masks,
        encode_dino,
        encode_text,
        load_dino,
        load_text_encoder,
    )
    from infer import load_model

    ensure_dinov3(dino_repo)
    dino, dino_info = load_dino(dino_repo, dino_weights, device)
    text_encoder, geo_info = load_text_encoder(geo_weights, device)
    text = encode_text(text_encoder, device).to(device)
    del text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = load_model(device)
    if model.fusion_mode != "equal":
        raise RuntimeError("Pinned inference model did not select equal CAM fusion.")
    support_image = images_dir / support_name
    support_features = encode_dino(dino, support_image, device).unsqueeze(0).to(device)
    support_mask_data = aligned_support_masks(
        masks_dir / f"{Path(support_name).stem}_label.bmp"
    )
    support_mask = torch.from_numpy(
        support_mask_data["binary_patch_nearest"].astype(np.float32)
    )[None, None].to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_records = []
    rows = []
    for index, query_name in enumerate(query_names, start=1):
        query_image = images_dir / query_name
        print(f"[{index:02d}/{len(query_names)}] {query_name}", flush=True)
        query_features = encode_dino(dino, query_image, device).unsqueeze(0).to(device)
        with torch.inference_mode():
            maps = model(
                support_features,
                query_features,
                support_mask,
                text,
                include_logits=False,
            )
        spatial = upsample_cam(maps["spatial_cam"])
        semantic = upsample_cam(maps["semantic_cam"])
        fused = upsample_cam(maps["selected_cam"])
        target = read_mask(masks_dir / f"{Path(query_name).stem}_label.bmp")

        spatial_result = counts(target, spatial >= threshold)
        semantic_result = counts(target, semantic >= threshold)
        fusion_result = counts(target, fused >= threshold)
        row = {
            "query_image": query_name,
            "spatial_iou": spatial_result["iou"],
            "semantic_iou": semantic_result["iou"],
            "equal_fusion_iou": fusion_result["iou"],
            "spatial_dice": spatial_result["dice"],
            "semantic_dice": semantic_result["dice"],
            "equal_fusion_dice": fusion_result["dice"],
            "tp": fusion_result["tp"],
            "fp": fusion_result["fp"],
            "fn": fusion_result["fn"],
            "tn": fusion_result["tn"],
        }
        panel_path = output_dir / f"{Path(query_name).stem}_panel.png"
        save_panel(
            query_image, target, spatial, semantic, fused, fused >= threshold, row, panel_path
        )
        np.savez_compressed(
            output_dir / f"{Path(query_name).stem}_cams.npz",
            spatial_cam=spatial.astype(np.float32),
            semantic_cam=semantic.astype(np.float32),
            equal_fusion_cam=fused.astype(np.float32),
            equal_fusion_mask=(fused >= threshold).astype(np.uint8),
            query_mask=target.astype(np.uint8),
            threshold=np.asarray(threshold, dtype=np.float32),
        )
        row["panel"] = str(panel_path)
        rows.append(row)
        image_records.append({
            "image_path": str(query_image),
            "query_image": query_name,
            "fused_cam": fused,
            "equal_fusion_iou": row["equal_fusion_iou"],
        })
        del query_features, maps
        if device.type == "cuda":
            torch.cuda.empty_cache()

    columns = list(rows[0])
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def aggregate(prefix: str) -> dict:
        key = f"{prefix}_iou"
        dice_key = f"{prefix}_dice"
        if prefix == "equal_fusion":
            totals = {
                name: sum(int(row[name]) for row in rows)
                for name in ("tp", "fp", "fn", "tn")
            }
            tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
            return {
                "macro_iou": float(np.mean([row[key] for row in rows])),
                "macro_dice": float(np.mean([row[dice_key] for row in rows])),
                "micro_iou": tp / max(1, tp + fp + fn),
                "micro_dice": 2 * tp / max(1, 2 * tp + fp + fn),
                **totals,
            }
        return {
            "macro_iou": float(np.mean([row[key] for row in rows])),
            "macro_dice": float(np.mean([row[dice_key] for row in rows])),
        }

    summary = {
        "benchmark": "Experiment A benchmark v1",
        "support_image": support_name,
        "query_count": len(rows),
        "fusion": "equal: 0.5 * spatial_cam + 0.5 * semantic_cam",
        "cam_threshold": threshold,
        "query_masks_used_for_inference": False,
        "spatial_cam_iou": aggregate("spatial"),
        "semantic_cam_iou": aggregate("semantic"),
        "equal_fusion_iou": aggregate("equal_fusion"),
        "dino": dino_info,
        "georsclip_text": geo_info,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    overview_path = output_dir / "overview.png"
    save_overview(image_records, overview_path)
    print(f"Results: {output_dir}", flush=True)
    print(json.dumps({
        "equal_fusion": summary["equal_fusion_iou"],
        "overview": str(overview_path),
    }, ensure_ascii=False, indent=2), flush=True)
    try:
        from IPython.display import Image as NotebookImage, display
        display(NotebookImage(filename=str(overview_path)))
        display(NotebookImage(filename=rows[0]["panel"]))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
