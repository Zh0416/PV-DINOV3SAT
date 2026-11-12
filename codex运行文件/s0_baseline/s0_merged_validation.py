"""Locked evaluation of the 185-image PV-512 merged validation set."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from s0_data import (canonical_sha256, checked_rgb, load_tokens, read_raster,
                     scene_key, sha256_file, write_json)


COHORT_COUNTS = {"PV-512-unused": 77, "PV-512TEST": 108}
CATEGORY_COUNTS = {"positive": 48, "hard_positive": 60}
METRICS = ("dice", "iou", "precision", "recall")


def find_merged_root(path: Path) -> Path:
    path = path.resolve()
    candidates = [path] if (path / "manifest.csv").is_file() else [
        p.parent for p in path.rglob("manifest.csv")
        if (p.parent / "images").is_dir() and (p.parent / "labels").is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one merged validation dataset under {path}; found {candidates}")
    return candidates[0]


def positive_mask(path: Path, expected_pixels: int) -> np.ndarray:
    mask = read_raster(path)
    if mask.shape != (512, 512) or mask.dtype != np.uint8 or not np.isin(mask, (0, 255)).all():
        raise ValueError(f"Expected a 512x512 uint8 0/255 label: {path}")
    binary = (mask == 255).astype(np.uint8)
    if int(binary.sum()) != expected_pixels or expected_pixels == 0:
        raise ValueError(f"Positive-pixel count differs from manifest: {path}")
    return binary


def _relative_file(root: Path, value: str, folder: str, sample_id: str) -> Path:
    relative = Path(value)
    expected = Path(folder) / f"{sample_id}.bmp"
    if relative.as_posix() != expected.as_posix():
        raise ValueError(f"Unexpected validation path: {value}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Missing validation file: {resolved}")
    return resolved


def load_merged_validation(root_arg: Path, selected: list[dict], selection_sha: str, output: Path,
                           expected_counts: dict[str, int] = COHORT_COUNTS,
                           expected_categories: dict[str, int] | None = CATEGORY_COUNTS,
                           reference_path: Path | None = None
                           ) -> tuple[list[dict], Path, str]:
    root = find_merged_root(root_arg)
    manifest = root / "manifest.csv"
    summary_path = root / "dataset_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("s0_selection_sha256") != selection_sha
            or summary.get("source_counts") != expected_counts
            or summary.get("sample_count") != sum(expected_counts.values())):
        raise ValueError("Merged dataset summary differs from the locked S0 selection or counts.")
    training_ids = {entry["id"] for entry in selected}
    training_scenes = {scene_key(entry["source_scene"]) for entry in selected}
    training_images = {entry["image_sha256"] for entry in selected}
    required = {"new_id", "image", "label", "source_dataset", "source_id",
                "source_category", "source_scene", "scene_in_s0_training",
                "foreground_pixels", "image_sha256", "label_sha256"}
    rows = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Merged validation manifest is missing required provenance columns.")
        for raw in reader:
            sample_id = raw["new_id"]
            cohort = raw["source_dataset"]
            if not sample_id.startswith("PV-") or cohort not in expected_counts:
                raise ValueError(f"Invalid merged validation sample: {sample_id}")
            image = _relative_file(root, raw["image"], "images", sample_id)
            label = _relative_file(root, raw["label"], "labels", sample_id)
            source_scene = scene_key(raw["source_scene"])
            same_scene = source_scene in training_scenes
            if (raw["scene_in_s0_training"].lower() != str(same_scene).lower()
                    or same_scene != (cohort == "PV-512-unused")):
                raise ValueError(f"Validation scene assignment disagrees with training: {sample_id}")
            if raw["source_id"] in training_ids or raw["image_sha256"] in training_images:
                raise ValueError(f"Training sample appears in validation: {sample_id}")
            if sha256_file(image) != raw["image_sha256"] or sha256_file(label) != raw["label_sha256"]:
                raise ValueError(f"Uploaded validation file differs from manifest: {sample_id}")
            checked_rgb(image)
            positive_mask(label, int(raw["foreground_pixels"]))
            rows.append({"id": sample_id, "image": raw["image"], "mask": raw["label"],
                         "image_sha256": raw["image_sha256"], "label_sha256": raw["label_sha256"],
                         "source_id": raw["source_id"], "source_dataset": cohort,
                         "source_category": raw["source_category"],
                         "source_scene": raw["source_scene"],
                         "foreground_pixels": int(raw["foreground_pixels"])})
    counts = {cohort: sum(row["source_dataset"] == cohort for row in rows)
              for cohort in expected_counts}
    categories = {category: sum(row["source_dataset"] == "PV-512TEST"
                                and row["source_category"] == category for row in rows)
                  for category in (expected_categories or {})}
    if (len({row["id"] for row in rows}) != len(rows) or counts != expected_counts
            or (expected_categories is not None and categories != expected_categories)
            or any(row["source_category"] not in ("positive", "hard_positive") for row in rows)):
        raise ValueError("Merged validation counts, categories, or IDs differ from the locked dataset.")
    lock = {"schema": "s0-merged-validation-v1", "selection_sha256": selection_sha,
            "manifest_sha256": sha256_file(manifest), "summary_sha256": sha256_file(summary_path),
            "rows": rows}
    lock["lock_sha256"] = canonical_sha256(lock)
    if reference_path is not None:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        for key in ("selection_sha256", "manifest_sha256", "summary_sha256", "lock_sha256"):
            if lock[key] != reference[key]:
                raise ValueError(f"Merged validation differs from committed reference: {key}")
    lock_path = output / "merged_validation_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise ValueError("Merged validation data changed since training started.")
    if not lock_path.exists():
        write_json(lock_path, lock)
    return rows, root, lock["lock_sha256"]


def _metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "dice": 2 * tp / max(1, 2 * tp + fp + fn),
            "iou": tp / max(1, tp + fp + fn),
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn)}


def summarize_merged(rows: list[dict]) -> dict:
    def group_summary(group: list[dict]) -> dict:
        by_scene = defaultdict(list)
        for item in group:
            by_scene[scene_key(item["source_scene"])].append(item)
        scenes = [{"source_scene": scene, "image_count": len(items),
                   **{metric: float(np.mean([item[metric] for item in items])) for metric in METRICS}}
                  for scene, items in sorted(by_scene.items())]
        totals = {name: sum(item[name] for item in group) for name in ("tp", "fp", "fn", "tn")}
        return {"image_count": len(group), "scene_count": len(scenes),
                "micro": _metrics(**totals),
                "macro_image": {metric: float(np.mean([item[metric] for item in group])) for metric in METRICS},
                "macro_scene": {metric: float(np.mean([scene[metric] for scene in scenes])) for metric in METRICS},
                "per_scene": scenes}

    independent = [row for row in rows if row["source_dataset"] == "PV-512TEST"]
    same_scene = [row for row in rows if row["source_dataset"] == "PV-512-unused"]
    if not independent or not same_scene:
        raise ValueError("Both merged validation cohorts are required.")
    categories = {category: group_summary([row for row in independent if row["source_category"] == category])
                  for category in ("positive", "hard_positive")}
    independent_summary = group_summary(independent)
    return {"all_validation": group_summary(rows),
            "new_scene_validation": independent_summary,
            "same_scene_validation": group_summary(same_scene),
            "new_scene_categories": categories,
            "selection_metric": "new_scene_validation.macro_scene.dice",
            "selection_score": independent_summary["macro_scene"]["dice"]}


@torch.inference_mode()
def evaluate_merged(model, support: torch.Tensor, support_mask: torch.Tensor,
                    rows: list[dict], root: Path, cache_dir: Path,
                    index: dict, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    results = []
    for entry in rows:
        query = load_tokens(cache_dir, index, entry["id"], device)
        target = torch.from_numpy(positive_mask(
            root / entry["mask"], entry["foreground_pixels"]))[None, None].to(device).bool()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(support, query, support_mask)["logits"]
        if logits.shape != target.shape or not torch.isfinite(logits).all():
            raise RuntimeError(f"Invalid validation logits: {entry['id']}")
        prediction = logits >= 0
        tp = int((prediction & target).sum().item())
        fp = int((prediction & ~target).sum().item())
        fn = int((~prediction & target).sum().item())
        tn = int(target.numel() - tp - fp - fn)
        results.append({"id": entry["id"], "source_id": entry["source_id"],
                        "source_dataset": entry["source_dataset"],
                        "source_category": entry["source_category"],
                        "source_scene": entry["source_scene"],
                        **_metrics(tp, fp, fn, tn)})
    return summarize_merged(results), results
