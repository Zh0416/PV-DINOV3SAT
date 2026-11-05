from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_boundary(values: np.ndarray, axis: int, boundary_index: int) -> dict[str, float]:
    differences = np.abs(np.diff(values, axis=axis))
    seam = differences[:, boundary_index] if axis == 1 else differences[boundary_index, :]
    interior = np.delete(differences, boundary_index, axis=axis).reshape(-1)
    interior_mean = float(interior.mean())
    return {
        "seam_mean_absolute_difference": float(seam.mean()),
        "interior_mean_absolute_difference": interior_mean,
        "mean_difference_ratio": float(seam.mean() / max(interior_mean, 1e-12)),
        "seam_p95_absolute_difference": float(np.percentile(seam, 95)),
        "interior_p95_absolute_difference": float(np.percentile(interior, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure CAM feature jumps across the stitched center tile boundaries.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_v1_results",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    metadata = json.loads((results_dir / "metadata.json").read_text(encoding="utf-8"))
    records = []
    for query in metadata["query_results"]:
        score_path = results_dir / "images" / Path(query["query_image"]).stem / "scores.npz"
        with np.load(score_path) as data:
            raw = data["S_fg_patch"]
            normalized = data["A_fg_patch"]
        if raw.shape != (64, 64) or normalized.shape != (64, 64):
            raise ValueError("Expected 64x64 score maps for {}.".format(query["query_image"]))
        records.append({
            "query_image": query["query_image"],
            "raw_cosine": {
                "vertical_center_boundary": summarize_boundary(raw, axis=1, boundary_index=31),
                "horizontal_center_boundary": summarize_boundary(raw, axis=0, boundary_index=31),
            },
            "normalized_cam": {
                "vertical_center_boundary": summarize_boundary(normalized, axis=1, boundary_index=31),
                "horizontal_center_boundary": summarize_boundary(normalized, axis=0, boundary_index=31),
            },
        })

    diagnostics = {
        "description": "Mean and 95th percentile of adjacent patch response differences across the center x/y seams, compared with all within-tile adjacent patch differences. Ratios are a visual QA diagnostic, not an artifact classifier; local scene content also changes across a seam.",
        "center_seam_patch_indices": {"vertical": 31, "horizontal": 31},
        "query_count": len(records),
        "query_masks_used": False,
        "results": records,
        "aggregate_median_mean_difference_ratio": {
            "raw_cosine_vertical": float(np.median([
                item["raw_cosine"]["vertical_center_boundary"]["mean_difference_ratio"] for item in records
            ])),
            "raw_cosine_horizontal": float(np.median([
                item["raw_cosine"]["horizontal_center_boundary"]["mean_difference_ratio"] for item in records
            ])),
            "normalized_cam_vertical": float(np.median([
                item["normalized_cam"]["vertical_center_boundary"]["mean_difference_ratio"] for item in records
            ])),
            "normalized_cam_horizontal": float(np.median([
                item["normalized_cam"]["horizontal_center_boundary"]["mean_difference_ratio"] for item in records
            ])),
        },
    }
    destination = results_dir / "tile_seam_diagnostics.json"
    destination.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics["aggregate_median_mean_difference_ratio"], indent=2))
    print("Diagnostics: {}".format(destination))


if __name__ == "__main__":
    main()
