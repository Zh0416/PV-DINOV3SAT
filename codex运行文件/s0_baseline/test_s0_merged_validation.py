"""Focused checks for the merged validation contract and scoring."""
from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from s0_data import sha256_file
from s0_merged_validation import (evaluate_merged, load_merged_validation,
                                  positive_mask, summarize_merged)
from train_s0 import train
from test_s0 import TinySegmenter


def fixture(root: Path) -> tuple[list[dict], Path]:
    records = [
        ("PV-000001", "PV-512-unused", "original_1", "positive", "train_scene", True),
        ("PV-000002", "PV-512TEST", "PV-001", "positive", "new_a", False),
        ("PV-000003", "PV-512TEST", "PV-002", "hard_positive", "new_b", False),
    ]
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    rows = []
    for number, (sample_id, cohort, source_id, category, scene, overlap) in enumerate(records, 1):
        image = root / "images" / f"{sample_id}.bmp"
        label = root / "labels" / f"{sample_id}.bmp"
        Image.fromarray(np.full((512, 512, 3), number, dtype=np.uint8), "RGB").save(image)
        pixels = np.zeros((512, 512), dtype=np.uint8)
        pixels[number:number + 4, 2:6] = 255
        Image.fromarray(pixels, "L").save(label)
        rows.append({"new_id": sample_id, "image": f"images/{sample_id}.bmp",
                     "label": f"labels/{sample_id}.bmp", "source_dataset": cohort,
                     "source_id": source_id, "source_category": category,
                     "source_scene": scene, "scene_in_s0_training": str(overlap).lower(),
                     "foreground_pixels": "16", "image_sha256": sha256_file(image),
                     "label_sha256": sha256_file(label)})
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "dataset_summary.json").write_text(json.dumps({
        "s0_selection_sha256": "locked", "source_counts": {"PV-512-unused": 1, "PV-512TEST": 2},
        "sample_count": 3}), encoding="utf-8")
    selected = [{"id": "support", "source_scene": "train_scene", "image_sha256": "other"}]
    return selected, manifest


class MergedValidationTests(unittest.TestCase):
    def test_label_conversion_and_locked_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "dataset"
            selected, manifest = fixture(root)
            output = Path(folder) / "output"
            rows, found_root, digest = load_merged_validation(
                root, selected, "locked", output,
                expected_counts={"PV-512-unused": 1, "PV-512TEST": 2},
                expected_categories={"positive": 1, "hard_positive": 1})
            self.assertEqual(found_root, root)
            self.assertEqual(len(rows), 3)
            self.assertEqual(digest, json.loads((output / "merged_validation_lock.json").read_text())["lock_sha256"])
            self.assertEqual(int(positive_mask(root / rows[0]["mask"], 16).sum()), 16)
            self.assertEqual(load_merged_validation(
                root, selected, "locked", output,
                expected_counts={"PV-512-unused": 1, "PV-512TEST": 2},
                expected_categories={"positive": 1, "hard_positive": 1})[2], digest)
            with (root / rows[0]["mask"]).open("wb") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "differs from manifest"):
                load_merged_validation(root, selected, "locked", output,
                    expected_counts={"PV-512-unused": 1, "PV-512TEST": 2},
                    expected_categories={"positive": 1, "hard_positive": 1})
            self.assertTrue(manifest.is_file())

    def test_scene_overlap_rejected_for_new_scene_cohort(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "dataset"
            selected, _ = fixture(root)
            selected[0]["source_scene"] = "new_a"
            with self.assertRaisesRegex(ValueError, "scene assignment"):
                load_merged_validation(root, selected, "locked", Path(folder) / "output",
                    expected_counts={"PV-512-unused": 1, "PV-512TEST": 2},
                    expected_categories={"positive": 1, "hard_positive": 1})

    def test_selection_uses_new_scene_macro_dice_not_same_scene_scores(self):
        base = {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "iou": 1.,
                "precision": 1., "recall": 1.}
        rows = [
            {**base, "id": "old", "source_dataset": "PV-512-unused",
             "source_category": "positive", "source_scene": "train", "dice": 0.0},
            {**base, "id": "new_1", "source_dataset": "PV-512TEST",
             "source_category": "positive", "source_scene": "a", "dice": 1.0},
            {**base, "id": "new_2", "source_dataset": "PV-512TEST",
             "source_category": "positive", "source_scene": "a", "dice": 1.0},
            {**base, "id": "new_3", "source_dataset": "PV-512TEST",
             "source_category": "hard_positive", "source_scene": "b", "dice": 0.0},
        ]
        summary = summarize_merged(rows)
        self.assertEqual(summary["all_validation"]["image_count"], 4)
        self.assertEqual(summary["new_scene_validation"]["image_count"], 3)
        self.assertEqual(summary["selection_score"], 0.5)
        self.assertEqual(summary["same_scene_validation"]["macro_image"]["dice"], 0.0)

    def test_evaluation_is_forward_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "dataset"
            selected, _ = fixture(root)
            rows, _, _ = load_merged_validation(root, selected, "locked", Path(folder) / "out",
                expected_counts={"PV-512-unused": 1, "PV-512TEST": 2},
                expected_categories={"positive": 1, "hard_positive": 1})
            class ConstantModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.bias = torch.nn.Parameter(torch.tensor(-1.0))

                def forward(self, support, query, mask):
                    return {"logits": self.bias.expand(1, 1, 512, 512)}

            model = ConstantModel()
            with patch("s0_merged_validation.load_tokens", return_value=torch.zeros(1, 1, 32, 32)):
                summary, per_image = evaluate_merged(model, torch.zeros(1, 1, 32, 32),
                    torch.ones(1, 1, 512, 512), rows, root, root, {}, torch.device("cpu"))
            self.assertEqual(len(per_image), 3)
            self.assertEqual(summary["new_scene_validation"]["scene_count"], 2)
            self.assertIsNone(model.bias.grad)

    def test_joint_training_evaluates_merged_set_and_selects_epoch_four(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8), "RGB").save(root / "image.png")
            positive = np.zeros((512, 512), dtype=np.uint8)
            positive[10:30, 20:40] = 1
            Image.fromarray(positive, "L").save(root / "positive.png")
            Image.fromarray(np.zeros_like(positive), "L").save(root / "empty.png")
            selected = [{"id": "support", "category": "正样本", "mask": "positive.png"},
                        {"id": "positive_query", "category": "正样本", "mask": "positive.png"},
                        {"id": "empty_query", "category": "困难负样本", "mask": "empty.png"}]
            for item in selected:
                item.update(image="image.png", source_scene="training")
            validation = [{"id": f"v{i}", "image": "image.png", "mask": "positive.png"}
                          for i in range(185)]
            args = argparse.Namespace(mode="train", smoke_steps=2, allow_cpu=True,
                source_scene_column="source_scene", epochs=4, output=root / "out",
                input_root=root, data_root=root, dino_weights=None, dino_repo=None,
                validation_manifest=None, validation_root=None, merged_validation_root=root)
            summary = {"selection_score": 0.75, "new_scene_validation": {"image_count": 108},
                       "same_scene_validation": {"image_count": 77}}
            with patch("train_s0.find_dataset_root", return_value=root), \
                 patch("train_s0.find_weights", return_value=root / "unused.pth"), \
                 patch("train_s0.inventory", return_value={}), \
                 patch("train_s0.lock_selection", return_value={"selected": selected, "selection_sha256": "locked"}), \
                 patch("train_s0.load_merged_validation", return_value=(validation, root, "validation")), \
                 patch("train_s0.cache_tokens", return_value={"weight_sha256": "weights"}), \
                 patch("train_s0.load_tokens", return_value=torch.ones(1, 1, 32, 32)), \
                 patch("train_s0.evaluate_merged", return_value=(summary, [{"id": "v"}] * 185)) as evaluate, \
                 patch("train_s0.S0Segmenter", TinySegmenter):
                train(args)
            self.assertEqual(evaluate.call_count, 1)
            checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 4)
            report = json.loads((args.output / "validation_epoch_04.json").read_text())
            self.assertEqual(report["validation_mode"], "merged_185")
            self.assertEqual(len(report["per_image"]), 185)
            self.assertEqual(report["summary"]["selection_score"], 0.75)


if __name__ == "__main__":
    unittest.main()
