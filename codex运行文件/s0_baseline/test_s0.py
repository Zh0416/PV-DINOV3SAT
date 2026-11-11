"""CPU checks for the S0 model, data lock, metrics, resume, and Kaggle bundle."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import argparse
import ast
import base64
import io
import zipfile
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

from build_kaggle_bundle import build, SOURCES
from s0_data import SUPPORT_ID, canonical_sha256, inventory, load_tokens, lock_selection
from s0_model import S0Segmenter, s0_loss
from train_s0 import (image_counts, load_training_state, metrics, memory_precheck, train,
                      configure_phase, load_validation, save_training_state, set_seed,
                      summarize, validation_key)


torch.set_num_threads(2)


class S0Tests(unittest.TestCase):
    def test_model_shape_and_live_projection_gradient(self):
        torch.manual_seed(1)
        model = S0Segmenter()
        support = torch.randn(1, 1024, 32, 32)
        query = torch.randn(1, 1024, 32, 32)
        mask = torch.zeros(1, 1, 512, 512)
        mask[:, :, 32:160, 48:192] = 1
        with torch.no_grad():
            output = model(support, query, mask)
        self.assertEqual(tuple(output["logits"].shape), (1, 1, 512, 512))
        self.assertEqual(tuple(output["cam"].shape), (1, 1, 32, 32))
        self.assertTrue(torch.isfinite(output["logits"]).all())
        model.zero_grad(set_to_none=True)
        cam = model(support, query, mask)["cam"]
        cam.square().mean().backward()
        self.assertGreater(float(model.visual_projection[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(support.grad)
        self.assertIsNone(query.grad)

    def test_positive_and_empty_query_loss(self):
        prediction = {"logits": torch.zeros(1, 1, 512, 512, requires_grad=True),
                      "cam": torch.zeros(1, 1, 32, 32, requires_grad=True)}
        positive = torch.zeros(1, 1, 512, 512)
        positive[:, :, 20:40, 30:50] = 1
        loss, _ = s0_loss(prediction, positive, False)
        loss.backward()
        self.assertGreater(float(prediction["cam"].grad.abs().sum()), 0)
        prediction["cam"].grad = None
        empty_loss, _ = s0_loss(prediction, torch.zeros_like(positive), True)
        empty_loss.backward()
        self.assertGreater(float(prediction["cam"].grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            s0_loss(prediction, positive, True)

    def test_full_decoder_training_gradient(self):
        torch.manual_seed(3)
        model = S0Segmenter()
        support = torch.randn(1, 1024, 32, 32)
        query = torch.randn(1, 1024, 32, 32)
        support_mask = torch.zeros(1, 1, 512, 512)
        support_mask[:, :, 16:144, 16:144] = 1
        target = torch.zeros_like(support_mask)
        target[:, :, 200:240, 300:340] = 1
        output = model(support, query, support_mask)
        loss, _ = s0_loss(output, target, False)
        loss.backward()
        self.assertGreater(float(model.decoder.output.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.visual_projection[0].weight.grad.abs().sum()), 0)

    def test_warmup_skips_decoder_and_trains_projection_for_both_query_types(self):
        model = S0Segmenter()
        self.assertEqual(configure_phase(model, 3), "warmup")
        features = torch.randn(1, 1024, 32, 32)
        mask = torch.zeros(1, 1, 512, 512)
        mask[:, :, 20:150, 30:180] = 1
        with patch.object(model.decoder, "forward", side_effect=AssertionError("Decoder must be skipped")):
            for hard_negative in (False, True):
                model.zero_grad(set_to_none=True)
                result = model(features, features.roll(1, -1), mask, include_logits=False)
                target = torch.zeros_like(mask) if hard_negative else mask
                loss, parts = s0_loss(result, target, hard_negative, phase="warmup")
                loss.backward()
                self.assertGreater(float(model.visual_projection[0].weight.grad.abs().sum()), 0)
                self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
                self.assertEqual(parts["dice"], 0)
        self.assertEqual(configure_phase(model, 4), "joint")
        self.assertTrue(all(p.requires_grad for p in model.decoder.parameters()))

    def test_scene_macro_selection_weights_scenes_equally(self):
        rows = [{"category": "positive", "source_scene": "many", **metrics({"tp": 10, "fp": 0, "fn": 0, "tn": 6})} for _ in range(9)]
        rows += [{"category": "positive", "source_scene": "one", **metrics({"tp": 0, "fp": 0, "fn": 10, "tn": 6})},
                 {"category": "empty", "source_scene": "empty_a", **metrics({"tp": 0, "fp": 8, "fn": 0, "tn": 8})},
                 {"category": "empty", "source_scene": "empty_b", **metrics({"tp": 0, "fp": 0, "fn": 0, "tn": 16})}]
        summary = summarize(rows)
        self.assertAlmostEqual(summary["positive_macro"]["dice"], 0.9)
        self.assertAlmostEqual(summary["positive_scene_macro"]["dice"], 0.5)
        self.assertAlmostEqual(validation_key(summary)[0], 0.25)

    def test_empty_metrics_and_selection_rule(self):
        empty = torch.zeros(1, 1, 4, 4)
        self.assertEqual(metrics(image_counts(-torch.ones_like(empty), empty))["dice"], 1)
        self.assertEqual(metrics(image_counts(torch.ones_like(empty), empty))["fp"], 16)
        rows = [{"category": "positive", "source_scene": "a", **metrics({"tp": 9, "fp": 1, "fn": 1, "tn": 5})},
                {"category": "empty", "source_scene": "b", **metrics({"tp": 0, "fp": 0, "fn": 0, "tn": 16})}]
        summary = summarize(rows)
        self.assertAlmostEqual(validation_key(summary)[0], 0.9)
        self.assertEqual(summary["empty"]["false_alarm_rate"], 0)

    def test_manifest_inventory_and_locked_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ids = [SUPPORT_ID, *[f"p{i:04d}" for i in range(1200)],
                   *[f"h{i:04d}" for i in range(51)]]
            rows = []
            for index, sample_id in enumerate(ids):
                category = "正样本" if index < 1201 else "困难负样本"
                for part in ("images", "labels"):
                    path = root / category / part / (sample_id + ".tif")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(sample_id.encode() + part.encode())
                rows.append({"id": sample_id, "category": category, "source_scene": "source_01",
                             "image": f"{category}/images/{sample_id}.tif"})
            with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("id", "category", "source_scene", "image"))
                writer.writeheader()
                writer.writerows(rows)
            (root / "dataset_summary.json").write_text("{}", encoding="utf-8")
            entries = inventory(root)
            output = root / "output"
            first = lock_selection(root, output, entries)
            second = lock_selection(root, output, entries)
            self.assertEqual(first, second)
            self.assertEqual(first["selected"][0]["id"], SUPPORT_ID)
            self.assertEqual(first["selected"][0]["source_scene"], "source_01")
            self.assertEqual(len(first["selected"]), 1250)
            self.assertEqual(len({e["id"] for e in first["selected"]}), 1250)
            self.assertEqual(first["selection_sha256"], canonical_sha256(
                {k: v for k, v in first.items() if k != "selection_sha256"}))
            with (output / "s0_selection_ids.csv").open("r", encoding="utf-8-sig") as stream:
                self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 1250)
            (root / "manifest.csv").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                lock_selection(root, output, entries)

    def test_checkpoint_resume_and_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            set_seed(42)
            model = torch.nn.Linear(3, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            loss = model(torch.randn(2, 3)).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            path = Path(folder) / "last.pt"
            save_training_state(path, model, optimizer, scaler, 1, (0.5, -0.1), 1,
                                0, "locked-selection", None, 1)
            expected = torch.randn(4)
            restored = torch.nn.Linear(3, 1)
            restored_optimizer = torch.optim.AdamW(restored.parameters())
            restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
            state = load_training_state(path, restored, restored_optimizer, restored_scaler,
                                        "locked-selection", None)
            torch.testing.assert_close(torch.randn(4), expected)
            torch.testing.assert_close(restored.weight, model.weight)
            self.assertEqual(state["completed_epoch"], 1)
            # The same next update must agree, including AdamW moments and step count.
            x = torch.tensor([[1., 2., 3.]])
            for current, optim in ((model, optimizer), (restored, restored_optimizer)):
                optim.zero_grad(set_to_none=True)
                current(x).square().sum().backward()
                optim.step()
            torch.testing.assert_close(restored.weight, model.weight, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                load_training_state(path, restored, restored_optimizer, restored_scaler,
                                    "changed-selection", None)

    def test_fp16_cache_loads_as_fp32_for_trainable_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.npy"
            import numpy as np
            np.save(path, np.ones((1024, 32, 32), dtype=np.float16))
            loaded = load_tokens(Path(folder), {"files": {"sample": {"file": path.name}}},
                                 "sample", torch.device("cpu"))
            self.assertEqual(loaded.dtype, torch.float32)
            self.assertEqual(tuple(loaded.shape), (1, 1024, 32, 32))

    def test_external_validation_lock_and_no_training_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            selected = [{"id": SUPPORT_ID, "source_scene": "train_scene", "image_sha256": "x", "mask_sha256": "y"}]
            records = []
            for sample_id, category in (("external_positive", "positive"), ("external_empty", "empty")):
                image = root / (sample_id + ".png")
                mask = root / (sample_id + "_mask.png")
                Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)).save(image)
                pixels = np.zeros((512, 512), dtype=np.uint8)
                if category == "positive":
                    pixels[4:8, 4:8] = 1
                Image.fromarray(pixels).save(mask)
                records.append({"id": sample_id, "category": category, "source_scene": "holdout",
                                "image": image.name, "mask": mask.name})
            manifest = root / "validation.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("id", "category", "source_scene", "image", "mask"))
                writer.writeheader()
                writer.writerows(records)
            output = root / "output"
            rows, validation_root, digest = load_validation(manifest, root, selected, output)
            self.assertEqual(len(rows), 2)
            self.assertEqual(validation_root, root)
            self.assertEqual(digest, json.loads((output / "validation_lock.json").read_text())["lock_sha256"])
            self.assertEqual(load_validation(manifest, root, selected, output)[2], digest)
            # Empty label files legitimately match across unrelated datasets.
            selected[0]["mask_sha256"] = rows[1]["mask_sha256"]
            self.assertEqual(load_validation(manifest, root, selected, output)[2], digest)
            with self.assertRaises(ValueError):
                load_validation(manifest, root, [{"id": "external_positive", "source_scene": "train_scene", "image_sha256": "x", "mask_sha256": "y"}], output)
            selected[0]["source_scene"] = "holdout"
            with self.assertRaisesRegex(ValueError, "source_scene overlaps"):
                load_validation(manifest, root, selected, output)

    def test_memory_precheck_restores_optimizer_parameters_and_rng(self):
        model = TinySegmenter()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        support = torch.ones(1, 1, 32, 32)
        target = torch.zeros(1, 1, 512, 512)
        target[:, :, 50:100, 50:100] = 1
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        report = memory_precheck(model, optimizer, scaler, support, target,
                                support, target, False, torch.device("cpu"))
        self.assertTrue(report["includes_adamw_state_allocation"])
        self.assertEqual(len(optimizer.state), 0)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_smoke_cold_hot_cache_initialization_consistency(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = [{"id": SUPPORT_ID, "category": "正样本", "mask": "positive.png"},
                       {"id": "positive_query", "category": "正样本", "mask": "positive.png"},
                       {"id": "empty_query", "category": "困难负样本", "mask": "empty.png"}]
            for record in records:
                record["source_scene"] = "training_scene"
                record["image"] = "image.png"
            Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)).save(root / "image.png")
            for name, positive in (("positive", True), ("empty", False)):
                pixels = np.zeros((512, 512), dtype=np.uint8)
                if positive:
                    pixels[30:50, 20:80] = 1
                Image.fromarray(pixels).save(root / f"{name}.png")
            args = argparse.Namespace(mode="smoke", smoke_steps=2, allow_cpu=True,
                source_scene_column="source_scene",
                epochs=20, output=root / "out", input_root=root, data_root=root,
                dino_weights=None, dino_repo=None, validation_manifest=None, validation_root=None)
            initial_states = []
            cache_calls = []
            def fake_cache(*unused, **kwargs):
                if not cache_calls:
                    torch.rand(500)  # Cold encoder construction consumes RNG.
                cache_calls.append(True)
                return {"weight_sha256": "weights"}
            def fake_model():
                model = TinySegmenter()
                initial_states.append({k: v.clone() for k, v in model.state_dict().items()})
                return model
            with patch("train_s0.find_dataset_root", return_value=root), \
                 patch("train_s0.find_weights", return_value=root / "unused.pth"), \
                 patch("train_s0.inventory", return_value={}), \
                 patch("train_s0.lock_selection", return_value={"selected": records, "selection_sha256": "locked"}), \
                 patch("train_s0.cache_tokens", side_effect=fake_cache), \
                 patch("train_s0.load_tokens", return_value=torch.ones(1, 1, 32, 32)), \
                 patch("train_s0.S0Segmenter", side_effect=fake_model):
                train(args)
                args.output = root / "hot_cache_run"
                train(args)
            for key in initial_states[0]:
                torch.testing.assert_close(initial_states[0][key], initial_states[1][key], rtol=0, atol=0)
            result = json.loads((args.output / "smoke_results.json").read_text())
            self.assertEqual(result["train"]["query_count"], 2)
            self.assertEqual(result["train"]["empty_count"], 1)
            self.assertFalse((args.output / "last.pt").exists())
            self.assertFalse((args.output / "best.pt").exists())

    def test_joint_selection_and_patience_start_after_warmup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)).save(root / "image.png")
            positive = np.zeros((512, 512), dtype=np.uint8)
            positive[10:30, 20:40] = 1
            Image.fromarray(positive).save(root / "positive.png")
            Image.fromarray(np.zeros_like(positive)).save(root / "empty.png")
            rows = [{"id": SUPPORT_ID, "category": "正样本", "mask": "positive.png"},
                    {"id": "p", "category": "正样本", "mask": "positive.png"},
                    {"id": "n", "category": "困难负样本", "mask": "empty.png"}]
            for row in rows:
                row.update(image="image.png", source_scene="training")
            args = argparse.Namespace(mode="train", smoke_steps=2, allow_cpu=True, source_scene_column="source_scene",
                epochs=20, output=root / "out", input_root=root, data_root=root,
                dino_weights=None, dino_repo=None, validation_manifest=None, validation_root=None)
            summary = {"positive_scene_macro": {"dice": 0.8}, "empty_scene_macro": {"false_positive_pixel_rate": 0.1}}
            with patch("train_s0.find_dataset_root", return_value=root), \
                 patch("train_s0.find_weights", return_value=root / "unused.pth"), \
                 patch("train_s0.inventory", return_value={}), \
                 patch("train_s0.lock_selection", return_value={"selected": rows, "selection_sha256": "locked"}), \
                 patch("train_s0.cache_tokens", return_value={"weight_sha256": "weights"}), \
                 patch("train_s0.load_tokens", return_value=torch.ones(1, 1, 32, 32)), \
                 patch("train_s0.load_validation", return_value=([{"id": "heldout"}], root, "validation")), \
                 patch("train_s0.evaluate", return_value=(summary, [])) as evaluation, \
                 patch("train_s0.S0Segmenter", TinySegmenter):
                train(args)
            status = json.loads((args.output / "run_status.json").read_text())
            self.assertEqual(status["best_epoch"], 4)
            self.assertEqual(status["completed_epoch"], 9)
            self.assertEqual(evaluation.call_count, 6)
            epoch3 = torch.load(args.output / "epoch_03.pt", weights_only=False)
            epoch4 = torch.load(args.output / "epoch_04.pt", weights_only=False)
            self.assertIsNone(epoch3["best_epoch"])
            self.assertEqual(epoch3["phase"], "warmup")
            self.assertEqual(epoch4["phase"], "joint")

    def test_bundle_contains_exact_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            output = build(Path(folder))
            script = (output / "run-s0-baseline.py").read_text(encoding="utf-8")
            metadata = json.loads((output / "kernel-metadata.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("base64.b64decode", script)
            self.assertEqual(set(manifest["sources"]), set(SOURCES))
            self.assertEqual(metadata["dataset_sources"], [])
            self.assertEqual(metadata["model_sources"], [])
            payload = next(node.args[0].value for node in ast.walk(ast.parse(script))
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                           and node.func.attr == "b64decode")
            extracted = Path(folder) / "extracted"
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
                self.assertEqual(set(archive.namelist()), set(SOURCES))
                for name in SOURCES:
                    self.assertEqual(archive.read(name), (Path(__file__).parent / name).read_bytes())
                archive.extractall(extracted)
            help_result = subprocess.run([sys.executable, str(extracted / "train_s0.py"), "--help"],
                                         capture_output=True, text=True, check=True)
            self.assertIn("--smoke-steps", help_result.stdout)
            original_script = script
            build(Path(folder))
            self.assertEqual(original_script, (output / "run-s0-baseline.py").read_text(encoding="utf-8"))


class TinySegmenter(torch.nn.Module):
    """Cheap fake encoder/decoder for testing orchestration, never used for results."""
    def __init__(self):
        super().__init__()
        self.visual_projection = torch.nn.Conv2d(1, 1, 1)
        self.decoder = torch.nn.Conv2d(1, 1, 1)

    def forward(self, support, query, mask, include_logits=True):
        cam = self.visual_projection(query[:, :1].float())
        logits = self.decoder(torch.nn.functional.interpolate(cam, (512, 512), mode="bilinear", align_corners=False)) if include_logits else None
        return {"cam": cam, "logits": logits}


if __name__ == "__main__":
    unittest.main()
