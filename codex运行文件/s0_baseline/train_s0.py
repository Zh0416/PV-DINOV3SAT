"""Train the S0 baseline on Kaggle T4 from frozen PV-512 DINO tokens."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import time
import shutil
import tempfile
from collections import defaultdict

import numpy as np
import torch

from s0_data import (SEED, SUPPORT_ID, cache_tokens, canonical_sha256,
                     checked_mask, checked_rgb, find_dataset_root, find_weights, inventory,
                     load_tokens, lock_selection, scene_key, sha256_file, write_json)
from s0_model import S0Segmenter, s0_loss
from s0_merged_validation import evaluate_merged, load_merged_validation


PROTOCOL = {
    "schema": "s0-pv512-training-v2", "seed": SEED, "support_id": SUPPORT_ID,
    "epochs_max": 20, "warmup_epochs": 3, "joint_start_epoch": 4,
    "batch_size": 1, "gradient_accumulation": 4,
    "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0, "precision": "CUDA FP16 autocast with GradScaler",
    "amp_initial_scale": 1024.0,
    "warmup_loss": "raw CAM BCEWithLogits(pos_weight=2.0), all query types, decoder frozen and skipped",
    "cam_temperature": 1.0,
    "positive_loss": "BCEWithLogits(pos_weight=2.0)+soft Dice+0.1*raw CAM BCE(pos_weight=2.0)",
    "hard_negative_loss": "BCEWithLogits(empty target)+0.1*raw CAM BCE(empty target); no Dice",
    "prediction_threshold": "logits >= 0", "empty_dice": "1 iff predicted mask empty, else 0",
    "validation_score": "merged PV-512TEST scene-macro Dice; legacy positive/empty validation keeps its original score; earliest exact tie wins",
    "validation_eligible": "joint phase only, start/reset best and patience at epoch 4",
    "early_stopping_patience": 5,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_validation(manifest: Path | None, root: Path | None,
                    selected: list[dict], output: Path) -> tuple[list[dict], Path | None, str | None]:
    if manifest is None:
        if (output / "validation_lock.json").exists():
            raise ValueError("Resume requires the original external validation manifest.")
        return [], None, None
    manifest = manifest.resolve()
    root = (root or manifest.parent).resolve()
    selected_ids = {entry["id"] for entry in selected}
    selected_image_hashes = {entry["image_sha256"] for entry in selected}
    training_scenes = {scene_key(entry["source_scene"]) for entry in selected}
    rows = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "category", "source_scene", "image", "mask"} <= set(reader.fieldnames or []):
            raise ValueError("External validation CSV needs id,category,source_scene,image,mask columns.")
        for row in reader:
            sample_id = row["id"].strip()
            category = row["category"].strip().lower()
            if (category not in ("positive", "empty") or not sample_id or sample_id in selected_ids
                    or sample_id in (".", "..") or any(c in sample_id for c in '/\\:')):
                raise ValueError(f"Invalid or overlapping validation sample: {sample_id}")
            source_scene = (row["source_scene"] or "").strip()
            if scene_key(source_scene) in training_scenes:
                raise ValueError(f"Validation source_scene overlaps training/support: {source_scene}")
            entry = {"id": sample_id, "category": category, "source_scene": source_scene}
            for field in ("image", "mask"):
                relative = Path(row[field].strip())
                path = (root / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f"Invalid external validation path: {row[field]}")
                entry[field] = relative.as_posix()
                entry[field + "_sha256"] = sha256_file(path)
            if entry["image_sha256"] in selected_image_hashes:
                raise ValueError(f"Validation file overlaps selected training files: {sample_id}")
            checked_mask(root / entry["mask"], positive=category == "positive")
            rows.append(entry)
    if len({r["id"] for r in rows}) != len(rows) or not any(r["category"] == "positive" for r in rows) or not any(r["category"] == "empty" for r in rows):
        raise ValueError("Validation needs unique IDs and both positive and empty queries.")
    lock = {"schema": "s0-external-validation-v1", "manifest_sha256": sha256_file(manifest),
            "rows": rows, "note": "User-provided holdout; geographic independence is not inferred by this code."}
    lock["lock_sha256"] = canonical_sha256(lock)
    path = output / "validation_lock.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != lock:
        raise ValueError("External validation manifest or files changed since training started.")
    if not path.exists():
        write_json(path, lock)
    return rows, root, lock["lock_sha256"]


def image_counts(logits: torch.Tensor, target: torch.Tensor) -> dict[str, int]:
    predicted = logits.detach() >= 0
    actual = target.bool()
    tp = int((predicted & actual).sum().item())
    fp = int((predicted & ~actual).sum().item())
    fn = int((~predicted & actual).sum().item())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": int(actual.numel() - tp - fp - fn)}


def metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    empty_both = tp + fp + fn == 0
    return {**counts,
            "dice": 1.0 if empty_both else 2 * tp / max(1, 2 * tp + fp + fn),
            "iou": 1.0 if empty_both else tp / max(1, tp + fp + fn),
            "precision": 1.0 if empty_both else tp / max(1, tp + fp),
            "recall": 1.0 if empty_both else tp / max(1, tp + fn)}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    positive = [r for r in rows if r["category"] == "positive"]
    empty = [r for r in rows if r["category"] != "positive"]
    counts = {key: sum(r[key] for r in rows) for key in ("tp", "fp", "fn", "tn")}
    result = {"query_count": len(rows), "positive_count": len(positive), "empty_count": len(empty),
              "micro": metrics(counts),
              "positive_macro": {key: float(np.mean([r[key] for r in positive])) for key in
                                 ("dice", "iou", "precision", "recall")} if positive else None,
              "empty": {"false_alarm_images": sum(r["fp"] > 0 for r in empty),
                        "false_alarm_rate": float(np.mean([r["fp"] > 0 for r in empty])),
                        "false_positive_pixels": sum(r["fp"] for r in empty),
                        "false_positive_pixel_rate": sum(r["fp"] for r in empty) / sum(r["fp"] + r["tn"] for r in empty),
                        "macro_dice": float(np.mean([r["dice"] for r in empty]))} if empty else None}
    positive_scenes, empty_scenes = defaultdict(list), defaultdict(list)
    for row in positive:
        positive_scenes[scene_key(row["source_scene"])].append(row)
    for row in empty:
        empty_scenes[scene_key(row["source_scene"])].append(row)
    result["per_positive_scene"] = [
        {"source_scene": name, "query_count": len(group),
         **{key: float(np.mean([r[key] for r in group])) for key in ("dice", "iou", "precision", "recall")}}
        for name, group in sorted(positive_scenes.items())]
    result["per_empty_scene"] = [
        {"source_scene": name, "query_count": len(group),
         "false_positive_pixel_rate": float(np.mean([r["fp"] / (r["fp"] + r["tn"]) for r in group])),
         "false_alarm_rate": float(np.mean([r["fp"] > 0 for r in group]))}
        for name, group in sorted(empty_scenes.items())]
    result["positive_scene_macro"] = {key: float(np.mean([r[key] for r in result["per_positive_scene"]]))
                                      for key in ("dice", "iou", "precision", "recall")} if positive_scenes else None
    result["empty_scene_macro"] = {key: float(np.mean([r[key] for r in result["per_empty_scene"]]))
                                   for key in ("false_positive_pixel_rate", "false_alarm_rate")} if empty_scenes else None
    return result


def validation_key(summary: dict) -> tuple[float, float]:
    return (summary["positive_scene_macro"]["dice"] - summary["empty_scene_macro"]["false_positive_pixel_rate"], 0.0)


def configure_phase(model, epoch: int) -> str:
    phase = "warmup" if epoch <= PROTOCOL["warmup_epochs"] else "joint"
    model.visual_projection.requires_grad_(True)
    model.decoder.requires_grad_(phase == "joint")
    return phase


@torch.inference_mode()
def evaluate(model: S0Segmenter, support: torch.Tensor, support_mask: torch.Tensor,
             rows: list[dict], root: Path, cache_dir: Path, index: dict,
             device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    results = []
    for entry in rows:
        query = load_tokens(cache_dir, index, entry["id"], device)
        target = torch.from_numpy(checked_mask(root / entry["mask"],
                                   positive=entry["category"] == "positive").copy())[None, None].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(support, query, support_mask)["logits"]
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Nonfinite validation logits: {entry['id']}")
        results.append({"id": entry["id"], "category": entry["category"],
                        "source_scene": entry["source_scene"],
                        **metrics(image_counts(logits, target))})
    return summarize(results), results


def atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_training_state(path: Path, model: S0Segmenter, optimizer: torch.optim.Optimizer,
                        scaler: torch.amp.GradScaler, epoch: int, best_key: tuple,
                        best_epoch: int | None, stale_epochs: int, selection_sha: str,
                        validation_sha: str | None, optimizer_steps: int,
                        run_identity: dict | None = None, skipped_updates: int = 0) -> None:
    atomic_checkpoint(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "scaler": scaler.state_dict(), "completed_epoch": epoch,
                             "phase": "warmup" if epoch <= PROTOCOL["warmup_epochs"] else "joint",
                             "best_key": best_key, "best_epoch": best_epoch,
                             "stale_epochs": stale_epochs, "optimizer_steps": optimizer_steps,
                             "skipped_updates": skipped_updates,
                             "rng": rng_state(), "selection_sha256": selection_sha,
                             "validation_sha256": validation_sha,
                             "protocol_sha256": canonical_sha256(PROTOCOL),
                             "run_identity": run_identity})


def load_training_state(path: Path, model: S0Segmenter, optimizer: torch.optim.Optimizer,
                        scaler: torch.amp.GradScaler, selection_sha: str,
                        validation_sha: str | None, run_identity: dict | None = None) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state["selection_sha256"] != selection_sha or state["validation_sha256"] != validation_sha
            or state["protocol_sha256"] != canonical_sha256(PROTOCOL)
            or state.get("run_identity") != run_identity):
        raise ValueError("Checkpoint selection/validation/protocol/source/encoder lock differs.")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return state


def optimizer_update(model, optimizer, scaler, device) -> bool:
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), PROTOCOL["gradient_clip_norm"])
    finite = bool(torch.isfinite(norm))
    if not finite and device.type != "cuda":
        raise RuntimeError("Nonfinite accumulated gradient on CPU.")
    # CUDA GradScaler skips an overflowing group and backs off its scale.
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return finite


def memory_precheck(model: S0Segmenter, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, support: torch.Tensor,
                    support_mask: torch.Tensor, query: torch.Tensor,
                    target: torch.Tensor, hard_negative: bool, device: torch.device) -> dict:
    # Include AdamW state allocation in the peak, then restore every mutable state.
    snapshot_dir = tempfile.TemporaryDirectory(prefix="s0-memory-precheck-")
    snapshot_path = Path(snapshot_dir.name) / "before.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng_state()}, snapshot_path)
    gradient_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    model.requires_grad_(True)  # Preflight measures the more expensive joint phase.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    try:
        for attempt in range(8):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(support, query, support_mask)
                loss, _ = s0_loss(prediction, target, hard_negative)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss in GPU memory precheck.")
            scaler.scale(loss).backward()
            updated = optimizer_update(model, optimizer, scaler, device)
            del prediction, loss
            if updated:
                break
        else:
            raise RuntimeError("Eight consecutive gradient overflows in precheck.")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report = {"dry_step_seconds": time.perf_counter() - start,
                  "includes_adamw_state_allocation": True, "parameters_restored_after_check": True,
                  "amp_overflow_retries": attempt,
                  "phase_measured": "joint",
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                  "total_device_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None}
        if device.type == "cuda" and report["peak_reserved_bytes"] > 0.9 * report["total_device_bytes"]:
            raise RuntimeError("Dry step reserved over 90% of GPU memory; refusing full training.")
        return report
    finally:
        before = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        model.load_state_dict(before["model"])
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(gradient_flags[name])
        optimizer.load_state_dict(before["optimizer"])
        scaler.load_state_dict(before["scaler"])
        optimizer.zero_grad(set_to_none=True)
        restore_rng(before["rng"])
        snapshot_dir.cleanup()


def train(args: argparse.Namespace) -> None:
    if args.mode != "prepare" and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("S0 full training requires CUDA; --allow-cpu is for local diagnostics only.")
    if not 1 <= args.epochs <= PROTOCOL["epochs_max"]:
        raise ValueError("S0 allows 1-20 epochs.")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.allow_cpu else "cpu")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = find_dataset_root(args.input_root, args.data_root)
    entries = inventory(root, args.source_scene_column)
    reference_lock = Path(__file__).parent / "protocol" / "s0_selection.json"
    selection = lock_selection(root, output, entries, reference_lock if reference_lock.exists() else None)
    selected = selection["selected"]
    support_entry, queries = selected[0], selected[1:]
    support_foreground_pixels = 0
    for entry in selected:
        mask = checked_mask(root / entry["mask"], positive=entry["category"] == "正样本")
        if entry["id"] == SUPPORT_ID:
            support_foreground_pixels = int(mask.sum())
            if mask.all():
                raise ValueError("Support must contain both foreground and background.")
        checked_rgb(root / entry["image"])
    write_json(output / "data_audit.json", {
        "selection_sha256": selection["selection_sha256"], "pairs_verified": len(selected),
        "mask_values": [0, 1], "image_size": [512, 512], "image_mode": "RGB",
        "positive_queries": len([e for e in queries if e["category"] == "正样本"]),
        "empty_hard_negative_queries": len([e for e in queries if e["category"] == "困难负样本"]),
        "support_id": SUPPORT_ID, "support_foreground_pixels": support_foreground_pixels,
        "source_scenes": sorted({e["source_scene"] for e in selected}),
        "source_counts": selection.get("source_counts"), "query_validation_performed": False})
    merged_arg = getattr(args, "merged_validation_root", None)
    if merged_arg is not None and args.validation_manifest is not None:
        raise ValueError("Choose either --merged-validation-root or --validation-manifest.")
    if merged_arg is not None:
        merged_rows, merged_root, validation_sha = load_merged_validation(
            merged_arg, selected, selection["selection_sha256"], output,
            reference_path=Path(__file__).parent / "protocol" / "merged_validation_reference.json")
        validation, validation_root = [], None
        validation_mode = "merged_185"
    else:
        if (output / "merged_validation_lock.json").exists():
            raise ValueError("Resume requires the original --merged-validation-root.")
        validation, validation_root, validation_sha = load_validation(
            args.validation_manifest, args.validation_root, selected, output)
        merged_rows, merged_root = [], None
        validation_mode = "external_positive_empty" if validation else "none"
    locked_protocol = {**PROTOCOL, "selection_sha256": selection["selection_sha256"],
                       "validation_sha256": validation_sha, "validation_mode": validation_mode}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != locked_protocol:
        raise ValueError("The existing run protocol differs; use the original options or a new output.")
    write_json(protocol_path, locked_protocol)
    if args.mode == "prepare":
        print(json.dumps({"selection_sha256": selection["selection_sha256"],
                          "positive_queries": 1199, "hard_negative_queries": 50,
                          "merged_validation_images": len(merged_rows),
                          "output": str(output)}, ensure_ascii=False), flush=True)
        return
    weights = find_weights(args.input_root, args.dino_weights)
    mode_path = output / "run_mode.json"
    mode_record = {"mode": args.mode}
    if mode_path.exists() and json.loads(mode_path.read_text(encoding="utf-8")) != mode_record:
        raise ValueError("Smoke and full training must use separate output directories.")
    write_json(mode_path, mode_record)
    if args.mode == "smoke":
        if not 2 <= args.smoke_steps <= 100:
            raise ValueError("Smoke requires 2-100 episodes, including both query classes.")
        hard_count = args.smoke_steps // 2
        queries = (queries[:args.smoke_steps - hard_count]
                   + [e for e in queries if e["category"] == "困难负样本"][:hard_count])
    cache_selected = [support_entry, *queries]
    cache_dir = output / "train_tokens"
    index = cache_tokens(root, cache_selected, cache_dir, args.dino_repo or output / "dinov3",
                         weights, device, selection["selection_sha256"])
    validation_cache = output / "validation_tokens"
    validation_index = None
    validation_entries = merged_rows if merged_rows else validation
    validation_data_root = merged_root if merged_rows else validation_root
    if validation_entries and args.mode != "smoke":
        validation_index = cache_tokens(validation_data_root, validation_entries, validation_cache,
                                        args.dino_repo or output / "dinov3", weights, device, validation_sha)
    support = load_tokens(cache_dir, index, SUPPORT_ID, device)
    support_mask = torch.from_numpy(checked_mask(root / support_entry["mask"], positive=True).copy())[None, None].to(device=device, dtype=torch.float32)
    # Encoder construction on a cache miss must not change decoder initialization.
    set_seed(SEED)
    model = S0Segmenter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PROTOCOL["learning_rate"],
                                  weight_decay=PROTOCOL["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=PROTOCOL["amp_initial_scale"])
    run_identity = {"weight_sha256": index["weight_sha256"],
                    "sources": {name: sha256_file(Path(__file__).parent / name)
                                for name in ("s0_data.py", "s0_model.py", "s0_merged_validation.py", "train_s0.py")},
                    "torch": str(torch.__version__), "device_type": device.type}
    write_json(output / "runtime.json", {**run_identity, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    last_path = output / "last.pt"
    best_key, best_epoch, stale_epochs, optimizer_steps, start_epoch = (-float("inf"), -float("inf")), None, 0, 0, 1
    skipped_updates = 0
    if last_path.exists():
        state = load_training_state(last_path, model, optimizer, scaler,
                                    selection["selection_sha256"], validation_sha, run_identity)
        start_epoch = state["completed_epoch"] + 1
        best_key, best_epoch = tuple(state["best_key"]), state["best_epoch"]
        stale_epochs, optimizer_steps = state["stale_epochs"], state["optimizer_steps"]
        skipped_updates = state["skipped_updates"]
    elif any(output.glob("epoch_*.json")):
        raise ValueError("Training logs exist without a resumable checkpoint.")
    elif args.mode == "train":
        save_training_state(last_path, model, optimizer, scaler, 0, best_key,
                            best_epoch, stale_epochs, selection["selection_sha256"],
                            validation_sha, optimizer_steps, run_identity, skipped_updates)
    if start_epoch > args.epochs or stale_epochs >= PROTOCOL["early_stopping_patience"]:
        print("No training epochs remain under the locked protocol.", flush=True)
        return
    first = queries[0]
    first_query = load_tokens(cache_dir, index, first["id"], device)
    first_target = torch.from_numpy(checked_mask(root / first["mask"], positive=True).copy())[None, None].to(device)
    precheck = memory_precheck(model, optimizer, scaler, support, support_mask,
                               first_query, first_target, False, device)
    write_json(output / "memory_precheck.json", precheck)
    print(json.dumps({"memory_precheck": precheck}), flush=True)
    del first_query, first_target
    timing_start = time.perf_counter()
    timing_initial_steps = optimizer_steps
    timing_reported = False
    timing_phase = None
    timed_episodes = 0
    last_epoch = 1 if args.mode == "smoke" else args.epochs
    for epoch in range(start_epoch, last_epoch + 1):
        # Smoke exercises the full decoder without counting as one training epoch.
        phase = configure_phase(model, 4 if args.mode == "smoke" else epoch)
        if phase != timing_phase:
            timing_phase, timing_start, timing_initial_steps = phase, time.perf_counter(), optimizer_steps
            timing_reported, timed_episodes = False, 0
        if epoch == PROTOCOL["joint_start_epoch"]:
            best_key, best_epoch, stale_epochs = (-float("inf"), -float("inf")), None, 0
        model.train()
        order = list(queries)
        random.Random(SEED + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        train_rows = []
        loss_total = 0.0
        for position, entry in enumerate(order):
            hard_negative = entry["category"] == "困难负样本"
            query = load_tokens(cache_dir, index, entry["id"], device)
            target = torch.from_numpy(checked_mask(root / entry["mask"], positive=not hard_negative).copy())[None, None].to(device)
            group_start = position // 4 * 4
            group_size = min(4, len(order) - group_start)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(support, query, support_mask, include_logits=phase == "joint")
                loss, _ = s0_loss(prediction, target, hard_negative, phase=phase)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss: {entry['id']}")
            scaler.scale(loss / group_size).backward()
            metric_logits = prediction["logits"] if phase == "joint" else torch.nn.functional.interpolate(
                prediction["cam"], size=(512, 512), mode="bilinear", align_corners=False)
            train_rows.append({"id": entry["id"], "category": "empty" if hard_negative else "positive",
                               "source_scene": entry["source_scene"],
                               **metrics(image_counts(metric_logits, target))})
            timed_episodes += 1
            loss_total += float(loss.detach())
            if (position + 1) % 4 == 0 or position + 1 == len(order):
                if optimizer_update(model, optimizer, scaler, device):
                    optimizer_steps += 1
                else:
                    skipped_updates += 1
                    print(f"AMP overflow: skipped group; scale={scaler.get_scale()}", flush=True)
                if optimizer_steps - timing_initial_steps == 100 and not timing_reported:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - timing_start
                    timing = {"optimizer_steps_in_segment": 100, "starting_step": timing_initial_steps,
                              "episodes_in_segment": timed_episodes, "phase": phase,
                              "seconds": elapsed, "seconds_per_optimizer_step": elapsed / 100,
                              "estimated_training_epoch_seconds_excluding_validation": elapsed / 100 * 313,
                              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
                    write_json(output / f"timing_{phase}_100_updates_from_{timing_initial_steps}.json", timing)
                    timing_reported = True
                    print(json.dumps({"timing": timing}), flush=True)
                if optimizer_steps % 25 == 0:
                    print(f"Epoch {epoch}: {position + 1}/{len(order)} episodes; optimizer step {optimizer_steps}", flush=True)
            del query, target, prediction, loss, metric_logits
        train_summary = summarize(train_rows)
        train_summary["mean_loss"] = loss_total / len(order)
        train_summary["optimizer_steps_total"] = optimizer_steps
        train_summary["amp_skipped_updates_total"] = skipped_updates
        record = {"epoch": epoch, "phase": phase, "train": train_summary, "validation": None,
                  "training_metric_source": "decoder logits" if phase == "joint" else "raw CAM only, decoder not evaluated",
                  "selection_sha256": selection["selection_sha256"]}
        if validation_entries and phase == "joint" and args.mode != "smoke":
            if merged_rows:
                summary, per_image = evaluate_merged(model, support, support_mask, merged_rows,
                                                     merged_root, validation_cache, validation_index, device)
                key = (summary["selection_score"], 0.0)
            else:
                summary, per_image = evaluate(model, support, support_mask, validation,
                                              validation_root, validation_cache, validation_index, device)
                key = validation_key(summary)
            record["validation"] = summary
            write_json(output / f"validation_epoch_{epoch:02d}.json",
                       {"summary": summary, "per_image": per_image, "validation_mode": validation_mode})
            improved = key[0] > best_key[0]
            if improved:
                best_key, best_epoch, stale_epochs = key, epoch, 0
                atomic_checkpoint(output / "best.pt", {"model": model.state_dict(),
                                  "epoch": epoch, "validation": summary,
                                  "selection_sha256": selection["selection_sha256"],
                                  "validation_sha256": validation_sha,
                                  "run_identity": run_identity,
                                  "protocol_sha256": canonical_sha256(PROTOCOL)})
            else:
                stale_epochs += 1
            record["selection_score"] = key[0]
            record["best_epoch"] = best_epoch
        if args.mode == "smoke":
            write_json(output / "smoke_results.json", {**record, "mode": "smoke",
                       "per_image": train_rows, "seconds": time.perf_counter() - timing_start,
                       "note": "Partial diagnostic only; no trained or best checkpoint exported."})
            print("S0 smoke completed; no full-training checkpoint written.", flush=True)
            return
        write_json(output / f"epoch_{epoch:02d}.json", record)
        write_json(output / f"training_epoch_{epoch:02d}.json", {"per_image": train_rows})
        save_training_state(last_path, model, optimizer, scaler, epoch, best_key,
                            best_epoch, stale_epochs, selection["selection_sha256"],
                            validation_sha, optimizer_steps, run_identity, skipped_updates)
        snapshot = output / f"epoch_{epoch:02d}.pt"
        shutil.copyfile(last_path, snapshot.with_suffix(".pt.tmp"))
        os.replace(snapshot.with_suffix(".pt.tmp"), snapshot)
        print(json.dumps({"epoch": epoch, "mean_loss": train_summary["mean_loss"],
                          "validation_score": record.get("selection_score"),
                          "best_epoch": best_epoch}, ensure_ascii=False), flush=True)
        if stale_epochs >= PROTOCOL["early_stopping_patience"]:
            break
    write_json(output / "run_status.json", {"completed_epoch": epoch,
               "best_epoch": best_epoch, "best_model_selected": best_epoch is not None,
               "validation_available": bool(validation_entries), "validation_mode": validation_mode,
               "optimizer_steps": optimizer_steps,
               "selection_sha256": selection["selection_sha256"],
               "timing_100_steps_completed_this_invocation": timing_reported,
               "note": "Without external validation, last.pt is resumable only and is not a selected best model."})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "smoke", "train"), default="train")
    parser.add_argument("--smoke-steps", type=int, default=8, help="Smoke episodes, 2-100, alternating class budgets")
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source-scene-column", default="source_scene")
    parser.add_argument("--dino-weights", type=Path)
    parser.add_argument("--dino-repo", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/s0_baseline"))
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--merged-validation-root", type=Path,
                        help="PV-512-MERGED-185 root or its Kaggle input wrapper")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--allow-cpu", action="store_true", help="Local diagnostic only")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
