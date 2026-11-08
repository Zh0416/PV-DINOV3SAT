from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


EXPERIMENT_DIR = Path(__file__).resolve().parent
SUPPORT_IMAGE = "PV08_331824_1195530.bmp"
QUERY_IMAGES = [
    "PV08_347185_1164244.bmp",
    "PV08_318322_1218788.bmp",
    "PV08_319542_1183980.bmp",
    "PV08_344239_1199475.bmp",
    "PV08_337694_1182766.bmp",
    "PV08_326513_1201232.bmp",
    "PV08_312621_1197085.bmp",
    "PV08_348179_1189176.bmp",
    "PV08_315404_1208378.bmp",
    "PV08_329573_1189671.bmp",
    "PV08_342992_1173495.bmp",
    "PV08_325206_1211345.bmp",
    "PV08_334930_1197013.bmp",
    "PV08_338249_1204612.bmp",
    "PV08_322505_1192968.bmp",
    "PV08_340747_1192732.bmp",
    "PV08_331361_1208263.bmp",
    "PV08_320381_1199142.bmp",
    "PV08_316218_1191721.bmp",
    "PV08_331752_1180424.bmp",
]
CLASSES = [
    ("PV", "ground-mounted solar panel arrays"),
    ("cropland", "agricultural fields"),
    ("greenhouse", "greenhouses"),
    ("blue_roof", "blue metal rooftops"),
    ("water", "water surfaces"),
    ("road", "roads"),
    ("bare_ground", "bare ground"),
]
TEMPLATES = [
    "a satellite image of {}",
    "an aerial image of {}",
    "an overhead view of {}",
]
WINDOW_SIZES = (128, 256)
WINDOW_STRIDE = 64
MODEL_ID = "Zilun/GeoRSCLIP"
CHECKPOINT_NAME = "RS5M_ViT-B-32.pt"
CHECKPOINT_URL = "https://huggingface.co/Zilun/GeoRSCLIP/resolve/main/ckpt/RS5M_ViT-B-32.pt?download=true"
CHECKPOINT_SHA256 = "129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01"
EXPECTED_CHECKPOINT_SIZE = 605222594
BENCHMARK_MANIFEST_SHA256 = "b0c8191a739d5cc23093660d2e1c0e79dce83b6043938b5ed76aa1eecce85f00"
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


def fixed_text_config() -> dict[str, Any]:
    return {
        "class_order": [name for name, _ in CLASSES],
        "classes": [
            {"name": name, "description": description}
            for name, description in CLASSES
        ],
        "templates": list(TEMPLATES),
        "expanded_prompts": {
            name: [template.format(description) for template in TEMPLATES]
            for name, description in CLASSES
        },
        "text_embedding_rule": (
            "L2-normalize each prompt embedding; average the 3 embeddings for each class; "
            "L2-normalize the class mean."
        ),
    }


def fixed_experiment_config() -> dict[str, Any]:
    return {
        "experiment": "E1 GeoRSCLIP semantic CAM",
        "scope": "Frozen zero-shot window-level semantic responses; no training or decoder.",
        "dataset_manifest": "../experiment_a/benchmark_v1.json",
        "dataset_manifest_sha256": BENCHMARK_MANIFEST_SHA256,
        "support_image": SUPPORT_IMAGE,
        "query_images": list(QUERY_IMAGES),
        "query_masks_read_during_inference": False,
        "backbone": {
            "repository": "https://github.com/om-ai-lab/RS5M",
            "model_page": "https://huggingface.co/Zilun/GeoRSCLIP",
            "model_id": MODEL_ID,
            "architecture": "OpenCLIP ViT-B/32, OpenAI CLIP configuration",
            "activation": "QuickGELU",
            "checkpoint": CHECKPOINT_NAME,
            "checkpoint_url": CHECKPOINT_URL.split("?", 1)[0],
            "checkpoint_sha256_expected": CHECKPOINT_SHA256,
            "checkpoint_size_bytes_expected": EXPECTED_CHECKPOINT_SIZE,
            "checkpoint_loading": "strict state-dict load; missing/unexpected keys must both be empty",
            "tokenizer": "open_clip.tokenize, context_length=77",
            "embedding": "final projected encode_image/encode_text embeddings, L2 normalized",
            "trainable": False,
            "eval_mode": True,
        },
        "image_preprocessing": {
            "official_source": "RS5M inference/inference_tool.py:get_preprocess(image_resolution=224, subset_name='clip', is_train=False)",
            "window_rgb": True,
            "resize": {"size": 224, "interpolation": "bicubic"},
            "center_crop": [224, 224],
            "to_tensor_scale": "uint8 / 255",
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
            "note": "Each source window is square; official Resize(224)+CenterCrop(224) is applied.",
        },
        "text": fixed_text_config(),
        "window_inference": {
            "source_image_size": [1024, 1024],
            "window_sizes": list(WINDOW_SIZES),
            "stride": WINDOW_STRIDE,
            "edge_policy": "include final start at image_size-window_size if the stride does not land on it",
            "window_resize": [224, 224],
            "window_batch_size": 32,
            "score": "cosine(image final projection, class final text projection)",
            "overlap_reconstruction": "sum window cosine over covered source pixels / integer coverage count",
            "scale_aggregation": "equal mean of the two reconstructed 7-class response maps",
            "E1_A": "S_PV",
            "E1_B": "S_PV - max(S_cropland, S_greenhouse, S_blue_roof, S_water, S_road, S_bare_ground)",
            "strongest_background": "argmax of the six reconstructed background class responses; explanatory prediction only",
            "interpretation": "window-level coarse localization; not pixel-boundary segmentation and not calibrated probability",
        },
        "evaluation": {
            "phase": "separate offline process after inference files and frozen config have been saved",
            "query_mask_values": {"background": 0, "photovoltaic": 12},
            "P_at_R95": "diagnostic operating point only; never fed back to inference",
        },
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def discover_data_root(data_root: Path | None) -> Path:
    expected = {SUPPORT_IMAGE, *QUERY_IMAGES}
    if data_root is not None:
        images_dir = data_root / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError("Missing images directory: {}".format(images_dir))
        available = {path.name for path in images_dir.glob("*.bmp")}
        if not expected.issubset(available):
            missing = sorted(expected - available)
            raise FileNotFoundError("Fixed benchmark images are missing: {}".format(missing))
        return data_root

    roots = [Path("/kaggle/input"), EXPERIMENT_DIR.parent / "experiment_a"]
    candidates: list[tuple[int, str, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for images_dir in root.rglob("images"):
            if not images_dir.is_dir():
                continue
            names = {path.name for path in images_dir.glob("*.bmp")}
            if expected.issubset(names) and (images_dir.parent / "mask").is_dir():
                candidates.append((len(names - expected), str(images_dir), images_dir.parent))
    if not candidates:
        raise FileNotFoundError("Could not locate the fixed 21-image benchmark under /kaggle/input or experiment_a.")
    candidates.sort()
    return candidates[0][2]


def validate_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "source": "fixed manifest values embedded from experiment_a/benchmark_v1.json",
            "sha256": BENCHMARK_MANIFEST_SHA256,
            "support_image": SUPPORT_IMAGE,
            "query_images": list(QUERY_IMAGES),
        }
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = json.loads(payload.decode("utf-8-sig"))
    actual_support = Path(manifest["support_image"]).name
    actual_queries = [Path(name).name for name in manifest["query_images"]]
    if digest != BENCHMARK_MANIFEST_SHA256:
        raise ValueError("benchmark_v1.json SHA-256 changed: {}".format(digest))
    if actual_support != SUPPORT_IMAGE or actual_queries != QUERY_IMAGES:
        raise ValueError("Embedded benchmark names do not match experiment_a/benchmark_v1.json.")
    return {
        "source": str(path),
        "sha256": digest,
        "support_image": actual_support,
        "query_images": actual_queries,
    }


def window_starts(length: int, window: int, stride: int) -> list[int]:
    if window <= 0 or stride <= 0 or length < window:
        raise ValueError("Invalid image/window/stride dimensions.")
    starts = list(range(0, length - window + 1, stride))
    final_start = length - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def make_window_boxes(width: int, height: int, window: int, stride: int) -> list[tuple[int, int, int, int]]:
    return [
        (x, y, x + window, y + window)
        for y in window_starts(height, window, stride)
        for x in window_starts(width, window, stride)
    ]


def reconstruct_scale(
    image_height: int,
    image_width: int,
    boxes: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild uniform window scores with rectangle differences and verify coverage."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have columns x0,y0,x1,y1")
    if scores.ndim != 2 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have one row per box")
    class_count = scores.shape[1]
    score_diff = np.zeros((class_count, image_height + 1, image_width + 1), dtype=np.float64)
    count_diff = np.zeros((image_height + 1, image_width + 1), dtype=np.int32)
    for box, score in zip(boxes, scores):
        x0, y0, x1, y1 = (int(x) for x in box)
        if not (0 <= x0 < x1 <= image_width and 0 <= y0 < y1 <= image_height):
            raise ValueError("Window box falls outside the source image: {}".format(box.tolist()))
        score_diff[:, y0, x0] += score
        score_diff[:, y0, x1] -= score
        score_diff[:, y1, x0] -= score
        score_diff[:, y1, x1] += score
        count_diff[y0, x0] += 1
        count_diff[y0, x1] -= 1
        count_diff[y1, x0] -= 1
        count_diff[y1, x1] += 1

    summed = np.cumsum(np.cumsum(score_diff, axis=1), axis=2)[:, :image_height, :image_width]
    coverage = np.cumsum(np.cumsum(count_diff, axis=0), axis=1)[:image_height, :image_width]
    if np.any(coverage <= 0):
        raise ValueError("Window grid left {} source pixels uncovered.".format(int(np.count_nonzero(coverage <= 0))))
    rebuilt = (summed / coverage[None, :, :]).astype(np.float32)
    if not np.isfinite(rebuilt).all():
        raise ValueError("Reconstructed response contains non-finite values.")
    return rebuilt, coverage


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size != s.size or not np.isfinite(s).all():
        raise ValueError("AP inputs must be same-size and finite.")
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    sorted_positive = y[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), y.size - 1]
    true_positive = np.cumsum(sorted_positive, dtype=np.int64)[ends]
    seen = ends + 1
    precision = true_positive / seen
    recall = true_positive / positives
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def precision_at_recall(y_true: np.ndarray, scores: np.ndarray, recall_target: float = 0.95) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return {"precision": None, "threshold": None, "recall": None}
    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    sorted_positive = y[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), y.size - 1]
    true_positive = np.cumsum(sorted_positive, dtype=np.int64)[ends]
    recall = true_positive / positives
    eligible = np.flatnonzero(recall >= recall_target)
    if eligible.size == 0:
        return {"precision": None, "threshold": None, "recall": None}
    # The required diagnostic uses the highest score threshold that still reaches target recall.
    selected = int(eligible[0])
    seen = int(ends[selected] + 1)
    return {
        "precision": float(true_positive[selected] / seen),
        "threshold": float(sorted_scores[ends[selected]]),
        "recall": float(recall[selected]),
    }


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size != s.size or not np.isfinite(s).all():
        raise ValueError("AUROC inputs must be same-size and finite.")
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    sorted_positive = y[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]) + 1]
    ends = np.r_[starts[1:], y.size]
    group_sizes = ends - starts
    group_positive = np.add.reduceat(sorted_positive, starts).astype(np.int64)
    average_rank = (starts + 1 + ends) / 2.0
    rank_sum = float(np.sum(group_positive * average_rank))
    u_stat = rank_sum - positives * (positives + 1) / 2.0
    return float(u_stat / (positives * negatives))


def _ensure_open_clip() -> None:
    try:
        import open_clip  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "open_clip_torch==2.24.0"])


def download_checkpoint(path: Path) -> None:
    if path.is_file() and path.stat().st_size == EXPECTED_CHECKPOINT_SIZE:
        if sha256_file(path) == CHECKPOINT_SHA256:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        import requests
    except ImportError as error:
        raise RuntimeError("requests is required to download the official GeoRSCLIP checkpoint.") from error
    with requests.get(CHECKPOINT_URL, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with partial.open("wb") as target:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if block:
                    target.write(block)
    size = partial.stat().st_size
    digest = sha256_file(partial)
    if size != EXPECTED_CHECKPOINT_SIZE or digest != CHECKPOINT_SHA256:
        raise ValueError("Official checkpoint integrity check failed: size={}, sha256={}".format(size, digest))
    partial.replace(path)


def load_geo_rsclip(checkpoint_path: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    _ensure_open_clip()
    import open_clip
    import torch

    # This reproduces the official OpenAI CLIP ViT-B/32 architecture while loading only RS5M weights.
    model = open_clip.create_model("ViT-B/32", pretrained=None, force_quick_gelu=True)
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError("Official checkpoint must contain a direct model state_dict.")
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError("Refusing partial GeoRSCLIP weights: missing={}, unexpected={}".format(missing, unexpected))
    model.eval().requires_grad_(False).to(device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("GeoRSCLIP must remain frozen and in eval mode.")

    visual_block = model.visual.transformer.resblocks[0]
    activation = type(visual_block.mlp.gelu).__name__
    if activation != "QuickGELU":
        raise RuntimeError("Expected the official OpenAI ViT-B/32 QuickGELU activation, found {}.".format(activation))
    model_config = open_clip.get_model_config("ViT-B/32") or {}
    details = {
        "open_clip_version": getattr(open_clip, "__version__", "unknown"),
        "architecture_name": "ViT-B/32",
        "architecture_config": model_config,
        "activation": activation,
        "tokenizer": "open_clip.tokenize, context_length=77",
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "model_training": bool(model.training),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "device": str(device),
    }
    return model, details


def official_preprocess() -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


def encode_text_classes(model: Any, device: Any) -> np.ndarray:
    import open_clip
    import torch
    import torch.nn.functional as functional

    prompts = [prompt for _, prompt_list in fixed_text_config()["expanded_prompts"].items() for prompt in prompt_list]
    tokens = open_clip.tokenize(prompts, context_length=77).to(device)
    with torch.inference_mode():
        embeddings = functional.normalize(model.encode_text(tokens).float(), dim=-1)
    embeddings = embeddings.reshape(len(CLASSES), len(TEMPLATES), -1)
    class_features = functional.normalize(embeddings.mean(dim=1), dim=-1)
    result = class_features.cpu().numpy().astype(np.float32)
    if result.shape[0] != len(CLASSES) or not np.isfinite(result).all():
        raise RuntimeError("Text embeddings are missing or non-finite.")
    return result


def encode_windows(
    model: Any,
    preprocess: Any,
    text_features: Any,
    windows: list[tuple[Image.Image, tuple[int, int, int, int]]],
    device: Any,
    batch_size: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    all_scores: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        batch = torch.stack([preprocess(image) for image, _ in windows[start : start + batch_size]])
        batch = batch.to(device, non_blocking=True)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = model.encode_image(batch).float()
            else:
                image_features = model.encode_image(batch).float()
            image_features = functional.normalize(image_features, dim=-1)
            scores = image_features @ text_features.T
        output = scores.detach().to(device="cpu", dtype=torch.float32).numpy()
        if not np.isfinite(output).all():
            raise RuntimeError("GeoRSCLIP emitted non-finite window scores.")
        all_scores.append(output)
    return np.concatenate(all_scores, axis=0)


def image_windows(image: Image.Image, size: int) -> list[tuple[Image.Image, tuple[int, int, int, int]]]:
    boxes = make_window_boxes(image.width, image.height, size, WINDOW_STRIDE)
    return [(image.crop(box), box) for box in boxes]


def run_inference(args: argparse.Namespace) -> Path:
    import torch

    data_root = discover_data_root(args.data_root)
    manifest_info = validate_manifest(args.manifest)
    images_dir = data_root / "images"
    output_dir = args.output_dir
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    frozen_config = fixed_experiment_config()
    frozen_config["runtime_data_root"] = str(data_root)
    frozen_config["manifest_validation"] = manifest_info
    frozen_config["inference_created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(output_dir / "experiment_config.json", frozen_config)
    config_sha = sha256_file(output_dir / "experiment_config.json")
    write_json(output_dir / "prompts.json", fixed_text_config())

    checkpoint_path = args.checkpoint or (Path("/kaggle/working/model") / CHECKPOINT_NAME)
    download_checkpoint(checkpoint_path)
    if sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA-256 does not match the official Hugging Face LFS object.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but Kaggle reports no available GPU.")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("The full 20-query E1 run requires a CUDA GPU; pass --allow-cpu only for a smoke run.")

    model, model_details = load_geo_rsclip(checkpoint_path, device)
    model_details["checkpoint_path"] = str(checkpoint_path)
    model_details["checkpoint_sha256"] = sha256_file(checkpoint_path)
    model_details["checkpoint_size_bytes"] = checkpoint_path.stat().st_size
    model_details["state_dict_parameter_keys"] = len(model.state_dict())
    write_json(output_dir / "model_load.json", model_details)

    text_features_np = encode_text_classes(model, device)
    np.save(output_dir / "text_class_embeddings.npy", text_features_np)
    text_features = torch.from_numpy(text_features_np).to(device=device, dtype=torch.float32)
    preprocess = official_preprocess()

    support_path = images_dir / SUPPORT_IMAGE
    with Image.open(support_path) as raw_support:
        support_image = raw_support.convert("RGB")
    if support_image.size != (1024, 1024):
        raise ValueError("Support image must be 1024x1024, got {}.".format(support_image.size))
    support_check: dict[str, Any] = {}
    for size in WINDOW_SIZES:
        first_window = image_windows(support_image, size)[:1]
        check_score = encode_windows(model, preprocess, text_features, first_window, device, 1)
        support_check[str(size)] = {
            "window_count": len(first_window),
            "finite": bool(np.isfinite(check_score).all()),
            "score_min": float(check_score.min()),
            "score_max": float(check_score.max()),
        }
        if not support_check[str(size)]["finite"]:
            raise RuntimeError("Support image basic-response check failed.")
    write_json(output_dir / "support_smoke_check.json", {
        "support_image": SUPPORT_IMAGE,
        "support_mask_read": False,
        "query_masks_read": False,
        "responses": support_check,
    })

    if len(QUERY_IMAGES) != 20 or len(set(QUERY_IMAGES)) != 20 or SUPPORT_IMAGE in QUERY_IMAGES:
        raise ValueError("The fixed benchmark must contain one support and 20 unique queries.")
    index: list[dict[str, Any]] = []
    for image_index, image_name in enumerate(QUERY_IMAGES, start=1):
        image_path = images_dir / image_name
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if image.size != (1024, 1024):
            raise ValueError("{} must be 1024x1024, got {}.".format(image_name, image.size))

        all_boxes: list[tuple[int, int, int, int, int]] = []
        all_window_scores: list[np.ndarray] = []
        scale_responses: list[np.ndarray] = []
        coverage_counts: list[dict[str, int]] = []
        for scale_index, size in enumerate(WINDOW_SIZES):
            windows = image_windows(image, size)
            boxes = [box for _, box in windows]
            scores = encode_windows(model, preprocess, text_features, windows, device, args.batch_size)
            for box in boxes:
                all_boxes.append((scale_index, *box))
            all_window_scores.append(scores)
            response, coverage = reconstruct_scale(
                image_height=image.height,
                image_width=image.width,
                boxes=np.asarray(boxes, dtype=np.int32),
                scores=scores,
            )
            scale_responses.append(response)
            coverage_counts.append({
                "windows": len(boxes),
                "minimum_pixel_coverage": int(coverage.min()),
                "maximum_pixel_coverage": int(coverage.max()),
                "uncovered_pixels": int(np.count_nonzero(coverage == 0)),
            })

        scale_array = np.stack(scale_responses).astype(np.float32)
        class_responses = scale_array.mean(axis=0, dtype=np.float64).astype(np.float32)
        e1_a = class_responses[0].copy()
        strongest_background_index = np.argmax(class_responses[1:], axis=0).astype(np.uint8) + 1
        strongest_background_score = np.max(class_responses[1:], axis=0).astype(np.float32)
        e1_b = (e1_a - strongest_background_score).astype(np.float32)
        if not all(np.isfinite(arr).all() for arr in (scale_array, class_responses, e1_a, e1_b, strongest_background_score)):
            raise RuntimeError("{} contains non-finite reconstructed class scores.".format(image_name))
        if not np.allclose(e1_a, class_responses[0], rtol=0, atol=0):
            raise RuntimeError("E1-A formula check failed.")
        expected_e1_b = class_responses[0] - np.max(class_responses[1:], axis=0)
        if not np.allclose(e1_b, expected_e1_b, rtol=1e-6, atol=1e-7):
            raise RuntimeError("E1-B formula check failed.")

        query_dir = raw_dir / Path(image_name).stem
        query_dir.mkdir(parents=True, exist_ok=True)
        boxes_with_scale = np.asarray(all_boxes, dtype=np.int32)
        window_scores = np.concatenate(all_window_scores, axis=0).astype(np.float32)
        score_path = query_dir / "scores.npz"
        np.savez_compressed(
            score_path,
            e1_a=e1_a,
            e1_b=e1_b,
            class_responses=class_responses,
            class_responses_by_scale=scale_array,
            strongest_background_index=strongest_background_index,
            strongest_background_score=strongest_background_score,
            window_boxes_scale_xyxy=boxes_with_scale,
            window_cosines=window_scores,
            class_order=np.asarray([name for name, _ in CLASSES]),
            window_sizes=np.asarray(WINDOW_SIZES, dtype=np.int32),
        )
        record = {
            "image": image_name,
            "size": [image.width, image.height],
            "window_count": int(window_scores.shape[0]),
            "scales": coverage_counts,
            "arrays": {
                "e1_a": ["H", "W"],
                "e1_b": ["H", "W"],
                "class_responses": ["7", "H", "W"],
                "class_responses_by_scale": ["2", "7", "H", "W"],
                "strongest_background_index": ["H", "W"],
                "strongest_background_score": ["H", "W"],
                "window_boxes_scale_xyxy": ["N", "5"],
                "window_cosines": ["N", "7"],
            },
            "score_ranges": {
                "e1_a": [float(e1_a.min()), float(e1_a.max())],
                "e1_b": [float(e1_b.min()), float(e1_b.max())],
            },
            "file": str(score_path.relative_to(output_dir)),
            "sha256": sha256_file(score_path),
        }
        index.append(record)
        print("[{}/20] saved {} ({:,} windows)".format(image_index, image_name, len(window_scores)), flush=True)

    write_json(output_dir / "inference_index.json", index)
    completion = {
        "status": "complete",
        "query_count": len(index),
        "query_images": list(QUERY_IMAGES),
        "config_sha256": config_sha,
        "model_load_sha256": sha256_file(output_dir / "model_load.json"),
        "text_embeddings_sha256": sha256_file(output_dir / "text_class_embeddings.npy"),
        "query_masks_read": False,
        "support_mask_read": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [
            {"path": item["file"], "sha256": item["sha256"]}
            for item in index
        ],
    }
    write_json(output_dir / "inference_complete.json", completion)
    print("INFERENCE COMPLETE: {} queries; masks were not read.".format(len(index)), flush=True)
    return output_dir


def load_query_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        mask = np.asarray(source.convert("L"), dtype=np.uint8)
    values = set(int(x) for x in np.unique(mask))
    if not values.issubset({0, 12}):
        raise ValueError("Unexpected query mask values in {}: {}".format(path, sorted(values)))
    if mask.shape != (1024, 1024):
        raise ValueError("Query mask must be 1024x1024: {}".format(path))
    return (mask == 12).astype(np.uint8)


def load_dino_cam(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        if "S_fg_upsampled" not in archive:
            raise KeyError("DINO CAM file lacks S_fg_upsampled: {}".format(path))
        values = np.asarray(archive["S_fg_upsampled"], dtype=np.float32)
    if values.shape != (1024, 1024) or not np.isfinite(values).all():
        raise ValueError("Invalid DINO S_fg map: {}".format(path))
    return values


def load_dino_prediction(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        values = np.asarray(source.convert("L"), dtype=np.uint8)
    if values.shape != (1024, 1024):
        raise ValueError("Invalid DINO block-24 prediction shape: {}".format(path))
    return (values > 0).astype(np.uint8)


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(values.max()),
    }


def metric_rows(
    labels_by_image: dict[str, np.ndarray],
    scores_by_image: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ap_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    methods = ["E1-A", "E1-B", "DINO-S_fg"]
    for method in methods:
        per_image: list[float] = []
        p95_precision: list[float] = []
        p95_recall: list[float] = []
        all_labels: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        for image_name in QUERY_IMAGES:
            labels = labels_by_image[image_name].reshape(-1)
            values = scores_by_image[image_name][method].reshape(-1)
            positives = int(labels.sum())
            negatives = int(labels.size - positives)
            ap = average_precision(labels, values)
            p95 = precision_at_recall(labels, values, 0.95)
            if p95["precision"] is not None:
                p95_precision.append(float(p95["precision"]))
                p95_recall.append(float(p95["recall"]))
            ap_rows.append({
                "image": image_name,
                "method": method,
                "positive_pixels": positives,
                "negative_pixels": negatives,
                "ap": ap,
                "ap_status": "ok" if ap is not None else "undefined_no_positive_pixels",
                "p_at_recall_95": p95["precision"],
                "threshold_at_recall_95": p95["threshold"],
                "actual_recall_at_threshold": p95["recall"],
                "p_at_r95_status": "ok" if p95["precision"] is not None else "undefined_missing_positive_or_negative_group",
            })
            if ap is not None:
                per_image.append(ap)
            all_labels.append(labels)
            all_scores.append(values)
            for group_name, group in (("foreground", labels == 1), ("background", labels == 0)):
                stats = distribution(values[group])
                score_rows.append({"image": image_name, "method": method, "group": group_name, **stats})
        pooled_labels = np.concatenate(all_labels)
        pooled_scores = np.concatenate(all_scores)
        summaries[method] = {
            "macro_ap": float(np.mean(per_image)) if per_image else None,
            "macro_ap_valid_images": len(per_image),
            "macro_ap_total_images": len(QUERY_IMAGES),
            "micro_ap": average_precision(pooled_labels, pooled_scores),
            "micro_positive_pixels": int(pooled_labels.sum()),
            "micro_negative_pixels": int(pooled_labels.size - pooled_labels.sum()),
            "macro_p_at_r95": float(np.mean(p95_precision)) if p95_precision else None,
            "p_at_r95_valid_images": len(p95_precision),
            "macro_actual_recall_at_selected_threshold": float(np.mean(p95_recall)) if p95_recall else None,
            "p_at_r95_total_images": len(QUERY_IMAGES),
            "pixels_are_not_independent_samples": True,
            "score_scope": "raw model scores; no per-image display normalization was used",
        }

    for image_name in QUERY_IMAGES:
        a = next(row["ap"] for row in ap_rows if row["image"] == image_name and row["method"] == "E1-A")
        b = next(row["ap"] for row in ap_rows if row["image"] == image_name and row["method"] == "E1-B")
        for row in ap_rows:
            if row["image"] == image_name and row["method"] in ("E1-A", "E1-B"):
                row["B_minus_A_ap"] = (float(b - a) if a is not None and b is not None else None)
    deltas = [row["B_minus_A_ap"] for row in ap_rows if row["method"] == "E1-A" and row["B_minus_A_ap"] is not None]
    summaries["E1-B_vs_E1-A"] = {
        "improved_images": int(sum(delta > 1e-12 for delta in deltas)),
        "degraded_images": int(sum(delta < -1e-12 for delta in deltas)),
        "unchanged_images": int(sum(abs(delta) <= 1e-12 for delta in deltas)),
        "undefined_images": len(QUERY_IMAGES) - len(deltas),
        "mean_per_image_ap_delta": float(np.mean(deltas)) if deltas else None,
    }
    return ap_rows, score_rows, summaries


def dino_complementarity(
    labels_by_image: dict[str, np.ndarray],
    predictions_by_image: dict[str, np.ndarray],
    scores_by_image: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    methods = ["E1-A", "E1-B", "DINO-S_fg"]
    for image_name in QUERY_IMAGES:
        gt = labels_by_image[image_name].reshape(-1).astype(bool)
        pred = predictions_by_image[image_name].reshape(-1).astype(bool)
        tp = pred & gt
        fp = pred & ~gt
        for method in methods:
            score = scores_by_image[image_name][method].reshape(-1)
            auc = binary_auroc(tp[tp | fp].astype(np.uint8), score[tp | fp])
            tp_stats = distribution(score[tp])
            fp_stats = distribution(score[fp])
            rows.append({
                "image": image_name,
                "method": method,
                "dino_tp_pixels": int(tp.sum()),
                "dino_fp_pixels": int(fp.sum()),
                "tp_mean": tp_stats["mean"],
                "tp_median": tp_stats["median"],
                "tp_p05": tp_stats["p05"],
                "tp_p95": tp_stats["p95"],
                "fp_mean": fp_stats["mean"],
                "fp_median": fp_stats["median"],
                "fp_p05": fp_stats["p05"],
                "fp_p95": fp_stats["p95"],
                "tp_vs_fp_auroc": auc,
                "auroc_status": "ok" if auc is not None else "undefined_missing_tp_or_fp_group",
            })
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method and row["tp_vs_fp_auroc"] is not None]
        summaries[method] = {
            "macro_per_image_auroc": float(np.mean([row["tp_vs_fp_auroc"] for row in method_rows])) if method_rows else None,
            "valid_images": len(method_rows),
            "total_images": len(QUERY_IMAGES),
            "tp_pixels": int(sum(row["dino_tp_pixels"] for row in rows if row["method"] == method)),
            "fp_pixels": int(sum(row["dino_fp_pixels"] for row in rows if row["method"] == method)),
            "interpretation": "Pixels are nested in images; report per-image AUROC and macro mean without pixel-level significance claims.",
        }
    summaries["DINO-S_fg_definition"] = "Existing DINOv3 spatial_cam_v1 S_fg_upsampled pure foreground cosine response, not cosine_fg-cosine_bg."
    return rows, summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_score_map(
    score: np.ndarray,
    path: Path,
    title: str,
    display_min: float,
    display_max: float,
    cmap: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 6), dpi=140)
    artist = axis.imshow(score, cmap=cmap, vmin=display_min, vmax=display_max, interpolation="nearest")
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def evaluate_results(args: argparse.Namespace) -> Path:
    inference_dir = args.inference_dir
    completion_path = inference_dir / "inference_complete.json"
    config_path = inference_dir / "experiment_config.json"
    if not completion_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Evaluation requires completed inference results and their frozen config.")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or completion.get("query_count") != 20:
        raise RuntimeError("Inference completion marker is invalid or incomplete.")
    if completion.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("Frozen inference configuration hash does not match completion marker.")
    if completion.get("query_masks_read") is not False:
        raise RuntimeError("Inference marker must confirm that query masks were not read.")
    if completion.get("query_images") != QUERY_IMAGES:
        raise RuntimeError("Inference manifest differs from fixed benchmark_v1 query order.")
    if completion.get("model_load_sha256") != sha256_file(inference_dir / "model_load.json"):
        raise RuntimeError("Model loading record hash does not match the completion marker.")
    if completion.get("text_embeddings_sha256") != sha256_file(inference_dir / "text_class_embeddings.npy"):
        raise RuntimeError("Cached text embedding hash does not match the completion marker.")
    frozen_config = json.loads(config_path.read_text(encoding="utf-8"))
    if frozen_config.get("support_image") != SUPPORT_IMAGE or frozen_config.get("query_images") != QUERY_IMAGES:
        raise RuntimeError("Frozen config does not match benchmark_v1.")
    if frozen_config.get("text") != fixed_text_config():
        raise RuntimeError("Frozen prompt text changed from the fixed E1 configuration.")
    model_load = json.loads((inference_dir / "model_load.json").read_text(encoding="utf-8"))
    if model_load.get("missing_keys") or model_load.get("unexpected_keys") or model_load.get("activation") != "QuickGELU":
        raise RuntimeError("GeoRSCLIP model loading record reports partial or incorrect weights/configuration.")
    text_embeddings = np.load(inference_dir / "text_class_embeddings.npy")
    if text_embeddings.shape[0] != 7 or not np.isfinite(text_embeddings).all():
        raise RuntimeError("Cached seven-class text embeddings are missing or non-finite.")

    expected_score_paths = {"raw/{}/scores.npz".format(Path(name).stem) for name in QUERY_IMAGES}
    file_records = completion.get("files", [])
    if {item.get("path") for item in file_records} != expected_score_paths:
        raise RuntimeError("Completion marker does not enumerate exactly the 20 raw query score files.")
    for item in file_records:
        score_path = inference_dir / item["path"]
        if not score_path.is_file() or sha256_file(score_path) != item.get("sha256"):
            raise RuntimeError("Raw query score file is missing or fails SHA-256 verification: {}".format(item.get("path")))

    data_root = discover_data_root(args.data_root)
    manifest_info = validate_manifest(args.manifest)
    images_dir = data_root / "images"
    masks_dir = data_root / "mask"
    dino_cam_dir = args.dino_cam_dir
    dino_prediction_dir = args.dino_prediction_dir
    output_dir = args.output_dir or (inference_dir.parent / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate every saved model response before the separate mask-reading pass below.
    labels_by_image: dict[str, np.ndarray] = {}
    scores_by_image: dict[str, dict[str, np.ndarray]] = {}
    dino_predictions: dict[str, np.ndarray] = {}
    dino_cams: dict[str, np.ndarray] = {}
    class_maps: dict[str, np.ndarray] = {}
    for image_name in QUERY_IMAGES:
        stem = Path(image_name).stem
        score_path = inference_dir / "raw" / stem / "scores.npz"
        if not score_path.is_file():
            raise FileNotFoundError("Missing raw inference arrays: {}".format(score_path))
        with np.load(score_path) as archive:
            required = {
                "e1_a", "e1_b", "class_responses", "class_responses_by_scale",
                "strongest_background_index", "strongest_background_score",
                "window_boxes_scale_xyxy", "window_cosines", "class_order",
            }
            if not required.issubset(archive.files):
                raise KeyError("Missing required arrays in {}: {}".format(score_path, sorted(required - set(archive.files))))
            a = np.asarray(archive["e1_a"])
            b = np.asarray(archive["e1_b"])
            classes = np.asarray(archive["class_responses"])
            by_scale = np.asarray(archive["class_responses_by_scale"])
            bg_index = np.asarray(archive["strongest_background_index"])
            bg_score = np.asarray(archive["strongest_background_score"])
            boxes = np.asarray(archive["window_boxes_scale_xyxy"])
            window_scores = np.asarray(archive["window_cosines"])
            class_order = [str(value) for value in archive["class_order"].tolist()]
        expected_class_order = [name for name, _ in CLASSES]
        if class_order != expected_class_order:
            raise ValueError("Class order mismatch for {}.".format(image_name))
        if any(value.dtype != np.float32 for value in (a, b, classes, by_scale, bg_score, window_scores)):
            raise TypeError("Raw response arrays must be float32 for {}.".format(image_name))
        if bg_index.dtype != np.uint8 or boxes.dtype != np.int32:
            raise TypeError("Background indices/coordinate records have unexpected dtypes for {}.".format(image_name))
        if a.shape != (1024, 1024) or b.shape != a.shape or classes.shape != (7, 1024, 1024) or by_scale.shape != (2, 7, 1024, 1024):
            raise ValueError("Invalid raw response shapes for {}.".format(image_name))
        if bg_index.shape != a.shape or bg_score.shape != a.shape:
            raise ValueError("Invalid strongest-background map shapes for {}.".format(image_name))
        if not all(np.isfinite(x).all() for x in (a, b, classes, by_scale, bg_score, window_scores)):
            raise ValueError("Non-finite raw inference arrays for {}.".format(image_name))
        if not np.array_equal(a, classes[0]) or not np.allclose(b, classes[0] - classes[1:].max(axis=0), rtol=1e-6, atol=1e-7):
            raise ValueError("Saved A/B scores do not match the frozen formulas for {}.".format(image_name))
        expected_bg_index = np.argmax(classes[1:], axis=0).astype(np.uint8) + 1
        if not np.array_equal(bg_index, expected_bg_index) or not np.allclose(bg_score, classes[1:].max(axis=0), rtol=1e-6, atol=1e-7):
            raise ValueError("Strongest-background maps do not match class responses for {}.".format(image_name))
        if boxes.shape != (window_scores.shape[0], 5) or window_scores.shape[1] != 7:
            raise ValueError("Window record coordinates/scores do not align for {}.".format(image_name))
        if not np.array_equal(classes, by_scale.mean(axis=0, dtype=np.float64).astype(np.float32)):
            raise ValueError("Equal two-scale aggregation check failed for {}.".format(image_name))
        for scale_index, window_size in enumerate(WINDOW_SIZES):
            selected = boxes[:, 0] == scale_index
            scale_boxes = boxes[selected, 1:]
            scale_scores = window_scores[selected]
            expected_boxes = np.asarray(make_window_boxes(1024, 1024, window_size, WINDOW_STRIDE), dtype=np.int32)
            if scale_boxes.shape != expected_boxes.shape or set(map(tuple, scale_boxes.tolist())) != set(map(tuple, expected_boxes.tolist())):
                raise ValueError("{} scale {} window geometry differs from the fixed protocol.".format(image_name, window_size))
            rebuilt, coverage = reconstruct_scale(1024, 1024, scale_boxes, scale_scores)
            if np.any(coverage <= 0) or not np.allclose(rebuilt, by_scale[scale_index], rtol=1e-6, atol=1e-7):
                raise ValueError("{} scale {} coverage/reconstruction verification failed.".format(image_name, window_size))

        scores_by_image[image_name] = {"E1-A": a, "E1-B": b}
        class_maps[image_name] = classes
        dino_predictions[image_name] = load_dino_prediction(dino_prediction_dir / (stem + ".png"))
        dino_cams[image_name] = load_dino_cam(dino_cam_dir / stem / "scores.npz")
        scores_by_image[image_name]["DINO-S_fg"] = dino_cams[image_name]

    # Query masks are opened only after all 20 response files pass checksum, formula, and geometry checks.
    for image_name in QUERY_IMAGES:
        stem = Path(image_name).stem
        labels_by_image[image_name] = load_query_mask(masks_dir / (stem + "_label.bmp"))

    ap_rows, score_rows, summary = metric_rows(labels_by_image, scores_by_image)
    complementarity_rows, complementarity_summary = dino_complementarity(labels_by_image, dino_predictions, scores_by_image)
    write_csv(output_dir / "ap_by_image.csv", ap_rows)
    write_csv(output_dir / "foreground_background_score_distributions.csv", score_rows)
    write_csv(output_dir / "dino_tp_fp_complementarity.csv", complementarity_rows)

    labels_all = np.concatenate([labels_by_image[name].reshape(-1) for name in QUERY_IMAGES])
    method_ranges: dict[str, tuple[float, float]] = {}
    for method in ("E1-A", "E1-B"):
        method_ranges[method] = (
            min(float(scores_by_image[name][method].min()) for name in QUERY_IMAGES),
            max(float(scores_by_image[name][method].max()) for name in QUERY_IMAGES),
        )
    category_min = min(float(class_maps[name].min()) for name in QUERY_IMAGES)
    category_max = max(float(class_maps[name].max()) for name in QUERY_IMAGES)
    b_limit = max(abs(method_ranges["E1-B"][0]), abs(method_ranges["E1-B"][1]))
    if b_limit == 0:
        b_limit = 1e-6
    display_ranges = {
        "E1-A": {"min": method_ranges["E1-A"][0], "max": method_ranges["E1-A"][1], "policy": "one global range across all 20 queries; no per-image min-max"},
        "E1-B": {"min": -b_limit, "max": b_limit, "policy": "one symmetric global range across all 20 queries; centered at zero"},
        "all_classes": {"min": category_min, "max": category_max, "policy": "one global range across all 20 queries and all 7 classes"},
    }
    write_json(output_dir / "display_ranges.json", display_ranges)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images_output = output_dir / "images"
    panels_output = output_dir / "panels"
    text_examples_output = output_dir / "text_sensitivity"
    panels_output.mkdir(parents=True, exist_ok=True)
    text_examples_output.mkdir(parents=True, exist_ok=True)
    ap_lookup = {(row["image"], row["method"]): row["ap"] for row in ap_rows}
    for image_index, image_name in enumerate(QUERY_IMAGES):
        stem = Path(image_name).stem
        with Image.open(images_dir / image_name) as source:
            image = np.asarray(source.convert("RGB"))
        label = labels_by_image[image_name]
        classes = class_maps[image_name]
        a = scores_by_image[image_name]["E1-A"]
        b = scores_by_image[image_name]["E1-B"]
        image_dir = images_output / stem
        for method, values, bounds, cmap in (
            ("E1-A", a, method_ranges["E1-A"], "viridis"),
            ("E1-B", b, (-b_limit, b_limit), "coolwarm"),
        ):
            plot_score_map(values, image_dir / (method.lower().replace("-", "_") + "_raw.png"), method + " raw cosine response", bounds[0], bounds[1], cmap)
            overlay = plt.figure(figsize=(6, 6), dpi=140)
            axis = overlay.add_subplot(111)
            axis.imshow(image)
            axis.imshow(values, cmap=cmap, vmin=bounds[0], vmax=bounds[1], alpha=0.48, interpolation="nearest")
            axis.set_title(method + " overlay")
            axis.axis("off")
            overlay.tight_layout()
            overlay.savefig(image_dir / (method.lower().replace("-", "_") + "_overlay.png"), bbox_inches="tight")
            plt.close(overlay)

        figure, axes = plt.subplots(1, 5, figsize=(20, 4.8), dpi=120)
        axes[0].imshow(image)
        axes[0].set_title("Original")
        dino_color_path = EXPERIMENT_DIR.parent / "spatial_cam_v1" / "benchmark_v1_results" / "images" / stem / "cam_color.png"
        with Image.open(dino_color_path) as dino_image:
            axes[1].imshow(dino_image.convert("RGB"))
        axes[1].set_title("DINOv3 block 24 S_fg CAM")
        axes[2].imshow(a, cmap="viridis", vmin=method_ranges["E1-A"][0], vmax=method_ranges["E1-A"][1])
        axes[2].set_title("E1-A raw cosine")
        axes[3].imshow(b, cmap="coolwarm", vmin=-b_limit, vmax=b_limit)
        axes[3].set_title("E1-B PV minus strongest background")
        axes[4].imshow(image)
        axes[4].imshow(np.ma.masked_where(label == 0, label), cmap="Reds", vmin=0, vmax=1, alpha=0.7)
        axes[4].set_title("Query truth (PV=12)")
        for axis in axes:
            axis.axis("off")
        figure.suptitle("{} | AP A={:.4f} B={:.4f}".format(image_name, ap_lookup.get((image_name, "E1-A"), float("nan")), ap_lookup.get((image_name, "E1-B"), float("nan"))))
        figure.tight_layout()
        figure.savefig(panels_output / (stem + ".jpg"), bbox_inches="tight")
        if image_index < 3:
            text_figure, text_axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=130)
            for axis, class_index in zip(text_axes, (0, 1, 4)):
                artist = axis.imshow(classes[class_index], cmap="viridis", vmin=category_min, vmax=category_max)
                axis.set_title("{}: {}".format(CLASSES[class_index][0], CLASSES[class_index][1]))
                axis.axis("off")
            text_figure.colorbar(artist, ax=list(text_axes), fraction=0.025, pad=0.025, label="raw cosine")
            text_figure.suptitle("Fixed text-sensitivity example: {}".format(image_name))
            text_figure.savefig(text_examples_output / (stem + ".jpg"), bbox_inches="tight")
            plt.close(text_figure)
        plt.close(figure)

    # Deterministic summary: fixed first-three text examples, plus a 20-query index in manifest order.
    text_rows: list[dict[str, Any]] = []
    for image_name in QUERY_IMAGES:
        classes = class_maps[image_name]
        for class_index in (0, 1, 4):
            values = classes[class_index]
            text_rows.append({
                "image": image_name,
                "class": CLASSES[class_index][0],
                "description": CLASSES[class_index][1],
                "mean_raw_response": float(values.mean()),
                "median_raw_response": float(np.median(values)),
                "p95_raw_response": float(np.quantile(values, 0.95)),
                "max_raw_response": float(values.max()),
            })
    write_csv(output_dir / "text_sensitivity.csv", text_rows)

    overview, overview_axes = plt.subplots(5, 4, figsize=(16, 20), dpi=110)
    for index, (axis, image_name) in enumerate(zip(overview_axes.flat, QUERY_IMAGES)):
        values = scores_by_image[image_name]["E1-B"]
        axis.imshow(values, cmap="coolwarm", vmin=-b_limit, vmax=b_limit)
        a_ap = ap_lookup.get((image_name, "E1-A"))
        b_ap = ap_lookup.get((image_name, "E1-B"))
        axis.set_title("{}\nAP A {:.3f} | B {:.3f}".format(Path(image_name).stem, a_ap if a_ap is not None else float("nan"), b_ap if b_ap is not None else float("nan")), fontsize=8)
        axis.axis("off")
    overview.suptitle("E1-B raw response | one global color range | fixed benchmark order", y=0.998)
    overview.tight_layout(rect=[0, 0, 1, 0.98])
    overview.savefig(output_dir / "overview.jpg", bbox_inches="tight")
    plt.close(overview)

    for image_name in QUERY_IMAGES:
        del scores_by_image[image_name]
    summary.update({
        "inference_config_sha256": completion["config_sha256"],
        "dataset_images": len(QUERY_IMAGES),
        "evaluation_masks": "read only after inference_complete.json and all score files were verified",
        "mask_values_observed": "0=background, 12=PV",
        "text_sensitivity_examples": QUERY_IMAGES[:3],
        "text_sensitivity_all_queries_indexed": True,
        "raw_response_not_probability": True,
        "pixel_level_significance_claims": False,
        "DINO_baseline": "DINOv3 block-24 predictions are thresholded binary masks; continuous reference is existing pure S_fg CAM only.",
        "display_ranges": display_ranges,
        "manifest_validation": manifest_info,
    })
    summary["DINO_TP_FP_complementarity"] = complementarity_summary
    summary["input_audit"] = {
        "manifest_query_images": list(QUERY_IMAGES),
        "evaluated_query_images": len(labels_by_image),
        "query_mask_positive_pixels": int(labels_all.sum()),
        "query_mask_negative_pixels": int(labels_all.size - labels_all.sum()),
        "DINO_prediction_files": len(dino_predictions),
        "all_class_responses_finite": True,
    }
    write_json(output_dir / "summary.json", summary)
    ap_by_image = {
        (row["image"], row["method"]): row["ap"]
        for row in ap_rows
    }
    b_deltas = [
        (row["B_minus_A_ap"], row["image"])
        for row in ap_rows
        if row["method"] == "E1-A" and row["B_minus_A_ap"] is not None
    ]
    b_deltas.sort()
    report_lines = [
        "# GeoRSCLIP Semantic CAM E1 Results",
        "",
        "Frozen GeoRSCLIP ViT-B/32, fixed prompts, 128/256 pixel windows, and 64 pixel stride. Scores are raw cosine responses or PV-minus-strongest-background differences; they are not probabilities or pixel-accurate masks.",
        "",
        "## AP",
        "",
        "| Method | Macro AP | Valid images | Micro AP | Macro P@R95 | Valid P@R95 images |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("E1-A", "E1-B", "DINO-S_fg"):
        values = summary[method]
        report_lines.append(
            "| {} | {} | {}/{} | {} | {} | {}/{} |".format(
                method,
                "{:.6f}".format(values["macro_ap"]) if values["macro_ap"] is not None else "undefined",
                values["macro_ap_valid_images"], values["macro_ap_total_images"],
                "{:.6f}".format(values["micro_ap"]) if values["micro_ap"] is not None else "undefined",
                "{:.6f}".format(values["macro_p_at_r95"]) if values["macro_p_at_r95"] is not None else "undefined",
                values["p_at_r95_valid_images"], values["p_at_r95_total_images"],
            )
        )
    comparison = summary["E1-B_vs_E1-A"]
    report_lines += [
        "",
        "E1-B AP changes: **{} improved, {} declined, {} unchanged, {} undefined**; mean per-image change **{}**.".format(
            comparison["improved_images"], comparison["degraded_images"], comparison["unchanged_images"], comparison["undefined_images"],
            "{:.6f}".format(comparison["mean_per_image_ap_delta"]) if comparison["mean_per_image_ap_delta"] is not None else "undefined",
        ),
        "",
        "P@R95 uses the highest raw-score threshold that still reaches recall 0.95. The selected threshold and achieved recall are diagnostics only and were not fed back to inference.",
        "",
        "## DINO TP/FP Complementarity",
        "",
        "DINO TP and FP are regions in the existing thresholded block-24 prediction. AUC compares E1 scores over those predicted-positive pixels. The DINO score reference is the saved pure foreground `S_fg_upsampled`, not a foreground-minus-background score.",
        "",
        "| Score | Macro per-image TP-vs-FP AUROC | Valid images | TP pixels | FP pixels |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("E1-A", "E1-B", "DINO-S_fg"):
        values = complementarity_summary[method]
        report_lines.append(
            "| {} | {} | {}/{} | {} | {} |".format(
                method,
                "{:.6f}".format(values["macro_per_image_auroc"]) if values["macro_per_image_auroc"] is not None else "undefined",
                values["valid_images"], values["total_images"], values["tp_pixels"], values["fp_pixels"],
            )
        )
    report_lines += [
        "",
        "Pixels are nested within images. Counts describe evaluated regions and are not independent sample sizes; no pixel-level significance test is reported.",
        "",
        "## Example Panels",
        "",
        "Every query panel is available under `panels/`. The files below are selected mechanically from the per-image E1-B minus E1-A AP change, not by visual preference:",
        "",
    ]
    if b_deltas:
        for label, item in (("Largest gain", b_deltas[-1]), ("Median change", b_deltas[len(b_deltas) // 2]), ("Largest decline", b_deltas[0])):
            delta, image_name = item
            report_lines.append("- {}: `{}` (AP change {:+.6f}; `panels/{}.jpg`).".format(label, image_name, delta, Path(image_name).stem))
    report_lines += [
        "",
        "## Outputs",
        "",
        "- `ap_by_image.csv`: per-query AP and P@R95 with selected threshold and achieved recall.",
        "- `foreground_background_score_distributions.csv`: raw score distributions for positive and negative pixels.",
        "- `dino_tp_fp_complementarity.csv`: per-image TP/FP distributions and AUROC.",
        "- `text_sensitivity.csv`: fixed PV/cropland/water responses on all 20 queries; first-three panels are in `text_sensitivity/`.",
        "- `overview.jpg`, `panels/`, and `images/`: fixed-range overview, truth comparisons, raw maps, and overlays.",
        "- `display_ranges.json`: global color ranges shared across all queries; no per-image stretch.",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_json(output_dir / "evaluation_complete.json", {
        "status": "complete",
        "query_count": len(QUERY_IMAGES),
        "config_sha256": completion["config_sha256"],
        "metrics": ["per-image AP", "macro AP", "micro AP", "P@R95", "DINO TP/FP AUROC"],
        "threshold_feedback_to_inference": False,
    })
    print("OFFLINE EVALUATION COMPLETE: {} queries.".format(len(QUERY_IMAGES)), flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen GeoRSCLIP E1 inference or its separate offline evaluation.")
    parser.add_argument("--mode", choices=("inference", "evaluate"), default="inference")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=EXPERIMENT_DIR.parent / "experiment_a" / "benchmark_v1.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--inference-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dino-cam-dir", type=Path, default=EXPERIMENT_DIR.parent / "spatial_cam_v1" / "benchmark_v1_results" / "images")
    parser.add_argument("--dino-prediction-dir", type=Path, default=EXPERIMENT_DIR.parent / "kaggle_run_output" / "layer_selection_v1" / "predictions" / "block_24_only")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-cpu", action="store_true", help="Explicitly allow small smoke runs on CPU.")
    args = parser.parse_args()
    if args.mode == "inference":
        if args.output_dir is None:
            args.output_dir = Path("/kaggle/working/georsclip_semantic_cam_e1/inference") if Path("/kaggle/working").is_dir() else EXPERIMENT_DIR / "inference"
    else:
        if args.inference_dir is None:
            args.inference_dir = EXPERIMENT_DIR / "inference"
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "inference":
        run_inference(args)
    else:
        evaluate_results(args)


if __name__ == "__main__":
    main()
