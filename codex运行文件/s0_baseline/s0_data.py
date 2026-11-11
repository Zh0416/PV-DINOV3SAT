"""PV-512 inventory, locked query selection, and frozen whole-image tokens."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch


SUPPORT_ID = "0807_Q_934ae716b5714f_r00024_c00076"
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
WEIGHT_NAME = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)
SEED = 42
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def find_dataset_root(search_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not (root / "manifest.csv").is_file():
            raise FileNotFoundError(f"Missing PV-512 manifest.csv: {root}")
        return root
    matches = sorted(path.parent for path in search_root.rglob("manifest.csv")
                     if (path.parent / "正样本").is_dir() and (path.parent / "困难负样本").is_dir())
    if len(matches) != 1:
        raise ValueError(f"Expected one PV-512 root under {search_root}; found {matches}")
    return matches[0].resolve()


def find_weights(search_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file() or path.name != WEIGHT_NAME:
            raise FileNotFoundError(f"Expected official SAT checkpoint {WEIGHT_NAME}: {path}")
        return path
    matches = sorted(search_root.rglob(WEIGHT_NAME))
    if len(matches) != 1:
        raise ValueError(f"Expected one SAT checkpoint under {search_root}; found {matches}")
    return matches[0].resolve()


def _find_subdirectory(category: Path, names: tuple[str, ...]) -> Path:
    matches = [path for path in category.iterdir() if path.is_dir() and path.name.lower() in names]
    if len(matches) != 1:
        raise ValueError(f"Expected one {names} folder under {category}; found {matches}")
    return matches[0]


def _mask_id(stem: str) -> str:
    for suffix in ("_label", "_mask", "_gt"):
        if stem.lower().endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _category_files(root: Path, category: str) -> dict[str, dict]:
    folder = root / category
    images = _find_subdirectory(folder, ("images", "image", "imgs"))
    masks = _find_subdirectory(folder, ("labels", "label", "masks", "mask", "annotations"))
    image_files = [p for p in images.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    mask_files = [p for p in masks.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    image_map = {p.stem: p for p in image_files}
    mask_map = {_mask_id(p.stem): p for p in mask_files}
    if len(image_map) != len(image_files) or len(mask_map) != len(mask_files):
        raise ValueError(f"Duplicate image/mask IDs in {category}.")
    if image_map.keys() != mask_map.keys():
        raise ValueError(f"Unpaired image/mask IDs in {category}.")
    return {sample_id: {"id": sample_id, "category": category,
                        "image": image_map[sample_id].relative_to(root).as_posix(),
                        "mask": mask_map[sample_id].relative_to(root).as_posix()}
            for sample_id in image_map}


def scene_key(scene: str) -> str:
    """Compare source raster names consistently across Windows/Kaggle mounts."""
    value = scene.strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].casefold()
    if not value or value in (".", ".."):
        raise ValueError("A real source_scene is required; no ID-derived scene is inferred.")
    return value


def _manifest_scenes(path: Path, known_ids: set[str], scene_column: str) -> dict[str, str]:
    found = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or scene_column not in reader.fieldnames:
            raise ValueError(f"PV-512 manifest requires the real source-scene column {scene_column!r}; use --source-scene-column for an explicitly identified equivalent.")
        for row in reader:
            matches = set()
            for value in row.values():
                if value is None:
                    continue
                name = value.strip().replace("\\", "/").rsplit("/", 1)[-1]
                stem = _mask_id(Path(name).stem)
                if stem in known_ids:
                    matches.add(stem)
            if len(matches) > 1:
                raise ValueError(f"Ambiguous IDs in manifest row: {matches}")
            if set(found) & matches:
                raise ValueError(f"Duplicate sample in manifest: {matches}")
            for sample_id in matches:
                scene = (row[scene_column] or "").strip()
                scene_key(scene)
                found[sample_id] = scene
    return found


def inventory(root: Path, scene_column: str = "source_scene") -> dict[str, dict]:
    positive = _category_files(root, "正样本")
    hard = _category_files(root, "困难负样本")
    if positive.keys() & hard.keys():
        raise ValueError("IDs occur in both positive and hard-negative classes.")
    if SUPPORT_ID not in positive:
        raise ValueError(f"Fixed support ID missing from positive class: {SUPPORT_ID}")
    entries = {**positive, **hard}
    scenes = _manifest_scenes(root / "manifest.csv", set(entries), scene_column)
    if set(scenes) != set(entries):
        missing = sorted(set(entries) - set(scenes))
        raise ValueError(f"PV-512 manifest does not list all image IDs; first missing: {missing[:5]}")
    return {sample_id: {**entry, "source_scene": scenes[sample_id],
                        "source_scene_column": scene_column} for sample_id, entry in entries.items()}


def _selection_key(category: str, sample_id: str) -> str:
    return hashlib.sha256(f"s0-pv512-v1|{SEED}|{category}|{sample_id}".encode()).hexdigest()


def _selected_entry(root: Path, entry: dict) -> dict:
    result = dict(entry)
    for field in ("image", "mask"):
        result[field + "_sha256"] = sha256_file(root / entry[field])
    return result


def lock_selection(root: Path, output: Path, entries: dict[str, dict], reference_lock: Path | None = None) -> dict:
    candidates = sorted((e for e in entries.values() if e["category"] == "正样本" and e["id"] != SUPPORT_ID),
                        key=lambda e: (_selection_key("positive", e["id"]), e["id"]))
    hard = sorted((e for e in entries.values() if e["category"] == "困难负样本"),
                  key=lambda e: (_selection_key("hard_negative", e["id"]), e["id"]))
    if len(candidates) < 1199 or len(hard) < 50:
        raise ValueError("PV-512 does not contain 1199 positive and 50 hard-negative query candidates.")
    selected = [entries[SUPPORT_ID], *candidates[:1199], *hard[:50]]
    manifest_path = root / "manifest.csv"
    summary_path = root / "dataset_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    proposed = {
        "schema": "s0-pv512-selection-v1", "selection_origin": "deterministic_generated_from_manifest",
        "seed": SEED, "support_id": SUPPORT_ID,
        "rule": "Sort eligible IDs by SHA256('s0-pv512-v1|42|<positive|hard_negative>|<id>'), tie by ID; choose first 1199 positive excluding support and first 50 hard negative.",
        "positive_budget_including_support": 1200, "positive_query_count": 1199,
        "hard_negative_query_count": 50,
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_summary_sha256": sha256_file(summary_path),
        "source_counts": {"positive": sum(e["category"] == "正样本" for e in entries.values()),
                          "hard_negative": sum(e["category"] == "困难负样本" for e in entries.values())},
        "selected": [_selected_entry(root, e) for e in selected],
    }
    proposed["selection_sha256"] = canonical_sha256(proposed)
    if reference_lock is not None:
        reference = json.loads(reference_lock.read_text(encoding="utf-8"))
        if reference != proposed:
            raise ValueError("PV-512 selection/source digests differ from the committed reference lock.")
    lock_path = output / "s0_selection.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing != proposed or existing.get("selection_sha256") != canonical_sha256(
                {k: v for k, v in existing.items() if k != "selection_sha256"}):
            raise ValueError("Existing S0 selection or source files differ; refusing to resume.")
    else:
        write_json(lock_path, proposed)
        with (output / "s0_selection_ids.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("role", "id", "category", "source_scene", "image", "mask", "image_sha256", "mask_sha256"))
            for index, entry in enumerate(proposed["selected"]):
                writer.writerow(("support" if index == 0 else "positive_query" if index < 1200 else "hard_negative_query",
                                 entry["id"], entry["category"], entry["source_scene"], entry["image"], entry["mask"],
                                 entry["image_sha256"], entry["mask_sha256"]))
    return proposed


def read_raster(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    with Image.open(path) as image:
        return np.asarray(image).copy()


def checked_rgb(path: Path) -> np.ndarray:
    pixels = read_raster(path)
    if pixels.shape != (512, 512, 3) or pixels.dtype != np.uint8:
        raise ValueError(f"Expected native uint8 RGB 512x512 image: {path}")
    return pixels


def checked_mask(path: Path, positive: bool | None = None) -> np.ndarray:
    pixels = read_raster(path)
    if pixels.shape != (512, 512) or not np.isin(pixels, (0, 1)).all():
        raise ValueError(f"Expected 512x512 0/1 mask: {path}")
    if positive is not None and bool(pixels.any()) != positive:
        raise ValueError(f"Foreground/empty mask disagrees with sample class: {path}")
    return pixels.astype(np.uint8)


def ensure_dino_repo(path: Path) -> Path:
    created = not path.exists()
    if created:
        subprocess.run(["git", "clone", "https://github.com/facebookresearch/dinov3.git", str(path)], check=True)
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if commit != DINOV3_COMMIT:
        if not created:
            raise ValueError(f"Existing --dino-repo must already be at {DINOV3_COMMIT}; found {commit}.")
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", DINOV3_COMMIT], check=True)
    if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("DINO source has modified tracked files; use a clean pinned checkout.")
    return path


def load_dino(repo: Path, weights: Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    weight_sha = sha256_file(weights)
    if weights.name != WEIGHT_NAME or not weight_sha.startswith("eadcf0ff"):
        raise ValueError("Official DINOv3-SAT checkpoint identity mismatch.")
    if str(repo.resolve()) not in sys.path:
        sys.path.insert(0, str(repo.resolve()))
    from dinov3.hub.backbones import dinov3_vitl16
    model = dinov3_vitl16(pretrained=True, weights=str(weights))
    model.eval().requires_grad_(False).to(device)
    return model, weight_sha


@torch.inference_mode()
def encode_image(model: torch.nn.Module, path: Path, device: torch.device) -> np.ndarray:
    pixels = checked_rgb(path)
    image = torch.from_numpy(pixels).permute(2, 0, 1).to(device, dtype=torch.float32) / 255
    image = (image - image.new_tensor(SAT_MEAN)[:, None, None]) / image.new_tensor(SAT_STD)[:, None, None]
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        layers = model.get_intermediate_layers(image[None], n=(23,), reshape=True, norm=True)
    if len(layers) != 1 or layers[0].shape != (1, 1024, 32, 32):
        raise ValueError("DINO block-24 tokens must have shape [1,1024,32,32].")
    result = layers[0][0].to(dtype=torch.float16, device="cpu").numpy()
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite frozen tokens: {path}")
    return result


def cache_tokens(root: Path, selected: list[dict], cache_dir: Path,
                 repo: Path, weights: Path, device: torch.device,
                 selection_sha: str) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    weight_sha = sha256_file(weights)
    if weights.name != WEIGHT_NAME or not weight_sha.startswith("eadcf0ff"):
        raise ValueError("Official DINOv3-SAT checkpoint identity mismatch.")
    expected = {"schema": "s0-dino-token-cache-v1", "selection_sha256": selection_sha,
                "weight_sha256": weight_sha, "dinov3_commit": DINOV3_COMMIT,
                "geometry": "whole 512x512 image, block 24 final LN, 1024x32x32 FP16 before L2"}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if any(index.get(k) != v for k, v in expected.items()):
            raise ValueError("Frozen token cache provenance differs from S0 selection/encoder.")
    else:
        index = {**expected, "files": {}}
    missing = []
    for entry in selected:
        sample_id = entry["id"]
        record = index["files"].get(sample_id)
        if sha256_file(root / entry["image"]) != entry["image_sha256"]:
            raise ValueError(f"Image changed after selection was locked: {sample_id}")
        if record is None:
            missing.append(entry)
        elif record["file"] != sample_id + ".npy" or sha256_file(cache_dir / record["file"]) != record["sha256"]:
            raise ValueError(f"Cached token digest changed: {sample_id}")
    if missing:
        encoder, confirmed_sha = load_dino(ensure_dino_repo(repo), weights, device)
        if confirmed_sha != weight_sha:
            raise ValueError("Checkpoint changed while loading encoder.")
        for number, entry in enumerate(missing, 1):
            tokens = encode_image(encoder, root / entry["image"], device)
            filename = entry["id"] + ".npy"
            path = cache_dir / filename
            temporary = cache_dir / (filename + ".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, tokens, allow_pickle=False)
            os.replace(temporary, path)
            index["files"][entry["id"]] = {"file": filename, "sha256": sha256_file(path)}
            write_json(index_path, index)
            if number % 25 == 0 or number == len(missing):
                print(f"Frozen token cache {number}/{len(missing)}", flush=True)
        if any(p.requires_grad or p.grad is not None for p in encoder.parameters()) or encoder.training:
            raise RuntimeError("Frozen encoder contract violated.")
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return index


def load_tokens(cache_dir: Path, index: dict, sample_id: str, device: torch.device) -> torch.Tensor:
    record = index["files"][sample_id]
    array = np.load(cache_dir / record["file"], allow_pickle=False)
    if array.shape != (1024, 32, 32) or array.dtype != np.float16 or not np.isfinite(array).all():
        raise ValueError(f"Invalid cached tokens: {sample_id}")
    return torch.from_numpy(array).unsqueeze(0).to(device=device, dtype=torch.float32)
