from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath


DATASET_NAME = "pv-dinov3sat-experiment-a"
WEIGHT_NAME = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
SUPPORT_NAME = "PV08_331824_1195530.bmp"
EXPECTED_SHA256_PREFIX = "eadcf0ff"
DINOV3_REPOSITORY = "https://github.com/facebookresearch/dinov3.git"
SPATIAL_BUFFER = 1000
SMOKE_QUERY_COUNT = 2


def sha256_prefix(path: Path, length: int = 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:length]


def archive_members(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def find_archive(input_root: Path, predicate, description: str) -> Path:
    matches = []
    for path in input_root.rglob("*.zip"):
        try:
            if predicate(archive_members(path)):
                matches.append(path)
        except zipfile.BadZipFile:
            continue
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected one {} ZIP archive under {}; found {}.".format(
                description, input_root, len(matches)
            )
        )
    return matches[0]


def member_map(archive: zipfile.ZipFile, predicate) -> dict[str, str]:
    entries = {
        PurePosixPath(name).name: name
        for name in archive.namelist()
        if predicate(PurePosixPath(name).name)
    }
    return entries


def select_smoke_names(image_names: list[str]) -> list[str]:
    support = SUPPORT_NAME
    if support not in image_names:
        raise FileNotFoundError("Support image is missing from the uploaded archive: {}".format(support))
    support_parts = support.removesuffix(".bmp").split("_")
    support_x, support_y = int(support_parts[-2]), int(support_parts[-1])
    candidates = []
    for name in image_names:
        if name == support:
            continue
        parts = name.removesuffix(".bmp").split("_")
        if len(parts) < 3:
            continue
        x, y = int(parts[-2]), int(parts[-1])
        distance = math.hypot(x - support_x, y - support_y)
        if distance > SPATIAL_BUFFER:
            candidates.append((distance, name))
    selected = [name for _, name in sorted(candidates)[:SMOKE_QUERY_COUNT]]
    if len(selected) != SMOKE_QUERY_COUNT:
        raise RuntimeError("Not enough query images outside the spatial buffer for the smoke test.")
    return [support, *selected]


def extract_smoke_subset(image_archive_path: Path, mask_archive_path: Path, dataset_root: Path) -> None:
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "mask"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(image_archive_path) as image_archive, zipfile.ZipFile(mask_archive_path) as mask_archive:
        image_entries = member_map(
            image_archive,
            lambda name: name.lower().endswith(".bmp") and not name.lower().endswith("_label.bmp"),
        )
        mask_entries = member_map(mask_archive, lambda name: name.lower().endswith("_label.bmp"))
        if len(image_entries) != 663 or len(mask_entries) != 663:
            raise RuntimeError("Expected 663 unique images and masks in the archives.")
        selected_names = select_smoke_names(sorted(image_entries))
        for image_name in selected_names:
            mask_name = Path(image_name).stem + "_label.bmp"
            if mask_name not in mask_entries:
                raise FileNotFoundError("Mask is missing from the uploaded archive: {}".format(mask_name))
            (images_dir / image_name).write_bytes(image_archive.read(image_entries[image_name]))
            (masks_dir / mask_name).write_bytes(mask_archive.read(mask_entries[mask_name]))
    print("Smoke-test subset: support={} queries={}".format(selected_names[0], selected_names[1:]), flush=True)


def stage_smoke_subset(source_root: Path, dataset_root: Path) -> None:
    source_images = source_root / "images"
    source_masks = source_root / "mask"
    image_names = sorted(path.name for path in source_images.glob("*.bmp"))
    mask_names = {path.name for path in source_masks.glob("*_label.bmp")}
    if len(image_names) != 663 or len(mask_names) != 663:
        raise RuntimeError("Expected 663 images and masks in the mounted dataset.")
    selected_names = select_smoke_names(image_names)
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "mask"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for image_name in selected_names:
        mask_name = Path(image_name).stem + "_label.bmp"
        if mask_name not in mask_names:
            raise FileNotFoundError("Mask is missing from the mounted dataset: {}".format(mask_name))
        shutil.copyfile(source_images / image_name, images_dir / image_name)
        shutil.copyfile(source_masks / mask_name, masks_dir / mask_name)
    print("Smoke-test subset: support={} queries={}".format(selected_names[0], selected_names[1:]), flush=True)


def main() -> None:
    input_root = Path("/kaggle/input")
    working_root = Path("/kaggle/working")
    dataset_root = working_root / "experiment_a_smoke_dataset"
    input_children = sorted(str(path) for path in input_root.iterdir()) if input_root.is_dir() else []
    print("Mounted Kaggle inputs: {}".format(input_children), flush=True)
    candidates = set()
    if input_root.is_dir():
        for image_dir in input_root.rglob("images"):
            mask_dir = image_dir.parent / "mask"
            if not image_dir.is_dir() or not mask_dir.is_dir():
                continue
            image_count = len(list(image_dir.glob("*.bmp")))
            mask_count = len(list(mask_dir.glob("*_label.bmp")))
            if image_count == 663 and mask_count == 663:
                candidates.add(image_dir.parent)

    if len(candidates) == 1:
        stage_smoke_subset(next(iter(candidates)), dataset_root)
    elif not candidates:
        image_archive = find_archive(
            input_root,
            lambda members: sum(
                name.lower().endswith(".bmp") and not name.lower().endswith("_label.bmp")
                for name in members
            ) == 663,
            "663-image",
        )
        mask_archive = find_archive(
            input_root,
            lambda members: sum(name.lower().endswith("_label.bmp") for name in members) == 663,
            "663-mask",
        )
        extract_smoke_subset(image_archive, mask_archive, dataset_root)
    else:
        raise FileNotFoundError("Found multiple matching 663-pair image/mask directories: {}".format(
            sorted(str(path) for path in candidates)
        ))

    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "mask"
    code_matches = list(input_root.rglob("run_experiment_a.py")) if input_root.is_dir() else []
    if len(code_matches) == 1:
        code_path = code_matches[0]
    elif len(code_matches) > 1:
        raise FileNotFoundError("Found multiple mounted copies of run_experiment_a.py: {}".format(
            [str(path) for path in code_matches]
        ))
    else:
        code_archive = find_archive(
            input_root,
            lambda members: any(Path(name).name == "run_experiment_a.py" for name in members),
            "experiment code",
        )
        code_dir = working_root / "experiment_a_code"
        code_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(code_archive) as archive:
            archive.extractall(code_dir)
        code_path = code_dir / "run_experiment_a.py"
    image_count = len(list(images_dir.glob("*.bmp")))
    mask_count = len(list(masks_dir.glob("*_label.bmp")))
    if image_count != SMOKE_QUERY_COUNT + 1 or mask_count != SMOKE_QUERY_COUNT + 1:
        raise RuntimeError("Expected exactly the smoke-test subset; found {} images and {} masks.".format(
            image_count, mask_count
        ))

    weight_matches = list(input_root.rglob(WEIGHT_NAME))
    if len(weight_matches) != 1:
        raise RuntimeError("Expected one mounted SAT ViT-L/16 checkpoint named {}; found {}".format(
            WEIGHT_NAME, len(weight_matches)
        ))
    weight_path = weight_matches[0]
    actual_prefix = sha256_prefix(weight_path)
    print("SAT checkpoint SHA-256 prefix: {}".format(actual_prefix), flush=True)
    if actual_prefix != EXPECTED_SHA256_PREFIX:
        raise RuntimeError("Checkpoint hash mismatch: expected {}, got {}. Inference was not started.".format(
            EXPECTED_SHA256_PREFIX, actual_prefix
        ))

    repo_path = working_root / "dinov3"
    subprocess.run(["git", "clone", "--depth", "1", DINOV3_REPOSITORY, str(repo_path)], check=True)
    if not code_path.is_file():
        raise FileNotFoundError("Experiment A runner was not found in the uploaded code bundle.")

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is unavailable in this Kaggle runtime.") from error
    print("PyTorch: {} | CUDA available: {}".format(torch.__version__, torch.cuda.is_available()), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("The Kaggle kernel has no GPU. Enable a GPU accelerator and rerun.")
    print("GPU: {}".format(torch.cuda.get_device_name(0)), flush=True)

    output_dir = working_root / "experiment_a_results"
    command = [
        sys.executable,
        str(code_path),
        "--data-root", str(dataset_root),
        "--dinov3-repo", str(repo_path),
        "--weights", str(weight_path),
        "--support", SUPPORT_NAME,
        "--spatial-buffer", str(SPATIAL_BUFFER),
        "--limit-query", str(SMOKE_QUERY_COUNT),
        "--device", "cuda",
        "--output-dir", str(output_dir),
    ]
    subprocess.run(command, check=True, cwd=working_root)
    print("Experiment outputs: {}".format(output_dir), flush=True)


if __name__ == "__main__":
    main()
