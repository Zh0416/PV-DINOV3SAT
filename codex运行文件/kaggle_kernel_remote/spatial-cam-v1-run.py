from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


DATASET_NAME = "pv-dinov3sat-experiment-a"
CODE_DIRECTORY = "spatial_cam_code"
WEIGHT_NAME = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
EXPECTED_SHA256_PREFIX = "eadcf0ff"
DINOV3_REPOSITORY = "https://github.com/facebookresearch/dinov3.git"
SUPPORT_NAME = "PV08_331824_1195530.bmp"
QUERY_NAMES = (
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
)


def sha256_prefix(path: Path, length: int = 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:length]


def find_dataset(input_root: Path) -> Path:
    expected = {SUPPORT_NAME, *QUERY_NAMES}
    candidates = set()
    for images_dir in input_root.rglob("images"):
        masks_dir = images_dir.parent / "mask"
        if not masks_dir.is_dir():
            continue
        image_names = {path.name for path in images_dir.glob("*.bmp")}
        mask_names = {path.name.removesuffix("_label.bmp") + ".bmp" for path in masks_dir.glob("*_label.bmp")}
        if image_names == expected and mask_names == expected:
            candidates.add(images_dir.parent)
    if len(candidates) != 1:
        raise FileNotFoundError("Expected one mounted directory containing exactly the benchmark 21 pairs; found {}.".format(
            sorted(str(path) for path in candidates)
        ))
    return next(iter(candidates))


def stage_cam_code(input_root: Path, working_root: Path) -> tuple[Path, Path]:
    script_matches = list(input_root.rglob("generate_spatial_cam.py"))
    if len(script_matches) != 1:
        raise FileNotFoundError("Expected one mounted spatial CAM script; found {}.".format(len(script_matches)))
    source_dir = script_matches[0].parent
    manifest_path = source_dir / "benchmark_v1.json"
    config_path = source_dir / "config.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Spatial CAM code bundle is missing benchmark_v1.json or config.json.")
    destination = working_root / CODE_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    staged_script = destination / script_matches[0].name
    staged_manifest = destination / manifest_path.name
    shutil.copyfile(script_matches[0], staged_script)
    shutil.copyfile(manifest_path, staged_manifest)
    shutil.copyfile(config_path, destination / config_path.name)
    readme = source_dir / "README.md"
    if readme.is_file():
        shutil.copyfile(readme, destination / readme.name)
    return staged_script, staged_manifest


def main() -> None:
    input_root = Path("/kaggle/input")
    working_root = Path("/kaggle/working")
    dataset_root = find_dataset(input_root)
    script_path, manifest_path = stage_cam_code(input_root, working_root)
    expected_names = {SUPPORT_NAME, *QUERY_NAMES}
    image_count = len(list((dataset_root / "images").glob("*.bmp")))
    mask_count = len(list((dataset_root / "mask").glob("*_label.bmp")))
    if image_count != len(expected_names) or mask_count != len(expected_names):
        raise RuntimeError("Expected exactly 21 image/mask pairs; found {} images and {} masks.".format(
            image_count, mask_count
        ))

    weight_matches = list(input_root.rglob(WEIGHT_NAME))
    if len(weight_matches) != 1:
        raise RuntimeError("Expected one mounted SAT ViT-L/16 checkpoint; found {}.".format(len(weight_matches)))
    weight_path = weight_matches[0]
    actual_prefix = sha256_prefix(weight_path)
    print("SAT checkpoint SHA-256 prefix: {}".format(actual_prefix), flush=True)
    if actual_prefix != EXPECTED_SHA256_PREFIX:
        raise RuntimeError("Checkpoint hash mismatch: expected {}, got {}.".format(
            EXPECTED_SHA256_PREFIX, actual_prefix
        ))

    repo_path = working_root / "dinov3"
    subprocess.run(["git", "clone", "--depth", "1", DINOV3_REPOSITORY, str(repo_path)], check=True)

    import torch

    print("PyTorch: {} | CUDA available: {}".format(torch.__version__, torch.cuda.is_available()), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("The Kaggle kernel has no GPU. Enable a GPU accelerator and rerun.")
    print("GPU: {}".format(torch.cuda.get_device_name(0)), flush=True)

    output_dir = working_root / "spatial_cam_v1_results"
    command = [
        sys.executable,
        str(script_path),
        "--data-root", str(dataset_root),
        "--manifest", str(manifest_path),
        "--dinov3-repo", str(repo_path),
        "--weights", str(weight_path),
        "--device", "cuda",
        "--output-dir", str(output_dir),
    ]
    subprocess.run(command, check=True, cwd=working_root)
    print("Spatial CAM output: {}".format(output_dir), flush=True)


if __name__ == "__main__":
    main()
