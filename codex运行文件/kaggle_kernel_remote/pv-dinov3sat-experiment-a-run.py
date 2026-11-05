from __future__ import annotations

import hashlib
import json
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


def extract_benchmark_dataset(
    image_archive_path: Path,
    mask_archive_path: Path,
    dataset_root: Path,
    expected_names: list[str],
) -> None:
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
        mask_names = {Path(name).stem.removesuffix("_label") + ".bmp" for name in mask_entries}
        expected = set(expected_names)
        if set(image_entries) != expected or mask_names != expected:
            raise RuntimeError("The uploaded archives must contain exactly the 21 benchmark image/mask pairs.")
        for image_name in expected_names:
            member_name = image_entries[image_name]
            (images_dir / image_name).write_bytes(image_archive.read(member_name))
        for image_name in expected_names:
            mask_name = Path(image_name).stem + "_label.bmp"
            member_name = mask_entries[mask_name]
            (masks_dir / mask_name).write_bytes(mask_archive.read(member_name))
    print("Extracted {} benchmark image/mask pairs; layer weights use the single support image.".format(
        len(expected_names)
    ), flush=True)


def stage_code_bundle(input_root: Path, code_dir: Path) -> tuple[Path, Path, Path]:
    required_names = {"run_experiment_a.py", "layer_selection_experiment.py", "benchmark_v1.json"}
    direct_bundles = []
    for layer_path in input_root.rglob("layer_selection_experiment.py"):
        parent = layer_path.parent
        paths = {name: parent / name for name in required_names}
        if all(path.is_file() for path in paths.values()):
            direct_bundles.append(paths)
    if len(direct_bundles) == 1:
        code_dir.mkdir(parents=True, exist_ok=True)
        staged = {}
        for name, source in direct_bundles[0].items():
            destination = code_dir / name
            destination.write_bytes(source.read_bytes())
            staged[name] = destination
        return (
            staged["run_experiment_a.py"],
            staged["layer_selection_experiment.py"],
            staged["benchmark_v1.json"],
        )
    if len(direct_bundles) > 1:
        raise FileNotFoundError("Found multiple mounted layer-selection code bundles.")

    bundles = []
    for path in input_root.rglob("*.zip"):
        try:
            members = archive_members(path)
        except zipfile.BadZipFile:
            continue
        by_basename = {}
        for member in members:
            by_basename.setdefault(PurePosixPath(member).name, []).append(member)
        if required_names.issubset(by_basename) and all(len(by_basename[name]) == 1 for name in required_names):
            bundles.append((path, by_basename))
    if len(bundles) != 1:
        raise FileNotFoundError("Expected one layer-selection code bundle; found {}.".format(len(bundles)))

    code_dir.mkdir(parents=True, exist_ok=True)
    bundle_path, by_basename = bundles[0]
    with zipfile.ZipFile(bundle_path) as archive:
        staged = {}
        for name in required_names:
            destination = code_dir / name
            destination.write_bytes(archive.read(by_basename[name][0]))
            staged[name] = destination
    return (
        staged["run_experiment_a.py"],
        staged["layer_selection_experiment.py"],
        staged["benchmark_v1.json"],
    )


def main() -> None:
    input_root = Path("/kaggle/input")
    working_root = Path("/kaggle/working")
    dataset_root = None
    expected_names = [SUPPORT_NAME, *QUERY_NAMES]
    input_children = sorted(str(path) for path in input_root.iterdir()) if input_root.is_dir() else []
    print("Mounted Kaggle inputs: {}".format(input_children), flush=True)
    candidates = set()
    if input_root.is_dir():
        for image_dir in input_root.rglob("images"):
            mask_dir = image_dir.parent / "mask"
            if not image_dir.is_dir() or not mask_dir.is_dir():
                continue
            image_names = {path.name for path in image_dir.glob("*.bmp")}
            mask_names = {path.name.removesuffix("_label.bmp") + ".bmp" for path in mask_dir.glob("*_label.bmp")}
            if image_names == set(expected_names) and mask_names == set(expected_names):
                candidates.add(image_dir.parent)

    if len(candidates) == 1:
        dataset_root = next(iter(candidates))
    elif not candidates:
        image_archive = find_archive(
            input_root,
            lambda members: sum(
                name.lower().endswith(".bmp") and not name.lower().endswith("_label.bmp")
                for name in members
            ) == len(expected_names),
            "benchmark-image",
        )
        mask_archive = find_archive(
            input_root,
            lambda members: sum(name.lower().endswith("_label.bmp") for name in members) == len(expected_names),
            "benchmark-mask",
        )
        dataset_root = working_root / "benchmark_v1_data"
        extract_benchmark_dataset(image_archive, mask_archive, dataset_root, expected_names)
    else:
        raise FileNotFoundError("Found multiple matching 21-pair benchmark directories: {}".format(
            sorted(str(path) for path in candidates)
        ))

    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "mask"
    image_count = len(list(images_dir.glob("*.bmp")))
    mask_count = len(list(masks_dir.glob("*_label.bmp")))
    if image_count != len(expected_names) or mask_count != len(expected_names):
        raise RuntimeError("Expected exactly the 21 fixed benchmark pairs; found {} images and {} masks.".format(
            image_count, mask_count
        ))

    code_dir = working_root / "layer_selection_code"
    code_path, layer_path, manifest_path = stage_code_bundle(input_root, code_dir)

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

    output_dir = working_root / "layer_selection_v1_results"
    command = [
        sys.executable,
        str(layer_path),
        "--data-root", str(dataset_root),
        "--dinov3-repo", str(repo_path),
        "--weights", str(weight_path),
        "--test-manifest", str(manifest_path),
        "--spatial-buffer", str(SPATIAL_BUFFER),
        "--train-episodes", "200",
        "--learning-rate", "0.05",
        "--seed", "20261105",
        "--device", "cuda",
        "--output-dir", str(output_dir),
    ]
    subprocess.run(command, check=True, cwd=working_root)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["test_manifest_version"] = 1
    summary["test_manifest"] = {
        "support_image": SUPPORT_NAME,
        "query_images": list(QUERY_NAMES),
        "spatial_buffer_coordinate_units": SPATIAL_BUFFER,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "benchmark_v1.json").write_text(json.dumps(summary["test_manifest"], indent=2), encoding="utf-8")
    print("Experiment outputs: {}".format(output_dir), flush=True)


if __name__ == "__main__":
    main()
