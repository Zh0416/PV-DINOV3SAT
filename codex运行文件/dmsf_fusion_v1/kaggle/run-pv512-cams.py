"""Run the pinned DMSF model on two PV-512 TIFF tiles and render CAMs.

The notebook only needs to set DMSF_CODE_ROOT and DMSF_PV512_ROOT. The
support/query basenames below are the pair selected for the current test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


SUPPORT_NAME = "0807_Q_934ae716b5714f_r00024_c00069.tif"
QUERY_NAME = "0807_Q_934ae716b5714f_r00024_c00075.tif"


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def find_unique(root: Path, basename: str) -> Path:
    matches = sorted(path for path in root.rglob(basename) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {basename!r} below {root}; found {len(matches)}: {matches[:5]}"
        )
    return matches[0]


def find_mask(image: Path, root: Path) -> Path:
    stem = image.stem
    suffixes = ("_label", "_mask", "_gt", "-label", "-mask")
    direct = []
    for suffix in suffixes:
        for extension in (image.suffix, ".tif", ".tiff", ".png", ".bmp"):
            direct.append(image.with_name(stem + suffix + extension))
    parent_names = {image.parent.name.lower()}
    for name in tuple(parent_names):
        if name in {"images", "image", "imgs"}:
            parent_names.update({"labels", "label", "masks", "mask", "annotations"})
    for parent_name in sorted(parent_names):
        parent = image.parent.parent / parent_name
        for suffix in ("", *suffixes):
            for extension in (image.suffix, ".tif", ".tiff", ".png", ".bmp"):
                direct.append(parent / (stem + suffix + extension))
    candidates = [path for path in dict.fromkeys(direct) if path.is_file()]
    if not candidates:
        normalized = {stem.lower(), *(stem.lower() + suffix for suffix in suffixes)}
        for path in root.rglob("*"):
            if not path.is_file() or path == image:
                continue
            if path.stem.lower() in normalized and any(
                token in str(path.parent).lower() for token in ("label", "mask", "annot")
            ):
                candidates.append(path)
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Could not identify one support mask for {image}; found {len(candidates)}: {candidates[:8]}"
        )
    return candidates[0]


def prepare_dino(repo: Path, weights_root: Path) -> Path:
    if not repo.is_dir():
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/dinov3.git", str(repo)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--detach", "6876159a11b4df116f30f667f8c9888617df0751"],
        check=True,
    )
    if weights_root.is_dir():
        matches = sorted(weights_root.rglob("dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one DINOv3 SAT checkpoint below {weights_root}; found {len(matches)}"
            )
        return matches[0]
    return weights_root


def render_outputs(output_npz: Path, query_image: Path, output_dir: Path) -> dict:
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(output_npz) as archive:
        spatial = np.asarray(archive["spatial_cam"], dtype=np.float32)
        semantic = np.asarray(archive["semantic_cam"], dtype=np.float32)
        fused = np.asarray(archive["fused_cam"], dtype=np.float32)
        mask = np.asarray(archive["mask_full"], dtype=np.uint8)
        weight = float(np.asarray(archive["spatial_weight"]).reshape(-1)[0])
    with Image.open(query_image) as image:
        rgb = np.asarray(image.convert("RGB"))

    def normalize(values: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        low, high = float(values.min()), float(values.max())
        return (values - low) / (high - low) if high > low else np.zeros_like(values)

    def resize(values: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255))
        return np.asarray(image.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR)) / 255.0

    cams = {"spatial_cam": resize(normalize(spatial)),
            "semantic_cam": resize(normalize(semantic)),
            "fused_cam": resize(normalize(fused))}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in cams.items():
        plt.imsave(output_dir / f"{name}.png", values, cmap="turbo", vmin=0, vmax=1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Query image")
    axes[0, 1].imshow(cams["spatial_cam"], cmap="turbo", vmin=0, vmax=1)
    axes[0, 1].set_title("Spatial CAM")
    axes[0, 2].imshow(cams["semantic_cam"], cmap="turbo", vmin=0, vmax=1)
    axes[0, 2].set_title("Semantic CAM")
    axes[1, 0].imshow(cams["fused_cam"], cmap="turbo", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Fused CAM (spatial weight={weight:.3f})")
    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(cams["fused_cam"], cmap="turbo", vmin=0, vmax=1, alpha=0.48)
    axes[1, 1].set_title("Fused overlay")
    axes[1, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("Predicted mask")
    for axis in axes.ravel():
        axis.set_axis_off()
    fig.suptitle("PV-512: spatial, semantic and fused CAM", fontsize=16, weight="bold")
    panel = output_dir / "cam_panel.png"
    fig.savefig(panel, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"panel": str(panel), "spatial_weight": weight, "resolution": list(rgb.shape[:2])}


def main() -> None:
    code_root = env_path("DMSF_CODE_ROOT", "/kaggle/working/PV-DINOV3SAT")
    dataset_root = env_path("DMSF_PV512_ROOT", "/kaggle/input/pv-512")
    if not (code_root / "codex运行文件" / "dmsf_fusion_v1" / "infer.py").is_file():
        raise FileNotFoundError("Set DMSF_CODE_ROOT to the mounted repository/code Dataset.")
    fusion_root = code_root / "codex运行文件" / "dmsf_fusion_v1"
    sys.path.insert(0, str(fusion_root))
    from infer import encode_raw_variable, load_model, predict_original_size
    import numpy as np
    import torch

    support_image = find_unique(dataset_root, SUPPORT_NAME)
    query_image = find_unique(dataset_root, QUERY_NAME)
    mask_override = os.environ.get("DMSF_SUPPORT_MASK")
    support_mask = Path(mask_override) if mask_override else find_mask(support_image, dataset_root)
    if not support_mask.is_file():
        raise FileNotFoundError(f"DMSF_SUPPORT_MASK does not exist: {support_mask}")
    output_dir = env_path("DMSF_OUTPUT_DIR", "/kaggle/working/pv512_cam_outputs")
    output_npz = output_dir / "prediction.npz"
    dino_repo = env_path("DMSF_DINO_REPO", "/kaggle/working/dinov3")
    dino_weights = prepare_dino(
        dino_repo,
        env_path("DMSF_DINO_WEIGHTS", "/kaggle/input/dinov3-vitl16-pretrain-sat493m"),
    )
    geo_weights = env_path("DMSF_GEO_WEIGHTS", "/kaggle/working/RS5M_ViT-B-32.pt")
    device = torch.device(os.environ.get("DMSF_DEVICE", "cuda"))

    inputs, geometry = encode_raw_variable(
        support_image, support_mask, query_image, dino_repo, dino_weights, geo_weights, device
    )
    model = load_model(device)
    result = predict_original_size(model, inputs, geometry)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **result)
    metadata = {
        "support_image": str(support_image),
        "support_mask": str(support_mask),
        "query_image": str(query_image),
        "letterbox_geometry": geometry,
        "output_resolution": list(result["logits_full"].shape),
        "checkpoint": str(fusion_root / "checkpoints" / "best.pt"),
    }
    (output_dir / "prediction.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    rendered = render_outputs(output_npz, query_image, output_dir)
    print(json.dumps({**metadata, **rendered}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
