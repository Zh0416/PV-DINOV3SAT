"""Inference for the pinned spatial/semantic/global-fusion checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS / 'dmsf_stage3'))
from stage3_model import Segmenter


TARGET_SIZE = 1024
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def version() -> dict:
    return json.loads((HERE / 'version.json').read_text(encoding='utf-8'))


def load_model(device: torch.device, checkpoint_path: Path | None = None) -> Segmenter:
    spec = version()
    for relative, expected in spec['source_sha256'].items():
        if source_sha256(EXPERIMENTS / relative) != expected:
            raise ValueError('Model source differs from pinned checkpoint: ' + relative)
    checkpoint_path = checkpoint_path or HERE / spec['checkpoint']
    if checkpoint_path.stat().st_size != spec['checkpoint_bytes'] or sha256(checkpoint_path) != spec['checkpoint_sha256']:
        raise ValueError('Checkpoint size or SHA-256 differs from pinned version.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (checkpoint['identity'] != spec['identity'] or checkpoint['seed'] != spec['seed']
            or checkpoint['epoch'] != spec['best_epoch']):
        raise ValueError('Checkpoint identity differs from pinned version.')
    for source, expected in [('dmsf.py', spec['source_sha256']['dmsf_v1/dmsf.py']),
                             ('stage3_model.py', spec['source_sha256']['dmsf_stage3/stage3_model.py'])]:
        if checkpoint['source_hashes'].get(source) != expected:
            raise ValueError('Checkpoint source manifest differs: ' + source)
    model = Segmenter(**{key: spec['identity'][key] for key in
                        ('branch', 'normalization', 'text_kind', 'beta_init')})
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval()


def load_feature_input(path: Path, device: torch.device) -> tuple[torch.Tensor, ...]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'support', 'query', 'mask', 'text'}:
            raise ValueError('Feature input requires support, query, mask and text arrays.')
        arrays = [archive[key].copy() for key in ('support', 'query', 'mask', 'text')]
    return validate_inputs(arrays, device)


def validate_inputs(arrays, device: torch.device) -> tuple[torch.Tensor, ...]:
    shapes = ((1, 1024, 64, 64), (1, 1024, 64, 64),
              (1, 1, 64, 64), (1, 512))
    for name, value, shape in zip(('support', 'query', 'mask', 'text'), arrays, shapes):
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'{name} must be finite float32 with shape {shape}.')
    if not np.isin(arrays[2], (0, 1)).all() or not arrays[2].any():
        raise ValueError('Support mask must be nonempty binary 0/1 at patch resolution.')
    if not np.allclose(np.linalg.norm(arrays[3], axis=1), 1, atol=1e-5):
        raise ValueError('Text embedding must be L2 normalized.')
    return tuple(torch.from_numpy(value).to(device) for value in arrays)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert('RGB')
        pixels = np.asarray(image).copy()
    if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) <= 0:
        raise ValueError('Input image must be a nonempty RGB image.')
    return pixels


def letterbox_geometry(height: int, width: int, target: int = TARGET_SIZE) -> dict:
    """Describe an aspect-preserving resize into the encoder's square canvas."""
    if height <= 0 or width <= 0 or target <= 0:
        raise ValueError('Image and target dimensions must be positive.')
    scale = min(target / width, target / height)
    resized_width = max(1, min(target, round(width * scale)))
    resized_height = max(1, min(target, round(height * scale)))
    left = (target - resized_width) // 2
    top = (target - resized_height) // 2
    return {
        'original_height': int(height), 'original_width': int(width),
        'canvas_size': int(target), 'resized_height': int(resized_height),
        'resized_width': int(resized_width), 'top': int(top), 'left': int(left),
        'scale': float(scale),
    }


def letterbox_rgb(pixels: np.ndarray, geometry: dict) -> np.ndarray:
    height, width = pixels.shape[:2]
    if (height, width) != (geometry['original_height'], geometry['original_width']):
        raise ValueError('RGB pixels and letterbox geometry disagree.')
    image = Image.fromarray(pixels, mode='RGB').resize(
        (geometry['resized_width'], geometry['resized_height']), Image.Resampling.BICUBIC)
    fill = tuple(int(round(value * 255)) for value in SAT_MEAN)
    canvas = Image.new('RGB', (geometry['canvas_size'], geometry['canvas_size']), fill)
    canvas.paste(image, (geometry['left'], geometry['top']))
    return np.asarray(canvas).copy()


def letterbox_support_mask(path: Path, geometry: dict) -> np.ndarray:
    """Read any grayscale foreground convention and align it to a 64x64 mask."""
    with Image.open(path) as image:
        raw = np.asarray(image.convert('L')).copy()
    if raw.shape != (geometry['original_height'], geometry['original_width']):
        raise ValueError('Support mask dimensions must match the support image.')
    binary = Image.fromarray((raw > 0).astype(np.uint8) * 255, mode='L').resize(
        (geometry['resized_width'], geometry['resized_height']), Image.Resampling.NEAREST)
    canvas = np.zeros((geometry['canvas_size'], geometry['canvas_size']), dtype=np.uint8)
    canvas[geometry['top']:geometry['top'] + geometry['resized_height'],
           geometry['left']:geometry['left'] + geometry['resized_width']] = np.asarray(binary)
    patch = F.interpolate(torch.from_numpy(canvas[None, None].astype(np.float32)),
                          size=(64, 64), mode='nearest')[0, 0].numpy()
    if patch.sum() <= 0:
        raise ValueError('Support foreground is empty after resize and patch alignment.')
    return patch[None, None].astype(np.float32)


@torch.no_grad()
def encode_dino_pixels(model, pixels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Encode a 1024px square canvas with the training-time four-crop geometry."""
    if pixels.shape != (TARGET_SIZE, TARGET_SIZE, 3) or pixels.dtype != np.uint8:
        raise ValueError('DINO pixels must be uint8 RGB 1024x1024.')
    image = torch.from_numpy(pixels).permute(2, 0, 1).to(device=device, dtype=torch.float32) / 255
    image = (image - image.new_tensor(SAT_MEAN)[:, None, None]) / image.new_tensor(SAT_STD)[:, None, None]
    tiles = []
    for top, left in [(0, 0), (0, 512), (512, 0), (512, 512)]:
        crop = image[None, :, top:top + 512, left:left + 512]
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            layers = model.get_intermediate_layers(crop, n=(23,), reshape=True, norm=True)
        if len(layers) != 1 or layers[0].shape != (1, 1024, 32, 32):
            raise ValueError('Unexpected block-24 token layout.')
        tiles.append(layers[0][0].float().cpu())
    output = torch.cat([torch.cat(tiles[:2], 2), torch.cat(tiles[2:], 2)], 1).contiguous()
    if output.shape != (1024, 64, 64) or not torch.isfinite(output).all():
        raise ValueError('Invalid DINO features.')
    return output


def restore_map_to_original(values: torch.Tensor, geometry: dict) -> np.ndarray:
    """Upsample a square-canvas map, remove padding, and restore source dimensions."""
    square = F.interpolate(values, size=(geometry['canvas_size'], geometry['canvas_size']),
                           mode='bilinear', align_corners=False)
    cropped = square[:, :, geometry['top']:geometry['top'] + geometry['resized_height'],
                     geometry['left']:geometry['left'] + geometry['resized_width']]
    restored = F.interpolate(cropped, size=(geometry['original_height'], geometry['original_width']),
                             mode='bilinear', align_corners=False)
    return restored[0, 0].detach().cpu().numpy().astype(np.float32)


def encode_raw(support_image: Path, support_mask: Path, query_image: Path,
               dino_repo: Path, dino_weights: Path, geo_weights: Path,
               device: torch.device) -> tuple[torch.Tensor, ...]:
    inputs, _ = encode_raw_variable(support_image, support_mask, query_image,
                                    dino_repo, dino_weights, geo_weights, device)
    return inputs


def encode_raw_variable(support_image: Path, support_mask: Path, query_image: Path,
                        dino_repo: Path, dino_weights: Path, geo_weights: Path,
                        device: torch.device) -> tuple[tuple[torch.Tensor, ...], dict]:
    """Encode arbitrary-size images through an aspect-preserving 1024px canvas."""
    sys.path.insert(0, str(EXPERIMENTS / 'dmsf_v1'))
    sys.path.insert(0, str(EXPERIMENTS / 'georsclip_semantic_cam_e1'))
    from feature_cache import (encode_text,
                               load_dino, load_text_encoder, DINOV3_COMMIT)

    actual_commit = subprocess.check_output(
        ['git', '-C', str(dino_repo), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_commit != DINOV3_COMMIT:
        raise ValueError('DINOv3 source checkout differs from the pinned encoder commit.')

    dino, _ = load_dino(dino_repo, dino_weights, device)
    text_encoder, _ = load_text_encoder(geo_weights, device)
    support_pixels = _load_rgb(support_image)
    query_pixels = _load_rgb(query_image)
    support_geometry = letterbox_geometry(*support_pixels.shape[:2])
    query_geometry = letterbox_geometry(*query_pixels.shape[:2])
    support = encode_dino_pixels(dino, letterbox_rgb(support_pixels, support_geometry), device).numpy()[None]
    query = encode_dino_pixels(dino, letterbox_rgb(query_pixels, query_geometry), device).numpy()[None]
    mask = letterbox_support_mask(support_mask, support_geometry)
    text = encode_text(text_encoder, device).numpy()
    return validate_inputs((support, query, mask, text), device), query_geometry


@torch.inference_mode()
def predict(model: Segmenter, inputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    maps = model(*inputs)
    if maps['semantic_cam'] is None:
        raise ValueError('Pinned version must contain a semantic CAM.')
    logits_full = F.interpolate(maps['logits'], size=(1024, 1024),
                                mode='bilinear', align_corners=False)
    result = {
        'spatial_cam': maps['spatial_cam'][0, 0].cpu().numpy(),
        'semantic_cam': maps['semantic_cam'][0, 0].cpu().numpy(),
        'fused_cam': maps['selected_cam'][0, 0].cpu().numpy(),
        'logits_patch': maps['logits'][0, 0].cpu().numpy(),
        'logits_full': logits_full[0, 0].cpu().numpy(),
        'mask_full': (logits_full[0, 0] >= 0).byte().cpu().numpy(),
        'spatial_weight': np.asarray(float(maps['spatial_weight'].cpu()), dtype=np.float32),
    }
    return result


@torch.inference_mode()
def predict_original_size(model: Segmenter, inputs: tuple[torch.Tensor, ...],
                          geometry: dict) -> dict[str, np.ndarray]:
    """Return CAMs and logits at the query image's original resolution."""
    maps = model(*inputs)
    if maps['semantic_cam'] is None:
        raise ValueError('Pinned version must contain a semantic CAM.')
    logits_square = F.interpolate(maps['logits'], size=(TARGET_SIZE, TARGET_SIZE),
                                 mode='bilinear', align_corners=False)
    result = {
        'spatial_cam': restore_map_to_original(maps['spatial_cam'], geometry),
        'semantic_cam': restore_map_to_original(maps['semantic_cam'], geometry),
        'fused_cam': restore_map_to_original(maps['selected_cam'], geometry),
        'logits_patch': maps['logits'][0, 0].cpu().numpy(),
        'logits_square': logits_square[0, 0].cpu().numpy(),
        'logits_full': restore_map_to_original(maps['logits'], geometry),
        'spatial_weight': np.asarray(float(maps['spatial_weight'].cpu()), dtype=np.float32),
    }
    result['mask_full'] = (result['logits_full'] >= 0).astype(np.uint8)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--features-npz', type=Path)
    source.add_argument('--support-image', type=Path)
    parser.add_argument('--support-mask', type=Path)
    parser.add_argument('--query-image', type=Path)
    parser.add_argument('--dino-repo', type=Path)
    parser.add_argument('--dino-weights', type=Path)
    parser.add_argument('--geo-weights', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != '.npz':
        parser.error('--output must end in .npz')
    device = torch.device(args.device)
    geometry = None
    if args.features_npz is None:
        required = (args.support_mask, args.query_image, args.dino_repo,
                    args.dino_weights, args.geo_weights)
        if any(value is None for value in required):
            parser.error('Raw images also require --support-mask, --query-image, --dino-repo, --dino-weights and --geo-weights.')
        inputs, geometry = encode_raw_variable(
            args.support_image, args.support_mask, args.query_image,
            args.dino_repo, args.dino_weights, args.geo_weights, device)
    else:
        inputs = load_feature_input(args.features_npz, device)
    model = load_model(device, args.checkpoint)
    result = predict_original_size(model, inputs, geometry) if geometry else predict(model, inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    metadata = {'version': version()['version'], 'checkpoint_sha256': version()['checkpoint_sha256'],
                'input': str(args.features_npz or args.query_image),
                'spatial_weight': float(result['spatial_weight']),
                'cam_resolution': [64, 64],
                'output_resolution': [int(result['logits_full'].shape[0]),
                                      int(result['logits_full'].shape[1])],
                'letterbox_geometry': geometry,
                'threshold': 'upsampled logits >= 0'}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
