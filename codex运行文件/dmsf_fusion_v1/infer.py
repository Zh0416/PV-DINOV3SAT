"""Inference for the pinned spatial/semantic/global-fusion checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS / 'dmsf_stage3'))
from stage3_model import Segmenter


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


def encode_raw(support_image: Path, support_mask: Path, query_image: Path,
               dino_repo: Path, dino_weights: Path, geo_weights: Path,
               device: torch.device) -> tuple[torch.Tensor, ...]:
    sys.path.insert(0, str(EXPERIMENTS / 'dmsf_v1'))
    sys.path.insert(0, str(EXPERIMENTS / 'georsclip_semantic_cam_e1'))
    from feature_cache import (aligned_support_masks, encode_dino, encode_text,
                               load_dino, load_text_encoder, DINOV3_COMMIT)

    actual_commit = subprocess.check_output(
        ['git', '-C', str(dino_repo), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_commit != DINOV3_COMMIT:
        raise ValueError('DINOv3 source checkout differs from the pinned encoder commit.')

    dino, _ = load_dino(dino_repo, dino_weights, device)
    text_encoder, _ = load_text_encoder(geo_weights, device)
    support = encode_dino(dino, support_image, device).numpy()[None]
    query = encode_dino(dino, query_image, device).numpy()[None]
    mask = aligned_support_masks(support_mask)['binary_patch_nearest'].astype(np.float32)[None, None]
    text = encode_text(text_encoder, device).numpy()
    return validate_inputs((support, query, mask, text), device)


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
    if args.features_npz is None:
        required = (args.support_mask, args.query_image, args.dino_repo,
                    args.dino_weights, args.geo_weights)
        if any(value is None for value in required):
            parser.error('Raw images also require --support-mask, --query-image, --dino-repo, --dino-weights and --geo-weights.')
        inputs = encode_raw(args.support_image, args.support_mask, args.query_image,
                            args.dino_repo, args.dino_weights, args.geo_weights, device)
    else:
        inputs = load_feature_input(args.features_npz, device)
    model = load_model(device, args.checkpoint)
    result = predict(model, inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    metadata = {'version': version()['version'], 'checkpoint_sha256': version()['checkpoint_sha256'],
                'input': str(args.features_npz or args.query_image),
                'spatial_weight': float(result['spatial_weight']),
                'cam_resolution': [64, 64], 'mask_resolution': [1024, 1024],
                'threshold': 'upsampled logits >= 0'}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
