"""Reusable frozen encoder/cache interface. No query label or CLIP image input.

Cache layout is float32 CHW DINO final-LayerNorm features BEFORE extra L2.
Inference uses four independently encoded 512px crops, not global attention.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from data_audit import sha256_file, write_json

PROMPT = 'a satellite image of ground-mounted solar panel arrays'
DINOV3_COMMIT = '6876159a11b4df116f30f667f8c9888617df0751'
WEIGHT_NAME = 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)
GEORSCLIP_SHA256 = '129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01'


def assert_frozen(model: nn.Module) -> None:
    if any(m.training for m in model.modules()):
        raise ValueError('Every encoder module must be in eval mode.')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Encoder must have no trainable parameters or gradients.')


def state_sha256(model: nn.Module) -> str:
    """Full parameter AND buffer hash, no sampling and no credential access."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().contiguous().cpu().numpy()
        h.update(name.encode()); h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def load_dino(repo: Path, weights: Path, device: torch.device) -> tuple[nn.Module, dict]:
    digest = sha256_file(weights)
    if weights.name != WEIGHT_NAME or not digest.startswith('eadcf0ff'):
        raise ValueError('Official SAT-493M ViT-L/16 checkpoint identity mismatch.')
    sys.path.insert(0, str(repo.resolve()))
    from dinov3.hub.backbones import dinov3_vitl16
    model = dinov3_vitl16(pretrained=True, weights=str(weights.resolve()))
    model.eval().requires_grad_(False).to(device)
    assert_frozen(model)
    return model, {'checkpoint': weights.name, 'checkpoint_sha256': digest,
                   'block_index_zero_based': 23, 'output': 'final LayerNorm, BEFORE extra L2',
                   'feature_channels': 1024, 'dtype_stored': 'float32',
                   'autocast': 'float16 on CUDA, disabled on CPU',
                   'mean': SAT_MEAN, 'std': SAT_STD,
                   'patch_size': 16, 'image_hw': [1024,1024], 'crop_hw': [512,512],
                   'grid_hw': [64,64], 'crop_order': ['top-left','top-right','bottom-left','bottom-right'],
                   'geometry_note': 'Independent crop attention/RoPE coordinates reset per crop; stitched grid is NOT whole-image encoding.'}


@torch.no_grad()
def encode_dino(model: nn.Module, image_path: Path, device: torch.device) -> torch.Tensor:
    assert_frozen(model)
    with Image.open(image_path) as image:
        if image.mode != 'RGB' or image.size != (1024,1024):
            raise ValueError('Expected native RGB 1024x1024 image.')
        pixels = np.asarray(image).copy()
    image = torch.from_numpy(pixels).permute(2,0,1).to(device=device, dtype=torch.float32) / 255
    image = (image - image.new_tensor(SAT_MEAN)[:,None,None]) / image.new_tensor(SAT_STD)[:,None,None]
    tiles = []
    for top, left in [(0,0),(0,512),(512,0),(512,512)]:
        crop = image[None,:,top:top+512,left:left+512]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            layers = model.get_intermediate_layers(crop, n=(23,), reshape=True, norm=True)
        if len(layers) != 1 or layers[0].shape != (1,1024,32,32):
            raise ValueError('Unexpected block-24 token layout.')
        tiles.append(layers[0][0].float().cpu())  # Deliberately no F.normalize.
    output = torch.cat([torch.cat(tiles[:2],2),torch.cat(tiles[2:],2)],1).contiguous()
    if output.shape != (1024,64,64) or not torch.isfinite(output).all():
        raise ValueError('Invalid DINO features.')
    return output


def aligned_support_masks(path: Path) -> dict[str, np.ndarray]:
    with Image.open(path) as image:
        raw = np.asarray(image).copy()
        if image.mode != 'L' or raw.shape != (1024,1024) or not set(np.unique(raw)) <= {0,12}:
            raise ValueError('Expected L 1024x1024 support mask with class IDs 0/12.')
    binary = torch.from_numpy((raw == 12).astype(np.float32))[None,None]
    nearest = F.interpolate(binary, size=(64,64), mode='nearest')[0,0]
    fractions = F.avg_pool2d(binary,16,16)[0,0]
    if nearest.sum() <= 0:
        raise ValueError('Support foreground empty after nearest downsampling.')
    return {'binary_full': binary[0,0].numpy().astype(np.uint8),
            'binary_patch_nearest': nearest.numpy().astype(np.uint8),
            'area_fraction_patch_diagnostic_only': fractions.numpy()}


def load_text_encoder(checkpoint: Path, device: torch.device) -> tuple[nn.Module, dict]:
    # Reuse the already verified strict RS5M loader; only encode_text is called.
    from run_e1 import download_checkpoint, load_geo_rsclip
    download_checkpoint(checkpoint)
    model, info = load_geo_rsclip(checkpoint, device)
    # Remove the image tower after strict full-checkpoint loading. Calling
    # encode_image becomes impossible in this branch, not merely unused by convention.
    del model.visual
    assert_frozen(model)
    info.update({'checkpoint_sha256': sha256_file(checkpoint), 'image_tower_removed': True,
                 'prompt': PROMPT, 'prompt_count': 1, 'embedding_dim': 512,
                 'prompt_rule': 'One fixed prompt, no ensemble and no support/query-specific rewriting.'})
    if info['checkpoint_sha256'] != GEORSCLIP_SHA256:
        raise ValueError('GeoRSCLIP hash mismatch.')
    return model, info


@torch.no_grad()
def encode_text(model: nn.Module, device: torch.device) -> torch.Tensor:
    import open_clip
    assert_frozen(model)
    token_ids = open_clip.tokenize([PROMPT], context_length=77).to(device)
    text = F.normalize(model.encode_text(token_ids).float(),dim=-1).cpu()
    if text.shape != (1,512) or not torch.isfinite(text).all() or not torch.allclose(text.norm(dim=-1),torch.ones(1),atol=1e-5):
        raise ValueError('Invalid normalized single-prompt text embedding.')
    return text


def export_cache(dino: nn.Module, text_encoder: nn.Module, data_root: Path,
                 image_names: list[str], support_names: list[str], output: Path,
                 device: torch.device, metadata: dict) -> dict:
    if len(set(image_names)) != len(image_names) or not set(support_names) <= set(image_names):
        raise ValueError('Unique feature names and support subset required.')
    if any(Path(n).name != n for n in image_names):
        raise ValueError('Image names must be safe basenames.')
    output.mkdir(parents=True,exist_ok=True)
    files = {}
    for name in image_names:
        features = encode_dino(dino, data_root/'images'/name, device)
        target = output / (Path(name).stem + '.features.npy')
        np.save(target, features.numpy(), allow_pickle=False)
        files[name] = {'feature_file': target.name, 'sha256': sha256_file(target),
                       'image_sha256': sha256_file(data_root/'images'/name),
                       'shape': list(features.shape), 'dtype': 'float32',
                       'token_norm_min_max': [float(features.norm(dim=0).min()),float(features.norm(dim=0).max())]}
        print('Cached frozen DINO:',name,flush=True)
    supports = {}
    for name in support_names:
        mask_path = data_root/'mask'/(Path(name).stem+'_label.bmp')
        target = output/(Path(name).stem+'.support_mask.npz')
        masks = aligned_support_masks(mask_path)
        np.savez_compressed(target, **masks)
        supports[name] = {'mask_file':target.name,'sha256':sha256_file(target),
                          'source_mask_sha256':sha256_file(mask_path),
                          'patch_foreground_count':int(masks['binary_patch_nearest'].sum()),
                          'rule':'class 12 ->1, nearest 1024->64 (source index 16*i). Area fractions are diagnostic only.'}
    text = encode_text(text_encoder, device)
    text_path = output/'single_prompt_text.npy'
    np.save(text_path,text.numpy(),allow_pickle=False)
    manifest = {'schema':'dmsf-frozen-features-v1','alignment_trained':False,
                'features_before_extra_l2':True, 'query_masks_read_by_export':False,
                'images': files, 'supports': supports,
                'text':{'file':text_path.name,'sha256':sha256_file(text_path),'shape':[1,512],'prompt':PROMPT},
                'metadata':metadata}
    write_json(output/'cache_manifest.json',manifest)
    return manifest


def load_episode(cache_dir: Path, support_name: str, query_name: str, device: torch.device):
    manifest = json.loads((cache_dir/'cache_manifest.json').read_text(encoding='utf8'))
    if manifest.get('schema') != 'dmsf-frozen-features-v1' or manifest.get('features_before_extra_l2') is not True:
        raise ValueError('Wrong cache schema or normalized feature cache.')
    metadata = manifest.get('metadata', {})
    dino_info = metadata.get('dino', {})
    text_info = metadata.get('georsclip_text', {})
    if (dino_info.get('checkpoint') != WEIGHT_NAME or
        not dino_info.get('checkpoint_sha256','').startswith('eadcf0ff') or
        dino_info.get('block_index_zero_based') != 23 or
        tuple(dino_info.get('mean',())) != SAT_MEAN or tuple(dino_info.get('std',())) != SAT_STD or
        metadata.get('dinov3_source_commit') != DINOV3_COMMIT or
        text_info.get('checkpoint_sha256') != GEORSCLIP_SHA256 or
        text_info.get('prompt_count') != 1 or manifest.get('text',{}).get('prompt') != PROMPT):
        raise ValueError('Cached encoder/preprocessing/prompt provenance differs from DMSF v1 contract.')
    if support_name == query_name:
        raise ValueError('Support/query must be different images.')
    def checked(entry, key):
        path = cache_dir/entry[key]
        if path.parent.resolve() != cache_dir.resolve() or sha256_file(path) != entry['sha256']:
            raise ValueError('Cache path/integrity mismatch.')
        return path
    arrays = []
    for name in [support_name,query_name]:
        array = np.load(checked(manifest['images'][name],'feature_file'),allow_pickle=False)
        if array.shape != (1024,64,64) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('Bad cached feature shape/dtype/value.')
        arrays.append(torch.from_numpy(array).unsqueeze(0).to(device))
    with np.load(checked(manifest['supports'][support_name],'mask_file'),allow_pickle=False) as archive:
        full = archive['binary_full']
        nearest = archive['binary_patch_nearest']
        if full.shape != (1024,1024) or not set(np.unique(full)) <= {0,1} or not np.array_equal(full[::16,::16],nearest) or nearest.sum() <= 0:
            raise ValueError('Bad cached support mask alignment.')
        mask = torch.from_numpy(full.astype(np.float32))[None,None].to(device)
    text = np.load(checked(manifest['text'],'file'),allow_pickle=False)
    if text.shape != (1,512) or text.dtype != np.float32 or not np.isfinite(text).all() or not np.allclose(np.linalg.norm(text,axis=-1),1,atol=1e-5):
        raise ValueError('Bad cached text embedding.')
    return (*arrays,mask,torch.from_numpy(text).to(device))
