"""Kaggle entry point for one arbitrary-size support/query pair.

Set the five DMSF_* environment variables in the notebook before running this
script. The code tree can be a mounted Kaggle dataset or a checked-out Git
repository; no training is performed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def find_file(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one {filename} under {root}; found {len(matches)}.')
    return matches[0]


def main() -> None:
    code_root = env_path('DMSF_CODE_ROOT', '/kaggle/working/PV-DINOV3SAT')
    if not (code_root / 'codex运行文件' / 'dmsf_fusion_v1' / 'infer.py').is_file():
        raise FileNotFoundError(
            'Mount the repository/code bundle at DMSF_CODE_ROOT or clone the branch in the notebook.')

    fusion_root = code_root / 'codex运行文件' / 'dmsf_fusion_v1'
    sys.path.insert(0, str(fusion_root))
    from infer import encode_raw_variable, load_model, predict_original_size
    import numpy as np
    import torch

    support_image = env_path('DMSF_SUPPORT_IMAGE', '/kaggle/input/my-selected-images/support.png')
    support_mask = env_path('DMSF_SUPPORT_MASK', '/kaggle/input/my-selected-images/support_mask.png')
    query_image = env_path('DMSF_QUERY_IMAGE', '/kaggle/input/my-selected-images/query.png')
    output = env_path('DMSF_OUTPUT', '/kaggle/working/dmsf_prediction.npz')
    dino_repo = env_path('DMSF_DINO_REPO', '/kaggle/working/dinov3')
    dino_weights = env_path('DMSF_DINO_WEIGHTS', '/kaggle/input/dinov3-vitl16-pretrain-sat493m')
    geo_weights = env_path('DMSF_GEO_WEIGHTS', '/kaggle/working/RS5M_ViT-B-32.pt')
    device = torch.device(os.environ.get('DMSF_DEVICE', 'cuda'))

    if not dino_repo.is_dir():
        subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/dinov3.git', str(dino_repo)], check=True)
    subprocess.run(['git', '-C', str(dino_repo), 'checkout', '--detach',
                    '6876159a11b4df116f30f667f8c9888617df0751'], check=True)
    if dino_weights.is_dir():
        dino_weights = find_file(dino_weights, 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')

    inputs, geometry = encode_raw_variable(
        support_image, support_mask, query_image, dino_repo, dino_weights,
        geo_weights, device)
    model = load_model(device)
    result = predict_original_size(model, inputs, geometry)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    metadata = {
        'checkpoint': str(fusion_root / 'checkpoints' / 'best.pt'),
        'query_image': str(query_image), 'support_image': str(support_image),
        'support_mask': str(support_mask), 'letterbox_geometry': geometry,
        'output_resolution': list(result['logits_full'].shape),
        'threshold': 'logits_full >= 0',
    }
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf8')
    print(f'Wrote {output}')
    print(f'Original query resolution: {result["logits_full"].shape[1]}x{result["logits_full"].shape[0]}')


if __name__ == '__main__':
    main()
