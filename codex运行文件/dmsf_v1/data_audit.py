"""Content audit and deterministic coordinate-group split; never select by masks.

Coordinates have UNKNOWN units/CRS and tile footprints. This is a provisional
coordinate-separated split, not proof of independent geographic regions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')


def coordinate_groups(names: list[str], radius: float = 1000.0) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    if radius <= 0 or not names or len(set(names)) != len(names):
        raise ValueError('Require positive radius and unique image names.')
    parsed = [re.fullmatch(r'(PV\d+)_(\d+)_(\d+)\.bmp', n) for n in names]
    if any(p is None for p in parsed) or len({p[1] for p in parsed}) != 1:
        raise ValueError('Require one known filename prefix; do not mix unverified coordinate systems.')
    xy = np.asarray([(int(p[2]), int(p[3])) for p in parsed], dtype=np.int64)
    distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    seen, groups = set(), []
    for start in range(len(names)):
        if start in seen:
            continue
        queue, group = [start], []
        seen.add(start)
        while queue:
            i = queue.pop()
            group.append(i)
            for j in np.flatnonzero(distances[i] <= radius):
                j = int(j)
                if j not in seen:
                    seen.add(j)
                    queue.append(j)
        groups.append(sorted(group))
    return groups, xy, distances


def create_split(names: list[str], legacy: set[str], radius: float = 1000.0) -> dict:
    groups, xy, distances = coordinate_groups(names, radius)
    records = []
    for indices in groups:
        members = [names[i] for i in indices]
        digest = hashlib.sha256(('dmsf-spatial-v1\n' + '\n'.join(members)).encode()).hexdigest()
        records.append({'group_id': digest[:16], 'allocation_hash': digest,
                        'members': members, 'legacy_exposed': bool(set(members) & legacy)})
    clean = sorted([r for r in records if not r['legacy_exposed']], key=lambda r: r['allocation_hash'])
    if len(clean) < 5:
        raise ValueError('Too few unexposed groups for held-out validation/test.')
    # 20% of unexposed GROUPS for test, 20% validation; counts are not mask-balanced.
    holdout = max(1, int(np.ceil(len(clean) * 0.2)))
    for i, r in enumerate(clean):
        r['split'] = 'test' if i < holdout else 'validation' if i < 2 * holdout else 'train'
    for r in records:
        if r['legacy_exposed']:
            r['split'] = 'legacy_development'
    assignments = {n: r['split'] for r in records for n in r['members']}
    cross = {}
    labels = sorted(set(assignments.values()))
    for i, a in enumerate(labels):
        ia = [k for k, n in enumerate(names) if assignments[n] == a]
        for b in labels[i+1:]:
            ib = [k for k, n in enumerate(names) if assignments[n] == b]
            cross[a + '__' + b] = float(distances[np.ix_(ia, ib)].min())
    return {'version': 1, 'status': 'provisional_coordinate_group_split',
            'units': 'filename coordinate units; CRS, physical units and footprint unknown',
            'rule': 'Connected components of Euclidean distance <=1000. Entire legacy-exposed components quarantined. Remaining groups sorted by SHA256(dmsf-spatial-v1 + sorted names); first ceil(20%) test, next ceil(20%) validation, remainder train.',
            'radius': radius, 'mask_values_or_metrics_used_for_assignment': False,
            'legacy_names': sorted(legacy), 'groups': records,
            'sample_counts': dict(Counter(assignments.values())),
            'group_counts': dict(Counter(r['split'] for r in records)),
            'minimum_cross_split_coordinate_distance': cross,
            'coordinate_min': xy.min(0).tolist(), 'coordinate_max': xy.max(0).tolist(),
            'test_access_policy': 'Only structural validation/hashes so far; no test CAM, loss, metric or visual panel. No test-based prompt/threshold/hyperparameter selection.',
            'limitation': 'Coordinate separation is verified; true region/date independence and non-overlapping footprints are NOT established without source georeferencing.'}


def audit(data_root: Path, legacy_manifest: Path, output: Path) -> dict:
    images = {p.name: p for p in (data_root / 'images').glob('*.bmp')}
    masks = {p.name.removesuffix('_label.bmp') + '.bmp': p for p in (data_root / 'mask').glob('*_label.bmp')}
    if set(images) != set(masks) or not images:
        raise ValueError('Unpaired files: ' + str(sorted(set(images) ^ set(masks))))
    names = sorted(images)
    legacy_obj = json.loads(legacy_manifest.read_text(encoding='utf8'))
    legacy = {legacy_obj['support_image'], *legacy_obj['query_images']}
    if not legacy <= set(names):
        raise ValueError('Legacy manifest names missing.')
    # Assign and save BEFORE opening any mask. Never stratify by foreground coverage.
    split = create_split(names, legacy)
    write_json(output / 'split_manifest.json', split)
    assignments = {n: g['split'] for g in split['groups'] for n in g['members']}
    records, codes, modes = [], Counter(), Counter()
    duplicates = defaultdict(list)
    for name in names:
        with Image.open(images[name]) as im:
            im.load()
            if im.size != (1024, 1024) or im.mode != 'RGB':
                raise ValueError('Unexpected image geometry/mode: ' + name)
            pixel_hash = hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
            info_keys = sorted(im.info)
        with Image.open(masks[name]) as im:
            im.load()
            if im.size != (1024, 1024) or im.mode != 'L':
                raise ValueError('Unexpected mask geometry/mode: ' + name)
            values = np.unique(np.asarray(im)).tolist()
            if not set(values) <= {0, 12}:
                raise ValueError('Unexpected mask codes: ' + name)
        codes.update(values)
        modes['RGB_image_L_mask'] += 1
        duplicates[pixel_hash].append(name)
        records.append({'image': name, 'split': assignments[name],
                        'image_bytes': images[name].stat().st_size,
                        'image_sha256': sha256_file(images[name]),
                        'pixel_sha256': pixel_hash,
                        'mask_bytes': masks[name].stat().st_size,
                        'mask_sha256': sha256_file(masks[name]),
                        'image_info_keys': info_keys})
    duplicate_groups = [v for v in duplicates.values() if len(v) > 1]
    crossing = [g for g in duplicate_groups if len({assignments[n] for n in g}) > 1]
    _, _, distances = coordinate_groups(names)
    np.fill_diagonal(distances, np.inf)
    result = {'paired_images': len(names), 'image_shape': [1024, 1024, 3],
              'mask_shape': [1024, 1024], 'mask_codes_observed': sorted(codes),
              'mask_audit_scope': 'Structure/allowed codes/hash only; no per-test label coverage or metrics used.',
              'duplicate_pixel_groups': duplicate_groups, 'cross_split_exact_duplicates': crossing,
              'nearest_coordinate_distance_quantiles': dict(zip(['min','p10','p25','median','p75','p90','max'], np.quantile(distances.min(1),[0,.1,.25,.5,.75,.9,1]).tolist())),
              'sample_counts': split['sample_counts'], 'group_counts': split['group_counts'],
              'split_sha256': sha256_file(output / 'split_manifest.json'),
              'georeferencing': 'BMP contains no verified CRS/GSD/tile footprint. Numerical distance is not metres.',
              'records': records}
    write_json(output / 'local_inventory.json', result)
    if crossing:
        raise ValueError('Exact duplicate pixels cross splits; revise grouping before training.')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--legacy-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    audit(a.data_root, a.legacy_manifest, a.output)
