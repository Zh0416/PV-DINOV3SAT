"""Kaggle GPU training/validation for controlled global scalar fusion."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import platform
import random
import sys
import tempfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'dmsf_stage2'))
sys.path.insert(0, str(HERE.parent / 'dmsf_v1'))

import numpy as np
import torch
from torch.nn import functional as F
from data_audit import sha256_file, write_json
from feature_cache import (DINOV3_COMMIT, PROMPT, WEIGHT_NAME, assert_frozen,
                           encode_text, load_dino, load_text_encoder, state_sha256)
from run_stage1 import ensure_dino_repo, find_data, verify_mounted_data
from stage2_data import StageData
from stage2_metrics import aggregate, confusion, from_counts
from stage3_model import Segmenter, gradient_norms, supervised_loss


def variants():
    for normalization in ('minmax', 'calibrated'):
        yield f'spatial_{normalization}', {'branch': 'spatial', 'normalization': normalization}
        for beta_name, beta in (('beta07', 0.7), ('weight085', math.log(.85 / .15))):
            for text_kind in ('real', 'random'):
                yield f'fusion_{normalization}_{beta_name}_{text_kind}', {
                    'branch': 'fusion', 'normalization': normalization,
                    'text_kind': text_kind, 'beta_init': beta}


@torch.no_grad()
def initialize_source_calibration(model, data, episodes, text, device):
    """Estimate raw class centers and scale from source TRAIN query masks only."""
    if model.normalization != 'calibrated':
        return {}
    model.eval()
    stats = {}
    for name, calibrator in [('spatial', model.spatial_calibrator),
                             ('semantic', model.semantic_calibrator)]:
        if calibrator is None:
            continue
        foreground, background = [], []
        for e in episodes:
            inputs, target = data.batch([e], 'train', text, device)
            score = model(*inputs)[name + '_raw']
            if target.bool().any() and (~target.bool()).any():
                foreground.append(float(score[target.bool()].mean()))
                background.append(float(score[~target.bool()].mean()))
        if not foreground:
            raise RuntimeError('No two-class source calibration episodes.')
        fg, bg = float(np.mean(foreground)), float(np.mean(background))
        threshold = (fg + bg) / 2
        temperature = max(.03, min(.5, abs(fg - bg) / 2))
        calibrator.source_threshold.fill_(threshold)
        calibrator.log_temperature.fill_(math.log(temperature))
        stats[name] = {'source_foreground_mean': fg, 'source_background_mean': bg,
                       'initial_source_threshold': threshold, 'initial_temperature': temperature,
                       'source_episode_count': len(foreground)}
    return stats


@torch.no_grad()
def evaluate(model, data, episodes, text, device, output=None):
    model.eval()
    rows = []
    for e in episodes:
        inputs = data.batch([e], 'validation', text, device, include_targets=False)
        maps = model(*inputs)
        score = F.interpolate(maps['logits'], (1024, 1024), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        target = data.label(e['query'], 'validation_query')
        counts = confusion(target, score >= 0)
        row = {**e, **counts, **from_counts(counts),
               'target_foreground_pixels': int(target.sum()),
               'spatial_cam_max': float(maps['spatial_cam'].max()),
               'spatial_cam_fraction_ge_half': float((maps['spatial_cam'] >= .5).float().mean()),
               'selected_cam_max': float(maps['selected_cam'].max()),
               'spatial_weight': float(maps['spatial_weight'])}
        if maps['semantic_cam'] is not None:
            row['semantic_cam_max'] = float(maps['semantic_cam'].max())
        rows.append(row)
    summary, groups = aggregate(rows)
    empty = [r for r in rows if r['target_foreground_pixels'] == 0]
    summary['empty_query_diagnostic'] = {
        'count': len(empty),
        'mean_spatial_cam_max': float(np.mean([r['spatial_cam_max'] for r in empty])) if empty else None,
        'mean_spatial_cam_fraction_ge_half': float(np.mean([r['spatial_cam_fraction_ge_half'] for r in empty])) if empty else None,
        'mean_selected_cam_max': float(np.mean([r['selected_cam_max'] for r in empty])) if empty else None}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'per_image.json', rows)
        write_json(output / 'per_group.json', groups)
        write_json(output / 'summary.json', summary)
    return summary


def train_one(name, spec, seed, data, schedule, validation, text, device, output, protocol, hashes):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = Segmenter(**spec, random_text_seed=protocol['random_text_seed'],
                      calibration_shrink=protocol['support_calibration_shrink']).to(device)
    source_calibration = initialize_source_calibration(
        model, data, schedule[0][:protocol['source_calibration_episodes']], text, device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=protocol['learning_rate'],
                                  weight_decay=protocol['weight_decay'])
    directory = output / f'seed_{seed}' / name
    directory.mkdir(parents=True, exist_ok=True)
    history, best, best_epoch, steps = [], -float('inf'), None, 0
    gradient_max = {}
    for epoch, episodes in enumerate(schedule, 1):
        model.train()
        loss_sum = 0.
        first_gradients = None
        for start in range(0, len(episodes), protocol['batch_size']):
            batch = episodes[start:start + protocol['batch_size']]
            inputs, target = data.batch(batch, 'train', text, device)
            optimizer.zero_grad(set_to_none=True)
            maps = model(*inputs)
            loss, _ = supervised_loss(maps, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            loss.backward()
            norms = gradient_norms(model)
            if first_gradients is None:
                first_gradients = norms
            for key, value in norms.items():
                if value is not None:
                    gradient_max[key] = max(gradient_max.get(key, 0.), value)
            torch.nn.utils.clip_grad_norm_(params, protocol['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            loss_sum += float(loss.detach()) * len(batch)
        validation_summary = evaluate(model, data, validation, text, device)
        score = validation_summary['macro_group']['dice']
        if score > best:
            best, best_epoch = score, epoch
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'identity': model.identity(), 'seed': seed, 'epoch': epoch,
                        'validation_group_dice': score, 'optimizer_steps': steps,
                        'source_hashes': hashes}, directory / 'best.pt')
        record = {'epoch': epoch, 'optimizer_steps': steps, 'train_loss': loss_sum / len(episodes),
                  'validation_macro_group_dice': score, 'validation_micro_foreground_iou': validation_summary['micro']['foreground_iou'],
                  'spatial_weight': float(model.dmsf.fusion_logit.detach().sigmoid()),
                  'first_batch_gradient_norms': first_gradients}
        history.append(record)
        write_json(directory / 'history.json', history)
        print(json.dumps({'seed': seed, 'variant': name, 'epoch': epoch,
                          'loss': record['train_loss'], 'val_group_dice': score,
                          'best_epoch': best_epoch}), flush=True)
    if any(value <= 0 for value in gradient_max.values()):
        raise RuntimeError('A trainable parameter group never received a gradient.')
    changed = {key: float((value.detach().cpu() - initial[key]).float().norm())
               for key, value in model.state_dict().items()}
    required = ['dmsf.visual_projection.0.weight', 'dmsf.spatial_refiner.net.2.weight',
                'head.0.weight', 'head.2.weight']
    if model.branch == 'fusion':
        required += ['dmsf.text_to_visual.0.weight', 'dmsf.text_projection.0.weight',
                     'dmsf.semantic_refiner.net.2.weight', 'dmsf.fusion_logit']
    if model.normalization == 'calibrated':
        required += ['spatial_calibrator.source_threshold', 'spatial_calibrator.log_temperature']
        if model.branch == 'fusion':
            required += ['semantic_calibrator.source_threshold', 'semantic_calibrator.log_temperature']
    if any(changed[key] <= 0 for key in required):
        raise RuntimeError('A required trainable parameter did not change.')
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
    restored = Segmenter(**spec, random_text_seed=protocol['random_text_seed'],
                         calibration_shrink=protocol['support_calibration_shrink']).to(device)
    restored.load_state_dict(checkpoint['state_dict'])
    final_val = evaluate(restored, data, validation, text, device, directory / 'validation')
    if abs(final_val['macro_group']['dice'] - best) > 1e-12:
        raise RuntimeError('Reloaded checkpoint differs from selection score.')
    result = {'variant': name, 'seed': seed, 'identity': model.identity(),
              'epochs': len(schedule), 'optimizer_steps': steps,
              'trainable_parameters': sum(p.numel() for p in params),
              'source_calibration': source_calibration,
              'best_epoch': best_epoch, 'best_validation_macro_group_dice': best,
              'validation': final_val, 'gradient_norm_max': gradient_max,
              'parameter_change_l2': changed,
              'best_checkpoint_sha256': sha256_file(directory / 'best.pt')}
    write_json(directory / 'training_summary.json', result)
    del model, restored, optimizer, initial
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(args):
    if not torch.cuda.is_available() or not Path('/kaggle').exists():
        raise RuntimeError('Real encoder training requires Kaggle GPU.')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf8'))
    split = json.loads((HERE / 'split_manifest.json').read_text(encoding='utf8'))
    episodes = json.loads((HERE / 'stage2_episode_proposal.json').read_text(encoding='utf8'))
    inventory = json.loads((HERE / 'local_inventory.json').read_text(encoding='utf8'))
    hashes = {p.name: sha256_file(p) for p in HERE.glob('*.py')}
    hashes.update({'protocol.json': sha256_file(HERE / 'protocol.json'),
                   'split_manifest.json': sha256_file(HERE / 'split_manifest.json'),
                   'stage2_episode_proposal.json': sha256_file(HERE / 'stage2_episode_proposal.json')})
    if hashes['split_manifest.json'] != protocol['split_sha256'] or hashes['split_manifest.json'] != episodes['split_sha256']:
        raise ValueError('Split integrity mismatch.')
    write_json(output / 'source_manifest.json', hashes)
    write_json(output / 'protocol.json', protocol)
    root = find_data(args.input_root, output, {r['image'] for r in inventory['records']})
    mounted = verify_mounted_data(root, inventory)
    if not mounted['complete_663_snapshot']:
        raise ValueError('Exact 663-pair snapshot required.')
    write_json(output / 'mounted_data_audit.json', mounted)
    data = StageData(root, output / 'cache', split, inventory,
                     hashes['protocol.json'], hashes['split_manifest.json'])
    support_audit = data.validate_supports(episodes, 'validation')
    write_json(output / 'validation_support_audit.json', support_audit)
    if not support_audit['valid']:
        raise RuntimeError('Invalid fixed validation supports.')
    validation = episodes['held_out']['validation']['episodes']
    schedules = {}
    for seed in protocol['seeds']:
        schedules[seed], excluded = data.training_schedule(protocol['epochs'], seed)
        if len(excluded) != 1:
            raise RuntimeError('Unexpected invalid training support count.')
    write_json(output / 'training_episode_schedules.json',
               {'seeds': schedules, 'excluded_supports': excluded})
    repo = Path(tempfile.mkdtemp(prefix='dmsf-stage3-runtime-')) / 'dinov3'
    ensure_dino_repo(repo)
    weights = list(args.input_root.rglob(WEIGHT_NAME))
    if len(weights) != 1:
        raise FileNotFoundError('One official SAT checkpoint required.')
    dino, dino_info = load_dino(repo, weights[0], device)
    encoder, text_info = load_text_encoder(repo.parent / 'RS5M_ViT-B-32.pt', device)
    before = {'dino': state_sha256(dino), 'georsclip_text': state_sha256(encoder)}
    text = encode_text(encoder, device).to(device)
    np.save(output / 'single_prompt_text.npy', text.cpu().numpy(), allow_pickle=False)
    data.manifest['metadata'] = {'dino': dino_info, 'georsclip_text': text_info,
                                  'prompt': PROMPT, 'dinov3_source_commit': DINOV3_COMMIT}
    data.export(data.names('train') + data.names('validation'), dino, device)
    results = {}
    for seed in protocol['seeds']:
        for name, spec in variants():
            results[f'seed_{seed}/{name}'] = train_one(
                name, spec, seed, data, schedules[seed], validation, text,
                device, output, protocol, hashes)
            write_json(output / 'progress.json', results)
    frozen = {}
    for name, module in [('dino', dino), ('georsclip_text', encoder)]:
        assert_frozen(module)
        after = state_sha256(module)
        if after != before[name]:
            raise RuntimeError('Frozen encoder changed: ' + name)
        frozen[name] = {'before_sha256': before[name], 'after_sha256': after,
                        'unchanged': True, 'trainable_parameters': 0, 'parameters_with_grad': 0}
    data.write_access(output / 'access_final.json')
    if data.test_unlocked or any(role.startswith('test') for _, role in data.access):
        raise RuntimeError('Previously used test partition was accessed.')
    write_json(output / 'results.json', {
        'status': 'completed', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'gpu': torch.cuda.get_device_name(0), 'protocol_sha256': hashes['protocol.json'],
        'split_sha256': hashes['split_manifest.json'], 'frozen_encoders': frozen,
        'validation_queries': len(validation), 'training_images': len(data.names('train')),
        'test_query_labels_used': False, 'test_used_for_selection': False,
        'results': results})
    print('STAGE3_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/dmsf_stage3'))
    args = parser.parse_args()
    main(args)
