"""Kaggle GPU feature export + synthetic VJP acceptance. ZERO training steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from data_audit import sha256_file, write_json
from dmsf import DMSF, gradient_audit
from feature_cache import (DINOV3_COMMIT, WEIGHT_NAME, assert_frozen, state_sha256,
                           load_dino, load_text_encoder, export_cache, load_episode)

HERE = Path(__file__).resolve().parent


def find_data(input_root: Path, working: Path, required: set[str]) -> Path:
    candidates = []
    for p in input_root.rglob('images'):
        if p.is_dir() and (p.parent/'mask').is_dir():
            names = {x.name for x in p.glob('PV08_*.bmp')}
            if required <= names:
                candidates.append((len(names),p.parent))
    if not candidates:
        for archive in input_root.rglob('pv663.zip'):
            # Keep an unexpanded input archive out of /kaggle/working outputs.
            # The disposable VM's temp directory is not an exported artifact.
            target = Path(tempfile.mkdtemp(prefix='dmsf-input-'))
            with zipfile.ZipFile(archive) as z:
                for item in z.infolist():
                    resolved = (target/item.filename).resolve()
                    if target.resolve() not in resolved.parents:
                        raise ValueError('Unsafe archive path.')
                z.extractall(target)
            return find_data(target,working,required)
        raise FileNotFoundError('No paired dataset contains requested exact filenames.')
    # Prefer complete source snapshot when both legacy subset and full dataset are mounted.
    return sorted(candidates,key=lambda item:(-item[0],str(item[1])))[0][1]


def verify_mounted_data(data_root: Path, inventory: dict) -> dict:
    expected = {r['image']:r for r in inventory['records']}
    images = {p.name:p for p in (data_root/'images').glob('*.bmp')}
    masks = {p.name.removesuffix('_label.bmp')+'.bmp':p for p in (data_root/'mask').glob('*_label.bmp')}
    if set(images) != set(masks) or not set(images) <= set(expected):
        raise ValueError('Mounted image/mask names disagree with each other or local inventory.')
    for name in sorted(images):
        r = expected[name]
        if sha256_file(images[name]) != r['image_sha256'] or sha256_file(masks[name]) != r['mask_sha256']:
            raise ValueError('Local/Kaggle byte mismatch: '+name)
        with Image.open(images[name]) as im:
            if im.size != (1024,1024) or im.mode != 'RGB': raise ValueError('Bad image structure.')
        with Image.open(masks[name]) as im:
            if im.size != (1024,1024) or im.mode != 'L' or not set(np.unique(np.asarray(im))) <= {0,12}:
                raise ValueError('Bad mask structure/codes.')
    return {'mounted_root':str(data_root),'paired_images':len(images),
            'all_mounted_image_and_mask_sha256_match_local':True,
            'complete_663_snapshot':len(images)==len(expected)==663,
            'masks_accessed_for_structural_integrity_only':True,
            'foreground_coverage_or_test_metrics_computed':False,
            'missing_local_names': sorted(set(expected)-set(images))}


def resolve_stage_data(input_root: Path, working: Path, required: set[str], config: dict,
                       download_dataset=None) -> Path:
    root = find_data(input_root,working,required)
    count = len(list((root/'images').glob('*.bmp')))
    if count == config['required_pair_count']:
        return root  # Full per-file SHA256 comparison still follows this resolution.
    parts = config['dataset_ref'].split('/')
    if len(parts) != 3 or not parts[2].isdigit():
        raise ValueError(f"Expected {config['required_pair_count']} pairs; mounted {count}; no pinned version available.")
    if download_dataset is None:
        import kagglehub
        download_dataset = kagglehub.dataset_download
    handle = '/'.join(parts[:2]) + '/versions/' + parts[2]
    print(f'Mounted input has {count} pairs; explicitly resolving historical dataset {handle}.',flush=True)
    pinned = Path(download_dataset(handle))
    root = find_data(pinned,working,required)
    count = len(list((root/'images').glob('*.bmp')))
    if count != config['required_pair_count']:
        raise ValueError(f'Historical dataset resolver returned {count} pairs; refusing wrong input.')
    return root


def ensure_dino_repo(path: Path) -> None:
    if not path.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',
                        'https://github.com/facebookresearch/dinov3.git',str(path)],check=True)
    subprocess.run(['git','-C',str(path),'checkout','--detach',DINOV3_COMMIT],check=True)
    actual = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual != DINOV3_COMMIT: raise RuntimeError('DINO source commit mismatch.')


def main(args) -> None:
    if not torch.cuda.is_available(): raise RuntimeError('Stage-1 real encoder acceptance requires Kaggle GPU.')
    torch.manual_seed(42); np.random.seed(42)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    inventory = json.loads((HERE/'local_inventory.json').read_text(encoding='utf8'))
    stage_config = json.loads((HERE/'stage_config.json').read_text(encoding='utf8'))
    benchmark = json.loads((HERE/'benchmark_v1.json').read_text(encoding='utf8'))
    support_name = benchmark['support_image']
    query_names = benchmark['query_images'][:2]
    selection = [support_name,*query_names]
    if args.selection_json:
        selection = json.loads(args.selection_json.read_text(encoding='utf8'))['image_names']
        if not {support_name,*query_names} <= set(selection):
            raise ValueError('Export selection must contain fixed engineering acceptance episodes.')
    root = resolve_stage_data(args.input_root,output,set(selection),stage_config)
    mounted = verify_mounted_data(root,inventory)
    if mounted['paired_images'] != stage_config['required_pair_count']:
        raise ValueError('Mounted dataset pair count differs from pinned stage configuration.')
    mounted['dataset_ref'] = stage_config['dataset_ref']
    write_json(output/'mounted_data_audit.json',mounted)
    print('Mounted paired files verified:',mounted['paired_images'],flush=True)
    repo = args.output.parent/'dinov3_pinned'
    ensure_dino_repo(repo)
    weights = list(args.input_root.rglob(WEIGHT_NAME))
    if len(weights) != 1: raise FileNotFoundError('Require one mounted official DINOv3 SAT checkpoint.')
    dino, dino_info = load_dino(repo,weights[0],device)
    text_encoder, text_info = load_text_encoder(args.output.parent/'RS5M_ViT-B-32.pt',device)
    encoders = {'dino':dino,'georsclip_text':text_encoder}
    before = {name:state_sha256(model) for name,model in encoders.items()}
    metadata = {'dino':dino_info,'georsclip_text':text_info,'dinov3_source_commit':DINOV3_COMMIT,
                'stage_config':stage_config,
                'code_sha256':{p.name:sha256_file(p) for p in sorted(HERE.glob('*.py'))},
                'python':platform.python_version(),'torch':torch.__version__,
                'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
                'seed':42,'local_inventory_sha256':sha256_file(HERE/'local_inventory.json'),
                'split_manifest_sha256':sha256_file(HERE/'split_manifest.json'),
                'selection_policy':'Fixed legacy support and first two manifest queries for engineering; no held-out test features in default smoke run.'}
    cache = output/'cache'
    manifest = export_cache(dino,text_encoder,root,selection,[support_name],cache,device,metadata)
    # Reload from disk, proving the cache is sufficient and detached from encoders.
    episodes = [load_episode(cache,support_name,q,device) for q in query_names]
    support,query,mask,text = [torch.cat([e[i] for e in episodes],dim=0) for i in range(4)]
    model = DMSF().to(device)
    report = gradient_audit(model,support,query,mask,text)
    with torch.no_grad():
        maps = model(support,query,mask,text)
        for i,name in enumerate(query_names):
            arrays = {k:maps[k][i,0].cpu().numpy() for k in
                      ['spatial_raw','semantic_raw','spatial_cam','semantic_cam','fused_cam']}
            full = F.interpolate(maps['fused_cam'][i:i+1],size=(1024,1024),mode='bilinear',align_corners=False)
            if full.shape != (1,1,1024,1024) or not torch.isfinite(full).all() or full.min()<0 or full.max()>1:
                raise RuntimeError('Upsampled fused CAM failed validation.')
            arrays['fused_cam_upsampled_untrained'] = full[0,0].cpu().numpy()
            np.savez_compressed(output/(Path(name).stem+'.untrained_probe_maps.npz'),**arrays)
    frozen = {}
    for name,encoder in encoders.items():
        assert_frozen(encoder)
        after = state_sha256(encoder)
        frozen[name] = {'eval_mode':True,'trainable_parameters':0,
                        'parameters_with_grad':sum(p.grad is not None for p in encoder.parameters()),
                        'state_sha256_before':before[name],'state_sha256_after':after,
                        'all_parameters_and_buffers_unchanged':before[name]==after}
        if before[name] != after: raise RuntimeError('Frozen encoder state changed.')
    report.update({'status':'passed','encoders':frozen,'episodes':[
        {'support':support_name,'query':q,'support_count':1} for q in query_names],
        'metadata':metadata,'dmsf_config':model.config_dict(),
        'trainable_dmsf_parameter_count':sum(p.numel() for p in model.parameters()),
        'cache_image_count':len(manifest['images']), 'mounted_data':mounted,
        'training_annotations_used':0,'support_masks_used_for_prototype':1,
        'supervised_training_performed':False,'segmentation_quality_claim':False,
        'query_masks_used_by_forward_or_probe':False,
        'map_warning':'Randomly initialized projections. These maps are engineering outputs, not learned alignment or evaluated segmentation.'})
    torch.save({'state_dict':model.state_dict(),'config':model.config_dict(),
                'alignment_trained':False,'optimizer_steps':0},output/'untrained_initial_state.pt')
    write_json(output/'acceptance_report.json',report)
    print(json.dumps({'status':report['status'],'gpu':metadata['gpu'],
                      'gradient_norms':report['gradient_norms'],'encoder_states_unchanged':True,
                      'optimizer_steps':0,'alignment_trained':False},indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working/dmsf_stage1'))
    parser.add_argument('--selection-json',type=Path)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        write_json(arguments.output/'failure.json',{'type':type(error).__name__,'message':str(error),
                  'optimizer_steps':0,'alignment_trained':False})
        raise
