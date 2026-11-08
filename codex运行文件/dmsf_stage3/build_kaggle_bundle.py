"""Package fixed source, protocol and manifests as a private Kaggle script."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
STAGE1 = HERE.parent / 'dmsf_v1'
STAGE2 = HERE.parent / 'dmsf_stage2'


def build():
    sources = {name: HERE / name for name in
               ('stage3_model.py', 'run_stage3.py', 'test_stage3.py')}
    for name in ('dmsf.py', 'feature_cache.py', 'data_audit.py', 'run_stage1.py'):
        sources[name] = STAGE1 / name
    for name in ('stage2_data.py', 'stage2_metrics.py'):
        sources[name] = STAGE2 / name
    sources['run_e1.py'] = HERE.parent / 'georsclip_semantic_cam_e1' / 'run_e1.py'
    for name in ('local_inventory.json', 'split_manifest.json', 'stage2_episode_proposal.json'):
        sources[name] = STAGE1 / 'audit' / name
    sources['protocol.json'] = HERE / 'protocol.json'
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(sources.items()):
            archive.writestr(name, path.read_bytes())
    script = '''# Generated source bundle; SHA-256 manifest is shipped beside this script.
import base64,io,pathlib,subprocess,sys,zipfile
root=pathlib.Path('/kaggle/working/dmsf_stage3_source');root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(root)
subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root),'-p','test_stage3.py','-v'],check=True)
subprocess.run([sys.executable,str(root/'run_stage3.py')],check=True,cwd=root)
'''.replace('PAYLOAD', repr(base64.b64encode(payload.getvalue()).decode()))
    out = HERE / 'kaggle_bundle'
    out.mkdir(exist_ok=True)
    (out / 'run-dmsf-stage3.py').write_text(script, encoding='utf8')
    metadata = {'id': 'zh0416/pv-dmsf-global-scalar-v1',
                'title': 'PV DMSF Global Scalar Fusion',
                'code_file': 'run-dmsf-stage3.py', 'language': 'python',
                'kernel_type': 'script', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
                'dataset_sources': ['zh0416/pv-dinov3sat-663-v1'],
                'model_sources': ['yadavdamodar/dinov3-vitl16-pretrain-sat493m/PyTorch/default/1'],
                'kernel_sources': [], 'competition_sources': []}
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf8')
    manifest = {'files': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in sources.items()},
                'generated_script_sha256': hashlib.sha256(script.encode()).hexdigest()}
    (out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    print('Built:', out)


if __name__ == '__main__':
    build()
