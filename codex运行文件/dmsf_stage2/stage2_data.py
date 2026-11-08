"""Split-aware feature/label access and deterministic cross-group episodes."""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'dmsf_v1'))

import numpy as np
from PIL import Image
import torch
from data_audit import sha256_file, write_json
from feature_cache import encode_dino


class StageData:
    def __init__(self, data_root, cache, split, inventory, protocol_sha, split_sha):
        self.root, self.cache = Path(data_root), Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.groups = {g['group_id']:g for g in split['groups']}
        self.membership = {n:(g['split'],g['group_id']) for g in split['groups'] for n in g['members']}
        self.inventory = {r['image']:r for r in inventory['records']}
        self.protocol_sha, self.split_sha = protocol_sha, split_sha
        self.test_unlocked = False
        self.access = Counter()
        self.mask_arrays = {}
        self.verified_features = set()
        self.manifest = {'schema':'dmsf-stage2-frozen-cache-v1', 'features_before_extra_l2':True,
                         'images':{}, 'test_extraction_after_checkpoint_lock':True}

    def names(self, split):
        return sorted(n for n,(s,_) in self.membership.items() if s == split)

    def authorize(self, name, role):
        split, _ = self.membership[name]
        if split == 'legacy_development':
            raise ValueError('Legacy samples are quarantined.')
        if split == 'test' and not self.test_unlocked:
            raise PermissionError('Test access requires a verified checkpoint lock.')
        if role != 'feature' and role not in (split+'_support', split+'_query'):
            raise PermissionError(f'Wrong label role: {split} / {role}.')
        self.access[(name,role)] += 1

    def unlock_test(self, lock_path, output_root):
        lock=json.loads(Path(lock_path).read_text(encoding='utf8'))
        if lock['protocol_sha256'] != self.protocol_sha or lock['split_sha256'] != self.split_sha:
            raise ValueError('Lock provenance mismatch.')
        if set(lock['checkpoints']) != {'spatial_only','true_text','random_text'}:
            raise ValueError('All planned ablations must be locked.')
        for entry in lock['checkpoints'].values():
            path = Path(output_root)/entry['file']
            if sha256_file(path) != entry['sha256']:
                raise ValueError('Checkpoint changed after selection.')
        self.test_unlocked=True
        self.lock_sha256=sha256_file(Path(lock_path))

    def label(self, name, role):
        self.authorize(name, role)
        if name not in self.mask_arrays:
            path=self.root/'mask'/(Path(name).stem+'_label.bmp')
            if sha256_file(path)!=self.inventory[name]['mask_sha256']:
                raise ValueError('Mask differs from locked inventory.')
            with Image.open(path) as im:
                pixels=np.asarray(im).copy()
                if im.mode!='L' or pixels.shape!=(1024,1024) or not set(np.unique(pixels))<={0,12}:
                    raise ValueError('Mask encoding mismatch.')
            self.mask_arrays[name]=(pixels==12).astype(np.uint8)
        return self.mask_arrays[name]

    def feature(self, name):
        self.authorize(name, 'feature')
        return self._feature(name)

    @lru_cache(maxsize=32)
    def _feature(self, name):
        entry=self.manifest['images'][name]
        path=self.cache/entry['file']
        if name not in self.verified_features:
            if sha256_file(path)!=entry['sha256']:
                raise ValueError('Cached feature checksum mismatch.')
            self.verified_features.add(name)
        array=np.load(path,allow_pickle=False)
        if array.shape!=(1024,64,64) or array.dtype!=np.float32 or not np.isfinite(array).all():
            raise ValueError('Cached feature shape/dtype/value mismatch.')
        return torch.from_numpy(array)

    def export(self, names, encoder, device):
        for i,name in enumerate(names):
            self.authorize(name, 'feature')
            image_path=self.root/'images'/name
            if sha256_file(image_path)!=self.inventory[name]['image_sha256']:
                raise ValueError('Image differs from locked inventory.')
            values=encode_dino(encoder,image_path,device)
            target=self.cache/(Path(name).stem+'.features.npy')
            np.save(target,values.numpy(),allow_pickle=False)
            self.manifest['images'][name]={'file':target.name,'sha256':sha256_file(target),
                'image_sha256':self.inventory[name]['image_sha256'], 'split':self.membership[name][0],
                'shape':[1024,64,64], 'dtype':'float32'}
            if i%25==0 or i+1==len(names):
                print(f'CACHE {i+1}/{len(names)} {self.membership[name][0]}',flush=True)
        write_json(self.cache/'cache_manifest.json',self.manifest)

    def validate_supports(self, proposal, split):
        invalid=[]
        for name in proposal['held_out'][split]['support_pool'].values():
            if not self.label(name,split+'_support')[::16,::16].any():invalid.append(name)
        return {'split':split,'support_count':len(proposal['held_out'][split]['support_pool']),
                'invalid_supports':invalid,'valid':not invalid}

    def training_schedule(self, epochs, seed):
        train=self.names('train')
        valid={n:bool(self.label(n,'train_support')[::16,::16].any()) for n in train}
        pools={g:sorted(n for n in record['members'] if valid.get(n,False))
               for g,record in self.groups.items() if record['split']=='train'}
        pools={g:names for g,names in pools.items() if names}
        rng=np.random.default_rng(seed)
        schedule=[]
        for epoch in range(epochs):
            episodes=[]
            for query in rng.permutation(train).tolist():
                qgroup=self.membership[query][1]
                candidates=sorted(g for g in pools if g!=qgroup)
                if not candidates:raise ValueError('No cross-group foreground training support.')
                sgroup=str(rng.choice(candidates));support=str(rng.choice(pools[sgroup]))
                episodes.append({'query':query,'query_group':qgroup,'support':support,'support_group':sgroup})
            schedule.append(episodes)
        return schedule,[n for n in train if not valid[n]]

    def batch(self, episodes, split, text, device, include_targets=True):
        for e in episodes:
            if e['support_group']==e['query_group']:
                raise ValueError('Support and query must come from different groups.')
            if any(self.membership[e[k]][0]!=split for k in ('query','support')):
                raise PermissionError('Episode crosses dataset split.')
        support=torch.stack([self.feature(e['support']) for e in episodes]).to(device)
        query=torch.stack([self.feature(e['query']) for e in episodes]).to(device)
        masks=torch.from_numpy(np.stack([self.label(e['support'],split+'_support')[::16,::16]
                                         for e in episodes])[:,None].astype(np.float32)).to(device)
        inputs=(support,query,masks,text.expand(len(episodes),-1))
        if not include_targets:return inputs
        target=torch.from_numpy(np.stack([self.label(e['query'],split+'_query')[::16,::16]
                                          for e in episodes])[:,None].astype(np.float32)).to(device)
        return inputs,target

    def write_access(self, path):
        write_json(path,{'test_unlocked':self.test_unlocked,
            'lock_sha256':getattr(self,'lock_sha256',None),
            'unique_feature_names':len({n for n,r in self.access if r=='feature'}),
            'accesses':[{'image':n,'role':r,'count':v} for (n,r),v in sorted(self.access.items())]})
