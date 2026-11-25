# -*- coding: utf-8 -*-
"""GPU SAM proposals on frozen source images; no manual prompts or robot imports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure, LAB
configure()
ROOT = LAB/'module_stroke_planner_20261104'
sys.path.insert(0, str(ROOT/'runtime'))
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main(ids,seeded=False):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this isolated inference experiment')
    torch.set_num_threads(4)
    model = sam_model_registry['vit_b'](checkpoint=str(ROOT/'weights/sam_vit_b_01ec64.pth')).cuda().eval()
    for ident in ids:
        folder = ROOT/'outputs'/ident
        folder.mkdir(parents=True, exist_ok=True)
        source = LAB/'stroke_unit_model_20261102/outputs'/ident/'source.png'
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        suffix='_seeded' if seeded else ''
        if (folder/f'sam_masks{suffix}.npz').exists():
            old=json.loads((folder/f'sam_manifest{suffix}.json').read_text('utf-8'))
            if old['source_sha256'] != digest:
                raise ValueError('Cached source mismatch')
            print(ident, 'cached', flush=True)
            continue
        rgb=np.array(Image.open(source).convert('RGB'))
        kwargs=dict(points_per_side=24)
        if seeded:
            doc=json.loads(source.with_name('curves.json').read_text('utf-8'))
            paths=[np.asarray(r['points'],float) for r in doc['paths']]
            paths.sort(key=lambda p:-np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
            seeds=[p[len(p)//2]/np.array([rgb.shape[1],rgb.shape[0]]) for p in paths[:128]]
            v=(np.arange(24)+.5)/24
            grid=np.stack(np.meshgrid(v,v),-1).reshape(-1,2)
            points=np.concatenate([grid,np.asarray(seeds)],0).astype(np.float32)
            kwargs=dict(points_per_side=None,point_grids=[points])
        generator = SamAutomaticMaskGenerator(model, points_per_batch=16,
            pred_iou_thresh=.85, stability_score_thresh=.90, crop_n_layers=0,
            min_mask_region_area=0,**kwargs)
        start=time.monotonic()
        with torch.inference_mode():
            masks=generator.generate(rgb)
        arrays={f'm{i:03}':m['segmentation'] for i,m in enumerate(masks)}
        np.savez_compressed(folder/f'sam_masks{suffix}.npz', **arrays)
        records=[dict(key=f'm{i:03}',area=m['area'],bbox=m['bbox'],
                      predicted_iou=m['predicted_iou'],stability_score=m['stability_score'])
                 for i,m in enumerate(masks)]
        save(folder/f'sam_manifest{suffix}.json',dict(id=ident, source=str(source),source_sha256=digest,
             records=records,seconds=time.monotonic()-start,device=torch.cuda.get_device_name(),
             no_manual_prompts=True,pretrained_SAM_not_finetuned=True,
             seed_strategy='grid plus 128 frozen-curve midpoint prompts' if seeded else '24x24 grid',
             masks_are_unverified_instance_proposals=True,robot_connected=False))
        print(ident, len(records), 'masks', round(time.monotonic()-start,1), 'seconds', flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ids',nargs='+',default=['G03_01','G05_03','T02'])
    parser.add_argument('--seeded',action='store_true')
    args=parser.parse_args();main(args.ids,args.seeded)
