# -*- coding: utf-8 -*-
"""Check per-section ink evidence on the frozen faint-ink comparison."""
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_faint_ink_anchor_guard_20261121 import MODULE as ANCHOR, ROOT as TRIAL
from evaluate_faint_ink_bridge_20261121 import LAB, LIMIT, MODULE as OLD, NATURAL, PREVIOUS, REGIONS
from evaluate_faint_ink_bridge_20261121 import ROOT as OLD_RUN, polygon_mask, read, render, sha, summarize, write_json
from gongbi_training.faint_ink_anchor_guard import guard_faint_ink_anchors
from gongbi_training.faint_ink_bridge import bridge_faint_ink
from gongbi_training.faint_ink_bridge_sections import bridge_faint_ink_sections


ROOT = LAB/'faint_ink_bridge_sections_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/faint_ink_bridge_sections.py'


def evaluate_image(rgb, baseline, direction, previous):
    added = bridge_faint_ink_sections(rgb, baseline, direction, limit=LIMIT) & (previous==0)
    improved = previous.copy()
    improved[added] = direction[added]
    return improved


def paper_regions(options):
    protocol = read(REGIONS/'protocol.json')
    results = {}
    for spec in protocol['specs']:
        if spec['kind']!='paper':
            continue
        name = spec['image']
        metadata = protocol['images'][name]
        assert sha(metadata['path'])==metadata['sha256']
        with Image.open(metadata['path']) as image, np.load(REGIONS/f'{name}.npz') as cached:
            rgb = np.array(image.convert('RGB'))
            baseline,direction = cached['baseline'],cached['direction']
        previous,_,_ = guard_faint_ink_anchors(rgb,baseline,direction,**options)
        old_bridge = bridge_faint_ink(rgb,baseline,direction,limit=LIMIT)
        previous[old_bridge] = direction[old_bridge]
        after = evaluate_image(rgb,baseline,direction,previous)
        region = polygon_mask(baseline.shape,spec['polygon']) & (baseline==0)
        results[spec['id']] = dict(pixels=int(region.sum()), before=int(((previous>0)&region).sum()),
                                   after=int(((after>0)&region).sum()))
    return results


def main():
    ROOT.mkdir(exist_ok=True)
    assert read(OLD_RUN/'protocol.json')['module']==sha(OLD)
    assert read(TRIAL/'protocol.json')['module']==sha(ANCHOR)
    old = read(OLD_RUN/'results.json')
    assert sha(OLD_RUN/'protocol.json')==old['protocol_sha256']
    verified = {}
    for directory in (PREVIOUS, NATURAL):
        manifest = read(directory/'screen_protocol.json')
        for name, meta in manifest['images'].items():
            path = Path(meta['path'])
            assert sha(path)==meta['sha256']
            cached = OLD_RUN/f'{directory.name}_{name}.npz'
            verified[f'{directory.name}/{name}'] = dict(source=str(path),source_sha=sha(path),
                                                         masks=str(cached),masks_sha=sha(cached))
    protocol = dict(module=sha(MODULE),previous=sha(OLD_RUN/'results.json'),limit=LIMIT,images=verified)
    lock = ROOT/'protocol.json'
    if lock.exists():
        assert read(lock)==protocol
    else:
        write_json(lock,protocol)
    images, totals = {}, {}
    for name, info in verified.items():
        with Image.open(info['source']) as image:
            rgb = np.array(image.convert('RGB'))
        with np.load(info['masks']) as masks:
            baseline,direction,previous = (masks[k] for k in ('baseline','direction','combined'))
        after = evaluate_image(rgb,baseline,direction,previous)
        variants = dict(baseline=baseline,direction=direction,combined=previous,sections=after)
        images[name] = rgb,variants
        totals[name] = {key:int(((mask>0)&(baseline==0)).sum()) for key,mask in variants.items()}
        np.savez_compressed(ROOT/f"{name.replace('/','__')}.npz",**variants)
    points = []
    for old_point in old['points']:
        x,y = old_point['xy']
        _,variants = images[old_point['image']]
        flags = {key:bool(mask[y,x]) for key,mask in variants.items()}
        assert flags['combined']==old_point['ink']['combined']
        points.append(dict(old_point,ink=flags))
    cohorts = {key:summarize([p for p in points if p['image'].startswith(directory.name+'/')])
               for key,directory in [('recent',PREVIOUS),('prior',NATURAL)]}
    negative = paper_regions(read(TRIAL/'protocol.json')['options']['core'])
    rock, tree = f'{PREVIOUS.name}/rounded', f'{NATURAL.name}/tree'
    sections = [(rock,3,'圆石弧线'),(rock,10,'岩层横线'),(rock,11,'岩面斜线'),
                (tree,6,'浅纸纹'),(tree,15,'枝条旁纸纹'),(tree,16,'细枝旁纸纹')]
    columns = ((None,'原图'),('direction','方向筛选'),('combined','双端续接'),('sections','逐截面复核'))
    render(images,points,ROOT/'comparison.png',sections=sections,columns=columns)
    locks = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:sha(locks[key])==locks[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    write_json(ROOT/'results.json',dict(metrics=cohorts,paper_regions=negative,totals=totals,points=points,
               protocol_sha256=sha(lock),baseline_hashes=hashes,independent_acceptance=False,
               adopted=False,robot_connected=False))
    print(cohorts)
    print(negative)
    print({key:(v['combined'],v['sections']) for key,v in totals.items()})
    print([(p['image'].split('/')[-1],p['group'],p['point']) for p in points
           if p['label']=='faint' and p['ink']['sections'] and not p['ink']['combined']])


if __name__=='__main__':
    main()
