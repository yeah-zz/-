# -*- coding: utf-8 -*-
"""Evaluate two-ended continuation against the frozen single-ended guard."""
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_faint_ink_anchor_guard_20261121 import MODULE as ANCHOR, NATURAL, ROOT as TRIAL
from screen_faint_ink_anchor_validation_20261121 import LAB, ROOT as PREVIOUS, read, sha, write_json
from validate_faint_ink_anchor_validation_20261121 import render
from validate_faint_ink_direction_20261121 import ROOT as REGIONS, polygon_mask
from gongbi_training.faint_ink_anchor_guard import guard_faint_ink_anchors
from gongbi_training.faint_ink_bridge import bridge_faint_ink


ROOT = LAB/'faint_ink_bridge_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/faint_ink_bridge.py'
LIMIT = 6


def reviewed_points(directory, result, images, totals):
    protocol = directory/'review_protocol.json'
    assert sha(protocol)==result['protocol_sha256']
    manifest = read(directory/'screen_protocol.json')
    latest = directory==PREVIOUS
    assert sha(ANCHOR)==read(TRIAL/'protocol.json')['module']
    options = read(TRIAL/'protocol.json')['options']['core']
    if latest:
        assert manifest['anchor_rule']==sha(ANCHOR) and manifest['options']==options
    for name, metadata in manifest['images'].items():
        assert sha(metadata['path'])==metadata['sha256']
        with Image.open(metadata['path']) as source:
            image = np.array(source.convert('RGB'))
        with np.load(directory/f'{name}.npz') as cached:
            baseline = cached['baseline']
            direction = cached['direction'] if latest else None
        if latest:
            with np.load(directory/f'{name}_anchor.npz') as cached:
                anchor = cached['anchor']
        else:
            with np.load(directory/f'{name}_direction.npz') as cached:
                direction = cached['roles']
            with np.load(TRIAL/f'{name}.npz') as cached:
                np.testing.assert_array_equal(cached['baseline'], baseline)
                np.testing.assert_array_equal(cached['previous'], direction)
                anchor = cached['core']
        bridge = bridge_faint_ink(image, baseline, direction, limit=LIMIT)
        union = anchor.copy()
        union[bridge] = direction[bridge]
        key = f'{directory.name}/{name}'
        variants = dict(baseline=baseline, direction=direction, anchor=anchor,
                        bridge=np.where(bridge,direction,baseline), combined=union)
        images[key] = image, variants
        totals[key] = {variant:int(((values>0)&(baseline==0)).sum()) for variant,values in variants.items()}
        np.savez_compressed(ROOT/f'{directory.name}_{name}.npz', **variants)
    points = []
    for point in result['points']:
        key = f"{directory.name}/{point['image']}"
        _, variants = images[key]
        x,y = point['xy']
        flags = {variant:bool(mask[y,x]) for variant,mask in variants.items()}
        points.append(dict(point, image=key, ink=flags))
    return points


def paper_regions(options):
    protocol = read(REGIONS/'protocol.json')
    results = {}
    for spec in protocol['specs']:
        if spec['kind']!='paper':
            continue
        name = spec['image']
        metadata = protocol['images'][name]
        assert sha(metadata['path'])==metadata['sha256']
        with Image.open(metadata['path']) as source, np.load(REGIONS/f'{name}.npz') as cached:
            rgb = np.array(source.convert('RGB'))
            baseline,direction = cached['baseline'],cached['direction']
        anchor,_,_ = guard_faint_ink_anchors(rgb,baseline,direction,**options)
        bridge = bridge_faint_ink(rgb,baseline,direction,limit=LIMIT)
        region = polygon_mask(baseline.shape,spec['polygon']) & (baseline==0)
        masks = dict(direction=direction>0,anchor=anchor>0,bridge=bridge,combined=(anchor>0)|bridge)
        results[spec['id']] = dict(pixels=int(region.sum()),new={key:int((region&value).sum())
                                                                for key,value in masks.items()})
    return results


def summarize(points):
    summary = {}
    for label in sorted({point['label'] for point in points}):
        chosen = [point for point in points if point['label']==label]
        summary[label] = dict(n=len(chosen),ink={key:sum(point['ink'][key] for point in chosen)
                                                 for key in chosen[0]['ink']})
    return summary


def main():
    ROOT.mkdir(exist_ok=True)
    protocols = dict(module=sha(MODULE),anchor=sha(ANCHOR),limit=LIMIT,
                     frozen_inputs={str(directory):sha(directory/'results.json')
                                    for directory in (PREVIOUS,NATURAL,REGIONS,TRIAL)})
    path = ROOT/'protocol.json'
    if path.exists():
        assert read(path)==protocols
    else:
        write_json(path,protocols)
    images, totals = {}, {}
    latest = reviewed_points(PREVIOUS,read(PREVIOUS/'results.json'),images,totals)
    prior = reviewed_points(NATURAL,read(NATURAL/'results.json'),images,totals)
    regions = paper_regions(read(TRIAL/'protocol.json')['options']['core'])
    rock, tree = f'{PREVIOUS.name}/rounded', f'{NATURAL.name}/tree'
    sections = [(rock,3,'圆石弧线'),(rock,10,'岩层横线'),(rock,11,'岩面斜线'),(rock,14,'短斜线'),
                (tree,6,'浅纸纹'),(tree,15,'枝条旁纸纹'),(tree,16,'细枝旁纸纹')]
    columns = ((None,'原图'),('direction','方向筛选'),('anchor','起点复核'),('combined','加双端续接'))
    render(images,latest+prior,ROOT/'comparison.png',sections=sections,columns=columns)
    locks = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:sha(locks[key])==locks[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    metrics = dict(latest=summarize(latest),prior=summarize(prior),paper_regions=regions)
    write_json(ROOT/'results.json', dict(metrics=metrics, totals=totals,points=latest+prior,
               baseline_hashes=hashes,protocol_sha256=sha(path),independent_acceptance=False,
               adopted=False,robot_connected=False))
    print(metrics)
    print({key:row for key,row in totals.items() if key.startswith(PREVIOUS.name)})
    print([(p['group'],p['point']) for p in latest if p['label']=='faint'
           and p['ink']['combined'] and not p['ink']['anchor']])


if __name__=='__main__':
    main()
