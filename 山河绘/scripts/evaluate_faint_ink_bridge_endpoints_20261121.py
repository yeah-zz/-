# -*- coding: utf-8 -*-
"""Compare stricter endpoint evidence on every cached cross-section addition."""
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_faint_ink_bridge_sections_20261121 import LAB, MODULE as SECTIONS, ROOT as SECTIONS_RUN
from validate_faint_ink_segment_holdout_20261121 import ROOT as HOLDOUT, lock_file, read, render
from validate_faint_ink_segment_holdout_20261121 import score_segment, sha, write_json
from gongbi_training.faint_ink_bridge_endpoints import bridge_faint_ink_verified_endpoints
from screen_faint_ink_natural_20261121 import grouped


ROOT = LAB/'faint_ink_bridge_endpoints_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/faint_ink_bridge_endpoints.py'
CONTEXT = 24


def filtered_output(source, baseline, direction, previous, sections, *, bridge=bridge_faint_ink_verified_endpoints):
    added = (sections>0)&(previous==0)
    output = previous.copy()
    # Only evaluate around existing additions. Padding covers feature kernels,
    # six intervening sections, and the extra source support at either end.
    for group in grouped(added):
        xs,ys = np.asarray(group['xy']).T
        left,top = int(max(xs.min()-CONTEXT,0)),int(max(ys.min()-CONTEXT,0))
        right,bottom = int(xs.max()+CONTEXT+1),int(ys.max()+CONTEXT+1)
        win = np.s_[top:bottom,left:right]
        rgb = np.array(source.crop((left,top,min(right,baseline.shape[1]),min(bottom,baseline.shape[0]))).convert('RGB'))
        confirmed = bridge(rgb,baseline[win],direction[win])
        kept = confirmed[ys-top,xs-left]
        output[ys[kept],xs[kept]] = sections[ys[kept],xs[kept]]
    return output


def evaluate(name, source_path, cached, previous_key, sections_key):
    with np.load(cached) as masks, Image.open(source_path) as source:
        baseline,direction,before,sections = (masks[key] for key in ('baseline','direction',previous_key,sections_key))
        result = filtered_output(source,baseline,direction,before,sections)
        if name.endswith('/rounded'):
            rgb = np.array(source.convert('RGB'))
            whole = bridge_faint_ink_verified_endpoints(rgb,baseline,direction)
            np.testing.assert_array_equal(result,np.where(whole & (sections>0),sections,before))
    variants = dict(baseline=baseline,prior_bridge=before,sections=sections,endpoints=result)
    counts = {key:int(((mask>0)&(before==0)).sum()) for key,mask in variants.items()}
    return variants,counts


def main():
    ROOT.mkdir(exist_ok=True)
    prior = read(SECTIONS_RUN/'protocol.json')
    assert sha(SECTIONS)==prior['module']
    sources = {}
    labels = []
    for key,meta in prior['images'].items():
        path = Path(meta['source'])
        assert sha(path)==meta['source_sha']
        cached = SECTIONS_RUN/f"{key.replace('/','__')}.npz"
        sources[f'previous/{key}'] = (path,cached,'combined','sections')
    for folder in (HOLDOUT,HOLDOUT/'landscape_addendum'):
        legacy = read(folder/'results.json')
        assert sha(folder/'review_protocol.json')==legacy['protocol_sha256']
        manifest = read(folder/'screen_protocol.json')
        assert manifest['module']==sha(SECTIONS)
        for key,meta in manifest['images'].items():
            path = Path(meta['path'])
            assert sha(path)==meta['sha256']
            sources[f'{folder.name}/{key}'] = (path,folder/f'{key}_scored.npz','previous','candidate')
        labels.extend(dict(row, image=f"{folder.name}/{row['image']}")
                      for row in read(folder/'review_protocol.json')['labels'])
    known_path = HOLDOUT/'segment_regression.json'
    for key,value in read(known_path)['scores'].items():
        title = {'round_rock':'圆石弧线','rock_horizontal':'岩层横线'}[key]
        labels.append(dict(image='previous/faint_ink_anchor_validation_20261121/rounded',
                           title=title,label='faint',trace=value['trace']))
    protocol = dict(module=sha(MODULE),sections=sha(SECTIONS),known_segments=sha(known_path),context=CONTEXT,
                    inputs={key:[sha(meta[0]),sha(meta[1])] for key,meta in sources.items()},
                    labels=labels,scope='Known examples used to develop and validate the endpoint gate.')
    lock_file(ROOT/'protocol.json',protocol)
    scores,totals,panels,images = {},{},[],{}
    display = {'圆石弧线','岩层横线','竖墨线旁底材','深墨旁浅色底材','青绿山体外的底材纹'}
    point_rows = []
    for name,params in sources.items():
        variants,totals[name] = evaluate(name,*params)
        for case in (case for case in labels if case['image']==name):
            coords, label, title = case['trace'],case['label'],case['title']
            scores[title] = dict(label=label,total=len(coords),
                                 hits={k:sum(bool(v[y,x]) for x,y in coords) for k,v in variants.items()})
            if label=='faint':
                scores[title]['segment'] = {k:score_segment(v,coords) for k,v in variants.items()}
            if title in display:
                if name not in images:
                    with Image.open(params[0]) as source:
                        images[name] = np.array(source.convert('RGB')),variants
                group = len(panels)+1
                panels.append((name,group,title))
                point_rows.extend(dict(image=name,group=group,xy=xy,label=label) for xy in coords)
        print(name,totals[name]['sections'],totals[name]['endpoints'],flush=True)
    panels.sort(key=lambda row: (row[2] not in {'圆石弧线','岩层横线'},row[1]))
    columns = ((None,'原图'),('prior_bridge','旧双端续接'),('sections','逐截面复核'),('endpoints','核对两端墨带'))
    render(images,point_rows,ROOT/'comparison.png',sections=panels,columns=columns)
    locks = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:sha(locks[key])==locks[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    write_json(ROOT/'results.json',dict(scores=scores,totals=totals,protocol_sha256=sha(ROOT/'protocol.json'),
               baseline_hashes=hashes,independent_acceptance=False,adopted=False,robot_connected=False))
    print({key:row for key,row in scores.items() if key in display})


if __name__=='__main__':
    main()
