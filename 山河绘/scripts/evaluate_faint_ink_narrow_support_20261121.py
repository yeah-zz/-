# -*- coding: utf-8 -*-
"""Audit off-target narrow bridge additions and score a support-run constraint."""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_faint_ink_narrow_20261121 import LAB, MODULE as NARROW, ROOT as PRIOR, apply_profile, evaluate, inputs, segment
from gongbi_training.faint_ink_narrow_support import bridge_faint_ink_narrow_supported


ROOT = LAB/'faint_ink_narrow_support_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/faint_ink_narrow_support.py'
# Judgments refer to the numbered source-only audit board for the 25 saved additions.
INK = {1:'圆石既有淡墨边缘', 6:'屋檐墨带', 10:'岩面已有深墨带'}
PAPER = {11:'浅底周期纹理', 12:'浅底周期纹理', 19:'无可辨笔线的暖色底面', 20:'无可辨笔线的浅色底面'}
WARM = 'color_holdout/wang'


def audit(prior):
    targets = [(name,xy) for name,positions in prior['added_positions'].items()
               if name!=WARM for xy in positions]
    assert len(targets)==25
    rows = []
    for index,(name,xy) in enumerate(targets,1):
        if index in INK:
            label,note = 'ink',INK[index]
        elif index in PAPER:
            label,note = 'paper',PAPER[index]
        else:
            label,note = 'unknown','边界、色面或纹理归属待核'
        rows.append(dict(index=index,image=name,xy=xy,label=label,note=note))
    return rows


def main(minimum_span):
    output = ROOT if minimum_span==3 else ROOT/f'span_{minimum_span}'
    output.mkdir(parents=True,exist_ok=True)
    prior = segment.read(PRIOR/'results.json')
    prior_lock = PRIOR/'protocol.json'
    assert prior['protocol_sha256']==segment.sha(prior_lock)
    assert segment.read(prior_lock)['module']==segment.sha(NARROW)
    labels = audit(prior)
    data, references = inputs()
    included = {row['image'] for row in references}
    data = {name:meta for name,meta in data.items() if name in included}
    lock = output/'protocol.json'
    settings = dict(module=segment.sha(MODULE),previous=segment.sha(PRIOR/'results.json'),
                    references=labels,scope='Known examples, source review frozen before support filtering.')
    if minimum_span!=3:
        settings['minimum_span'] = minimum_span
    segment.lock_file(lock,settings)
    bridge = partial(bridge_faint_ink_narrow_supported,minimum_span=minimum_span)
    scores,totals,images,points,panels,retained = {},{},{},[],[],{}
    show = {'暖色画面淡横线','青绿山面斜线','圆石弧线','岩层横线'}
    for name,(source,cache,stages) in data.items():
        old = evaluate(source,cache,stages)
        color = old['adaptive']
        with Image.open(source) as im, np.load(cache) as arrays:
            rgb = np.array(im.convert('RGB'))
            basis,direction = arrays['baseline'],arrays['direction']
        unrestricted = apply_profile(rgb,basis,direction,color,True)
        additions = np.argwhere((unrestricted>0)&(color==0))[:,::-1].tolist()
        assert additions==prior['added_positions'][name]
        proposed = apply_profile(rgb,basis,direction,color,True,bridge=bridge)
        restricted = np.where(unrestricted==proposed,proposed,color)
        new = (restricted>0)&(color==0)
        retained[name] = np.argwhere(new)[:,::-1].tolist()
        totals[name] = dict(previous=len(prior['added_positions'][name]),supported=int(new.sum()))
        variants = dict(baseline=basis,color=color,unrestricted=unrestricted,supported=restricted)
        for row in (r for r in references if r['image']==name):
            title,trace,label = (row[k] for k in ('title','trace','label'))
            scores[title] = dict(label=label,n=len(trace),covered={k:sum(bool(v[y,x]) for x,y in trace)
                                                                  for k,v in variants.items()})
            if label=='faint':
                scores[title]['segment'] = {k:segment.score_segment(v,trace) for k,v in variants.items()}
            if title in show:
                images[name] = rgb,variants
                index = len(panels)+1
                panels.append((name,index,title))
                points.extend(dict(image=name,group=index,xy=xy,label=label) for xy in trace)
        for row in (r for r in labels if r['image']==name):
            x,y = row['xy']
            row['retained'] = bool(new[y,x])
            if row['label']=='paper':
                title = f"目标外复核 {row['index']:02d}"
                images[name] = rgb,variants
                index = len(panels)+1
                panels.append((name,index,title))
                points.append(dict(image=name,group=index,xy=[x,y],label='paper'))
        if name==WARM:
            trace = [[x,87] for x in range(97,111)]
            short_trace = segment.score_segment(restricted,trace)
            # This is a post-hoc corridor check. The original 15-point trace still controls acceptance.
            segment.write_json(output/'warm_alternative_corridor.json',dict(trace=trace,score=short_trace))
        print(name,totals[name],flush=True)
    panels.sort(key=lambda row:(row[2] not in show,row[2]!='暖色画面淡横线',row[1]))
    columns = ((None,'原图'),('color','设色方案'),('unrestricted','上一轮细线方案'),('supported','要求连续候选'))
    segment.render(images,points,output/'comparison.png',sections=panels,columns=columns)
    baseline_lock = segment.read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:segment.sha(baseline_lock[key])==baseline_lock[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    segment.write_json(output/'results.json',dict(audit=labels,scores=scores,totals=totals,retained_positions=retained,
                        baseline_hashes=hashes,protocol_sha256=segment.sha(lock),
                        independent_acceptance=False,adopted=False,robot_connected=False))
    print({key:value for key,value in scores.items() if value['label']=='faint'})
    print([(r['index'],r['label'],r['retained']) for r in labels])


if __name__=='__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--minimum-span',type=int,default=3)
    main(cli.parse_args().minimum_span)
