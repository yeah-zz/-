# -*- coding: utf-8 -*-
"""Evaluate color-aware texture on known natural lines and paper impostors."""
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_faint_ink_bridge_endpoints_20261121 import MODULE as ENDPOINT, ROOT as ENDPOINT_RUN, filtered_output
from validate_faint_ink_endpoints_holdout_20261121 import ROOT as RECENT, segment
from gongbi_training.faint_ink_color_texture import bridge_faint_ink_color_texture


LAB = segment.LAB
ROOT = LAB/'faint_ink_color_texture_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/faint_ink_color_texture.py'


def inputs():
    old = segment.read(ENDPOINT_RUN/'protocol.json')
    old_result = segment.read(ENDPOINT_RUN/'results.json')
    assert segment.sha(ENDPOINT)==old['module']
    assert segment.sha(ENDPOINT_RUN/'protocol.json')==old_result['protocol_sha256']
    sources = {}
    for name, (source_hash, mask_hash) in old['inputs'].items():
        cohort, rest = name.split('/',1)
        if cohort=='previous':
            artifact, image = rest.split('/',1)
            manifest = segment.read(LAB/artifact/'screen_protocol.json')
            source = Path(manifest['images'][image]['path'])
            mask = LAB/'faint_ink_bridge_sections_20261121'/f'{artifact}__{image}.npz'
            stages = ('combined','sections')
        else:
            folder = segment.ROOT if cohort==segment.ROOT.name else segment.ROOT/'landscape_addendum'
            source = Path(segment.read(folder/'screen_protocol.json')['images'][rest]['path'])
            mask = folder/f'{rest}_scored.npz'
            stages = ('previous','candidate')
        assert segment.sha(source)==source_hash and segment.sha(mask)==mask_hash
        sources[name] = (source,mask,stages)

    reviewed = segment.read(RECENT/'review_protocol.json')
    assert segment.sha(RECENT/'review_protocol.json')==segment.read(RECENT/'results.json')['review_sha256']
    manifest = segment.read(RECENT/'screen_protocol.json')['images']
    for name,meta in manifest.items():
        source = Path(meta['path'])
        assert segment.sha(source)==meta['sha256']
        sources[f'color_holdout/{name}'] = (source,RECENT/f'{name}_comparison.npz',('previous','sections'))
    labels = old['labels'][:] + [dict(row,image=f"color_holdout/{row['image']}") for row in reviewed['references']]
    labels += [
        dict(image=f'{segment.ROOT.name}/ink', title='上一轮剩余印刷网纹', trace=[[3077,1112]],
             label='paper_review'),
        dict(image='landscape_addendum/wangximen_101', title='上一轮剩余设色过渡', trace=[[220,145]],
             label='unknown'),
    ]
    return sources,labels


def evaluate(source, masks, stages):
    with Image.open(source) as image, np.load(masks) as data:
        baseline,direction,previous,sections = (data[key] for key in ('baseline','direction',*stages))
        last = filtered_output(image,baseline,direction,previous,sections)
        directional = filtered_output(image,baseline,direction,previous,sections,
                                      bridge=partial(bridge_faint_ink_color_texture,reject_periodic=False))
        adaptive = filtered_output(image,baseline,direction,previous,sections,
                                   bridge=bridge_faint_ink_color_texture)
    return dict(baseline=baseline,prior_bridge=previous,sections=sections,endpoints=last,
                directional=directional,adaptive=adaptive)


def main():
    ROOT.mkdir(exist_ok=True)
    sources,labels = inputs()
    protocol = dict(module=segment.sha(MODULE), previous=segment.sha(ENDPOINT),
                    images={name:[segment.sha(info[0]),segment.sha(info[1])] for name,info in sources.items()},
                    labels=labels,scope='Previously viewed samples used for development.')
    protocol_path = ROOT/'protocol.json'
    segment.lock_file(protocol_path,protocol)
    images,totals,scores,points,panels = {},{},{},[],[]
    shown = ('青绿山面斜线','暖色画面淡横线','圆石弧线','岩层横线','竖墨线旁底材','上一轮剩余印刷网纹')
    for name,params in sources.items():
        variants = evaluate(*params)
        before = variants['prior_bridge']
        totals[name] = {k:int(((m>0)&(before==0)).sum()) for k,m in variants.items() if k!='baseline'}
        for row in (r for r in labels if r['image']==name):
            title,trace,label = (row[k] for k in ('title','trace','label'))
            scores[title] = dict(label=label,total=len(trace),
                                 hits={key:sum(bool(m[y,x]) for x,y in trace) for key,m in variants.items()})
            if label=='faint':
                scores[title]['segment'] = {key:segment.score_segment(m,trace) for key,m in variants.items()}
            if title not in shown:
                continue
            if name not in images:
                with Image.open(params[0]) as image:
                    images[name] = np.array(image.convert('RGB')),variants
            index = len(panels)+1
            panels.append((name,index,title))
            mark = 'paper' if label=='paper_review' else label
            points.extend(dict(image=name,group=index,xy=xy,label=mark) for xy in trace)
    panels.sort(key=lambda item:shown.index(item[2]))
    columns = ((None,'原图'),('endpoints','旧两端复核'),('directional','沿线估计纹理'),('adaptive','加周期网纹检查'))
    segment.render(images,points,ROOT/'comparison.png',sections=panels,columns=columns)
    locks = segment.read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:segment.sha(locks[key])==locks[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    segment.write_json(ROOT/'results.json',dict(scores=scores,totals=totals,baseline_hashes=hashes,
                       protocol_sha256=segment.sha(protocol_path),independent_acceptance=False,
                       adopted=False,robot_connected=False))
    for key,row in scores.items():
        if row['label']!='unknown':
            print(key,row)
    print({name:values for name,values in totals.items() if any(values.values())})


if __name__=='__main__':
    main()
