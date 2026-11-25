# -*- coding: utf-8 -*-
"""Freeze narrow-support variants and test them on new source crops."""
import argparse
from functools import partial

import numpy as np
from PIL import Image

import validate_faint_ink_segment_holdout_20261121 as segment
from evaluate_faint_ink_bridge_endpoints_20261121 import filtered_output
from evaluate_faint_ink_bridge_sections_20261121 import evaluate_image
from evaluate_faint_ink_color_texture_20261121 import MODULE as COLOR, ROOT as COLOR_RUN
from evaluate_faint_ink_narrow_20261121 import MODULE as NARROW, ROOT as NARROW_RUN, apply_profile
from evaluate_faint_ink_narrow_support_20261121 import MODULE as SUPPORT, ROOT as SUPPORT_RUN
from gongbi_training.faint_ink_anchor_guard import guard_faint_ink_anchors
from gongbi_training.faint_ink_bridge import bridge_faint_ink
from gongbi_training.faint_ink_color_texture import bridge_faint_ink_color_texture
from gongbi_training.faint_ink_narrow_support import bridge_faint_ink_narrow_supported


ROOT = segment.LAB/'faint_ink_narrow_support_holdout_20261121'
PUBLISHED = segment.LAB/'cunfa_role_reference_20261112/references'
TEACHING = segment.LAB/'clear_rock_sources_20261117/views'
SOURCES = {
    'aam_hemp': PUBLISHED/'aam_long_hemp.png',
    'hkma_axe': PUBLISHED/'hkma_axe.png',
    'hkma_hemp': PUBLISHED/'hkma_hemp.png',
    'npm_hemp': PUBLISHED/'npm_named_hemp.png',
    'pine': TEACHING/'F1500_pine_needles.png',
    'trunk': TEACHING/'F1500_pine_trunk.png',
}
CONTROLS = {'aam_hemp': (1,4,8), 'hkma_axe': (1,2,6,7), 'hkma_hemp': (2,4,5,7), 'pine': (1,)}
GAPS = (3,5)


def support_lock():
    specs = [(NARROW_RUN,NARROW), (COLOR_RUN,COLOR), (SUPPORT_RUN/'span_5',SUPPORT)]
    locked = {}
    for folder,module in specs:
        protocol = folder/'protocol.json'
        assert segment.sha(protocol)==segment.read(folder/'results.json')['protocol_sha256']
        assert segment.sha(module)==segment.read(protocol)['module']
        locked[str(module)] = dict(sha=segment.sha(module),protocol=segment.sha(protocol))
    return locked


def screen():
    frozen = support_lock()
    ROOT.mkdir(exist_ok=True)
    previous = segment.read(COLOR_RUN/'protocol.json')['images']
    original = {v[0] for v in previous.values()}
    assert all(segment.sha(p) not in original for p in SOURCES.values())
    segment.lock_file(ROOT/'support_lock.json',frozen)
    segment.ROOT = ROOT
    segment.SOURCES = SOURCES
    segment.make_screen()


def run_variants(source, baseline, direction, previous):
    rgb = np.array(source.convert('RGB'))
    sections = evaluate_image(rgb,baseline,direction,previous)
    color = filtered_output(source,baseline,direction,previous,sections,bridge=bridge_faint_ink_color_texture)
    unrestricted = apply_profile(rgb,baseline,direction,color,True)
    variants = dict(baseline=baseline,prior_bridge=previous,color=color,unrestricted=unrestricted)
    for span in (3,5):
        bridge = partial(bridge_faint_ink_narrow_supported,minimum_span=span)
        proposed = apply_profile(rgb,baseline,direction,color,True,bridge=bridge)
        variants[f'span{span}'] = np.where(unrestricted==proposed,proposed,color)
    return rgb,variants


def erase_center(baseline, item, length):
    trace = item['trace']
    first = (len(trace)-length)//2
    target = trace[first:first+length]
    role = item['role']
    erased = baseline.copy()
    for x,y in target:
        assert baseline[y,x]==role
        across = erased[y] if item['axis']=='vertical' else erased[:,x]
        coordinate = x if item['axis']=='vertical' else y
        across[coordinate] = 0
        for direction in (-1,1):
            for offset in range(1,5):
                index = coordinate+offset*direction
                if not 0<=index<len(across) or across[index]!=role:
                    break
                across[index] = 0
    return erased,target


def test_control(source, baseline, item, length, options):
    trace = item['trace']
    rgb = np.array(source.convert('RGB'))
    erased,target = erase_center(baseline,item,length)
    comparison = segment.compare(rgb,erased)
    direction = comparison['direction']
    previous,_,_ = guard_faint_ink_anchors(rgb,erased,direction,**options)
    recovered = bridge_faint_ink(rgb,erased,direction,limit=segment.LIMIT)
    previous[recovered] = direction[recovered]
    _,variants = run_variants(source,erased,direction,previous)
    variants['direction'] = direction
    scores = {key:dict(covered=sum(int(m[y,x]==item['role']) for x,y in target),
                       connected=segment.score_segment(m,trace)['connected'])
              for key,m in variants.items()}
    return rgb,variants,target,scores


def score():
    frozen = support_lock()
    assert segment.read(ROOT/'support_lock.json')==frozen
    options = segment.lock_options()
    manifest = segment.read(ROOT/'screen_protocol.json')['images']
    bank = segment.read(ROOT/'controlled_screen.json')
    references = [dict(bank[name][index-1],image=name,group=index)
                  for name,indices in CONTROLS.items() for index in indices]
    lock = ROOT/'review_protocol.json'
    segment.lock_file(lock,dict(algorithms=frozen,screen=segment.sha(ROOT/'screen.json'),
                      sources=segment.sha(ROOT/'screen_protocol.json'),
                      masks={name:segment.sha(ROOT/f'{name}.npz') for name in manifest},
                      references=references,gap_lengths=list(GAPS),
                      scope='Original ink traces reviewed before scoring. Artificial gaps in roles only.'))
    natural,controls,images,panels,points = {},[],{},[],[]
    for name,meta in manifest.items():
        assert segment.sha(meta['path'])==meta['sha256']
        with Image.open(meta['path']) as source,np.load(ROOT/f'{name}.npz') as cached:
            baseline,direction,previous = (cached[key] for key in ('baseline','direction','previous'))
            _,outputs = run_variants(source,baseline,direction,previous)
            natural[name] = {key:np.argwhere((mask>0)&(outputs['color']==0))[:,::-1].tolist()
                             for key,mask in outputs.items() if key in ('unrestricted','span3','span5')}
            for item in (row for row in references if row['image']==name):
                for length in GAPS:
                    rgb,variants,target,scores = test_control(source,baseline,item,length,options)
                    title = f"{name} {item['group']:02d} · {length}像素断口"
                    controls.append(dict(image=name,group=item['group'],title=title,length=length,targets=target,
                                         before=segment.score_segment(baseline,item['trace']),scores=scores))
                    key = f"{name}/{item['group']}/{length}"
                    if length==3 and item['group']==CONTROLS[name][0]:
                        images[key] = rgb,variants
                        panels.append((key,1,title))
                        points.extend(dict(image=key,group=1,xy=xy,label='faint') for xy in target)
    columns = ((None,'原图'),('baseline','受控缺口'),('unrestricted','细线候选'),('span5','要求5像素跨度'))
    segment.render(images,points,ROOT/'controlled_comparison.png',sections=panels,columns=columns)
    summary = {}
    for length in GAPS:
        active = [row for row in controls if row['length']==length and row['before']['connected']
                  and not row['scores']['baseline']['connected']]
        summary[length] = dict(cases=len(active),pixels=length*len(active),variants={
            key:dict(covered=sum(r['scores'][key]['covered'] for r in active),
                     connected=sum(r['scores'][key]['connected'] for r in active))
            for key in ('direction','prior_bridge','color','unrestricted','span3','span5')})
    baseline_lock = segment.read(segment.LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:segment.sha(baseline_lock[key])==baseline_lock[key+'_sha256']
              for key in ('approved500','working489')}
    assert all(hashes.values())
    result = dict(summary=summary,natural=natural,controls=controls,baseline_hashes=hashes,
                  review_sha256=segment.sha(lock),independent_acceptance=False,adopted=False,robot_connected=False)
    segment.write_json(ROOT/'results.json',result)
    print(summary)
    print({k:{v:len(p) for v,p in values.items()} for k,values in natural.items()})
    print([(r['image'],r['group'],r['length'],r['scores']['span3']['connected'],r['scores']['span5']['connected'])
           for r in controls if r['scores']['span3']['connected']!=r['scores']['span5']['connected']])


if __name__=='__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('stage',choices=['screen','score'])
    if cli.parse_args().stage=='screen':
        screen()
    else:
        score()
