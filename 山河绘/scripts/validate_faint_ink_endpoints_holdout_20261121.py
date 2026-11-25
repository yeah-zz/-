# -*- coding: utf-8 -*-
"""Freeze and evaluate the endpoint check on separate archived source material."""
import argparse

import numpy as np
from PIL import Image

import validate_faint_ink_segment_holdout_20261121 as segment
from evaluate_faint_ink_bridge_endpoints_20261121 import filtered_output, MODULE, ROOT as DEVELOPMENT
from evaluate_faint_ink_bridge_sections_20261121 import evaluate_image


ROOT = segment.LAB/'faint_ink_endpoints_holdout_20261121'
ARCHIVE = segment.LAB/'guohua_boundary_reference_20261116/views'
FRAMES = segment.LAB/'clear_rock_sources_20261117/views'
SOURCES = {
    'dai': ARCHIVE/'dai-jin_returning-late-from-a-spring-outing_6_input.png',
    'qiu': ARCHIVE/'qiu-ying_fairyland-of-peach-blossoms_8_input.png',
    'wang': ARCHIVE/'wang-shimin_clearing-of-rivers-and-mountains-after-snow_6_input.png',
    'h14': ARCHIVE/'H_14_input.png',
    'ni': segment.LAB/'whole_rock_evidence_20261117/views/nizan_8.png',
    'foreground': FRAMES/'F1500_foreground_rock.png',
    'wash': FRAMES/'F1875_wash_core.png',
    'slope': FRAMES/'F360_mountain_face.png',
}


def endpoint_lock():
    root = DEVELOPMENT/'protocol.json'
    result = segment.read(DEVELOPMENT/'results.json')
    assert segment.sha(root)==result['protocol_sha256']
    manifest = segment.read(root)
    assert segment.sha(MODULE)==manifest['module']
    return dict(module=manifest['module'],development=segment.sha(root))


def screen():
    lock = endpoint_lock()
    ROOT.mkdir(exist_ok=True)
    taken = {values[0] for values in segment.read(DEVELOPMENT/'protocol.json')['inputs'].values()}
    assert all(segment.sha(path) not in taken for path in SOURCES.values())
    segment.lock_file(ROOT/'endpoint_lock.json',lock)
    segment.ROOT = ROOT
    segment.SOURCES = SOURCES
    segment.make_screen()


def references():
    # The traces follow ink visible in the source-only screen and native crops.
    known = [
        dict(image='qiu',title='青绿山面斜线',label='faint',
             trace=[[197,182],[198,183],[199,184],[199,185],[199,186],[200,187],
                    [201,188],[202,189],[202,190],[202,191]]),
        dict(image='wang',title='暖色画面淡横线',label='faint',trace=[[x,87] for x in range(96,111)]),
        dict(image='qiu',title='青绿斜线外的浅色面',label='paper',trace=[[196,y] for y in range(185,190)]),
        dict(image='wang',title='横线外的浅色底材',label='paper',trace=[[x,91] for x in range(104,111)]),
    ]
    screen_data = segment.read(ROOT/'screen.json')
    unknown = {'qiu': {1:'山面暖色交界',2:'青绿灰带',4:'树冠碎纹',5:'树枝边界',6:'色面过渡'},
               'wang': {2:'浅墨转折待核',3:'横线前段待核',4:'斜线暗点待核',5:'树纹交叠',6:'树枝交汇'}}
    return known + [dict(image=name,title=title,label='unknown',trace=screen_data[name]['groups'][group-1]['xy'])
                    for name,groups in unknown.items() for group,title in groups.items()]


def score():
    assert segment.read(ROOT/'endpoint_lock.json')==endpoint_lock()
    segment.lock_options()
    inputs = segment.read(ROOT/'screen_protocol.json')['images']
    labels = references()
    review = ROOT/'review_protocol.json'
    segment.lock_file(review,dict(screen=segment.sha(ROOT/'screen.json'),
                      endpoint=segment.sha(ROOT/'endpoint_lock.json'),
                      manifest=segment.sha(ROOT/'screen_protocol.json'),
                      masks={name:segment.sha(ROOT/f'{name}.npz') for name in inputs},
                      references=labels,scope='Original image references locked before either new variant.'))
    images, totals, scores, panels, points = {},{},{},[],[]
    for name,meta in inputs.items():
        assert segment.sha(meta['path'])==meta['sha256']
        with Image.open(meta['path']) as source, np.load(ROOT/f'{name}.npz') as cached:
            rgb = np.array(source.convert('RGB'))
            baseline,direction,previous = (cached[key] for key in ('baseline','direction','previous'))
            sections = evaluate_image(rgb,baseline,direction,previous)
            endpoints = filtered_output(source,baseline,direction,previous,sections)
        variants = dict(baseline=baseline,direction=direction,previous=previous,sections=sections,endpoints=endpoints)
        images[name] = rgb,variants
        totals[name] = {key:np.argwhere((mask>0)&(previous==0))[:,::-1].tolist()
                        for key,mask in variants.items() if key in ('sections','endpoints')}
        np.savez_compressed(ROOT/f'{name}_comparison.npz',**variants)
    for index,row in enumerate(labels,1):
        name,title,trace,label = (row[k] for k in ('image','title','trace','label'))
        _,variants = images[name]
        scores[title] = dict(label=label,n=len(trace),counts={k:sum(bool(v[y,x]) for x,y in trace)
                                                            for k,v in variants.items()})
        if label=='faint':
            scores[title]['segment'] = {k:segment.score_segment(v,trace) for k,v in variants.items()}
        points.extend(dict(image=name,group=index,xy=xy,label=label) for xy in trace)
        if label!='unknown':
            panels.append((name,index,title))
    columns = ((None,'原图'),('previous','旧双端续接'),('sections','逐截面复核'),('endpoints','核对两端墨带'))
    segment.render(images,points,ROOT/'comparison.png',sections=panels,columns=columns)
    lock = segment.read(segment.LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:segment.sha(lock[key])==lock[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    segment.write_json(ROOT/'results.json',dict(scores=scores,additions=totals,review_sha256=segment.sha(review),
                       baseline_hashes=hashes,independent_reviewer=False,adopted=False,robot_connected=False))
    print({key:value for key,value in scores.items() if value['label']!='unknown'})
    print({key:{kind:len(coords) for kind,coords in row.items()} for key,row in totals.items()})


if __name__=='__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('stage',choices=['screen','score'])
    if cli.parse_args().stage=='screen':
        screen()
    else:
        score()
