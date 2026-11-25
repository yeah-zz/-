# -*- coding: utf-8 -*-
"""Screen archived sources for natural gaps, then score frozen bridge variants."""
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_faint_ink_anchor_guard_20261121 import MODULE as ANCHOR, ROOT as TRIAL
from evaluate_faint_ink_bridge_20261121 import LIMIT, MODULE as OLD, ROOT as BRIDGE_RUN
from evaluate_faint_ink_bridge_sections_20261121 import LAB, MODULE, ROOT as SECTIONS_RUN
from evaluate_faint_ink_bridge_sections_20261121 import evaluate_image, read, render, sha, write_json
from evaluate_ink_paper_width_20261119 import predict_extra
from screen_faint_ink_natural_20261121 import grouped
from validate_faint_ink_direction_20261121 import compare
from gongbi_training.faint_ink_anchor_guard import guard_faint_ink_anchors
from gongbi_training.faint_ink_bridge import _reach, bridge_faint_ink
from gongbi_training.ink_paper_join_guard import protect_ink_joins
from gongbi_training.ink_paper_persistent_stop import persistent_paper_extension
from gongbi_training.ink_paper_width import separate_narrow_paper


ROOT = LAB/'faint_ink_segment_holdout_20261121'
MUSEUM = LAB/'cunfa_fresh_validation_20261111/sources'
ALBUM = LAB/'cunfa_cross_source_20261117/original_images'
VIEWS = LAB/'guohua_faint_sources_20261116/views'
SOURCES = {
    'scroll': MUSEUM/'231419_0.jpg',
    'peaks': MUSEUM/'231773_0.jpg',
    'tea': VIEWS/'tang-yin_making-tea_2_input.png',
    'album_13a': ALBUM/'qingpu_13_1.jpeg',
    'album_13b': ALBUM/'qingpu_13_2.jpeg',
    'album_20': ALBUM/'qingpu_20_0.jpeg',
    'album_21': ALBUM/'qingpu_21_0.jpeg',
    'ink': ALBUM/'ntmofa_6_0.jpeg',
}


def lock_options():
    assert sha(MODULE)==read(SECTIONS_RUN/'protocol.json')['module']
    assert sha(OLD)==read(BRIDGE_RUN/'protocol.json')['module']
    trial = read(TRIAL/'protocol.json')
    assert sha(ANCHOR)==trial['module']
    return trial['options']['core']


def lock_file(path, data):
    if path.exists():
        assert read(path)==data
    else:
        write_json(path,data)


def two_sided_geometry(roles, direction):
    """Locate proposals between same-role detections without looking at image tone."""
    eligible = np.zeros_like(roles,dtype=bool)
    for transposed in (False,True):
        base = roles.T if transposed else roles
        proposals = direction.T if transposed else direction
        found = np.zeros_like(base,dtype=bool)
        for role in (1,2):
            anchor = base==role
            gap = (base==0)&(proposals==role)
            left = _reach(anchor,gap,1,LIMIT)
            right = np.logical_or.accumulate(_reach(anchor,gap,-1,LIMIT),axis=0)
            for i, reached in enumerate(left):
                found |= reached & right[LIMIT-i-1]
        eligible |= found.T if transposed else found
    return eligible


def old_stages(image, roles, options):
    narrow,_,_ = separate_narrow_paper(image,roles)
    opened,_,_ = persistent_paper_extension(image,roles,narrow)
    baseline,_,_ = protect_ink_joins(image,roles,narrow,opened,span=3,check_dark_edge=True)
    direction = compare(image,baseline)['direction']
    previous,_,_ = guard_faint_ink_anchors(image,baseline,direction,**options)
    bridge = bridge_faint_ink(image,baseline,direction,limit=LIMIT)
    previous[bridge] = direction[bridge]
    return baseline,direction,previous


def preview(name, rgb, groups):
    if not groups:
        return
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    for start in range(0,min(len(groups),24),8):
        page = groups[start:start+8]
        board = Image.new('RGB',(964,160*((len(page)+1)//2)),'#f5f1e7')
        pen = ImageDraw.Draw(board)
        for j,item in enumerate(page):
            left,top = (j%2)*482+6,(j//2)*160+4
            xs,ys = zip(*item['xy'])
            x0,y0,x1,y1 = min(xs)-8,min(ys)-8,max(xs)+9,max(ys)+9
            crop = Image.fromarray(rgb).crop((x0,y0,x1,y1))
            scale = max(1,min(224//crop.width,128//crop.height))
            crop = crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.NEAREST)
            board.paste(crop,(left,top+22))
            annotated = crop.copy()
            d = ImageDraw.Draw(annotated)
            for index,(x,y) in enumerate(item['xy'],1):
                u,v = (x-x0)*scale,(y-y0)*scale
                d.rectangle((u,v,u+scale-1,v+scale-1),outline='#e0731b')
                if scale>=8:
                    d.text((u,v),str(index),fill='#127b90',stroke_width=1,stroke_fill='white')
            board.paste(annotated,(left+234,top+22))
            pen.text((left,top),f'{name} {start+j+1:02d} · {len(xs)}像素',font=font,fill='#263c34')
        board.save(ROOT/f'{name}_screen_{start//8+1}.png')


def make_screen():
    options = lock_options()
    ROOT.mkdir(exist_ok=True)
    known = {m['source_sha'] for m in read(SECTIONS_RUN/'protocol.json')['images'].values()}
    known |= {m['original_sha256'] for m in read(SECTIONS_RUN/'holdout/screen_protocol.json')['images'].values()}
    images = {key:dict(path=str(path),sha256=sha(path)) for key,path in SOURCES.items()}
    assert all(row['sha256'] not in known for row in images.values())
    lock_file(ROOT/'screen_protocol.json',dict(module=sha(MODULE),old=sha(OLD),anchor=sha(ANCHOR),images=images,
              selection='Only geometrically two-ended proposals still missed by the old scheme.'))
    screened = {}
    for name,(image,roles) in predict_extra(dict(images=images)).items():
        baseline,direction,previous = old_stages(image,roles,options)
        eligible = two_sided_geometry(baseline,direction)&(previous==0)
        groups = grouped(eligible)
        np.savez_compressed(ROOT/f'{name}.npz',baseline=baseline,direction=direction,previous=previous)
        screened[name] = dict(groups=groups,candidates=int(eligible.sum()))
        preview(name,image,groups)
        print(name,len(groups),int(eligible.sum()),flush=True)
    lock_file(ROOT/'screen.json',screened)


def score_segment(mask, trace):
    # Score connectivity only inside the immediate corridor around the marked ink.
    points = np.asarray(trace,dtype=np.int32)
    x0,y0 = np.maximum(points.min(axis=0)-2,0)
    x1,y1 = np.minimum(points.max(axis=0)+3,mask.shape[::-1])
    local = points-[x0,y0]
    corridor = np.zeros((y1-y0,x1-x0),np.uint8)
    cv2.polylines(corridor,[local.astype(np.int32)],False,1,1)
    corridor = cv2.dilate(corridor, np.ones((3,3),np.uint8)).astype(bool)
    window = mask[y0:y1,x0:x1]
    start,end = local[0],local[-1]
    has_connection = False
    for role in (1,2):
        _, components = cv2.connectedComponents((corridor & (window==role)).astype(np.uint8),connectivity=8)
        start_component = components[start[1],start[0]]
        end_component = components[end[1],end[0]]
        has_connection |= bool(start_component and start_component==end_component)
    hits = [bool(mask[y,x]) for x,y in trace]
    return dict(connected=has_connection,core=sum(hits),length=len(hits))


def score_holdout(labels):
    lock_options()
    manifest = read(ROOT/'screen_protocol.json')
    assert manifest['module']==sha(MODULE)
    lock = ROOT/'review_protocol.json'
    lock_file(lock,dict(screen=sha(ROOT/'screen.json'),manifest=sha(ROOT/'screen_protocol.json'),
                        masks={name:sha(ROOT/f'{name}.npz') for name in manifest['images']},
                        labels=labels,scope='Reviewed original source before cross-section output.'))
    images, totals, scores, points, panels = {}, {}, {}, [], []
    for name,meta in manifest['images'].items():
        assert sha(meta['path'])==meta['sha256']
        with Image.open(meta['path']) as image, np.load(ROOT/f'{name}.npz') as cached:
            rgb = np.array(image.convert('RGB'))
            baseline,direction,previous = (cached[key] for key in ('baseline','direction','previous'))
        candidate = evaluate_image(rgb,baseline,direction,previous)
        variants = dict(baseline=baseline,direction=direction,previous=previous,candidate=candidate)
        images[name] = rgb,variants
        added = (candidate>0)&(previous==0)
        totals[name] = dict(added=int(added.sum()),coordinates=np.argwhere(added)[:,::-1].tolist())
        np.savez_compressed(ROOT/f'{name}_scored.npz',**variants)
    for index,case in enumerate(labels,1):
        name,trace,label = case['image'],case['trace'],case['label']
        _,variants = images[name]
        flags = {key:sum(bool(mask[y,x]) for x,y in trace) for key,mask in variants.items()}
        scores[case['title']] = dict(label=label,hits=flags,total=len(trace))
        if label=='faint':
            scores[case['title']]['segment'] = {key:score_segment(mask,trace) for key,mask in variants.items()}
        points.extend(dict(image=name,group=index,xy=xy,label=label) for xy in trace)
        panels.append((name,index,case['title']))
    if panels:
        columns = ((None,'原图'),('direction','方向筛选'),('previous','旧双端续接'),('candidate','逐截面复核'))
        render(images,points,ROOT/'comparison.png',sections=panels,columns=columns)
    locks = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:sha(locks[key])==locks[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    write_json(ROOT/'results.json',dict(scores=scores,totals=totals,protocol_sha256=sha(lock),
               baseline_hashes=hashes,independent_reviewer=False,adopted=False,robot_connected=False))
    print(scores)
    print({name:row['added'] for name,row in totals.items()})


if __name__=='__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('stage',choices=['screen','score'])
    args = cli.parse_args()
    if args.stage=='screen':
        make_screen()
    else:
        score_holdout(read(ROOT/'source_labels.json'))
