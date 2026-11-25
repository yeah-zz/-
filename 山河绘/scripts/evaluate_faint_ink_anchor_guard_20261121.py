# -*- coding: utf-8 -*-
"""Compare source-backed anchor rules on faint lines and paper texture."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_faint_ink_direction_filter_20261121 import MODULE as DIRECTION, REFERENCE, ROOT as REGRESSION
from evaluate_ink_paper_extension_20261121 import write_json
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_width_20261119 import score
from validate_faint_ink_natural_20261121 import ROOT as NATURAL
from validate_ink_paper_join_guard_20261121 import local_reference
from gongbi_training.faint_ink_anchor_guard import guard_faint_ink_anchors


ROOT = LAB/'faint_ink_anchor_trial_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/faint_ink_anchor_guard.py'
ADAPTIVE = dict(min_anchor_length=2,min_ink_depth=20,texture_weight=2.5,strong_ridge_floor=18)
OPTIONS = dict(run=dict(min_anchor_length=2),
               fixed=dict(min_anchor_length=2,min_ink_depth=30),
               adaptive=ADAPTIVE, core=dict(ADAPTIVE,dark_core_floor=55))


def comparing(images, points):
    sections = [('rocks',2,'岩面淡线'),('banks',2,'岸边淡线'),('tree',1,'浓墨线'),
                ('tree',6,'浅纸纹'),('tree',15,'枝旁纸纹'),('tree',16,'背景纸纹')]
    board = Image.new('RGB',(1040,190*len(sections)+34),'#f5f1e7')
    pen = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for row,(name,group,title) in enumerate(sections):
        image,variants = images[name]
        selected = [item for item in points if item['image']==name and item['group']==group]
        xs,ys = zip(*(item['xy'] for item in selected))
        x0,y0,x1,y1 = min(xs)-7,min(ys)-7,max(xs)+8,max(ys)+8
        region = (slice(y0,y1),slice(x0,x1))
        scale = min(242//(x1-x0),151//(y1-y0))
        for col,(key,label) in enumerate([(None,'原图'),('previous','上轮方向筛选'),
                                          ('fixed','固定深度'),('core','检查起点')]):
            view = image[region].copy()
            if key:
                ink = variants['baseline'][region]>0
                addition = (variants[key][region]>0)&~ink
                view[ink] = (.60*view[ink]+.40*np.array([34,137,111])).astype(np.uint8)
                view[addition] = (.32*view[addition]+.68*np.array([37,148,202])).astype(np.uint8)
            crop = Image.fromarray(view).resize(((x1-x0)*scale,(y1-y0)*scale),Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(crop)
            for item in selected:
                if item['label']=='unknown':
                    continue
                x,y = item['xy']
                u,v = (x-x0)*scale,(y-y0)*scale
                color = '#d34a85' if item['label']=='paper' else '#fa9023'
                draw.rectangle((u,v,u+scale-1,v+scale-1),outline=color,width=1)
            left,top = 260*col+6,190*row+4
            pen.text((left,top),f'{title} · {label}',font=font,fill='#263c34')
            board.paste(crop,(left,top+26))
    pen.text((8,board.height-27),'橙框为墨迹参考，粉框为纸面参考，蓝色覆盖为新增墨迹。',font=font,fill='#263c34')
    board.save(ROOT/'comparison.png')


def main():
    ROOT.mkdir(exist_ok=True)
    reviewed = read(NATURAL/'results.json')
    assert sha(NATURAL/'review_protocol.json')==reviewed['protocol_sha256']
    assert sha(DIRECTION)==read(NATURAL/'screen_protocol.json')['direction_rule']
    assert sha(DIRECTION)==read(REGRESSION/'protocol.json')['module']
    natural = read(NATURAL/'screen_protocol.json')['images']
    controls = read(REFERENCE/'protocol.json')
    all_sources = {**natural, **controls['images']}
    inputs = {}
    for name in all_sources:
        if name in natural:
            inputs[name]=(NATURAL/f'{name}.npz',NATURAL/f'{name}_direction.npz','roles')
        else:
            inputs[name]=(REGRESSION/f'{name}.npz',REGRESSION/f'{name}.npz','anchored')
    protocol = dict(module=sha(MODULE), audit=sha(NATURAL/'results.json'), options=OPTIONS,
                    arrays={key:[sha(paths[0]),sha(paths[1])] for key,paths in inputs.items()},
                    variance_scope='Candidate settings compared on known examples and prior regressions.')
    path = ROOT/'protocol.json'
    if path.exists():
        assert read(path)==protocol
    else:
        write_json(path,protocol)

    images, changes = {}, {}
    for name,meta in all_sources.items():
        assert meta['sha256']==sha(meta['path'])
        with Image.open(meta['path']) as opened:
            image = np.array(opened.convert('RGB'))
        source, prior, key = inputs[name]
        with np.load(source) as f, np.load(prior) as other:
            baseline, previous = f['baseline'],other[key]
        additions = (baseline==0)&(previous>0)
        variants = dict(baseline=baseline,previous=previous)
        changes[name] = dict(previous=int(additions.sum()))
        for variant,args in OPTIONS.items():
            result,keep,reject = guard_faint_ink_anchors(image,baseline,previous,**args)
            np.testing.assert_array_equal(keep | reject, additions)
            np.testing.assert_array_equal(result[~reject],previous[~reject])
            variants[variant] = result
            changes[name][variant] = int(keep.sum())
        images[name] = image, variants
        np.savez_compressed(ROOT/f'{name}.npz',**variants)
    points = []
    for datum in reviewed['points']:
        x,y = datum['xy']
        _, variants = images[datum['image']]
        ink = {key:bool(value[y,x]) for key,value in variants.items()}
        assert ink['previous']==datum['ink']['direction']
        points.append(dict(datum,ink=ink))
    metrics = {}
    for label in ('faint','ink','paper','unknown'):
        chosen = [item for item in points if item['label']==label]
        metrics[label] = dict(n=len(chosen), ink={key:sum(item['ink'][key] for item in chosen)
                                                for key in ('previous',*OPTIONS)})
    regions = {}
    for spec in controls['specs']:
        name = spec['image']
        image,variants = images[name]
        _,mask,rows,transposed = local_reference(spec,image,controls['images'][name]['crop'])
        regions[spec['id']] = {label:score(value.T if transposed else value,mask,rows)
                               for label,value in variants.items()}
    comparing(images,points)
    locks = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:sha(locks[key])==locks[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    output = dict(points=points,metrics=metrics,changes=changes,regions=regions,baseline_hashes=hashes,
                  protocol_sha256=sha(path),independent_validation=False,adopted=False,robot_connected=False)
    write_json(ROOT/'results.json',output)
    print(metrics)
    print(changes)
    print({k:(v['rows_with_both_rails'],v['gap_ink_pixels'])
           for k,v in regions['X_gap'].items()})


if __name__ == '__main__':
    main()
