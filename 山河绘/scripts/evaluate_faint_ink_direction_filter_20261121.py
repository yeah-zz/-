# -*- coding: utf-8 -*-
"""Evaluate direction filtering of the frozen faint-ink proposals."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_faint_ink_continuation_20261121 import MODULE as CONTINUATION, ROOT as OLD
from evaluate_ink_paper_extension_20261121 import write_json
from evaluate_ink_paper_regions_20261118 import render
from evaluate_ink_paper_seams_20261118 import LAB, UPSTREAM, read, sha, stats
from evaluate_ink_paper_width_20261119 import score
from validate_ink_paper_join_guard_20261121 import ROOT as REFERENCE, local_reference
from gongbi_training.faint_ink_continuation import recover_faint_ink
from gongbi_training.faint_ink_direction_filter import filter_faint_ink_direction


ROOT = LAB/'faint_ink_direction_trial_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/faint_ink_direction_filter.py'
SETTINGS = dict(source_direction=dict(min_coherence=0.45, aligned=False),
                anchored=dict(min_coherence=0.45, aligned=True),
                strict=dict(min_coherence=0.50, aligned=True))


def compare(image, baseline, previous):
    variants = dict(baseline=baseline, previous=previous)
    proposals = (baseline == 0) & (previous > 0)
    counts = dict(previous=int(proposals.sum()))
    for name, option in SETTINGS.items():
        result, accepted, rejected = filter_faint_ink_direction(image, baseline, previous, **option)
        np.testing.assert_array_equal(accepted | rejected, proposals)
        np.testing.assert_array_equal(result[~rejected], previous[~rejected])
        np.testing.assert_array_equal(result[rejected], baseline[rejected])
        variants[name] = result
        counts[name] = int(accepted.sum())
    return variants, counts


def box_detail(image, variants, box, filename):
    width, height = 335, 300
    board = Image.new('RGB', (width * 3, height + 40), '#f5f1e7')
    pen = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    old = variants['baseline'] > 0
    for index, (name, title) in enumerate([(None,'原图'),('previous','上轮补点'),('anchored','按方向续接')]):
        painted = image.copy()
        if name:
            extra = (variants[name] > 0) & ~old
            painted[old] = (.62 * painted[old] + .38 * np.array([34,137,111])).astype(np.uint8)
            painted[extra] = (.36 * painted[extra] + .64 * np.array([42,155,201])).astype(np.uint8)
        tile = Image.fromarray(painted).crop(box)
        scale = min((width-15)/tile.width, (height-34)/tile.height)
        tile = tile.resize((round(tile.width*scale), round(tile.height*scale)), Image.Resampling.NEAREST)
        x = index*width+7
        pen.text((x,4),title,font=font,fill='#263c34')
        board.paste(tile,(x,30))
    pen.text((7,height+8),'蓝色为新增预测墨迹，绿色为原预测墨迹。',font=font,fill='#263c34')
    board.save(filename)


def main():
    ROOT.mkdir(exist_ok=True)
    source = read(REFERENCE/'protocol.json')
    previous = read(OLD/'results.json')
    assert sha(OLD/'protocol.json') == previous['protocol_sha256']
    assert read(OLD/'protocol.json')['module'] == sha(CONTINUATION)
    h_prediction = UPSTREAM/'predictions/H_band_paper.npz'
    h_reference = LAB/'cunfa_band_checked_20261112/references/H_14.npz'
    lock = dict(module=sha(MODULE), previous=sha(OLD/'results.json'),
                inputs={name:sha(OLD/f'{name}.npz') for name in source['images']},
                H14=[sha(h_prediction),sha(h_reference)], settings=SETTINGS, independent=False)
    path = ROOT/'protocol.json'
    if path.exists():
        assert read(path)==lock
    else:
        write_json(path,lock)
    images, changes = {}, {}
    for name, meta in source['images'].items():
        assert sha(meta['path']) == meta['sha256']
        with Image.open(meta['path']) as opened:
            image = np.array(opened.convert('RGB'))
        with np.load(OLD/f'{name}.npz') as candidates:
            variants, changes[name] = compare(image,candidates['baseline'],candidates['longer'])
        images[name] = image, variants
        np.savez_compressed(ROOT/f'{name}.npz', **variants)

    regions, panels = {}, []
    for spec in source['specs']:
        name = spec['image']
        image, variants = images[name]
        view, mask, rows, transverse = local_reference(spec, image, source['images'][name]['crop'])
        measured = {key: score(labels.T if transverse else labels,mask,rows) for key,labels in variants.items()}
        assert measured['previous'] == previous['regions'][spec['id']]['longer']
        regions[spec['id']] = measured
        panels.append((view,image,mask.T if transverse else mask,
                       {k: variants[k] for k in ('baseline','previous','anchored','strict')}))
    render(panels, ROOT/'comparison.png', ['原图参考','冻结模型','上轮补点','按方向续接','高门槛'])
    water_image, mesh = images['xia']
    box_detail(water_image,mesh,(146,154,167,176),ROOT/'water_detail.png')
    audited = []
    for item in read(OLD/'outside_audit.json'):
        image, variants = images[item['image']]
        x0,y0,x1,y1 = item['box']
        old_ink = variants['baseline'][y0:y1,x0:x1]>0
        measured = {key:int((~old_ink & (pred[y0:y1,x0:x1]>0)).sum())
                    for key,pred in variants.items() if key!='baseline'}
        audited.append(dict(item,new_in_box=measured))
    target = next(item for item in audited if item['image']=='mi' and item['box'][1]<100)
    image, variants = images['mi']
    box_detail(image,variants,target['box'],ROOT/'outside_detail.png')
    with np.load(h_prediction) as p, np.load(h_reference) as f:
        im,base,target = p['image'],p['prediction'],f['band_targets']
        np.testing.assert_array_equal(im,f['image'])
    pre,_ = recover_faint_ink(im,base,limit=6,allow_tips=True,min_path_contrast=12)
    variants, changes['H14'] = compare(im,base,pre)
    h = {key: stats(base,pred,np.isin(target,[1,2]),target==0) for key,pred in variants.items()}
    baselines = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {k:sha(baselines[k])==baselines[k+'_sha256'] for k in ('approved500','working489')}
    assert all(hashes.values())
    outcome = dict(regions=regions, additions=changes, audit=audited,H14=h,
                   baseline_hashes=hashes,protocol_sha256=sha(path),independent_validation=False,
                   adopted=False,robot_connected=False)
    write_json(ROOT/'results.json',outcome)
    print(changes)
    print({k: [v['rows_with_both_rails'],v['longest_8_connected_rows'],v['gap_ink_pixels']]
           for k,v in regions['X_gap'].items()})
    print(audited)
    print(h)


if __name__ == '__main__':
    main()
