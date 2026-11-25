# -*- coding: utf-8 -*-
"""Evaluate faint centerline recovery and source-paper false additions."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_ink_paper_extension_20261121 import write_json
from evaluate_ink_paper_regions_20261118 import render
from evaluate_ink_paper_seams_20261118 import LAB, UPSTREAM, read, sha, stats
from evaluate_ink_paper_width_20261119 import score
from validate_ink_paper_join_guard_20261121 import ROOT as PRIOR, local_reference
from gongbi_training.faint_ink_continuation import recover_faint_ink


ROOT = LAB/'faint_ink_continuation_trial_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/faint_ink_continuation.py'
SETTINGS = dict(double=dict(allow_tips=False), single=dict(allow_tips=True),
                coherent=dict(allow_tips=True, min_path_contrast=12),
                longer=dict(allow_tips=True, min_path_contrast=12, limit=6))


def compare(image, baseline):
    variants = dict(baseline=baseline)
    counts = dict(baseline=0)
    for name, settings in SETTINGS.items():
        updated, restored = recover_faint_ink(image, baseline, **settings)
        assert np.all(baseline[restored] == 0)
        np.testing.assert_array_equal(updated[~restored], baseline[~restored])
        variants[name] = updated
        counts[name] = int(restored.sum())
    return variants, counts


def detail(image, variants, points, crop, destination):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    board = Image.new('RGB', (1050, 356), '#f5f1e7')
    d = ImageDraw.Draw(board)
    x0, y0, x1, y1 = crop
    scale = 13
    for index, (name, title) in enumerate([(None,'原图'),('baseline','基线墨纹'),('coherent','整段检验'),
                                           ('longer','允许多续一步')]):
        view = image.copy()
        if name:
            old = variants['baseline'] > 0
            new = (variants[name] > 0) & ~old
            view[old] = (.60 * view[old] + .40 * np.array([34, 137, 111])).astype(np.uint8)
            view[new] = (.35 * view[new] + .65 * np.array([31, 142, 198])).astype(np.uint8)
        tile = Image.fromarray(view).crop(crop).resize(((x1-x0)*scale,(y1-y0)*scale),Image.Resampling.NEAREST)
        td = ImageDraw.Draw(tile)
        for x,y in points:
            px,py=(x-x0)*scale,(y-y0)*scale
            td.rectangle((px,py,px+scale-1,py+scale-1),outline='#ec8a2e',width=1)
        left = 8 + 261 * index
        d.text((left,5),title,font=font,fill='#263c34')
        board.paste(tile,(left,29))
    d.text((8,330),'橙框标出下侧淡水纹的原图参考位置。绿色为原墨迹预测，蓝色为本轮补回的墨迹。',font=font,fill='#263c34')
    board.save(destination)


def main():
    ROOT.mkdir(exist_ok=True)
    reference = read(PRIOR/'protocol.json')
    previous = read(PRIOR/'results.json')
    assert sha(PRIOR/'protocol.json') == previous['protocol_sha256']
    h_prediction = UPSTREAM/'predictions/H_band_paper.npz'
    h_reference = LAB/'cunfa_band_checked_20261112/references/H_14.npz'
    protocol = dict(module=sha(MODULE), previous=sha(PRIOR/'results.json'), settings=SETTINGS,
                    inputs={name:sha(PRIOR/f'{name}.npz') for name in reference['images']},
                    h_prediction=sha(h_prediction), h_reference=sha(h_reference), independent=False)
    path = ROOT/'protocol.json'
    if path.exists():
        assert read(path) == protocol
    else:
        write_json(path, protocol)
    images, changes = {}, {}
    for name, meta in reference['images'].items():
        assert sha(meta['path']) == meta['sha256']
        with Image.open(meta['path']) as source:
            image = np.array(source.convert('RGB'))
        with np.load(PRIOR/f'{name}.npz') as f:
            baseline = f['protected']
        variants, changes[name] = compare(image, baseline)
        images[name] = image, variants
        np.savez_compressed(ROOT/f'{name}.npz', **variants)

    regions, panels, rails = {}, [], {}
    for spec in reference['specs']:
        name = spec['image']
        image, variants = images[name]
        view, mask, rows, transverse = local_reference(spec, image, reference['images'][name]['crop'])
        results = {k: score(p.T if transverse else p, mask, rows) for k,p in variants.items()}
        assert results['baseline'] == previous['regions'][spec['id']]['scores']['protected']
        regions[spec['id']] = results
        if spec['id']=='X_gap':
            rails = {}
            for k,pred in variants.items():
                rails[k] = [int(x+reference['images'][name]['crop'][0]) for x,_,hi,_,lower in rows
                            if not pred[max(lower-1,hi+1):lower+2,x].any()]
            pts = [(x,y) for x,_,_,_,y in rows if x+reference['images'][name]['crop'][0] in rails['baseline']]
            detail(image, variants, pts, (146,154,166,176), ROOT/'water_line_detail.png')
        shown = {k: variants[k] for k in ('baseline','double','coherent','longer')}
        panels.append((view, image, mask.T if transverse else mask, shown))
    render(panels, ROOT/'comparison.png', ['原图参考', '基线', '只补两端相连', '整段检验', '多续一步'])
    with np.load(h_prediction) as f, np.load(h_reference) as ref:
        image, baseline, truth = f['image'], f['prediction'], ref['band_targets']
        np.testing.assert_array_equal(image, ref['image'])
    candidates, changes['H14'] = compare(image, baseline)
    checked = {key: stats(baseline, labels, np.isin(truth,[1,2]), truth==0) for key,labels in candidates.items()}

    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key: sha(lock[key]) == lock[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    result = dict(regions=regions, missing_lower_columns=rails, changes=changes, H14=checked,
                  baseline_hashes=hashes, protocol_sha256=sha(path), independent_validation=False,
                  adopted=False, robot_connected=False)
    write_json(ROOT/'results.json', result)
    print(rails)
    print({key:[value['rows_with_both_rails'],value['longest_8_connected_rows'],value['gap_ink_pixels']]
           for key,value in regions['X_gap'].items()})
    print(changes)
    print(checked)


if __name__ == '__main__':
    main()
