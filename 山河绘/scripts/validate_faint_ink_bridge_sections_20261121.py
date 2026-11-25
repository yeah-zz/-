# -*- coding: utf-8 -*-
"""Score a source-reviewed holdout with the previously frozen bridge variants."""
import numpy as np
from PIL import Image

from evaluate_faint_ink_anchor_guard_20261121 import ROOT as TRIAL
from evaluate_faint_ink_bridge_sections_20261121 import MODULE, ROOT, evaluate_image
from evaluate_faint_ink_bridge_20261121 import LIMIT, read, render, sha, summarize, write_json
from gongbi_training.faint_ink_anchor_guard import guard_faint_ink_anchors
from gongbi_training.faint_ink_bridge import bridge_faint_ink


HOLDOUT = ROOT/'holdout'
# Group references marked on source-only previews. Labels refer to line presence, not stroke role.
REVIEW = {
    'silk_landscape': {1: ('unknown', '横向深带，岸线与底材未分'),
                       2: ('unknown', '横向深带，岸线与底材未分'),
                       4: ('unknown', '山林交界色面'),
                       7: ('unknown', '山林交界色面')},
    'layered_rocks': {2: ('ink', '屋顶连续横线'),
                      5: ('unknown', '山石深浅交界'),
                      8: ('unknown', '岩面浅色过渡'),
                      11: ('ink', '屋舍横边'),
                      12: ('ink', '较细的横向轮廓'),
                      13: ('ink', '岩面内部斜弧墨线'),
                      15: ('unknown', '梯状笔迹下缘'),
                      16: ('ink', '屋舍两条连续横边')},
}


def main():
    manifest_path = HOLDOUT/'screen_protocol.json'
    manifest = read(manifest_path)
    assert sha(MODULE)==manifest['module']==read(ROOT/'protocol.json')['module']
    screen_path = HOLDOUT/'screen.json'
    screen = read(screen_path)
    labels = [dict(image=name, group=group, xy=xy, label=label, note=note)
              for name, choices in REVIEW.items() for group,(label,note) in choices.items()
              for xy in screen[name][group-1]['xy']]
    protocol = dict(manifest=sha(manifest_path), screen=sha(screen_path),
                    arrays={key:sha(HOLDOUT/f'{key}.npz') for key in manifest['images']},
                    labels=labels, notes='Source-only labels fixed before running either bridge.')
    path = HOLDOUT/'review_protocol.json'
    if path.exists():
        assert read(path)==protocol
    else:
        write_json(path,protocol)
    options = read(TRIAL/'protocol.json')['options']['core']
    images, totals = {}, {}
    for name, info in manifest['images'].items():
        assert sha(info['path'])==info['sha256']
        assert sha(info['original'])==info['original_sha256']
        with Image.open(info['path']) as im, np.load(HOLDOUT/f'{name}.npz') as cached:
            rgb = np.array(im.convert('RGB'))
            baseline,direction = cached['baseline'],cached['direction']
        previous,_,_ = guard_faint_ink_anchors(rgb, baseline, direction, **options)
        bridge = bridge_faint_ink(rgb, baseline, direction, limit=LIMIT)
        previous[bridge] = direction[bridge]
        after = evaluate_image(rgb, baseline, direction, previous)
        variants = dict(baseline=baseline,direction=direction,combined=previous,sections=after)
        images[name] = rgb,variants
        totals[name] = {key:int(((mask>0)&(baseline==0)).sum()) for key,mask in variants.items()}
        totals[name]['added_xy'] = [[int(x),int(y)] for y,x in np.argwhere((after>0)&(previous==0))]
        np.savez_compressed(HOLDOUT/f'{name}_out.npz',**variants)
    points = []
    for point in labels:
        x,y = point['xy']
        _,variants = images[point['image']]
        flags = {key:bool(mask[y,x]) for key,mask in variants.items()}
        assert not flags['baseline'] and flags['direction']
        points.append(dict(point,ink=flags))
    picks = [(n,g,t) for n,g,t in [('layered_rocks',2,'屋顶横线'),('layered_rocks',11,'屋舍横边'),
                 ('layered_rocks',12,'横向轮廓'),('layered_rocks',13,'岩面斜弧线'),
                 ('layered_rocks',16,'屋舍横边'),('silk_landscape',1,'深带待核')]]
    columns = ((None,'原图'),('direction','方向筛选'),('combined','双端续接'),('sections','逐截面复核'))
    render(images,points,HOLDOUT/'comparison.png',sections=picks,columns=columns)
    result = dict(metrics=summarize(points),totals=totals,points=points,review_protocol_sha256=sha(path),
                  passed_independent_acceptance=False)
    write_json(HOLDOUT/'results.json',result)
    print(result['metrics'])
    print(totals)


if __name__=='__main__':
    main()
