# -*- coding: utf-8 -*-
"""Compare whole-object roles against source-reviewed rock-face and branch associations."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_automatic_ink_routing_20261121 import ROOT as AUTO, contrast_field, metric, role_hint, shortlist
from evaluate_guided_ink_reconstruction_20261121 import ROOT as GUIDED, TARGETS
from evaluate_object_stroke_gate_20261121 import ROOT as GATE
from compare_whole_scene_repair_20261121 import ARCHIVE, ROOT as WHOLE
from gongbi_training.automatic_ink_routing import automatic_ink_route
from gongbi_training.structural_ink_routing import route_owned_ink
from gongbi_training.structure_candidate_rank import rank_owned_candidates
from validate_faint_ink_narrow_support_holdout_20261121 import segment


LAB = segment.LAB
ROOT = LAB/'structural_ink_ranking_20261122'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/structural_ink_routing.py'
ATTACHMENT_WEIGHT = .8
# These are source-reviewed areas, not endpoint coordinates or stroke traces.
# The tree boxes identify the visible branch neighborhood and its trunk attachment.
ZONES = {
    'rock_ridge': dict(start=[259,303,275,313],end=[276,332,286,344],owner='m020',
                       basis='The inner blue-green face between two darker rock boundaries.'),
    'pine_right': dict(start=[290,133,304,144],end=[354,165,369,178],owner='tree_outline',
                       basis='The upper main-trunk shoulder and the descending branch before the next fork.'),
    'pine_left': dict(start=[221,149,237,159],end=[277,148,290,157],owner='tree_outline',
                      basis='The left horizontal branch and its meeting area on the trunk.'),
}


def rectangle(shape, box):
    mask = np.zeros(shape,np.uint8)
    x0,y0,x1,y1=box
    mask[y0:y1,x0:x1] = 1
    return mask


def tree_outline(plan, shape):
    layer=np.zeros(shape,np.uint8)
    tree=next(m for m in plan['modules'] if m['id']==8)
    cv2.fillPoly(layer,[np.rint(poly).astype(np.int32) for poly in tree['outlines']],1)
    return layer


def draw_compare(rgb, views):
    tile_w, tile_h = 285,184
    board=Image.new('RGB',(900,224*len(views)+33),'#f5f1e7')
    pen=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for row,(target,zone,owner,before,after) in enumerate(views):
        x0,y0,x1,y1=target['crop']
        sz=(x1-x0,y1-y0)
        scale=min(tile_w/sz[0],tile_h/sz[1])
        for col,(name,item) in enumerate((('结构区域',None),('上轮自动首选',before),('按归属和端区重排',after))):
            source=rgb[y0:y1,x0:x1].copy()
            if item is None:
                local=owner[y0:y1,x0:x1]>0
                source[local]=(.8*source[local]+.2*np.array([21,152,132])).astype(np.uint8)
            tile=Image.fromarray(source).resize(tuple(round(v*scale) for v in sz),Image.Resampling.NEAREST)
            ink=ImageDraw.Draw(tile)
            if item is None:
                for key,color in [('start','#ec9c28'),('end','#58b9c6')]:
                    a,b,c,d=zone[key]
                    ink.rectangle([(round((a-x0)*scale),round((b-y0)*scale)),
                                   (round((c-1-x0)*scale),round((d-1-y0)*scale))],outline=color,width=2)
            else:
                q=item['points']
                xy=[(round((x-x0)*scale),round((y-y0)*scale)) for x,y in q]
                color='#169ab5' if item['status']=='route_found' else '#e89124'
                if len(xy)>1:
                    ink.line(xy,fill=color,width=2)
            left,top=col*300+5,row*224+5
            pen.text((left,top),f"{target['title']} · {name}",font=font,fill='#263c34')
            board.paste(tile,(left,top+25))
    pen.text((6,board.height-26),'浅绿色显示归属区域。橙框、蓝框为起侧和止侧区域。橙线为存在歧义的首选。',font=font,fill='#263c34')
    board.save(ROOT/'comparison.png')


def main():
    ROOT.mkdir(exist_ok=True)
    previous=segment.read(AUTO/'results.json')
    assert segment.sha(AUTO/'protocol.json')==previous['protocol_sha256']
    lock=segment.read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes={key:segment.sha(lock[key])==lock[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    source=ARCHIVE/'G03_01/source.png'
    anchors=ARCHIVE/'G03_01/selected_module_masks.npz'
    mask_file=WHOLE/'G03_01.npz'
    data=segment.read(GATE/'references.json')
    assert segment.sha(mask_file)==data['inputs']['g_scene']['frozen_array_sha256']
    protocol=ROOT/'protocol.json'
    segment.lock_file(protocol,dict(module=segment.sha(MODULE),source=segment.sha(source),mask=segment.sha(mask_file),
                      ranking=segment.sha(MODULE.with_name('structure_candidate_rank.py')),
                      attachment_weight=ATTACHMENT_WEIGHT,
                      archived_parts=segment.sha(anchors),references=segment.sha(GATE/'references.json'),
                      baseline=lock['working489_sha256'],zones=ZONES,
                      scope='Source-reviewed structural neighborhoods and archived object masks. Reference traces for evaluation only.'))
    with Image.open(source) as im, np.load(mask_file) as f, np.load(anchors) as parts:
        rgb=np.array(im.convert('RGB'))
        mask=f['baseline']
        rock=parts['m020']>0
    plan=segment.read(lock['working489'])
    shape=mask.shape
    owners=dict(m020=rock,tree_outline=tree_outline(plan,shape))
    field=contrast_field(rgb)
    refs={row['id']:row['trace'] for row in data['rows']}
    scored,views,paths={},[],[]
    for target in TARGETS:
        key=target['id']
        zone=ZONES[key]
        owner=owners[zone['owner']]
        hint=role_hint(plan,target,shape)
        options=(4,2,4,.10) if target['role']=='structure' else (8,3,2,.03)
        baseline=automatic_ink_route(rgb,mask,crop=target['crop'],hints=hint,gate_options=options,include_routes=True)
        owner_only=automatic_ink_route(rgb,mask*owner,crop=target['crop'],hints=hint,gate_options=options)
        routed=route_owned_ink(rgb,mask,owner,crop=target['crop'],start_zone=rectangle(shape,zone['start']),
                              end_zone=rectangle(shape,zone['end']),hints=hint)
        ranked=rank_owned_candidates([baseline,routed],owner,zone['start'],zone['end'],
                                     attachment_weight=ATTACHMENT_WEIGHT)
        results = {name:metric(result,mask,refs[key],field) for name,result in
                   [('prior',baseline),('owner_only',owner_only),('structure',routed),('ranked',ranked)]}
        results['structure']['shortlist']=shortlist(routed,mask,refs[key])
        results['ranked']['shortlist']=shortlist(ranked,mask,refs[key])
        scored[key]=results
        if ranked['status']=='route_found' and results['ranked']['radius3']['connected']:
            paths.append(dict(id=key,role=target['role'],module=target['module'],
                              points=ranked['points'].tolist(),
                              origin='assisted structure regions with automatic point choice'))
        views.append((target,zone,owner,baseline,ranked))
    draw_compare(rgb,views)
    segment.write_json(ROOT/'paths.json',dict(paths=paths,offline=True,adopted=False))
    segment.write_json(ROOT/'results.json',dict(scores=scored,baseline_hashes=hashes,protocol_sha256=segment.sha(protocol),
                       independent_acceptance=False,adopted=False,robot_connected=False))
    for key,row in scored.items():
        print(key,{stage:[r['status'],r['endpoints'],r['radius1']['covered_near'],r['radius3']['covered_near'],
                           r['radius3']['connected'],r.get('shortlist')] for stage,r in row.items()})
    print('exported', [p['id'] for p in paths])


if __name__=='__main__':
    main()
