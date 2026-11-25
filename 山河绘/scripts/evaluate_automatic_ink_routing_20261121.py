# -*- coding: utf-8 -*-
"""Compare local automatic endpoint extraction with and without archived role hints."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_whole_scene_repair_20261121 import ROOT as WHOLE, raster_curves
from evaluate_guided_ink_reconstruction_20261121 import ROOT as GUIDED, TARGETS, describe, to_mask
from evaluate_object_stroke_gate_20261121 import ROOT as GATE, trace_evidence
from gongbi_training.automatic_ink_routing import automatic_ink_route
from gongbi_training.local_rock_patch import contrast_field
from validate_faint_ink_narrow_support_holdout_20261121 import segment


LAB = segment.LAB
ROOT = LAB/'automatic_ink_ownership_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/automatic_ink_routing.py'


def role_hint(plan, target, shape):
    roles = {'branch'} if target['role']=='branch' else {'structure','cun_candidate'}
    eligible = [row for row in plan['curves']
                if row['module']==target['module'] and row['stroke_role'] in roles]
    hints,_ = raster_curves({'curves':eligible},shape)
    return hints


def tree_appearance(image, plan):
    chroma = cv2.cvtColor(image.astype(np.float32)/255,cv2.COLOR_RGB2LAB)[...,1:]
    profiles = []
    for roles in ({'trunk','branch'},{'foliage'}):
        points = [np.asarray(c['points']) for c in plan['curves']
                  if c['module']==8 and c['stroke_role'] in roles and not c['role_uncertain']]
        coords = np.unique(np.rint(np.concatenate(points)).astype(np.int32),axis=0)
        profiles.append(np.median(chroma[coords[:,1],coords[:,0]],axis=0))
    wood,foliage = (np.linalg.norm(chroma-p,axis=-1) for p in profiles)
    return .85*np.clip((wood-foliage+1.2)/5,0,1)


def shortlist(result, mask, trace):
    measured = [trace_evidence(trace,to_mask(item['points'],mask.shape),3)
                for item in result.get('routes',[])]
    return dict(count=len(measured),matching_rank=[i+1 for i,r in enumerate(measured) if r['connected']])


def metric(result, mask, trace, field):
    route = result['points']
    values = describe(route,mask,trace,field)
    endpoints = route[[0,-1]].tolist() if len(route)>1 else []
    return dict(values,status=result['status'],endpoints=endpoints,
                competing_costs=[round(v['cost'],4) for v in result['alternatives']])


def chart(rgb, rows):
    board = Image.new('RGB',(1064,240*len(rows)+32),'#f5f1e7')
    pen = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    titles = ('原图','自动选点加角色参考','加入枝叶色度','扩大端点搜索')
    for j,(target,hinted,colored,dense) in enumerate(rows):
        x0,y0,x1,y1=target['crop']
        sx,sy=x1-x0,y1-y0
        scale=min(251/sx,199/sy)
        for i,path in enumerate((None,hinted,colored,dense)):
            img=rgb[y0:y1,x0:x1].copy()
            if path is not None and len(path['points'])>1:
                points=(path['points']-[x0,y0]).astype(np.int32)
                cv2.polylines(img,[points],False,(222,126,33) if path['status']=='ambiguous' else (32,157,182),1)
                for pos in (points[0],points[-1]):
                    cv2.circle(img,tuple(pos),1,(228,64,104),-1)
                if path['status']=='ambiguous' and len(path.get('alternatives',[]))>1:
                    other=path['alternatives'][1]['points']-[x0,y0]
                    cv2.polylines(img,[other.astype(np.int32)],False,(219,156,74),1)
            x,y=i*266+6,j*240+6
            pen.text((x,y),f"{target['title']} · {titles[i]}",font=font,fill='#263c34')
            tile=Image.fromarray(img).resize((int(sx*scale),int(sy*scale)),Image.Resampling.NEAREST)
            board.paste(tile,(x,y+24))
    pen.text((8,board.height-25),'蓝线为选出的路径，橙线为有归属歧义的路径，红点为端点。',font=font,fill='#263c34')
    board.save(ROOT/'comparison.png')


def main():
    ROOT.mkdir(exist_ok=True)
    frozen=segment.read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes={k:segment.sha(frozen[k])==frozen[k+'_sha256'] for k in ('approved500','working489')}
    assert all(hashes.values())
    source=segment.read(WHOLE/'protocol.json')['images']['G03_01']
    assert segment.sha(source['path'])==source['sha256']
    guide=segment.read(GUIDED/'protocol.json')
    guides=segment.read(GATE/'references.json')
    assert segment.sha(GATE/'references.json')==guide['reference_sha256']
    mask_file=WHOLE/'G03_01.npz'
    assert segment.sha(mask_file)==guide['evidence_sha256']
    queries=[{k:t[k] for k in ('id','title','module','role','crop')} for t in TARGETS]
    lock=ROOT/'protocol.json'
    segment.lock_file(lock,dict(module=segment.sha(MODULE),image=source,mask=guide['evidence_sha256'],
                    working_plan=frozen['working489_sha256'],queries=queries,
                    endpoint_search=[8,3,2,.03],appearance_weight=.85,
                    scope='Same manual inspection boxes. No manual endpoints or interior reference in selection.'))
    with Image.open(source['path']) as image,np.load(mask_file) as cached:
        rgb=np.array(image.convert('RGB'))
        mask=cached['baseline']
    plan=segment.read(frozen['working489'])
    manual={c['id']:c['points'] for c in segment.read(GUIDED/'candidate_paths.json')['paths']}
    refs={row['id']:row['trace'] for row in guides['rows']}
    field=contrast_field(rgb)
    appearance=tree_appearance(rgb,plan)
    metrics,rows,paths={},[],[]
    for target in queries:
        key=target['id']
        hints=role_hint(plan,target,mask.shape)
        auto=automatic_ink_route(rgb,mask,crop=target['crop'])
        hinted=automatic_ink_route(rgb,mask,crop=target['crop'],hints=hints,include_routes=True)
        owner=appearance if target['role']=='branch' else None
        colored=automatic_ink_route(rgb,mask,crop=target['crop'],hints=hints,ownership_cost=owner)
        dense=automatic_ink_route(rgb,mask,crop=target['crop'],hints=hints,
                                  gate_options=(8,3,2,.03),include_routes=True)
        metrics[key]={name:metric(route,mask,refs[key],field)
                      for name,route in [('source_only',auto),('role_hint',hinted),
                                         ('appearance',colored),('dense_endpoints',dense)]}
        for name,route in [('role_hint',hinted),('dense_endpoints',dense)]:
            metrics[key][name]['shortlist_diagnostic'] = shortlist(route,mask,refs[key])
        old=dict(points=np.asarray(manual[key]),status='manual') if key in manual else None
        if old is not None:
            metrics[key]['manual']=metric(dict(old,alternatives=[]),mask,refs[key],field)
        if hinted['status']=='route_found' and metrics[key]['role_hint']['radius3']['connected']:
            paths.append(dict(id=key,module=target['module'],role=target['role'],
                              points=hinted['points'].tolist()))
        rows.append((target,hinted,colored,dense))
    chart(rgb,rows)
    segment.write_json(ROOT/'candidate_paths.json',dict(paths=paths,ready_for_robot=False,adopted=False))
    segment.write_json(ROOT/'results.json',dict(metrics=metrics,baseline_hashes=hashes,protocol_sha256=segment.sha(lock),
                        independent_acceptance=False,adopted=False,robot_connected=False))
    for key,row in metrics.items():
        print(key,{k:[v['status'],v['endpoints'],v['radius1']['covered_near'],v['radius3']['connected'],
                      v.get('shortlist_diagnostic')] for k,v in row.items()},flush=True)
    print('exported', [r['id'] for r in paths])


if __name__=='__main__':
    main()
