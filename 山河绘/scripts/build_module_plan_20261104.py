# -*- coding: utf-8 -*-
"""Auditable offline module/whole-curve comparisons. No physical execution exports."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import argparse
import colorsys
import hashlib
import html
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from module_stroke_core_20261104 import CONFIG,propose_modules,assemble,schedule,route_metrics,length,raster,at,samples

ROOT=LAB/'module_stroke_planner_20261104'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)

def save(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)

def color(i):
    return tuple(round(v*210) for v in colorsys.hsv_to_rgb((i*.61803398875)%1,.8,1))

def baseline_route(curves):
    order=sorted(range(len(curves)),key=lambda i:-length(curves[i]['points']))
    cursor=np.array([0.,495.]);route=[];recent=[]
    for i in order:
        p=curves[i]['points'];reverse=np.linalg.norm(p[-1]-cursor)<np.linalg.norm(p[0]-cursor)
        if reverse:p=p[::-1]
        wet=raster(recent[-12:],(496,496),3)
        route.append(dict(curve=i,module=curves[i]['module'],reverse=bool(reverse),
            travel_px=float(np.linalg.norm(p[0]-cursor)),
            predicted_recent_overlap_fraction=float(at(wet,samples(p,2.)).mean()),
            predicted_start_contact_with_recent_ink=bool(at(wet,p[:1])[0])))
        recent.append(p);cursor=p[-1]
    return route

def distance_aware_route(curves):
    """Recompute prior length/(1+travel/10) heuristic on the identical curve set."""
    p0=np.array([r['points'][0] for r in curves]);p1=np.array([r['points'][-1] for r in curves])
    lens=np.array([length(r['points']) for r in curves]);remaining=np.ones(len(curves),bool)
    cursor=np.array([0.,495.]);route=[];recent=[]
    while remaining.any():
        d0=np.linalg.norm(p0-cursor,axis=1);d1=np.linalg.norm(p1-cursor,axis=1)
        dist=np.minimum(d0,d1);score=np.where(remaining,lens/(1+dist/10.),-np.inf)
        i=int(np.argmax(score));reverse=bool(d1[i]<d0[i]);p=curves[i]['points'][::-1] if reverse else curves[i]['points']
        wet=raster(recent[-12:],(496,496),3)
        route.append(dict(curve=i,module=curves[i]['module'],reverse=reverse,travel_px=float(dist[i]),
            predicted_recent_overlap_fraction=float(at(wet,samples(p,2.)).mean()),
            predicted_start_contact_with_recent_ink=bool(at(wet,p[:1])[0])))
        remaining[i]=False;cursor=p[-1];recent.append(p)
    return route

def drawing(curves,route=None,count=None):
    im=np.full((496,496,3),255,np.uint8)
    if route is None:
        route=[dict(curve=i,reverse=False) for i in range(len(curves))]
    if count is not None:route=route[:count]
    for s in route:
        r=curves[s['curve']]
        p=r['points'][::-1] if s.get('reverse') else r['points']
        cv2.polylines(im,[np.rint(p).astype(np.int32)],False,color(r['module']),1,cv2.LINE_AA)
    return im

def export_html(folder, payload):
    template=(Path(__file__).with_name('module_plan_viewer_20261104.html')).read_text('utf-8')
    data=json.dumps(payload,ensure_ascii=False,default=serial,allow_nan=False).replace('</','<\\/')
    (folder/'index.html').write_text(template.replace('__DATA__',data),encoding='utf-8')

def build(ident):
    start=time.monotonic();folder=ROOT/'outputs'/ident
    frozen=LAB/'stroke_unit_model_20261102/outputs'/ident
    source=frozen/'source.png';pathfile=frozen/'curves.json'
    doc=json.loads(pathfile.read_text('utf-8'))
    rgb=np.array(Image.open(source).convert('RGB').resize((496,496),Image.Resampling.LANCZOS))
    factor=np.array([496/doc['width'],496/doc['height']])
    paths=[np.asarray(r['points'],float)*factor for r in doc['paths']]
    manifest=json.loads((folder/'sam_manifest_seeded.json').read_text('utf-8'))
    assert manifest['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(folder/'sam_masks_seeded.npz') as masks:
        modules,ownership=propose_modules(rgb,paths,masks,manifest['records'])
    old=[dict(points=p,module=o['module'],source_curve_ids=[i]) for i,(p,o) in enumerate(zip(paths,ownership))]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)/255.
    new,diagnostics=assemble(paths,gray,ownership)
    print(ident,'assigned',len(modules),'regions, assembled',len(paths),'->',len(new),flush=True)
    route_long=baseline_route(old)
    route_old=distance_aware_route(old)
    module_only=schedule(old,modules,gray.shape)
    route=schedule(new,modules,gray.shape)
    print(ident,'scheduled',round(time.monotonic()-start,1),'s',flush=True)
    metrics_old=route_metrics(route_old,old);metrics=route_metrics(route,new)
    mm_per_px=186.8/495 # comparison convention, not a current physical calibration
    def geom(curves):
        lens=np.array([length(r['points'])*mm_per_px for r in curves])
        return dict(count=len(curves),under_2mm=int((lens<2).sum()),over_10mm=int((lens>10).sum()),
            median_mm=float(np.median(lens)),max_mm=float(lens.max()),total_length_mm=float(lens.sum()))
    stats=dict(id=ident,baseline=geom(old),assembled=geom(new),baseline_route=metrics_old,module_route=metrics,
        global_long_first_control=route_metrics(route_long,old),
        reorder_only_ablation=route_metrics(module_only,old),
        modules=len(modules),sam_owned_modules=sum(m['sam_key'] is not None for m in modules),
        unresolved_local_regions=sum(m['sam_key'] is None for m in modules),
        unresolved_source_curves=sum(o['uncertain'] for o in ownership),
        accepted_joins=len(diagnostics['accepted']),pending_continuations=len(diagnostics['pending']),
        no_original_curve_deleted=True,ready_for_robot=False,robot_connected=False,
        stroke_identity_accuracy=None,instance_segmentation_accuracy=None,
        new_pixel_extraction_training=False,physical_world_model_trained=False,
        same_previously_seen_source_images=True,new_blind_test=False,
        reference_mm_per_pixel=mm_per_px,reference_scale_is_not_robot_calibration=True,
        runtime_seconds=time.monotonic()-start,
        baseline_is_recomputed_prior_distance_aware_heuristic_not_actual_execution_log=True)
    stats['release_status']='offline_candidate_only'
    stats['unresolved_risks']=[
        'SAM proposals are not verified rock/tree instances; some object boundaries remain unresolved',
        'Many short curves remain; this is not complete artist-stroke recovery',
        'Geometric recent-ink overlap is not calibrated wetness or actual ink spread',
        'No fixed camera registration or measured stroke-outcome predictor']
    stats['recent_ink_proxy_worsened']=metrics['predicted_start_recent_contacts']>metrics_old['predicted_start_recent_contacts']
    assert metrics['module_reentries']==0
    before_len=sum(length(p) for p in paths)
    after_len=sum(length(r['points']) for r in new)
    assert after_len+1e-6>=before_len
    assert after_len-before_len<=sum(r['gap'] for r in diagnostics['accepted'])+1e-5
    save(folder/'statistics.json',stats)
    save(folder/'assembly_diagnostics.json',diagnostics)
    descriptions=[{k:v for k,v in m.items() if k!='mask'} for m in modules]
    plan=dict(schema='offline-module-full-curve-v1',id=ident,image_size=[496,496],
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_curves=str(pathfile),source_curves_sha256=hashlib.sha256(pathfile.read_bytes()).hexdigest(),
        coordinate_system='normalized image pixels, no robot coordinates',
        configuration=CONFIG,modules=descriptions,ownership=ownership,curves=new,actions=route,
        ready_for_robot=False,robot_connected=False,physical_pressure=None,physical_speed=None,
        state_observation='planned-command coverage only; no camera input',
        world_model='geometric rollout interface; physical predictor absent',
        actual_observation_required_for_closed_loop=True,
        brush_ink_state_observed=False,stroke_direction_verified=False,
        no_original_curve_deleted=True,artist_stroke_identity_verified=False)
    (folder/'offline_plan.json').write_text(json.dumps(plan,ensure_ascii=False,default=serial,indent=2),encoding='utf-8')
    np.savez_compressed(folder/'selected_module_masks.npz',**{f'm{m["id"]:03}':m['mask'] for m in modules})
    overlay=(rgb*.8+255*.2).astype(np.uint8)
    for m in reversed(modules):
        if m['sam_key'] is None:continue
        mask=m['mask'];overlay[mask]=(overlay[mask]*.62+np.array(color(m['id']))*.38).astype(np.uint8)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,color(m['id']),1)
    Image.fromarray(rgb).save(folder/'source.png')
    Image.fromarray(overlay).save(folder/'module_candidates.png')
    Image.fromarray(drawing(new)).save(folder/'assembled_modules.png')
    first=100
    panels=[(rgb,'原图'),(overlay,'自动模块候选（未经实例标注验证）'),
            (drawing(old,route_old,first),f'旧距离感知排序：前{first}笔'),
            (drawing(new,route,first),f'模块内完成：前{first}笔')]
    board=Image.new('RGB',(2016,565),'#f5f3ec');d=ImageDraw.Draw(board)
    for i,(im,title) in enumerate(panels):
        board.paste(Image.fromarray(im),(i*504,42));d.text((i*504+5,10),title,font=FONT,fill='black')
    d.text((8,541),f'{ident}｜完整线条保留：{len(old)} → {len(new)}条；仅几何轨迹预览，非实画，未复现GWM',font=FONT,fill='black')
    board.save(folder/'comparison.jpg',quality=95)
    payload=dict(id=ident,old=old,new=new,oldRoute=route_old,longRoute=route_long,route=route,modules=descriptions,stats=stats)
    export_html(folder,payload)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 496">',
         '<title>Offline module-ordered centerlines; not executable robot commands</title>']
    for step,s in enumerate(route):
        r=new[s['curve']];p=r['points'][::-1] if s['reverse'] else r['points']
        points=' '.join(f'{x:.3f},{y:.3f}' for x,y in p)
        rgbc=','.join(map(str,color(r['module'])))
        svg.append(f'<polyline id="s{step:04}" data-module="{r["module"]}" points="{points}" fill="none" stroke="rgb({rgbc})" stroke-width=".7"/>')
    svg.append('</svg>');(folder/'ordered_preview.svg').write_text('\n'.join(svg),encoding='utf-8')
    return stats

def main(ids):
    cv2.setNumThreads(2)
    records=[build(i) for i in ids]
    save(ROOT/'results.json',dict(records=records,robot_connected=False,new_training_performed=False,
        physical_predictor='absent; needs registered before/action/after data'))
    links=''.join(f'<li><a href="outputs/{html.escape(r["id"])}/index.html">{html.escape(r["id"])} · 顺序动画与数据</a></li>' for r in records)
    (ROOT/'index.html').write_text('<meta charset="utf-8"><title>模块与连续笔画离线实验</title><h1>模块与连续笔画离线实验</h1>'
        '<p>预训练SAM对象候选＋原图支持的曲线连接＋模块内完成的几何规划。不是实画回放，不是已训练的GWM。</p><ul>'+links+'</ul>',encoding='utf-8')
    print(json.dumps(records,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ids',nargs='+',default=['G03_01','G05_03','T02'])
    main(parser.parse_args().ids)
