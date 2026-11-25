# -*- coding: utf-8 -*-
"""Compare frozen faint-ink repair against archived full-scene drawing plans."""
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from validate_faint_ink_narrow_support_holdout_20261121 import run_variants, segment, support_lock
from evaluate_ink_paper_width_20261119 import predict_extra
from screen_faint_ink_natural_20261121 import grouped


LAB = segment.LAB
ROOT = LAB/'whole_scene_repair_benchmark_20261121'
ARCHIVE = LAB/'module_stroke_planner_20261104/outputs'
WORKS = ('G03_01','G05_03','T02')


def raster_curves(plan, shape):
    coverage = np.zeros(shape,np.uint8)
    curve_masks = []
    for curve in plan['curves']:
        xy = np.asarray(np.rint(curve['points']),dtype=np.int32)
        path = np.zeros(shape,np.uint8)
        if len(xy)==1:
            cv2.circle(path,tuple(xy[0]),0,1,-1)
        elif len(xy)>1:
            cv2.polylines(path,[xy],False,1,1)
        coverage |= path
        curve_masks.append(path.astype(bool))
    return coverage.astype(bool),curve_masks


def order_metrics(plan):
    actions = plan['actions']
    module_runs = [module for module,_ in groupby(a['module'] for a in actions)]
    left = set(module_runs)
    roles = {}
    for row in plan['curves']:
        role = row.get('stroke_role','unverified')
        roles[role] = roles.get(role,0)+1
    lengths = [float(np.linalg.norm(np.diff(np.asarray(c['points']),axis=0),axis=1).sum())
               for c in plan['curves']]
    reversals = sum(a['module']==b['module'] and a.get('phase',0)>b.get('phase',0)
                    for a,b in zip(actions,actions[1:]))
    return dict(strokes=len(actions),module_runs=len(module_runs),modules=len(left),
                module_reentries=len(module_runs)-len(left),phase_reversals=reversals,
                roles=roles,shorter_than_5_px=sum(n<5 for n in lengths))


def assess(color, patched, coverage, curve_masks):
    added = (color==0) & (patched>0)
    near = cv2.dilate(coverage.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    points = np.argwhere(added)[:,::-1].tolist()
    touched = cv2.dilate(added.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    joins = []
    for role in (1,2):
        _,ids = cv2.connectedComponents((color==role).astype(np.uint8),connectivity=8)
        for group in grouped(added & (patched==role)):
            x,y = np.asarray(group['xy']).T
            blob = np.zeros(color.shape,np.uint8)
            blob[y,x] = 1
            surround = cv2.dilate(blob,np.ones((3,3),np.uint8)).astype(bool)
            components = int(np.unique(ids[surround & (ids>0)]).size)
            if components>=2:
                joins.append(dict(pixels=len(x),components=components,box=group['box']))
    return dict(added=int(added.sum()),coordinates=points,near_plan=int((added&near).sum()),
                touched_curves=sum(bool((path&touched).any()) for path in curve_masks),
                possible_joins=joins)


def figures(frames):
    tile = 344
    canvas = Image.new('RGB',(3*(tile+12),len(frames)*(tile+41)+40),'#f5f1e7')
    pen = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for row,(name,rgb,coverage,color,added) in enumerate(frames):
        plan = np.full_like(rgb,253)
        plan[coverage] = [44,54,51]
        overlay = rgb.copy()
        overlay[color>0] = .63*overlay[color>0] + .37*np.array([37,111,102])
        halo = cv2.dilate(added.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        overlay[halo] = .3*overlay[halo] + .7*np.array([215,74,28])
        for column,(picture,title) in enumerate([(rgb,'原图'),(plan,'归档绘画轨迹'),(overlay,'候选补线位置')]):
            left,top=column*(tile+12)+5,row*(tile+41)+6
            pen.text((left,top),f'{name} · {title}',font=font,fill='#24352e')
            canvas.paste(Image.fromarray(picture).resize((tile,tile),Image.Resampling.LANCZOS),(left,top+25))
    pen.text((6,canvas.height-29),'绿色为现有提取，橙色扩大显示补点位置。中栏来自已归档的绘画轨迹。',font=font,fill='#24352e')
    canvas.save(ROOT/'whole_comparison.png')


def module_review(source, plan):
    targets = {2:'中央主山石',8:'松树与树石交界'}
    rgb = np.asarray(source)
    figure = Image.new('RGB',(1062,606),'#f5f1e7')
    pen = ImageDraw.Draw(figure)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    palette = dict(main_contour=(35,67,57),trunk=(35,67,57),branch=(82,126,113),
                   foliage=(164,107,37),cun_candidate=(164,107,37),detail=(137,126,107))
    reviews = []
    for row,(identifier,title) in enumerate(targets.items()):
        curves = [(index,c) for index,c in enumerate(plan['curves']) if c['module']==identifier]
        actions = [a for a in plan['actions'] if a['module']==identifier]
        positions = np.concatenate([np.asarray(c['points']) for _,c in curves],axis=0)
        low = np.maximum(np.floor(positions.min(axis=0)).astype(int)-9,0)
        high = np.minimum(np.ceil(positions.max(axis=0)).astype(int)+10,rgb.shape[1::-1])
        x0,y0 = low
        x1,y1 = high
        region = rgb[y0:y1,x0:x1].copy()
        summary = dict(name=title,module=identifier,count=len(curves),phases={},curve_ids=[i for i,_ in curves],
                       uncertain_boundary=[i for i,c in curves if c['object_boundary_uncertain']],
                       uncertain_role=[i for i,c in curves if c['role_uncertain']])
        for action in actions:
            phase = str(action['phase'])
            summary['phases'][phase] = summary['phases'].get(phase,0)+1
        reviews.append(summary)
        for col,(key,label) in enumerate([('source','原图'),('major','先画结构'),('all','其后叠加细部')]):
            picture = region.copy() if key=='source' else np.full_like(region,253)
            if key!='source':
                for _,curve in curves:
                    role = curve['stroke_role']
                    if key=='major' and role not in {'main_contour','trunk'}:
                        continue
                    loc = np.asarray(np.rint(np.asarray(curve['points'])-low),dtype=np.int32)
                    if len(loc)>1:
                        cv2.polylines(picture,[loc],False,palette.get(role,(92,116,106)),1,cv2.LINE_AA)
            tile = Image.fromarray(picture)
            scale = min(338/tile.width,245/tile.height)
            tile = tile.resize((round(tile.width*scale),round(tile.height*scale)),Image.Resampling.LANCZOS)
            left,top = col*354+7,row*280+6
            pen.text((left,top),f'{title} · {label}',font=font,fill='#24352e')
            figure.paste(tile,(left,top+25))
    pen.text((7,figure.height-30),'棕色为皴笔或枝叶候选。笔画角色取自现有规划，仍需按原图复核。',font=font,fill='#24352e')
    figure.save(ROOT/'module_review.png')
    segment.write_json(ROOT/'module_review.json',reviews)


def main():
    ROOT.mkdir(exist_ok=True)
    frozen = support_lock()
    options = segment.lock_options()
    locks = segment.read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:segment.sha(locks[key])==locks[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    sources = {name:ARCHIVE/name/'source.png' for name in WORKS}
    paths = {name:ARCHIVE/name/'offline_plan.json' for name in WORKS}
    paths['G03_01'] = Path(locks['working489'])
    images = {name:dict(path=str(path),sha256=segment.sha(path)) for name,path in sources.items()}
    lock = ROOT/'protocol.json'
    segment.lock_file(lock,dict(algorithms=frozen,images=images,
                 plans={name:segment.sha(path) for name,path in paths.items()},
                 scope='Offline whole-image routing decision. No plan updates.'))
    output,frames = {},[]
    for name,(rgb,roles) in predict_extra(dict(images=images)).items():
        baseline,direction,previous = segment.old_stages(rgb,roles,options)
        _,variants = run_variants(Image.fromarray(rgb),baseline,direction,previous)
        current,candidate = variants['color'],variants['span5']
        plan = segment.read(paths[name])
        coverage,curve_masks = raster_curves(plan,baseline.shape)
        reports = {key:assess(current,variants[key],coverage,curve_masks) for key in ('unrestricted','span3','span5')}
        output[name] = dict(plan=order_metrics(plan),effects=reports,
                            base_pixels=int((baseline>0).sum()),color_pixels=int((current>0).sum()))
        frames.append((name,rgb,coverage,current,(current==0)&(candidate>0)))
        np.savez_compressed(ROOT/f'{name}.npz',role_baseline=baseline,plan_coverage=coverage,**variants)
        print(name,{k:{field:r[field] for field in ('added','near_plan','touched_curves')}
                   for k,r in reports.items()},output[name]['plan'],flush=True)
    figures(frames)
    with Image.open(sources['G03_01']) as source:
        module_review(source.convert('RGB'),segment.read(paths['G03_01']))
    segment.write_json(ROOT/'results.json',dict(images=output,baseline_hashes=hashes,protocol_sha256=segment.sha(lock),
                       independent_acceptance=False,adopted=False,robot_connected=False))


if __name__=='__main__':
    main()
