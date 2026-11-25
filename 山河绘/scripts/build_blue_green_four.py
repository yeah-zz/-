"""Offline exhibition set: generated references -> calibrated blue/green brush paths."""
from pathlib import Path
import sys
import json
from collections import Counter
import copy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import imgio, vision, ink_control, shuimo, job, watercolor
from original_d_adapter import build_strokes, TraceConfig
from geometry import CanvasMap, check_waypoints
from config import ArmConfig, CanvasConfig, PALETTE
from color_profile import ColorProfile
from run_mountain_v4_robot import safe_transfers
from run_color_experiment import module_lines
from run_json_painting import make_bundle, load_artwork, manifest_for
from painting_session import save
from scripts import build_filled_landscape as filling

OUT = ROOT / '输出' / '蓝绿山水四幅_20261001'
INSET = .80  # Plan all marks together inside the same centered artwork rectangle.


def contour_strokes(source, work, info, scale):
    """Trace visible edges, not the medial skeleton of broad watercolor areas."""
    small = cv2.resize(source, (820, round(source.shape[0]*820/source.shape[1])), interpolation=cv2.INTER_AREA)
    smooth = cv2.bilateralFilter(small, 7, 38, 4)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 45, 95, L2gradient=True)
    pixels = set(zip(*np.nonzero(edge)[::-1]))
    neighbors = {}
    for x,y in pixels:
        links=[]
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            q=(x+dx,y+dy)
            if q not in pixels:continue
            if dx and dy and ((x+dx,y) in pixels or (x,y+dy) in pixels):continue
            links.append(q)
        neighbors[(x,y)] = links
    seen=set(); paths=[]
    starts=sorted(pixels,key=lambda p:(len(neighbors[p])==2,p[1],p[0]))
    for start in starts:
        for first in neighbors[start]:
            if (start,first) in seen:continue
            path=[start];a,b=start,first
            while True:
                seen.add((a,b));seen.add((b,a));path.append(b)
                if len(neighbors[b])!=2:break
                options=[q for q in neighbors[b] if (b,q) not in seen]
                if not options:break
                a,b=b,options[0]
            paths.append(path)
    top,bottom,left,right=info['pad']
    ch=work.shape[0]-top-bottom;cw=work.shape[1]-left-right
    result=[]
    for path in paths:
        pts=np.asarray(path,np.float32)
        length=np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()*cw/small.shape[1]*scale
        if length<1.2:continue
        reduced=cv2.approxPolyDP(pts.reshape(-1,1,2),.65,False).reshape(-1,2)
        # Resample for smooth, explicit pressure ramps.
        dense=[]
        for a,b in zip(reduced,reduced[1:]):
            n=max(1,int(np.ceil(np.linalg.norm(b-a)/1.5)))
            dense.extend(np.linspace(a,b,n,endpoint=False))
        dense.append(reduced[-1]);pts=np.asarray(dense)
        xy=pts/[small.shape[1],small.shape[0]]*[cw,ch]+[left,top]
        t=np.linspace(0,1,len(xy))
        q=np.rint(pts).astype(int)
        darkness=1-gray[q[:,1],q[:,0]]/255.
        peak=.10+.16*np.clip(darkness,0,1)
        ramp=np.minimum(np.clip(t/.10,0,1),np.clip((1-t)/.10,0,1))
        pressure=.06+(peak-.06)*ramp
        stroke=dict(points=xy.tolist(),press=pressure.tolist(),speed=[30.]*len(xy),
                    pass_=1,pal_index=shuimo.ink_palette_index(),layer=0 if length>20 else 1,
                    name='reference contour',role='visible_contour',design_origin='edge-trace-820px',
                    tail_mm=0.,lift_mm=4.)
        result.append(ink_control.refresh_stats(stroke,scale))
    return result,dict(algorithm='visible-edge-820px',strokes=len(result))


def build(folder):
    name = folder.name
    reference = folder / '参考原图.png'
    source = vision.load_image(str(reference))
    work, info = vision.prepare_image(source, work_size=1500)
    if info.get('used_corners'):
        raise ValueError('This generator requires the existing full-canvas mapping')
    mapper = CanvasMap(info['prepared_size'])
    scale = mapper.mm_per_px
    local_scale = scale * INSET
    center = (np.array(info['prepared_size'])-1)/2
    print('TRACE ' + name, flush=True)
    # Coarser tracing preserves the large composition without reproducing every
    # generated watercolor fleck. Do not bridge unrelated texture fragments.
    base, stats = contour_strokes(source, work, info, local_scale)
    print('INK %s %d' % (name,len(base)), flush=True)
    ink_mask = np.zeros(work.shape[:2], np.uint8)
    for stroke in base:
        cv2.polylines(ink_mask, [np.rint(stroke['points']).astype(np.int32)], False, 1,
                      max(1, int(np.ceil((stroke['width_mm'] + .5)/local_scale))))
        stroke['points'] = ((np.array(stroke['points'])-center)*INSET+center).tolist()
        ink_control.refresh_stats(stroke, scale)
    policy = ink_control.InkPolicy()
    base = ink_control.order_for_ink(ink_control.split_long_strokes(base, scale, policy), scale, policy)
    for index, stroke in enumerate(base):
        stroke['id'] = '%s-ink-%04d' % (name, index)
    steps, ink_stats = ink_control.build_plan(base, shuimo.ink_palette_index(), scale, policy)
    conditioned = [step['stroke'] for step in steps if step['kind'] == 'stroke']
    points, _ = job.steps_to_waypoints(steps, mapper)
    points = safe_transfers(points)
    assert check_waypoints(points)[0]

    # Reuse the current landscape's color motion settings; do not alter calibration.
    donor = next((ROOT/'输出'/'层峦松水_双蓝长笔版_20260930_172520').glob('*_计划.json'))
    donor_plan = json.loads(donor.read_text(encoding='utf-8'))
    profile = ColorProfile(**donor_plan['modules']['color']['motion_candidate']).to_dict()
    width = shuimo.press_to_width_mm(profile['press_mm'])
    b,g,r = cv2.split(work.astype(np.float32))
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    green = (g-r>5) & (g-b>1) & (gray<235) & (gray>45)
    blue = (b-r>6) & (gray<246) & (gray>85) & ~green
    colors = []
    report = {}
    filling.MAX_STROKE_MM = 55.
    filling.BRIDGE_MAX_MM = 1.5
    filling.BRIDGE_MIN_INSIDE = 1.
    # Paint restrained bands, preserve mist/water and separate the two pigments.
    for color, raw, angle in [('blue', blue, 78), ('green', green, 12)]:
        radius = max(1, round(.35/local_scale))
        mask = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2))
        erosion = max(1, int(np.ceil((width/2+.15)/local_scale)))
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erosion+1,)*2))
        mask[ink_mask>0] = 0
        mask[gray<55] = 0
        pitch = max(2, round(max(width*1.3, 1.15)/local_scale))
        strokes, coverage = filling.fill_paths_rot(mask, local_scale, color, pitch, width,
                                                   .48 if color=='blue' else .64, angle)
        for index, stroke in enumerate(strokes):
            stroke['id'] = '%s-%s-%04d' % (name, color, index)
            stroke['points'] = ((np.array(stroke['points'])-center)*INSET+center).tolist()
        colors.extend(strokes)
        report[color] = dict(strokes=len(strokes), estimated_mask_coverage=coverage)

    geometry = dict(image_size=list(info['prepared_size']), mm_per_px=scale,
                    image_corners=mapper.image_corners)
    module = dict(format='sparse-color-design-v1', **geometry, strokes=colors,
                  regions=[], stroke_count=len(colors), palette_bgr=watercolor.PALETTE,
                  motion_candidate=profile, reload_strokes_per_load=8)
    for color in ('blue', 'green'):
        lines, skipped = module_lines(module, color, profile)
        if skipped or not lines:
            raise ValueError('%s %s: skipped=%s count=%d' % (name,color,skipped,len(lines)))
    plan = dict(format='painting-workflow-design-v1', ready_for_execution=False,
                subject='reference', title=name, reference_image=str(reference.resolve()),
                **geometry, stages=['outline','color'],
                modules=dict(outline=dict(format='outline-design-v1', strokes=conditioned, **geometry), color=module),
                sequence=[dict(kind='run_module',module='outline'),
                          dict(kind='manual_wash_wipe_and_load',next_stage='color'),
                          dict(kind='run_module',module='color')],
                artwork_scale=INSET, fill_report=report,
                note='Blue and green pigments only after black ink. Preview uses exported paths, not source pixels; optical widths and wet diffusion are unverified.')
    meta = dict(mode='shuimo-framework', profile='mountain', ink_policy='balanced',
                image=str(reference.resolve()), image_size=geometry['image_size'],
                canvas_corners=[list(p) for p in CanvasConfig.ROBOT_CORNERS], paper_z=ArmConfig.PAPER_Z,
                strokes=len(base), quality_stats=stats,
                ink_execution=dict(base_strokes=base, policy=ink_stats['ink_policy'], mm_per_px=scale,
                    calibration=dict(rpy=list(ArmConfig.RPY), user=ArmConfig.USER_INDEX,
                        tool=ArmConfig.TOOL_INDEX, speed_factor=ArmConfig.SPEED_FACTOR, palette=PALETTE)))
    ink_job = dict(format='arm-painting job', version=1, meta=meta, waypoints=points)
    save(folder/(name+'_计划.json'), plan)
    save(folder/(name+'_黑线任务.json'), ink_job)
    bundle_path = folder/(name+'_完整绘画.json')
    save(bundle_path, make_bundle(plan, ink_job))
    ink, checked_plan, checked_strokes, checked_meta = load_artwork(bundle_path)
    manifest_for(checked_strokes, checked_plan, checked_meta)
    preview, paths = watercolor.render_image(work, module, include_ink=False)
    ink_view = np.full_like(work, 255)
    for stroke in conditioned:
        pts = np.rint(stroke['points']).astype(int)
        for i, (a, z) in enumerate(zip(pts, pts[1:])):
            width_px = max(1, round((shuimo.press_to_width_mm(stroke['press'][i])+
                                    shuimo.press_to_width_mm(stroke['press'][i+1]))/(2*scale)))
            cv2.line(preview, tuple(a), tuple(z), (38,38,38), width_px, cv2.LINE_AA)
            cv2.line(ink_view, tuple(a), tuple(z), (38,38,38), width_px, cv2.LINE_AA)
            cv2.line(paths, tuple(a), tuple(z), (38,38,38), 1, cv2.LINE_AA)
    for suffix, raster in [('实际路径预览',preview),('黑线预览',ink_view),('中心路径',paths)]:
        imgio.imwrite(str(folder/(name+'_'+suffix+'.png')),raster)
    result = dict(name=name, black=len(base), blue=report['blue']['strokes'], green=report['green']['strokes'],
                  bundle=str(bundle_path), validation='passed; no robot connection',
                  canvas_mm=list(mapper.canvas_size_mm()), paper_z=ArmConfig.PAPER_Z,
                  color_press_mm=profile['press_mm'])
    save(folder/'离线检查.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__ == '__main__':
    from unittest.mock import patch
    with patch('robot.E6Robot.__init__', side_effect=AssertionError('Offline generation only')):
        results = [build(folder) for folder in sorted(OUT.iterdir()) if folder.is_dir() and (folder/'参考原图.png').exists()]
    save(OUT/'四幅清单.json',results)
