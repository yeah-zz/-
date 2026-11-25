"""Deterministic blue mountain-face shading; offline, no hardware imports.

Polygons are drawn in the ORIGINAL prepared design, never from phone geometry.
The preview renders exactly the thin-brush paths; no fictitious broad wash.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import color_pass
import job
import vision
from config import ArmConfig, CanvasConfig
from geometry import CanvasMap

GAP = 8.0  # nominal center-to-ink; registration still requires onsite checking
WIDTH = 1.0
EDGE = 25.0
# Mountain faces and their downhill directions, in prepared source pixels.
FACES = [
    ('left-back', [(260,345),(305,420),(425,500),(310,520),(225,465)], (.35,1)),
    ('left-middle', [(183,490),(250,555),(330,595),(230,615),(150,570)], (.55,1)),
    ('main-back', [(1065,390),(1120,465),(1225,550),(1120,600),(1000,555)], (.4,1)),
    ('right-back', [(1345,480),(1420,550),(1490,580),(1385,605),(1285,550)], (.6,1)),
    ('center-middle', [(590,535),(660,595),(780,650),(635,677),(540,608)], (.5,1)),
    ('left-low', [(330,660),(415,717),(460,745),(330,776),(268,738)], (.5,1)),
    ('main-front', [(572,665),(650,748),(790,820),(655,848),(540,800),(505,740)], (.42,1)),
    ('right-middle', [(1510,650),(1560,725),(1520,790),(1410,805),(1390,750)], (-.35,1)),
    ('left-front', [(280,819),(365,893),(540,970),(370,1002),(233,938),(210,881)], (.65,1)),
    ('right-front', [(1310,865),(1425,930),(1530,975),(1360,1010),(1240,967),(1180,934)], (.65,1)),
]

def inputs(job_path, image_path):
    _, meta = job.load_job(str(job_path))
    if (meta.get('profile') != 'mountain' or meta.get('ink_policy') != 'balanced'
            or meta['canvas_corners'] != [list(p) for p in CanvasConfig.ROBOT_CORNERS]
            or meta['paper_z'] != ArmConfig.PAPER_Z):
        raise ValueError('Ink job/calibration mismatch')
    work, info = vision.prepare_image(vision.load_image(str(image_path)), work_size=1800)
    if list(info['prepared_size']) != [1659,1139] or meta['image_size'] != [1659,1139]:
        raise ValueError('Authored faces require original 1659 x 1139 design')
    mapper = CanvasMap(meta['image_size'], info['image_corners'])
    # Conservative metric scale: smallest singular value of affine map.
    # Current calibration has small perspective; use a sampled lower bound.
    yy,xx=np.mgrid[0:1139:40,0:1659:40]
    p=np.c_[xx.ravel(), yy.ravel()].astype(float)
    a=mapper.to_robot(p); dx=mapper.to_robot(p+[1,0])-a; dy=mapper.to_robot(p+[0,1])-a
    scale=float(np.linalg.svd(np.stack([dx,dy],axis=2),compute_uv=False).min())*.99
    ink=(cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)<215).astype(np.uint8)*255
    for s in meta['ink_execution']['base_strokes']:
        cv2.polylines(ink,[np.rint(s['points']).astype(np.int32)],False,255,5)
    # Include actual executed ink centerlines, with conservative 1 mm footprint.
    raw=json.loads(Path(job_path).read_text(encoding='utf8'))
    inv=np.linalg.inv(mapper.matrix)
    prev=None
    for wp in raw['waypoints']:
        if wp.get('pen_up',True): prev=None; continue
        pix=cv2.perspectiveTransform(np.array([[[wp['x'],wp['y']]]],np.float32),inv)[0,0]
        if prev is not None: cv2.line(ink,tuple(np.rint(prev).astype(int)),tuple(np.rint(pix).astype(int)),255,max(3,int(np.ceil(1/scale))))
        prev=pix
    clearance=cv2.distanceTransform(255-ink,cv2.DIST_L2,5)*scale
    return meta,work,mapper,clearance,scale

def dense_points(points, step=.8):
    points=np.asarray(points,float)
    out=[]
    for a,b in zip(points,points[1:]):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/step)))
        out.extend(a+(b-a)*np.arange(n)[:,None]/n)
    return np.asarray(out+[points[-1]])

def clearance_for(points, mapper, clearance):
    dense=dense_points(points)
    ij=np.rint(dense).astype(int)
    h,w=clearance.shape
    if (ij<0).any() or (ij[:,0]>=w).any() or (ij[:,1]>=h).any():
        raise ValueError('Path outside source')
    nominal=float(clearance[ij[:,1],ij[:,0]].min())
    xy=mapper.to_robot(dense)
    poly=np.asarray(CanvasConfig.ROBOT_CORNERS,np.float32)
    edge=min(cv2.pointPolygonTest(poly,tuple(map(float,p)),True) for p in xy)
    return nominal,edge

def build(job_path='artifacts/mountain/mountain_v6_job.json', image_path='artifacts/reference/desired_mountain_reference.png'):
    meta,work,mapper,clearance,scale=inputs(job_path,image_path)
    h,w=clearance.shape
    # Deliberately broad visual exclusion around the TWO accidental stains.
    # This is an exclusion only, not a photo-derived robot registration.
    clearance[240:530,440:820]=0
    # Existing robot trial and surrounding region also remain untouched.
    clearance[350:510,1430:1659]=0
    strokes=[]
    for name,polygon,direction in FACES:
        region=np.zeros((h,w),np.uint8)
        cv2.fillPoly(region,[np.asarray(polygon,np.int32)],255)
        direction=np.asarray(direction,float); direction/=np.linalg.norm(direction)
        normal=np.array([direction[1],-direction[0]])
        poly=np.asarray(polygon,float)
        for u in np.arange((poly@normal).min(),(poly@normal).max(),1.35/scale):
            t=np.arange((poly@direction).min(),(poly@direction).max(),.7)
            p=u*normal+t[:,None]*direction
            ij=np.rint(p).astype(int)
            valid=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
            safe=np.zeros(len(p),bool)
            q=ij[valid]
            safe[valid]=(region[q[:,1],q[:,0]]>0)&(clearance[q[:,1],q[:,0]]>=GAP+.3)
            for a,b in color_pass._runs(safe):
                if b-a<2: continue
                endpoints=np.array([p[a],p[b-1]])
                xy=mapper.to_robot(endpoints)
                length=float(np.linalg.norm(xy[1]-xy[0]))
                # A single straight MovL eliminates all intermediate paint dwells.
                if not 5<=length<=32: continue
                gap,edge=clearance_for(endpoints,mapper,clearance)
                if edge<EDGE: continue
                strokes.append(dict(id='%s-%03d'%(name,len(strokes)),face=name,color='blue',target_tone='light',
                    points=endpoints.tolist(),length_mm=length,brush_width_mm_estimate=WIDTH,
                    min_ink_clearance_mm=gap,canvas_edge_clearance_mm=edge))
    if len(strokes)<40: raise ValueError('Too little area for full composition')
    plan=dict(format='blue-wash-v2',ready_for_execution=False,
        reason='Needs supervised hover registration and manual loading each batch',
        ink_job=str(Path(job_path).resolve()),ink_job_sha256=color_pass._digest(job_path),
        ink_image=str(Path(image_path).resolve()),ink_image_sha256=color_pass._digest(image_path),
        image_size=meta['image_size'],image_corners=mapper.image_corners,
        canvas_corners=meta['canvas_corners'],paper_z_snapshot=meta['paper_z'],
        rpy_snapshot=list(ArmConfig.RPY),user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,
        mm_per_px=meta['mm_per_px'],ink_gap_mm=GAP,brush_width_mm_estimate=WIDTH,
        excluded_source_rectangles=[[440,240,820,530],[1430,350,1659,510]],strokes=strokes)
    return plan,work

if __name__=='__main__':
    if Path('artifacts/blue/mountain_v6_blue_wash_plan.json').exists() or Path('artifacts/blue/mountain_v6_blue_wash_preview.png').exists():
        raise SystemExit('Refusing to overwrite an existing reviewed plan/preview; create a separately reviewed revision')
    plan,work=build()
    color_pass.save_plan('artifacts/blue/mountain_v6_blue_wash_plan.json',plan)
    color_pass.render_preview(work,plan['strokes'],plan['mm_per_px'],'artifacts/blue/mountain_v6_blue_wash_preview.png')
    print(len(plan['strokes']),'strokes;',round(sum(s['length_mm'] for s in plan['strokes'])),'mm')
