# -*- coding: utf-8 -*-
"""Small junction-loop hypotheses, supported by source pixels, never a learned truth."""
import numpy as np,cv2,networkx as nx
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from module_stroke_core_20261104 import samples,length
from source_stroke_width_20261104 import fields,sample

def inspect_cycles(rgb,paths):
    h,w=rgb.shape[:2];mask=np.zeros((h,w),np.uint8)
    for p in paths:cv2.polylines(mask,[np.rint(p).astype(np.int32)],False,1,1)
    sk=skeletonize(mask>0);ys,xs=np.nonzero(sk);graph=nx.Graph()
    for y,x in zip(ys,xs):
        u=(int(x),int(y));graph.add_node(u)
        for dx,dy in [(1,0),(0,1),(1,1),(-1,1)]:
            xx,yy=x+dx,y+dy
            if 0<=xx<w and 0<=yy<h and sk[yy,xx]:
                # Avoid artificial triangles made solely by diagonal shortcuts.
                if dx and dy and (sk[y,xx] or sk[yy,x]):continue
                graph.add_edge(u,(int(xx),int(yy)))
    gray,bg=fields(rgb);candidates=[]
    for vertices in nx.cycle_basis(graph):
        p=np.asarray(vertices,np.float32);area=abs(cv2.contourArea(p));extent=np.ptp(p,axis=0)
        if area>8 or max(extent)>5 or len(p)>16:continue
        lo=p.min(0);hi=p.max(0);center=p.mean(0)
        # Dense subpixel probing evaluates small holes without inventing pixels.
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+.01,.25),np.arange(lo[1],hi[1]+.01,.25))
        q=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
        inside=np.array([cv2.pointPolygonTest(p,tuple(v),True)>.28 for v in q])
        border=samples(np.r_[p,p[:1]],.2);boundary=float(np.median(sample(gray,border)))
        evidence=float(np.median(sample(bg,border)-sample(gray,border)))
        external=sum(any(v not in vertices for v in graph.neighbors(u)) for u in vertices)
        if inside.any():
            interior=sample(gray,q[inside]);hole_delta=float(np.quantile(interior,.8)-boundary)
        else:hole_delta=None
        if hole_delta is None:reason='insufficient interior resolution';take=False
        elif hole_delta>.025:reason='visible bright hole: preserve';take=False
        elif external<1:reason='isolated closed motif: preserve';take=False
        elif evidence<.012 and boundary>.38:reason='insufficient source ink evidence';take=False
        else:reason='small branch junction, source interior has no bright separation';take=True
        candidates.append(dict(vertices=p,center=center,area_px2=area,hole_brightness_delta=hole_delta,
            boundary_intensity=boundary,source_contrast=evidence,external_branch_vertices=external,
            eligible=take,reason=reason,source_geometry_verified_not_artist_identity=True))
    accepted=[]
    for r in sorted(candidates,key=lambda r:(r['area_px2'],float(r['center'][1]),float(r['center'][0]))):
        if not r['eligible']:continue
        p=r['vertices'];lo=p.min(0)-.6;hi=p.max(0)+.6
        if any(np.all(hi>=a['vertices'].min(0)-.6) and np.all(lo<=a['vertices'].max(0)+.6) for a in accepted):
            r['eligible']=False;r['reason']='overlapping junction candidate: defer';continue
        accepted.append(r)
    return candidates,accepted

def apply_cycles(rgb,rows):
    paths=[np.asarray(r['points'],float) for r in rows];candidates,accepted=inspect_cycles(rgb,paths)
    work=[{**r,'points':samples(p,min(.3,max(.01,length(p)/8))) if np.linalg.norm(p[0]-p[-1])<.1 else samples(p,.3)} for r,p in zip(rows,paths)]
    applied=[]
    for k,cycle in enumerate(accepted):
        polygon=cycle['vertices'];center=cycle['center'];affected=[]
        for i,r in enumerate(work):
            q=r['points']
            near=((q>=polygon.min(0)-.6)&(q<=polygon.max(0)+.6)).all(1)
            hit=np.zeros(len(q),bool)
            for j in np.flatnonzero(near):hit[j]=cv2.pointPolygonTest(polygon,tuple(q[j].astype(float)),True)>=-.55
            if hit.any():affected.append((i,hit))
        # A loop may not be turned into an isolated physical ink dot. Each
        # affected object's zero-length loop must have an adjacent surviving arm.
        surviving={work[i]['module'] for i,hit in affected if (~hit).any()}
        if any(work[i]['module'] not in surviving for i,hit in affected if hit.all()):
            cycle['eligible']=False;cycle['reason']='no same-object surviving arm; defer';continue
        for i,hit in affected:work[i]['points'][hit]=center
        applied.append(dict(eligible_cycle_index=k,center=center,
            source_ids=sorted({v for i,_ in affected for v in work[i]['source_curve_ids']}),
            artist_stroke_identity_verified=False))
    active=[r for r in work if length(r['points'])>1e-5];collapsed=[r for r in work if length(r['points'])<=1e-5]
    for r in collapsed:
        same=[j for j,a in enumerate(active) if a['module']==r['module']]
        if not same:raise AssertionError('No provenance recipient for junction')
        j=min(same,key=lambda j:float(np.linalg.norm(active[j]['points']-r['points'][0],axis=1).min()))
        active[j]['source_curve_ids']=active[j]['source_curve_ids']+r['source_curve_ids']
    for r in active:
        p=r['points'];keep=np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-7];p=p[keep]
        q=cv2.approxPolyDP(p.astype(np.float32),.12,False).reshape(-1,2).astype(float)
        q[0]=p[0];q[-1]=p[-1]
        r['points']=q if len(q)>=2 and length(q)>1e-7 else p
    assert sorted(v for r in active for v in r['source_curve_ids'])==sorted(v for r in rows for v in r['source_curve_ids'])
    return active,dict(small_cycle_candidates=candidates,eligible_cycles=len(accepted),applied_junctions=applied,
        collapsed_zero_length_curves_absorbed=len(collapsed),input_curves=len(rows),output_curves=len(active),
        raw_geometry_archived=True,no_bright_hole_knowingly_collapsed=True,artist_identity_accuracy=None,ready_for_robot=False)
