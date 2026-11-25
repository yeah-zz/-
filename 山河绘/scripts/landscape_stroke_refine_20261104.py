# -*- coding: utf-8 -*-
"""Image-supported curve containment and continuation. Offline image pixels only.

This does not identify artistic cun styles or learn robot/ink dynamics. It
keeps a provenance partition and exposes ambiguous continuation decisions.
"""
import math
from collections import Counter
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter,gaussian_filter1d
from module_stroke_core_20261104 import samples as base_samples,length,tangent
from source_stroke_width_20261104 import fields,sample,width_profile

CONFIG=dict(duplicate_distance_px=.70,duplicate_max_distance_px=1.25,
            duplicate_parallel_cos=.94,duplicate_fraction=.96,
            gap_px=2.5,turn_degrees=45.,corner_degrees=95.,
            competition_margin=.14,minimum_bridge_support=.94,
            normal_recentering_limit_px=.75,simplify_deviation_px=.18)

def unit(v):return v/max(float(np.linalg.norm(v)),1e-8)

def samples(p,step=1.):
    p=np.asarray(p,float);arc=length(p)
    if np.linalg.norm(p[0]-p[-1])<.1 and arc>1e-8:step=min(step,arc/8)
    return base_samples(p,step)

def curve_tangents(q):
    t=np.gradient(q,axis=0)
    return t/np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-8)

def same_band(gray,a,b):
    """Do not absorb adjacent strokes with a visible bright valley between them."""
    a=np.asarray(a);b=np.asarray(b)
    values=np.stack([sample(gray,a*(1-t)+b*t) for t in np.linspace(0,1,9)],axis=1)
    ends=np.maximum(values[:,0],values[:,-1])
    # A local bright separation is positive evidence for two distinct bands.
    valley=values[:,1:-1].max(1)-ends
    return valley<=.025

def deduplicate(rgb,paths,owners=None):
    gray,_=fields(rgb)
    arrays=[np.asarray(p,float) for p in paths]
    owners=[0]*len(arrays) if owners is None else list(owners)
    q=[samples(p,.35) for p in arrays];t=[curve_tangents(p) for p in q]
    trees=[cKDTree(p) for p in q];lens=np.array([length(p) for p in arrays])
    kept=[];mapping={};events=[];rejections=Counter()
    for i in sorted(range(len(arrays)),key=lambda k:(-lens[k],k)):
        found=None
        for j in kept:
            if owners[i]!=owners[j]:continue
            # Never collapse a closed motif onto an open line near its edge.
            closed_i=np.linalg.norm(arrays[i][0]-arrays[i][-1])<.1
            closed_j=np.linalg.norm(arrays[j][0]-arrays[j][-1])<.1
            if closed_i!=closed_j:continue
            d,ix=trees[j].query(q[i]);close=d<=CONFIG['duplicate_distance_px']
            if close.mean()<CONFIG['duplicate_fraction'] or d.max()>CONFIG['duplicate_max_distance_px']:continue
            aligned=abs((t[i]*t[j][ix]).sum(1))>=CONFIG['duplicate_parallel_cos']
            if aligned.mean()<.92:rejections['different_tangent']+=1;continue
            if same_band(gray,q[i],q[j][ix]).mean()<.98:rejections['visible_bright_gap']+=1;continue
            # Matched points must progress monotonically along the same curve.
            # Prevent an unrelated local zigzag being absorbed merely by proximity.
            delta=np.diff(ix)
            forward=(delta>=-2).mean();backward=(delta<=2).mean()
            if max(forward,backward)<.97:rejections['nonmonotone_match']+=1;continue
            found=j
            events.append(dict(absorbed=i,representative=j,max_deviation_px=float(d.max()),
                matched_fraction=float(close.mean()),reason='contained, tangent-aligned, no bright separation',
                artist_same_stroke_verified=False))
            break
        if found is None:kept.append(i);mapping[i]=[i]
        else:mapping[found].append(i)
    result=[dict(points=arrays[i],source_curve_ids=mapping[i],module=owners[i]) for i in kept]
    assert sorted(k for row in result for k in row['source_curve_ids'])==list(range(len(arrays)))
    return result,dict(events=events,rejections=dict(rejections),input=len(arrays),output=len(result),
        raster_identity_not_claimed=True,maximum_absorbed_curve_distance_px=CONFIG['duplicate_max_distance_px'])

def bridge_path(a,b,ta,tb):
    gap=float(np.linalg.norm(b-a))
    n=max(5,int(gap/.12)+1);u=np.linspace(0,1,n)[:,None]
    # ta/tb point inwards into their existing curves. Hermite connector continues
    # out of curve a, then into curve b; endpoints remain precisely on old paths.
    m0=-ta*gap;m1=tb*gap
    return (2*u**3-3*u**2+1)*a+(u**3-2*u**2+u)*m0+(-2*u**3+3*u**2)*b+(u**3-u**2)*m1

def join_curves(rgb,rows):
    gray,bg=fields(rgb);paths=[np.asarray(r['points'],float) for r in rows]
    ends=[]
    for i,p in enumerate(paths):
        if length(p)<1. or np.linalg.norm(p[0]-p[-1])<.1:continue
        for side in [0,1]:
            q=samples(p if side==0 else p[::-1],.25)
            at=min(len(q)-1,max(1,int(min(2.,length(q)*.35)/.25)))
            ends.append(dict(i=i,side=side,p=q[0],t=unit(q[at]-q[0])))
    pairs=sorted(cKDTree([e['p'] for e in ends]).query_pairs(CONFIG['gap_px'])) if ends else []
    degree=Counter()
    for a,b in pairs:
        if ends[a]['i']!=ends[b]['i'] and np.linalg.norm(ends[a]['p']-ends[b]['p'])<1.:
            degree[a]+=1;degree[b]+=1
    candidates=[];rejected=Counter()
    for a,b in pairs:
        ea,eb=ends[a],ends[b];i,j=ea['i'],eb['i']
        if i==j:continue
        if rows[i]['module']!=rows[j]['module']:rejected['different_owner']+=1;continue
        v=eb['p']-ea['p'];gap=float(np.linalg.norm(v));facing=float(ea['t']@(-eb['t']))
        corner=gap<.9 and degree[a]==degree[b]==1
        angle=CONFIG['corner_degrees'] if corner else CONFIG['turn_degrees']
        if facing<math.cos(math.radians(angle)):rejected['turn_or_parallel']+=1;continue
        if gap>.75 and min(float(-ea['t']@unit(v)),float(eb['t']@unit(v)))<.6:
            rejected['sideways_bridge']+=1;continue
        bridge=bridge_path(ea['p'],eb['p'],ea['t'],eb['t'])
        g=sample(gray,bridge);contrast=sample(bg,bridge)-g
        support=float(((contrast>.010)|(g<.30)).mean())
        if gap>.15 and support<CONFIG['minimum_bridge_support']:
            rejected['unsupported_gap']+=1;continue
        score=(facing+1)/2-.12*gap+.15*support
        candidates.append(dict(a=a,b=b,i=i,j=j,gap_px=gap,score=score,
            source_support=support,corner=corner,bridge=bridge))
    ranks={}
    for k,c in enumerate(candidates):
        for e in [c['a'],c['b']]:ranks.setdefault(e,[]).append((c['score'],k))
    for e in ranks:ranks[e].sort(reverse=True)
    parent=list(range(len(rows)));links={};accepted=[];pending=[]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for k,c in sorted(enumerate(candidates),key=lambda x:-x[1]['score']):
        a,b=c['a'],c['b']
        if a in links or b in links:continue
        if any(ranks[e][0][1]!=k or len(ranks[e])>1 and ranks[e][0][0]-ranks[e][1][0]<CONFIG['competition_margin'] for e in [a,b]):
            pending.append({**c,'reason':'ambiguous competing continuations'});continue
        x,y=root(c['i']),root(c['j'])
        if x==y:rejected['new_loop']+=1;continue
        parent[y]=x;links[a]=(b,c['bridge']);links[b]=(a,c['bridge'][::-1]);accepted.append(c)
    byend={(ends[a]['i'],ends[a]['side']):((ends[b]['i'],ends[b]['side']),bridge) for a,(b,bridge) in links.items()}
    visited=set();result=[]
    def walk(start):
        cur=start;parts=[];source=[];members=[]
        while cur[0] not in visited:
            i,side=cur;visited.add(i);members.append(i)
            parts.append(paths[i] if side==0 else paths[i][::-1]);source+=rows[i]['source_curve_ids']
            nex=byend.get((i,1-side))
            if nex is None:break
            cur,bridge=nex;parts.append(bridge)
        result.append(dict(points=np.concatenate(parts),source_curve_ids=source,module=rows[start[0]]['module'],
            constituent_curves=members,artist_stroke_identity_verified=False))
    for i in range(len(rows)):
        for side in [0,1]:
            if i not in visited and (i,side) not in byend:walk((i,side))
    for i in range(len(rows)):
        if i not in visited:walk((i,0))
    assert sorted(k for r in result for k in r['source_curve_ids'])==sorted(k for r in rows for k in r['source_curve_ids'])
    return result,dict(accepted=accepted,pending=pending,rejections=dict(rejected),input=len(rows),output=len(result))

def refine(rgb,paths,owners=None,recenter=True):
    """Generic curved rock/tree treatment; no straightening or length pruning."""
    flags=[recenter]*len(paths) if isinstance(recenter,bool) else list(recenter)
    profiles=[width_profile(rgb,p,recenter=flag) for p,flag in zip(paths,flags)]
    centered=[];recenter_scales=[]
    for source,p,flag in zip(paths,profiles,flags):
        q=p['original_sample_points'];delta=p['points']-q
        if flag:
            # Independent cross-section centers fluctuate on antialiased paper.
            # Smooth the bounded displacement, not the original rock turns.
            delta=gaussian_filter1d(median_filter(delta,size=(5,1),mode='nearest'),1.,axis=0,mode='nearest')
        if np.linalg.norm(q[0]-q[-1])<.1:delta[[0,-1]]=0
        alpha=1.
        while length(q+alpha*delta)>length(np.asarray(source,float))*1.01+1e-6 and alpha>1/128:alpha*=.5
        centered.append(q+alpha*delta);recenter_scales.append(alpha)
    rows,dedup=deduplicate(rgb,centered,owners)
    rows,joined=join_curves(rgb,rows)
    # Approximation only inside an existing stroke, never a new cross-stroke link.
    count_before=sum(len(r['points']) for r in rows)
    for row in rows:
        p=row['points'];q=cv2.approxPolyDP(p.astype(np.float32),CONFIG['simplify_deviation_px'],False).reshape(-1,2).astype(float)
        q[0]=p[0];q[-1]=p[-1]
        if len(q)<2 or length(q)<1e-8 or (np.linalg.norm(p[0]-p[-1])<.1 and len(q)<4):q=p.copy()
        row['points']=q
    return rows,dict(deduplication=dedup,continuation=joined,input=len(paths),output=len(rows),
        vertices_before_simplification=count_before,vertices_after_simplification=sum(len(r['points']) for r in rows),
        normal_shift_limit_px=CONFIG['normal_recentering_limit_px'],simplification_tolerance_px=CONFIG['simplify_deviation_px'],
        recenter_displacement_smoothed=True,recenter_length_growth_limit_fraction=.01,recenter_scales=recenter_scales,
        no_short_stroke_length_pruning=True,physical_pressure_mm=None,ready_for_robot=False,
        cun_style_recognition_trained=False)
