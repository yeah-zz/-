# -*- coding: utf-8 -*-
"""Offline object-proposal ownership, complete-curve assembly and hierarchical rollout.

Only image coordinates. SAM masks are proposals, not verified rock instances.
Rollout is geometric, not a learned ink/brush dynamics model. No robot imports.
"""
import math
from collections import Counter
import cv2
import numpy as np
from scipy.spatial import cKDTree

CONFIG=dict(min_mask_area=420, max_mask_fraction=.38, min_foreground_fraction=.22,
            ownership_fraction=.80, gap_px=3., tangent_degrees=60.,
            unique_corner_degrees=115., match_margin=.12, bridge_support=.90,
            footprint_width_px=1.6, wet_history_actions=12)

def length(p):
    return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())

def samples(p, step=1.):
    p=np.asarray(p,float)
    d=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if d[-1]<1e-8:
        return p[:1]
    unique=np.r_[True,np.diff(d)>1e-8]
    t=np.linspace(0,d[-1],max(2,int(d[-1]/step)+1))
    return np.column_stack([np.interp(t,d[unique],p[unique,k]) for k in [0,1]])

def tangent(p):
    p=samples(p,.5)
    at=min(len(p)-1,8)
    v=p[at]-p[0]
    return v/max(np.linalg.norm(v),1e-8)

def raster(paths, shape, thickness=1):
    a=np.zeros(shape,np.uint8)
    for p in paths:
        cv2.polylines(a,[np.rint(p).astype(np.int32)],False,1,thickness)
    return a

def at(field,p):
    xy=np.rint(p).astype(int)
    xy[:,0]=np.clip(xy[:,0],0,field.shape[1]-1)
    xy[:,1]=np.clip(xy[:,1],0,field.shape[0]-1)
    return field[xy[:,1],xy[:,0]]

def propose_modules(rgb, paths, masks, records):
    """No image-specific IDs, polygons, crop labels, or position-based semantic names."""
    h,w=rgb.shape[:2]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)/255.
    fg=(gray<.90)|(hsv[:,:,1]>35)
    proposals=[]
    for record in records:
        mask=cv2.resize(masks[record['key']].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        area=int(mask.sum())
        if area<CONFIG['min_mask_area'] or area>h*w*CONFIG['max_mask_fraction']:
            continue
        if float(fg[mask].mean())<CONFIG['min_foreground_fraction']:
            continue
        # Only suppress near-identical masks. Nested part/whole proposals stay
        # visible; ownership and hierarchy below expose segmentation uncertainty.
        if any((mask&p['mask']).sum()/max(1,(mask|p['mask']).sum())>.90 for p in proposals):
            continue
        proposals.append(dict(mask=mask,area=area,sam_key=record['key'],
            predicted_iou=record['predicted_iou'],stability_score=record['stability_score']))
    # Collapse nested roof/base/rock-face masks into their proposed whole object.
    # This is an explicit part/whole heuristic, not semantic class supervision.
    proposals.sort(key=lambda r:r['area'])
    parents={}
    for i,p in enumerate(proposals):
        containing=[j for j,q in enumerate(proposals) if j>i and q['area']<p['area']*8
                    and (p['mask']&q['mask']).sum()/p['area']>.92]
        if containing:
            parents[i]=min(containing,key=lambda j:proposals[j]['area'])
    def top(i):
        while i in parents:
            i=parents[i]
        return i
    roots=sorted(set(top(i) for i in range(len(proposals))))
    proposals=[dict(**proposals[i],part_whole_proposal=True) for i in roots]
    proposals.sort(key=lambda r:r['area'])
    expanded=[cv2.dilate(p['mask'].astype(np.uint8),np.ones((5,5),np.uint8)) for p in proposals]
    ownership=[]
    for path in paths:
        q=samples(path)
        fits=[(i,float(at(mask,q).mean())) for i,mask in enumerate(expanded)]
        good=[(i,v) for i,v in fits if v>=CONFIG['ownership_fraction']]
        if good:
            i,v=good[0]  # tightest supporting mask; never cut a curve at mask rims
            ownership.append(dict(module=i,fraction=v,uncertain=False))
        else:
            ownership.append(dict(module=-1,fraction=max([v for _,v in fits]+[0.]),uncertain=True))
    used=sorted(set(r['module'] for r in ownership)-{-1})
    remap={v:i for i,v in enumerate(used)}
    modules=[dict(**proposals[v],id=remap[v],kind='SAM visual instance proposal',verified=False) for v in used]
    for r in ownership:
        if r['module']>=0:
            r['module']=remap[r['module']]
    # The unresolved set is explicitly spatial, not relabelled as rock/tree.
    # Components are bounded local neighborhoods; no arbitrary force assignment
    # to a confidently classified object across white space.
    pending=[i for i,r in enumerate(ownership) if r['module']<0]
    groups=[]
    for i in sorted(pending,key=lambda i:-length(paths[i])):
        c=samples(paths[i]).mean(0)
        choices=[(np.linalg.norm(c-g['center']),j) for j,g in enumerate(groups)]
        best=min(choices,default=(1e9,-1))
        if best[0]<32.:
            g=groups[best[1]]
            g['ids'].append(i)
            # Keep the initial anchor, preventing a chain drifting across scene.
        else:
            groups.append(dict(center=c,ids=[i]))
    for g in groups:
        index=len(modules)
        mask=raster([paths[i] for i in g['ids']],(h,w),3)>0
        modules.append(dict(id=index,mask=mask,area=int(mask.sum()),sam_key=None,
                            kind='unresolved local region',verified=False))
        for i in g['ids']:
            ownership[i]['module']=index
    return modules, ownership

def assemble(paths, gray, owners):
    """Global reciprocal endpoint matching; no length cap, no deletion.

    Degree-two corners can continue despite a large turn. At junctions, only
    unambiguous tangent continuation is permitted. Parallel strokes stay separate.
    """
    paths=[np.asarray(p,float) for p in paths]
    ends=[]
    for i,p in enumerate(paths):
        if length(p)<1. or np.linalg.norm(p[0]-p[-1])<.1:
            continue
        for side in [0,1]:
            q=p if side==0 else p[::-1]
            ends.append(dict(i=i,side=side,xy=q[0],t=tangent(q)))
    pairs=sorted(cKDTree([r['xy'] for r in ends]).query_pairs(CONFIG['gap_px'])) if ends else []
    degree=Counter()
    for a,b in pairs:
        if ends[a]['i']!=ends[b]['i'] and np.linalg.norm(ends[a]['xy']-ends[b]['xy'])<1.5:
            degree[a]+=1
            degree[b]+=1
    bg=cv2.GaussianBlur(gray,(0,0),3.)
    ink=(bg-gray>.009)|(gray<.38)
    candidates=[]
    rejected=Counter()
    for a,b in pairs:
        ea,eb=ends[a],ends[b]
        i,j=ea['i'],eb['i']
        if i==j:
            continue
        if owners[i]['module']!=owners[j]['module']:
            rejected['different_module']+=1
            continue
        delta=eb['xy']-ea['xy'];gap=float(np.linalg.norm(delta))
        facing=float(ea['t']@(-eb['t']))
        unique_corner=gap<1.5 and degree[a]==degree[b]==1
        threshold=CONFIG['unique_corner_degrees'] if unique_corner else CONFIG['tangent_degrees']
        if facing<math.cos(math.radians(threshold)):
            rejected['turn_or_parallel']+=1
            continue
        if gap>1.5:
            axis=delta/gap
            if min(float(-ea['t']@axis),float(eb['t']@axis))<.75:
                rejected['sideways_bridge']+=1
                continue
        bridge=np.linspace(ea['xy'],eb['xy'],max(3,int(gap*8)+1))
        support=float(at(ink,bridge).mean())
        if gap>.1 and support<CONFIG['bridge_support']:
            rejected['unsupported_gap']+=1
            continue
        score=(facing+1)/2-.045*gap+.10*support
        candidates.append(dict(a=a,b=b,i=i,j=j,gap=gap,score=score,support=support,
                               unique_corner=unique_corner,angle=math.degrees(math.acos(np.clip(facing,-1,1)))))
    ranks={}
    for k,c in enumerate(candidates):
        for e in [c['a'],c['b']]:
            ranks.setdefault(e,[]).append((c['score'],k))
    for e in ranks:
        ranks[e].sort(reverse=True)
    parent=list(range(len(paths)));links={};accepted=[];pending=[]
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for k,c in sorted(enumerate(candidates),key=lambda z:-z[1]['score']):
        a,b=c['a'],c['b']
        if a in links or b in links:
            continue
        if any(ranks[e][0][1]!=k or len(ranks[e])>1 and ranks[e][0][0]-ranks[e][1][0]<CONFIG['match_margin'] for e in [a,b]):
            pending.append(dict(**c,reason='competing continuations'))
            continue
        x,y=root(c['i']),root(c['j'])
        if x==y:
            rejected['new_closed_loop']+=1
            continue
        parent[y]=x;links[a]=b;links[b]=a
        accepted.append(c)
    byend={(ends[a]['i'],ends[a]['side']):(ends[b]['i'],ends[b]['side']) for a,b in links.items()}
    visited=set();out=[]
    def walk(start):
        current=start;parts=[];ids=[];orientations=[]
        while current[0] not in visited:
            i,side=current;visited.add(i)
            parts.append(paths[i] if side==0 else paths[i][::-1])
            ids.append(i);orientations.append(side)
            if (i,1-side) not in byend:
                break
            current=byend[(i,1-side)]
        out.append(dict(points=np.concatenate(parts),source_curve_ids=ids,
                        module=owners[ids[0]]['module'],uncertain_module=any(owners[i]['uncertain'] for i in ids),
                        source_reversed=orientations,artist_stroke_identity_verified=False))
    for i in range(len(paths)):
        for side in [0,1]:
            if i not in visited and (i,side) not in byend:
                walk((i,side))
    for i in range(len(paths)):
        if i not in visited:
            walk((i,0))
    assert sorted(i for r in out for i in r['source_curve_ids'])==list(range(len(paths)))
    old=raster(paths,gray.shape);new=raster([r['points'] for r in out],gray.shape)
    near=cv2.dilate(new,np.ones((3,3),np.uint8))
    missing=int(np.count_nonzero((old>0)&(near==0)))
    if missing:
        raise ValueError('Geometry lost')
    return out,dict(accepted=accepted,pending=pending,rejections=dict(rejected),
        original_count=len(paths),assembled_count=len(out),no_original_curve_deleted=True,
        original_raster_missing_within_one_pixel=missing,
        new_bridge_pixels=int(((new>0)&(old==0)).sum()),complete_stroke_truth_available=False)

def schedule(curves, modules, shape, use_modules=True):
    """Pick next stroke using geometric consequences, with an active module lock.

    A footprint and recent-command mask are hypothetical state only; never
    described as observed ink or calibrated fluid dynamics.
    """
    remaining=set(range(len(curves)));route=[];cursor=np.array([0.,shape[0]-1.])
    sampled=[samples(r['points'],2.) for r in curves]
    arc_lengths=[length(r['points']) for r in curves]
    active=None;done_modules=[];recent=[];covered=np.zeros(shape,np.uint8)
    while remaining:
        if use_modules and not any(curves[i]['module']==active for i in remaining):
            if active is not None:
                done_modules.append(active)
            available=sorted(set(curves[i]['module'] for i in remaining))
            # Prefer SAM-owned modules before explicitly unresolved local regions.
            confident=[m for m in available if modules[m]['sam_key'] is not None]
            available=confident or available
            def module_cost(m):
                ids=[i for i in remaining if curves[i]['module']==m]
                ends=np.concatenate([curves[i]['points'][[0,-1]] for i in ids])
                return float(np.linalg.norm(ends-cursor,axis=1).min())
            active=min(available,key=module_cost)
        ids=[i for i in remaining if not use_modules or curves[i]['module']==active]
        wet=np.zeros(shape,np.uint8)
        for p in recent[-CONFIG['wet_history_actions']:]:
            cv2.polylines(wet,[np.rint(p).astype(np.int32)],False,1,3)
        choices=[]
        for i in ids:
            p=curves[i]['points'];q=sampled[i]
            novel=1-float(at(covered,q).mean())
            overlap=float(at(wet,q).mean())
            for reverse in [False,True]:
                start=p[-1] if reverse else p[0]
                travel=float(np.linalg.norm(start-cursor))
                endpoint_pool=float(at(wet,start.reshape(1,2))[0])
                benefit=math.sqrt(max(arc_lengths[i],.1))*(.35+.65*novel)
                cost=benefit/(1+travel/20.)-5.*overlap-8.*endpoint_pool
                choices.append((cost,-i,not reverse,i,reverse,travel,overlap,endpoint_pool))
        _,_,_,i,reverse,travel,overlap,pool=max(choices)
        p=curves[i]['points'][::-1] if reverse else curves[i]['points']
        route.append(dict(curve=i,module=curves[i]['module'],reverse=reverse,
                          travel_px=travel,predicted_recent_overlap_fraction=overlap,
                          predicted_start_contact_with_recent_ink=bool(pool),
                          rollout='uncalibrated geometric footprint; not actual ink'))
        cv2.polylines(covered,[np.rint(p).astype(np.int32)],False,1,1)
        recent.append(p);cursor=p[-1];remaining.remove(i)
    assert sorted(r['curve'] for r in route)==list(range(len(curves)))
    return route

def route_metrics(route,curves):
    labels=[r['module'] for r in route]
    switches=sum(a!=b for a,b in zip(labels,labels[1:]))
    completed=set();returns=0
    for a,b in zip(labels,labels[1:]):
        if a!=b:
            completed.add(a)
            returns+=int(b in completed)
    return dict(strokes=len(route),module_switches=switches,module_reentries=returns,
                air_travel_px=sum(r['travel_px'] for r in route),
                predicted_start_recent_contacts=sum(r['predicted_start_contact_with_recent_ink'] for r in route),
                predicted_overlap_mean=float(np.mean([r['predicted_recent_overlap_fraction'] for r in route])) if route else 0.,
                actual_drawing_time_seconds=None,actual_blot_reduction=None)
