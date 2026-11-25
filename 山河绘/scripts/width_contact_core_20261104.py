# -*- coding: utf-8 -*-
"""Resolution-aware optional drawing plan. Pure image/paper geometry, no robot.

Preserves extraction in a separate complete archive. Deferred details are NOT
classified as erroneous strokes. Width envelopes are hypotheses, not physics.
"""
import math
import cv2
import numpy as np

CONFIG=dict(ppm=8.,curve_tolerance_mm=.08,structural_candidate_length_mm=8.,
            closed_preserve_length_mm=2.,overlap_defer=.80,
            dense_short_overlap=.45,dense_other_overlap=.65,
            minimum_endpoint_relative_width=.20,ramp_max_mm=.8)

def length(p):return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())

def validate_path(p):
    p=np.asarray(p,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all():
        raise ValueError('Finite polyline with >=2 XY points required')
    p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-9]]
    if len(p)<2:raise ValueError('Zero-length path')
    return p

def bounded_simplify(points,tolerance=.08):
    """RDP in millimeters; preserves endpoints, never joins different strokes."""
    p=validate_path(points)
    def recurse(a):
        if len(a)<=2:return a
        delta=a[-1]-a[0];denom=float(delta@delta)
        t=np.clip((a-a[0])@delta/max(denom,1e-15),0,1)
        distances=np.linalg.norm(a-(a[0]+t[:,None]*delta),axis=1)
        index=int(distances.argmax())
        if distances[index]<=tolerance:return a[[0,-1]]
        return np.concatenate([recurse(a[:index+1])[:-1],recurse(a[index:])])
    return recurse(p)

def max_deviation(original,simplified):
    p=validate_path(original);q=validate_path(simplified)
    a=q[:-1];delta=np.diff(q,axis=0);den=(delta*delta).sum(1)
    t=np.clip(((p[:,None,:]-a[None,:,:])*delta[None,:,:]).sum(2)/np.maximum(den,1e-15),0,1)
    nearest=a[None,:,:]+t[:,:,None]*delta[None,:,:]
    return float(np.linalg.norm(p[:,None,:]-nearest,axis=2).min(1).max())

def contact_envelope(points):
    p=validate_path(points);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    step=.12;at=np.linspace(0,arc[-1],max(3,int(math.ceil(arc[-1]/step))+1))
    q=np.column_stack([np.interp(at,arc,p[:,k]) for k in [0,1]])
    closed=bool(np.linalg.norm(p[-1]-p[0])<1e-6)
    if closed:relative=np.ones(len(q))
    else:
        ramp=min(CONFIG['ramp_max_mm'],arc[-1]*.3)
        phase=np.clip(np.minimum(at,arc[-1]-at)/max(ramp,1e-8),0,1)
        phase=phase*phase*(3-2*phase)
        relative=CONFIG['minimum_endpoint_relative_width']+(1-CONFIG['minimum_endpoint_relative_width'])*phase
    return dict(points=q,arclength_mm=at,relative_width=relative,closed=closed,
                physical_compression_mm=None,pressure_to_width_mapping=None,
                envelope_is_design_prior=True,no_extension_outside_source_curve=True)

def swept_mask(points,width_mm,canvas_mm=(186.8,186.8),taper=True,ppm=None):
    if not np.isfinite(width_mm) or width_mm<=0:raise ValueError('Positive width required')
    ppm=CONFIG['ppm'] if ppm is None else ppm
    p=validate_path(points)
    if (p.min(0)<-.001).any() or (p.max(0)>np.array(canvas_mm)+.001).any():raise ValueError('Outside paper')
    env=contact_envelope(p)
    q=env['points'];widths=env['relative_width']*width_mm if taper else np.full(len(q),width_mm)
    pad=int(math.ceil(width_mm*ppm))+4
    full_shape=(int(math.ceil(canvas_mm[1]*ppm))+2*pad+1,int(math.ceil(canvas_mm[0]*ppm))+2*pad+1)
    xy=q*ppm+pad
    radius=widths*ppm/2
    lo=np.maximum(np.floor(xy.min(0)-radius.max()-2).astype(int),0)
    hi=np.minimum(np.ceil(xy.max(0)+radius.max()+3).astype(int),np.array(full_shape[::-1]))
    # Supersampling avoids treating subpixel tips as the same full-width disks.
    scale=2;local=(xy-lo)*scale
    mask=np.zeros(tuple(((hi-lo)*scale)[::-1]),np.uint8)
    for i in range(len(local)-1):
        delta=local[i+1]-local[i];norm=np.linalg.norm(delta)
        if norm<1e-9:continue
        normal=np.array([-delta[1],delta[0]])/norm
        a=normal*radius[i]*scale;b=normal*radius[i+1]*scale
        poly=np.array([local[i]+a,local[i+1]+b,local[i+1]-b,local[i]-a])
        cv2.fillConvexPoly(mask,np.rint(poly).astype(np.int32),255)
    for xy0,r in zip(local,radius):cv2.circle(mask,tuple(np.rint(xy0).astype(int)),max(1,int(round(r*scale))),255,-1)
    mask=cv2.resize(mask,(int(hi[0]-lo[0]),int(hi[1]-lo[1])),interpolation=cv2.INTER_AREA)>=128
    return mask,(slice(lo[1],hi[1]),slice(lo[0],hi[0])),full_shape,pad,env

def coverage(paths,width,taper=True,canvas_mm=(186.8,186.8)):
    counts=None;pad=None
    for p in paths:
        mask,sl,shape,pad,_=swept_mask(p,width,canvas_mm,taper)
        if counts is None:counts=np.zeros(shape,np.uint16)
        counts[sl][mask]+=1
    if counts is None:raise ValueError('At least one path required')
    return counts,pad

def coverage_metrics(counts):
    occupied=counts>0;pixels=int(occupied.sum());repeat=int((counts>=2).sum())
    return dict(geometric_covered_mm2=pixels/CONFIG['ppm']**2,
                geometric_repeated_mm2=repeat/CONFIG['ppm']**2,
                repeated_fraction=repeat/max(1,pixels),max_other_stroke_exposures=int(counts.max()),
                actual_ink_simulation=False,measured_drawing_improvement=None)

def plan_for_width(curves,route,width,canvas_mm=(186.8,186.8)):
    paths=[validate_path(c['points']) for c in curves]
    lengths=np.array([length(p) for p in paths])
    protected={i for i,p in enumerate(paths) if lengths[i]>=CONFIG['structural_candidate_length_mm']
        or (np.linalg.norm(p[0]-p[-1])<1e-6 and lengths[i]>=CONFIG['closed_preserve_length_mm'])}
    module_order=list(dict.fromkeys(curves[r['curve']]['module'] for r in route))
    counts=None;decisions=[];selected=[];deferred=[];profiles={}
    for module in module_order:
        ids=[i for i,c in enumerate(curves) if c['module']==module]
        ids.sort(key=lambda i:(i not in protected,-lengths[i],i))
        for i in ids:
            mask,sl,shape,pad,env=swept_mask(paths[i],width,canvas_mm,True)
            if counts is None:counts=np.zeros(shape,np.uint16)
            before=counts[sl];area=max(1,int(mask.sum()))
            fraction=float((mask&(before>0)).sum()/area)
            is_short=lengths[i]<width*2.5
            crowded=fraction>CONFIG['overlap_defer'] or (
                i not in protected and ((is_short and fraction>CONFIG['dense_short_overlap'])
                                       or fraction>CONFIG['dense_other_overlap']))
            defer=bool(crowded and i not in protected)
            decision=dict(curve=i,module=module,length_mm=float(lengths[i]),
                prior_footprint_overlap_fraction=fraction,protected_long_or_closed_candidate=i in protected,
                disposition='deferred_detail' if defer else 'selected',
                reason='detail cannot be separated at assumed width' if defer else 'preserved structural candidate' if i in protected else 'new geometric coverage',
                unresolved_structural_overlap=bool(i in protected and fraction>.65),
                same_artist_stroke_verified=False,detail_is_error=False)
            decisions.append(decision)
            if defer:deferred.append(i)
            else:
                selected.append(i);counts[sl][mask]+=1
                profiles[i]=dict(arclength_mm=env['arclength_mm'].tolist(),
                    relative_width=env['relative_width'].tolist(),physical_compression_mm=None,
                    outside_curve_extension_mm=0.,closed_seam_pending=env['closed'],
                    pressure_mapping_measured=False)
    assert sorted(selected+deferred)==list(range(len(curves)))
    assert protected.issubset(selected)
    return dict(selected=selected,deferred=deferred,decisions=decisions,profiles=profiles,
        protected_curve_ids=sorted(protected),counts=counts,pad=pad,
        complete_extraction_preserved=True,automatic_detail_simplification=True,
        unverified_module_boundaries=True,ready_for_robot=False,
        deferred_is_not_extraction_correction=True,assumed_width_mm=width,
        structural_conflicts=sum(d['unresolved_structural_overlap'] for d in decisions))
