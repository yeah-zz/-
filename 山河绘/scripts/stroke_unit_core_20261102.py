# -*- coding: utf-8 -*-
"""Image-space stroke relations. Never emits robot coordinates or commands."""
from collections import defaultdict
from pathlib import Path
import json
import math
import cv2
import numpy as np
from scipy.spatial import cKDTree
from prepare_full_ink_trial_20261102 import join_existing

FEATURES=['log_len_a','log_len_b','gap','facing','axis_agreement','a_to_gap','b_to_gap',
          'overlap_a','overlap_b','mean_distance','p90_distance','band_fraction',
          'valley_excess','dark_a','dark_b','contrast_a','contrast_b',
          'curvature_a','curvature_b','curve_match','chord_a','chord_b','gap_ink_support','length_ratio']


def length(p):return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())


def resample(p,step=1.,limit=64):
    p=np.asarray(p,np.float64)
    p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-8]]
    if len(p)<2:return np.repeat(p[:1],2,axis=0)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    t=np.linspace(0,arc[-1],min(limit,max(2,int(np.ceil(arc[-1]/step))+1)))
    return np.column_stack([np.interp(t,arc,p[:,j]) for j in range(2)])


def unit(v):return v/max(float(np.linalg.norm(v)),1e-8)


def tangent(p):
    p=resample(p,.7,48)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    at=min(5.,arc[-1]);q=np.array([np.interp(at,arc,p[:,j]) for j in range(2)])
    return unit(q-p[0])


def sample(gray,p):
    p=np.asarray(p,np.float32)
    return cv2.remap(gray,p[:,0].reshape(-1,1),p[:,1].reshape(-1,1),cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT101).ravel()


def band_evidence(gray,a,b):
    va=sample(gray,a);vb=sample(gray,b);bright=np.maximum(va,vb);peak=bright.copy()
    for t in [.2,.4,.6,.8]:peak=np.maximum(peak,sample(gray,a*(1-t)+b*t))
    return peak<=bright+.018,np.maximum(0,peak-bright)


def pair_features(p,q,gray,background=None):
    p=resample(p,.8,48);q=resample(q,.8,48)
    lp=length(p);lq=length(q);ta=tangent(p);tb=tangent(q)
    delta=q[0]-p[0];gap=float(np.linalg.norm(delta));axis=unit(delta)
    distances=np.linalg.norm(p[:,None,:]-q[None,:,:],axis=2)
    near=distances.argmin(axis=1);da=distances.min(axis=1);db=distances.min(axis=0)
    pa=unit(p[-1]-p[0]);qa=unit(q[-1]-q[0]);band,valley=band_evidence(gray,p,q[near])
    gp=np.gradient(p,axis=0);gp/=np.maximum(np.linalg.norm(gp,axis=1,keepdims=True),1e-8)
    gq=np.gradient(q,axis=0);gq/=np.maximum(np.linalg.norm(gq,axis=1,keepdims=True),1e-8)
    agreement=float(np.mean(abs((gp*gq[near]).sum(1))))
    vp=sample(gray,p);vq=sample(gray,q)
    if background is None:background=cv2.GaussianBlur(gray,(0,0),3.)
    cp=sample(background,p)-vp;cq=sample(background,q)-vq
    curvature=lambda a:float(np.cross(unit(a[min(3,len(a)-1)]-a[0]),unit(a[-1]-a[0])))
    ca=curvature(p);cb=curvature(q)
    bridge=np.linspace(p[0],q[0],max(3,int(gap*3)+1));v=sample(gray,bridge)
    contrast=sample(background,bridge)-v
    ink=float(np.mean((contrast>.008)|(v<.4)))
    x=[np.log1p(lp)/4,np.log1p(lq)/4,min(gap,30)/10,float(ta@(-tb)),agreement,
       float(ta@axis) if gap>.01 else 0,float(tb@(-axis)) if gap>.01 else 0,
       float((da<1.3).mean()),float((db<1.3).mean()),min(float(da.mean()),30)/10,
       min(float(np.quantile(da,.9)),30)/10,float(band.mean()),float(np.mean(valley))*5,
       float(vp.mean()),float(vq.mean()),float(cp.mean())*5,float(cq.mean())*5,
       ca,cb,abs(ca+cb),float(np.linalg.norm(p[-1]-p[0]))/max(lp,1e-6),
       float(np.linalg.norm(q[-1]-q[0]))/max(lq,1e-6),ink,min(lp,lq)/max(lp,lq,1e-6)]
    return np.asarray(x,np.float32)


def deduplicate_exact(paths):
    seen=set();out=[];removed=0
    for i,p in enumerate(paths):
        chunk=[]
        for a,b in zip(p[:-1],p[1:]):
            if np.linalg.norm(a-b)<1e-8:continue
            key=tuple(sorted((tuple(a),tuple(b))))
            if key in seen:
                removed+=1
                if len(chunk)>1:out.append(dict(points=np.array(chunk),source_ids=[i]))
                chunk=[]
            else:
                seen.add(key)
                if not chunk:chunk=[a]
                chunk.append(b)
        if len(chunk)>1:out.append(dict(points=np.array(chunk),source_ids=[i]))
    return out,removed


def is_closed(p):return len(p)>3 and np.linalg.norm(p[0]-p[-1])<.1


def match_whole_band(p,q,gray):
    p=resample(p,.7,80);q=resample(q,.7,80)
    dist,ix=cKDTree(q).query(p)
    band,valley=band_evidence(gray,p,q[ix])
    gp=np.gradient(p,axis=0);gp/=np.maximum(np.linalg.norm(gp,axis=1,keepdims=True),1e-8)
    gq=np.gradient(q,axis=0);gq/=np.maximum(np.linalg.norm(gq,axis=1,keepdims=True),1e-8)
    parallel=abs((gp*gq[ix]).sum(1))>.92
    good=(dist<=1.25)&band&parallel
    dark=sample(gray,p);contrast=sample(cv2.GaussianBlur(gray,(0,0),3.),p)-dark
    evidence=float(((contrast>.009)|(dark<.4)).mean())
    if is_closed(p):
        # Keep closed contours unless they are a second near-identical closed contour.
        if not is_closed(q):return False,dict(reason='closed_contour_identity_uncertain')
    okay=good.mean()>=.96 and band.mean()>=.98 and evidence>=.7
    return bool(okay),dict(covered_fraction=float(good.mean()),same_ink_band=float(band.mean()),
        source_ink_support=evidence,median_distance=float(np.median(dist)))


def union_raster(shape,paths):
    mask=np.zeros(shape,np.uint8)
    for p in paths:cv2.polylines(mask,[np.rint(p).astype(np.int32)],False,1,1,cv2.LINE_8)
    return mask.astype(bool)


def process(rgb,paths,predict,thresholds):
    """predict Nx24 -> Nx3: continuous / same-ink-band duplicate / separate."""
    native_shape=rgb.shape[:2];factor=496./max(native_shape)
    gray=cv2.cvtColor(cv2.resize(rgb,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    background=cv2.GaussianBlur(gray,(0,0),3.)
    records,exact_removed=deduplicate_exact(paths)
    # Keep the previous clearly supported exact-junction routing as the fair
    # baseline. The learned module handles what remains, rather than undoing it.
    records,baseline_joins=join_existing(records)
    baseline_lengths=[length(r['points'])*factor*186.8/495 for r in records]
    baseline_closed=int(sum(is_closed(r['points']) for r in records))
    normalized=[r['points']*factor for r in records]
    sampled=[resample(p,1.,96) for p in normalized]
    lengths=np.array([length(p) for p in normalized]);n=len(records)
    cloud=np.concatenate(sampled);owners=np.concatenate([np.full(len(p),i,np.int32) for i,p in enumerate(sampled)])
    tree=cKDTree(cloud);candidate_pairs=set()
    for i,p in enumerate(sampled):
        dist,near=tree.query(p,k=6,distance_upper_bound=1.35)
        candidates=set(owners[near[np.isfinite(dist)]].tolist())
        for j in candidates:
            if i!=j and (lengths[j]>lengths[i] or lengths[j]==lengths[i] and j<i):candidate_pairs.add((i,j))
    pairs=sorted(candidate_pairs)
    x=np.stack([pair_features(normalized[i],normalized[j],gray,background) for i,j in pairs]) if pairs else np.empty((0,len(FEATURES)),np.float32)
    probs=predict(x) if len(x) else np.empty((0,3))
    duplicate_probabilities=probs.copy()
    lookup=defaultdict(list)
    for (i,j),prob in zip(pairs,probs):
        if prob[1]>=thresholds['duplicate']:lookup[i].append((float(prob[1]),j))
    retained=set();dup=[]
    for i in sorted(range(n),key=lambda i:(-lengths[i],i)):
        removed=False
        for probability,j in sorted(lookup[i],reverse=True):
            if j not in retained:continue
            okay,evidence=match_whole_band(normalized[i],normalized[j],gray)
            if okay:
                dup.append(dict(removed_record=i,represented_by=j,probability=probability,
                    source_ids=records[i]['source_ids'],**evidence));removed=True;break
        if not removed:retained.add(i)
    endpoint=[]
    for i in sorted(retained):
        if is_closed(normalized[i]):continue
        for side in [0,1]:endpoint.append((i,side,normalized[i][0 if side==0 else -1]))
    candidates=[]
    if endpoint:
        et=cKDTree([e[2] for e in endpoint])
        for a,b in sorted(et.query_pairs(1.5)):
            i,si,pa=endpoint[a];j,sj,pb=endpoint[b]
            if i==j:continue
            p=normalized[i] if si==0 else normalized[i][::-1]
            q=normalized[j] if sj==0 else normalized[j][::-1]
            if min(lengths[i],lengths[j])<1.2:continue
            f=pair_features(p,q,gray,background)
            if f[3]<-.3:continue
            gap=float(np.linalg.norm(pa-pb))
            if gap>.05 and f[22]<.95:continue
            candidates.append((a,b,f,gap))
    x=np.stack([r[2] for r in candidates]) if candidates else np.empty((0,len(FEATURES)),np.float32)
    probs=predict(x) if len(x) else np.empty((0,3))
    rankings=defaultdict(list)
    for k,((a,b,_,gap),prob) in enumerate(zip(candidates,probs)):
        rankings[a].append((float(prob[0]),k));rankings[b].append((float(prob[0]),k))
    linked={};joins=[]
    for k,((a,b,_,gap),prob) in sorted(enumerate(zip(candidates,probs)),key=lambda v:-v[1][1][0]):
        probability=float(prob[0])
        if probability<thresholds['continuous']:continue
        if a in linked or b in linked:continue
        ranks=[sorted(rankings[e],reverse=True) for e in (a,b)]
        if any(r[0][1]!=k or len(r)>1 and r[0][0]-r[1][0]<.12 for r in ranks):continue
        ea=endpoint[a][:2];eb=endpoint[b][:2];linked[a]=b;linked[b]=a
        joins.append(dict(end_a=list(ea),end_b=list(eb),probability=probability,
                          gap_normalized_px=gap,artist_stroke_identity_verified=False))
    by_end={endpoint[a][:2]:endpoint[b][:2] for a,b in linked.items()}
    visited=set();out=[]
    def walk(start):
        current=start;parts=[];ids=[];record_ids=[]
        while current[0] not in visited:
            i,side=current;visited.add(i);p=records[i]['points'] if side==0 else records[i]['points'][::-1]
            if parts and np.linalg.norm(parts[-1][-1]-p[0])<1e-7:p=p[1:]
            parts.append(p);ids+=records[i]['source_ids'];record_ids.append(i)
            if (i,1-side) not in by_end:break
            current=by_end[(i,1-side)]
        out.append(dict(points=np.concatenate(parts),source_ids=sorted(set(ids)),record_ids=record_ids))
    for i in sorted(retained):
        for side in (0,1):
            if i not in visited and (i,side) not in by_end:walk((i,side))
    for i in sorted(retained):
        if i not in visited:walk((i,0))
    if visited!=retained:raise AssertionError('Unrepresented retained source record')
    out.sort(key=lambda r:-length(r['points']))
    old=union_raster(native_shape,paths);new=union_raster(native_shape,[r['points'] for r in out])
    near=cv2.distanceTransform((~new).astype(np.uint8),cv2.DIST_L2,5)
    tolerance=2./factor
    coverage=float((near[old]<=tolerance).mean()) if old.any() else 1.
    minimum=float(np.min([r.get('covered_fraction',1.) for r in dup])) if dup else 1.
    lengths_mm=[length(r['points'])*factor*186.8/495 for r in out]
    old_lengths=[length(p)*factor*186.8/495 for p in paths]
    stats=dict(input_graph_segments=len(paths),old_routed_curve_units=len(records),
        baseline_exact_joins=len(baseline_joins),exact_duplicate_edges_removed=exact_removed,
        learned_joins=len(joins),learned_gap_joins=sum(j['gap_normalized_px']>.05 for j in joins),
        source_band_duplicates_removed=len(dup),output_curve_units=len(out),
        raw_graph_under_2mm=sum(v<2 for v in old_lengths),
        old_under_2mm=sum(v<2 for v in baseline_lengths),new_under_2mm=sum(v<2 for v in lengths_mm),
        old_median_mm=float(np.median(baseline_lengths)),new_median_mm=float(np.median(lengths_mm)),
        assumed_square_size_mm=186.8,original_candidate_coverage_at_2_normalized_px=coverage,
        coverage_is_not_ground_truth_accuracy=True,minimum_removed_path_band_coverage=minimum,
        unverified_short_units=sum(v<2 for v in lengths_mm),ready_for_robot=False,
        input_closed_paths=baseline_closed,output_closed_paths=int(sum(is_closed(r['points']) for r in out)))
    scores=dict(duplicate_probabilities=duplicate_probabilities,
        duplicate_pair_ids=np.array(pairs,dtype=np.int32).reshape(-1,2),
        endpoint_probabilities=probs,
        endpoint_pair_ids=np.array([[r[0],r[1]] for r in candidates],dtype=np.int32).reshape(-1,2))
    stats['raw_model_duplicate_candidates']=int((duplicate_probabilities[:,1]>=thresholds['duplicate']).sum()) if len(duplicate_probabilities) else 0
    stats['raw_model_continuation_candidates']=int((probs[:,0]>=thresholds['continuous']).sum()) if len(probs) else 0
    return out,dict(duplicate_decisions=dup,join_decisions=joins,probability_arrays=scores),stats
