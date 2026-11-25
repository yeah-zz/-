# -*- coding: utf-8 -*-
"""Native-pixel fine-line proposals using frozen tree routing and source valleys."""
import cv2,numpy as np
from scipy.spatial import cKDTree
from module_stroke_core_20261104 import samples,length
from source_stroke_width_20261104 import width_profile,fields,sample
from landscape_stroke_refine_20261104 import deduplicate

def restore(rgb,paths,tree_probability,box):
    x0,y0,x1,y1=box;pad=16
    xa,ya=max(0,x0-pad),max(0,y0-pad);xb,yb=min(rgb.shape[1],x1+pad),min(rgb.shape[0],y1+pad)
    crop=rgb[ya:yb,xa:xb];gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected=detector.detect(gray)[0]
    source_gray,_=fields(rgb);existing=cKDTree(np.concatenate([samples(p,.5) for p in paths]))
    accepted=[];pending=[]
    candidates=[]
    if detected is not None:candidates=[('LSD',l.reshape(2,2).astype(float)+[xa,ya]) for l in detected[:,0,:]]
    # Ridge/pixel chains can follow curved needles that a straight-line detector
    # splits or misses. Both sources face the identical source/novelty gates.
    from native_fine_faint_20261101 import proposals
    _,_,chains,_=proposals(crop)
    candidates += [('native_dark_chain',np.asarray(r['points'],float)+[xa,ya]) for r in chains if r['origin']=='fine_dark' and r['accepted']]
    for ident,(origin,p) in enumerate(candidates):
        arc=length(p)
        if not 5<=arc<=30:continue
        if not (x0<=p[:,0].mean()<x1 and y0<=p[:,1].mean()<y1):continue
        # LSD returns an intensity edge, which can sit >.75 native pixels from
        # a needle's center. Locate a shared normal valley before width sampling.
        axis=(p[-1]-p[0])/max(np.linalg.norm(p[-1]-p[0]),1e-8);normal=np.array([-axis[1],axis[0]])
        if origin=='LSD':
            seed=samples(p,.5);offsets=np.arange(-2.,2.01,.25)
            cross=seed[:,None,:]+offsets[None,:,None]*normal
            values=sample(source_gray,cross.reshape(-1,2)).reshape(len(seed),-1)
            search=(abs(offsets)<=1.75);ix=np.flatnonzero(search)
            shifts=offsets[ix[values[:,search].argmin(1)]]
            middle=shifts[max(0,len(shifts)//5):max(1,len(shifts)*4//5)]
            p=p+normal*float(np.median(middle))
        profile=width_profile(rgb,p);q=profile['points'];good=profile['observed']
        xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,rgb.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,rgb.shape[0]-1)
        route=float((tree_probability[xy[:,1],xy[:,0]]>=.65).mean())
        width=float(np.median(profile['width_px']))
        radius=max(2.,width*1.6);center=sample(source_gray,q)
        bilateral=np.minimum(sample(source_gray,q+normal*radius),sample(source_gray,q-normal*radius))-center
        support=float((bilateral>.012).mean());dist=existing.query(q)[0];novel=float((dist>1.35).mean())
        take=route>=.85 and good.mean()>=.85 and .65<=width<=3.2 and support>=.80 and novel>=.65 and dist.min()<14
        row=dict(id=ident,points=q,length_px=arc,source_bilateral_support=support,
            source_width_px=width,observed_width_fraction=float(good.mean()),tree_route_fraction=route,
            novel_fraction=novel,minimum_old_curve_distance_px=float(dist.min()),accepted=bool(take),
            identity_verified=False,origin=origin+' with frozen learned tree routing')
        (accepted if take else pending).append(row)
    if accepted:
        unique,diag=deduplicate(rgb,[r['points'] for r in accepted])
        final=[]
        for row in unique:
            p=row['points'];q=samples(p,.4)
            # Earlier added candidates also count as coverage; do not add the two
            # sides of a thin needle merely because both differ from the old plan.
            if final:
                nearest=cKDTree(np.concatenate([samples(r['points'],.4) for r in final])).query(q)[0]
                if (nearest<1.).mean()>.55:continue
            final.append(dict(points=p,native_proposal_ids=[accepted[i]['id'] for i in row['source_curve_ids']],
                source_curve_ids=[],new_native_detail=True,artist_stroke_identity_verified=False))
    else:final=[];diag={}
    return final,dict(raw_detected_lines=0 if detected is None else len(detected),raw_dark_chains=len(candidates)-(0 if detected is None else len(detected)),source_filtered_candidates=len(accepted),accepted_new_details=len(final),
        accepted=accepted,pending=pending,deduplication=diag,roi=box,
        frozen_tree_routing_is_not_verified_semantic_mask=True,new_training_used_for_tree_routing=False,
        new_details_not_expert_ground_truth=True,no_paper_or_wash_edge_added_without_bilateral_valley_test=True)
