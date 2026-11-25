# -*- coding: utf-8 -*-
"""Conservative whole-curve source evidence selection, no stroke-count budget.

KEEP/UNCERTAIN both remain in the default output. Only explicit weak source
evidence and whole-curve same-band duplicates leave it, with a full archive.
Roles below are geometric protection flags, not learned cun-style labels.
"""
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from landscape_stroke_refine_20261104 import samples,length,curve_tangents
from source_stroke_width_20261104 import sample

CONFIG=dict(noise_floor=.0007,noise_multiplier=2.6,contrast_floor=.0025,
    main_extent_px=7.,main_length_px=14.,ridge_support_keep=.48,
    weak_reject_support=.12,blank_max_contrast=.0015,
    duplicate_fraction=.96,duplicate_distance_px=1.6,duplicate_max_distance_px=2.2,
    duplicate_alignment=.94,visible_gap_margin=.010,duplicate_band_contrast_floor=.015)

def fields(rgb):
    rgbf=rgb.astype(np.float32)/255;gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    scale=max(1.,max(rgb.shape[:2])/496)
    smooth=cv2.GaussianBlur(gray,(0,0),.5)
    hp=abs(gray-smooth);noise=cv2.GaussianBlur(cv2.medianBlur(hp,5),(0,0),max(1,scale))
    noise=np.maximum(CONFIG['noise_floor'],noise*1.48)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8
    hessians=[]
    for sigma in [.65,1.15,2.0,3.0*scale]:
        g=cv2.GaussianBlur(gray,(0,0),sigma)
        a=cv2.Sobel(g,cv2.CV_32F,2,0,ksize=3)/4;b=cv2.Sobel(g,cv2.CV_32F,0,2,ksize=3)/4
        c=cv2.Sobel(g,cv2.CV_32F,1,1,ksize=3)/4
        hessians.append((a,b,c))
    context_bg=cv2.GaussianBlur(gray,(0,0),max(1.,2*scale))
    return dict(rgb=rgbf,gray=gray,noise=noise,gx=gx,gy=gy,scale=scale,hessians=hessians,context_bg=context_bg)

def evidence(p,f):
    scale=f['scale'];q=samples(p,max(.4,.5*scale));arc=length(np.asarray(p,float))/scale
    smooth=gaussian_filter1d(q,1.,axis=0,mode='nearest');t=curve_tangents(smooth);n=np.column_stack([-t[:,1],t[:,0]])
    gray=f['gray'];center=sample(gray,q);rgbcenter=np.stack([sample(f['rgb'][:,:,k],q) for k in range(3)],1)
    ridge=np.full(len(q),-1.,float);boundary=ridge.copy();neutral=ridge.copy()
    for radius in sorted(set([1.,1.5,2.5,4.,1.5*scale,3*scale,6*scale])):
        left=sample(gray,q+n*radius);right=sample(gray,q-n*radius)
        ridge=np.maximum(ridge,np.minimum(left,right)-center)
        boundary=np.maximum(boundary,np.maximum(abs(left-center),abs(right-center)))
        lc=np.stack([sample(f['rgb'][:,:,k],q+n*radius) for k in range(3)],1)
        rc=np.stack([sample(f['rgb'][:,:,k],q-n*radius) for k in range(3)],1)
        neutral=np.maximum(neutral,np.minimum(lc-rgbcenter,rc-rgbcenter).min(1))
    noise=sample(f['noise'],q);threshold=np.maximum(CONFIG['contrast_floor'],noise*CONFIG['noise_multiplier'])
    support=(ridge>threshold)&(neutral>threshold*.35)
    # An outline can have ink/color on only one side. Aligned gradient and
    # continuous shape guard such edges, rather than demanding a dark valley.
    grad=np.column_stack([sample(f['gx'],q),sample(f['gy'],q)])
    magnitude=np.linalg.norm(grad,axis=1)
    aligned=abs((grad*n).sum(1))/np.maximum(magnitude,1e-8)
    edge=(boundary>threshold*2)&(aligned>.65)&(magnitude>noise*.5)
    directional=np.zeros(len(q),bool)
    for aa,bb,cc in f['hessians']:
        a=sample(aa,q);b=sample(bb,q);c=sample(cc,q)
        across=a*n[:,0]**2+2*c*n[:,0]*n[:,1]+b*n[:,1]**2
        along=a*t[:,0]**2+2*c*t[:,0]*t[:,1]+b*t[:,1]**2
        directional|=(across>np.maximum(.0002,noise*.15))&(across>abs(along)*1.7)
    context=np.maximum(ridge,0)
    # Source peaks within a small localization tolerance protect a reference
    # that runs near, rather than exactly through, a faint stroke center.
    for shift in [-1.,1.]:
        near=q+n*shift*scale
        context=np.maximum(context,sample(f['context_bg'],near)-sample(gray,near))
    span=float(np.linalg.norm(q[-1]-q[0]))/scale;extent=float(np.linalg.norm(np.ptp(q,axis=0)))/scale
    angles=np.unwrap(np.arctan2(t[:,1],t[:,0]));turn=abs(np.diff(angles))
    return dict(points=q,tangents=t,length_px=arc,extent_px=extent,span_px=span,
        ridge_support=float(support.mean()),edge_support=float(edge.mean()),
        near_ink_support=float((context>threshold*.8).mean()),
        directional_support=float(directional.mean()),
        median_contrast=float(np.median(ridge)),q90_contrast=float(np.quantile(ridge,.9)),
        q95_boundary=float(np.quantile(boundary,.95)),median_noise=float(np.median(noise)),
        coherence=float(np.mean(turn<.35)) if len(turn) else 1.,
        closed=bool(np.linalg.norm(q[0]-q[-1])<.1*scale))

def filter_curves(rgb,rows):
    f=fields(rgb);scale=f['scale'];paths=[np.asarray(r['points'],float) for r in rows]
    features=[evidence(p,f) for p in paths]
    # Spatial context does not force connections or change any kept geometry.
    allq=np.concatenate([r['points'] for r in features]);owner=np.concatenate([np.full(len(r['points']),i) for i,r in enumerate(features)])
    tree=cKDTree(allq)
    decisions=[]
    for i,r in enumerate(features):
        q=r['points'];dist,ix=tree.query(q,k=min(16,len(allq)))
        if dist.ndim==1:dist=dist[:,None];ix=ix[:,None]
        other=np.where(owner[ix]!=i,dist,np.inf).min(1)
        isolated=float((other>3*scale).mean())>.8
        long_shape=r['length_px']>=CONFIG['main_length_px'] and r['extent_px']>=CONFIG['main_extent_px']
        strong_ridge=r['ridge_support']>=CONFIG['ridge_support_keep'] and r['directional_support']>=.35
        contour_guard=long_shape and (r['edge_support']>=.25 or r['near_ink_support']>=.30)
        cun_guard=r['coherence']>=.75 and r['length_px']>=3 and r['directional_support']>=.25 and (r['ridge_support']>=.30 or r['near_ink_support']>=.45)
        tiny_true=r['ridge_support']>=.65 # short needles/dots can be genuine ink
        reasons=[];disposition='uncertain'
        if strong_ridge or contour_guard or cun_guard or tiny_true:
            disposition='keep';reasons.append('source-supported ridge or protected continuous outline/texture')
        else:
            strict_blank=r['q95_boundary']<CONFIG['blank_max_contrast'] and r['near_ink_support']<.05
            low_evidence=r['ridge_support']<.12 and r['edge_support']<.12 and r['near_ink_support']<.18
            weak_grain=r['q90_contrast']<max(.003,r['median_noise']*1.8)
            short_irregular=r['length_px']<5 and (r['coherence']<.65 or r['extent_px']<1.6)
            if strict_blank:
                disposition='reject';reasons.append('no visible source evidence along entire curve')
            elif low_evidence and weak_grain and short_irregular and isolated:
                disposition='reject';reasons.append('isolated irregular grain with low adaptive source evidence')
            elif r['length_px']<5 and r['q90_contrast']<=.0015 and r['median_contrast']<=0 and r['edge_support']<.20 and r['directional_support']<.25:
                disposition='reject';reasons.append('short off-ridge candidate: no independent dark valley, edge, or directional source support')
            else:reasons.append('source evidence ambiguous: retain by default')
        decisions.append(dict(curve=i,disposition=disposition,reasons=reasons,
            contour_guard=bool(contour_guard),texture_guard=bool(cun_guard),
            strong_ridge=bool(strong_ridge),isolated=bool(isolated),
            metrics={k:v for k,v in r.items() if k not in ['points','tangents']},
            cun_style_verified=False,source_role_verified=False))
    # Whole-curve duplicate suppression: compare a single longer representative,
    # require tangent agreement and no independently visible bright separation.
    kept=[];duplicates=[]
    for i in sorted(range(len(rows)),key=lambda i:(decisions[i]['disposition']!='keep',-features[i]['length_px'],i)):
        if decisions[i]['disposition']=='reject':continue
        r=features[i];q=r['points'];found=None
        for j in kept:
            a=features[j]
            if rows[i].get('module',0)!=rows[j].get('module',0):continue
            if a['length_px']+1e-6<r['length_px'] or a['closed']!=r['closed']:continue
            if (q.max(0)<a['points'].min(0)-2.2*scale).any() or (q.min(0)>a['points'].max(0)+2.2*scale).any():continue
            d,ix=cKDTree(a['points']).query(q)
            if (d<=CONFIG['duplicate_distance_px']*scale).mean()<.96 or d.max()>CONFIG['duplicate_max_distance_px']*scale:continue
            aligned=abs((r['tangents']*a['tangents'][ix]).sum(1))>=CONFIG['duplicate_alignment']
            if aligned.mean()<.94:continue
            paired=a['points'][ix]
            vals=np.stack([sample(f['gray'],q*(1-t)+paired*t) for t in np.linspace(0,1,13)],1)
            base=np.maximum(vals[:,0],vals[:,-1]);maximum=vals[:,1:-1].max(1)
            noise=sample(f['noise'],(q+paired)/2)
            # High local noise is uncertainty, not permission to merge a gap.
            has_gap=maximum>base+CONFIG['visible_gap_margin']
            if has_gap.mean()>.02:continue
            # Both candidates must belong to one bounded dark band: two outer
            # flanks must be lighter than its interior. A monotone wash edge
            # without a shared valley fails this additional check.
            normal=paired-q;normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-8)
            fallback=np.column_stack([-r['tangents'][:,1],r['tangents'][:,0]])
            normal[d<.1]=fallback[d<.1]
            band_valley=np.full(len(q),-1.,float)
            for radius in [1.,2.,4.]:
                left=sample(f['gray'],q-normal*radius*scale);right=sample(f['gray'],paired+normal*radius*scale)
                band_valley=np.maximum(band_valley,np.minimum(left,right)-vals.min(1))
            if (band_valley>np.maximum(CONFIG['duplicate_band_contrast_floor'],noise)).mean()<.85:continue
            if decisions[j]['disposition']!='keep':continue
            found=j;duplicates.append(dict(curve=i,representative=j,maximum_distance_px=float(d.max()),
                matched_fraction=float((d<=1.6*scale).mean()),bright_gap_fraction=float(has_gap.mean()),
                source_same_artist_stroke_verified=False));break
        if found is None:kept.append(i)
        else:
            decisions[i].update(disposition='duplicate',representative=found,reasons=['whole curve covered by one aligned source band without a visible gap'])
    selected=[i for i,d in enumerate(decisions) if d['disposition'] in ['keep','uncertain']]
    removed=[i for i,d in enumerate(decisions) if d['disposition'] in ['reject','duplicate']]
    assert sorted(selected+removed)==list(range(len(rows)))
    return dict(selected=selected,removed=removed,decisions=decisions,duplicates=duplicates,
        configuration=CONFIG,geometry_of_selected_curves_unchanged=True,uncertain_retained_by_default=True,
        original_complete_archive_required=True,semantic_cun_recognition=False,ready_for_robot=False)
