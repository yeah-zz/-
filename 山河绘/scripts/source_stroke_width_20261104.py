# -*- coding: utf-8 -*-
"""Source-supported straight stroke grouping and transverse image width targets.

No robot imports or pressure calibration. Local architectural ROI is supplied
by the experiment, not claimed to be automatically recognized semantics.
"""
import math
import numpy as np
import cv2
from scipy.ndimage import median_filter
from module_stroke_core_20261104 import samples,length,assemble

CONFIG=dict(line_residual_px=.55,line_group_residual_px=.65,max_line_angle_degrees=7.,
            maximum_supported_gap_px=3.,support_fraction=.85,cross_section_radius_px=4.,
            minimum_contrast=.018,maximum_observed_width_px=6.5,minimum_observed_width_px=.55)

def sample(field,p):
    p=np.asarray(p,np.float32)
    return cv2.remap(field,p[:,0:1],p[:,1:2],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101).ravel()

def fields(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    return gray,cv2.GaussianBlur(gray,(0,0),3.)

def fit_line(p,limit=.55):
    p=samples(p,.5)
    if length(p)<2:return None
    center=p.mean(0);_,_,v=np.linalg.svd(p-center,full_matrices=False);axis=v[0]
    normal=np.array([-axis[1],axis[0]])
    if np.max(abs((p-center)@normal))>limit:return None
    t=(p-center)@axis
    return dict(center=center,axis=axis,normal=normal,points=np.array([center+t.min()*axis,center+t.max()*axis]))

def supported(gray,bg,p):
    q=samples(p,.18);g=sample(gray,q);contrast=sample(bg,q)-g
    return float(((contrast>.010)|(g<.38)).mean())

def reconstruct_lines(rgb,paths):
    """Merge collinear segments, including T-junction pieces, not parallel rows."""
    gray,bg=fields(rgb)
    originals=[np.asarray(p,float) for p in paths]
    # Align legacy edge-biased points to the measured basin center BEFORE
    # fitting/grouping. Rendering an entire band around its edge inflates ink.
    paths=[width_profile(rgb,p)['points'] for p in originals]
    groups=[dict(points=np.asarray(p,float),ids=[i],line=fit_line(p),fitted=False) for i,p in enumerate(paths)]
    decisions=[]
    while True:
        options=[]
        for i,a in enumerate(groups):
            if a['line'] is None:continue
            fa=a['line']
            for j in range(i+1,len(groups)):
                b=groups[j]
                if b['line'] is None:continue
                fb=b['line'];angle=abs(float(fa['axis']@fb['axis']))
                if angle<math.cos(math.radians(CONFIG['max_line_angle_degrees'])):continue
                # Both endpoint bands must stay on the same center, preventing
                # many small sideward merges drifting onto the next roof line.
                offsets=abs((fb['points']-fa['center'])@fa['normal'])
                if offsets.max()>CONFIG['line_group_residual_px']:continue
                ta=(fa['points']-fa['center'])@fa['axis'];tb=(fb['points']-fa['center'])@fa['axis']
                gap=max(0.,max(ta.min(),tb.min())-min(ta.max(),tb.max()))
                if gap>CONFIG['maximum_supported_gap_px']:continue
                if gap>.3:
                    if ta.max()<tb.min():ends=[ta.max(),tb.min()]
                    else:ends=[tb.max(),ta.min()]
                    bridge=fa['center']+np.asarray(ends)[:,None]*fa['axis']
                    if supported(gray,bg,bridge)<.95:continue
                original=np.concatenate([paths[k] for k in a['ids']+b['ids']])
                centered=original-original.mean(0);_,_,v=np.linalg.svd(centered,full_matrices=False)
                axis=v[0];normal=np.array([-axis[1],axis[0]]);center=original.mean(0)
                error=float(np.max(abs((original-center)@normal)))
                if error>CONFIG['line_group_residual_px']:continue
                t=(original-center)@axis;p=np.array([center+t.min()*axis,center+t.max()*axis])
                ink=supported(gray,bg,p)
                if ink<CONFIG['support_fraction']:continue
                options.append((error+.12*gap,-length(p),i,j,p,error,ink))
        if not options:break
        _,_,i,j,p,error,ink=min(options,key=lambda r:r[:4]);a,b=groups[i],groups[j]
        ids=a['ids']+b['ids'];decisions.append(dict(source_curve_ids=ids,kind='same-axis source-supported straight candidate',
            max_source_point_deviation_px=error,source_ink_support=ink,artist_identity_verified=False))
        groups[i]=dict(points=p,ids=ids,line=fit_line(p),fitted=True);groups.pop(j)
    # Leave curved eaves as curves. Only reuse conservative endpoint joining,
    # with source support and competition checks; no manual template strokes.
    joined,diagnostics=assemble([g['points'] for g in groups],gray,[dict(module=0,uncertain=False) for _ in groups])
    result=[]
    for row in joined:
        indices=row['source_curve_ids'];ids=[i for k in indices for i in groups[k]['ids']]
        result.append(dict(points=row['points'],source_curve_ids=ids,
            straight_replacement=any(groups[k]['fitted'] for k in indices),artist_stroke_identity_verified=False))
    assert sorted(i for r in result for i in r['source_curve_ids'])==list(range(len(paths)))
    return result,dict(straight_group_events=decisions,endpoint_join_diagnostics=diagnostics,
        input_curves=len(paths),output_curves=len(result),all_input_curve_ids_accounted=True,
        source_normal_recentering_limit_px=.75,
        geometry_replacement_not_exact_pixel_preservation=True,uncertain_continuations=diagnostics['pending'])

def width_profile(rgb,path,recenter=True):
    gray,bg=fields(rgb)
    path=np.asarray(path,float)
    step=min(.5,length(path)/8) if np.linalg.norm(path[0]-path[-1])<.1 and length(path)>1e-8 else .5
    q=samples(path,step);n=len(q)
    tang=np.gradient(q,axis=0);tang/=np.maximum(np.linalg.norm(tang,axis=1,keepdims=True),1e-8)
    normals=np.column_stack([-tang[:,1],tang[:,0]])
    offsets=np.linspace(-4.,4.,65);valid=[];widths=[];contrasts=[];neighbor=[];shifts=[]
    for point,normal in zip(q,normals):
        cross=point+offsets[:,None]*normal
        values=sample(gray,cross)
        # Background can differ on the two sides of a colored rock; model a
        # local linear background instead of applying one global dark threshold.
        left=float(np.quantile(values[:8],.8));right=float(np.quantile(values[-8:],.8))
        base=np.linspace(left,right,len(values));signal=base-values
        middle=np.flatnonzero(abs(offsets)<=.75);peak=int(middle[np.argmax(signal[middle])]);height=float(signal[peak])
        lo=hi=peak;threshold=height*.5
        while lo>0 and signal[lo-1]>=threshold:lo-=1
        while hi<len(signal)-1 and signal[hi+1]>=threshold:hi+=1
        inside=lo>0 and hi<len(signal)-1
        if inside:
            l=offsets[lo]-(signal[lo]-threshold)/max(signal[lo]-signal[lo-1],1e-8)*.125
            r=offsets[hi]+(signal[hi]-threshold)/max(signal[hi]-signal[hi+1],1e-8)*.125
            width=float(r-l)
        else:width=float('nan')
        good=bool(height>=CONFIG['minimum_contrast'] and inside and
            CONFIG['minimum_observed_width_px']<=width<=CONFIG['maximum_observed_width_px'])
        # The central line must actually pass near this basin, not along its edge.
        good=good and bool(signal[32]>=height*.45)
        peaks=[k for k in range(1,64) if signal[k]>height*.6 and signal[k]>=signal[k-1] and signal[k]>signal[k+1]]
        neighbor.append(bool(any(abs(offsets[k]-offsets[peak])>1. for k in peaks)))
        valid.append(good);widths.append(width if good else np.nan);contrasts.append(max(0.,height))
        shifts.append(float(np.clip((l+r)/2,-.75,.75)) if good else 0.)
    valid=np.asarray(valid,bool);widths=np.asarray(widths,float)
    if valid.any():
        fill=np.interp(np.arange(n),np.flatnonzero(valid),widths[valid]);fallback='same_curve_neighbor_interpolation'
    else:
        fill=np.ones(n);fallback='unmeasured_1px_display_placeholder'
    fill=median_filter(fill,size=min(5,n if n%2 else max(1,n-1)),mode='nearest')
    corrected=q+np.asarray(shifts)[:,None]*normals if recenter else q.copy()
    if np.linalg.norm(np.asarray(path)[0]-np.asarray(path)[-1])<1e-6:
        corrected[0]=q[0];corrected[-1]=q[-1]
    return dict(points=corrected,original_sample_points=q,normal_shift_px=np.asarray(shifts),
        width_px=fill,observed=valid,contrast=np.asarray(contrasts),
        neighboring_basin=np.asarray(neighbor,bool),fallback=fallback,
        normal_recentering_applied=bool(recenter),
        width_is_image_target_not_brush_measurement=True,pressure_mm=None,
        pixel_resolution_limits_subpixel_width_accuracy=True)

def render_profiles(shape,profiles,scale=4,flatten_width_px=None,relieve_junctions=False):
    """Finite image-space bands only. Junction cap is an unmeasured target."""
    height,width=shape;counts=np.zeros((height*scale,width*scale),np.uint16)
    image=np.full((*counts.shape,3),255,np.uint8);events=[];unknown=np.zeros(counts.shape,np.uint8)
    for index,row in enumerate(profiles):
        q=np.asarray(row['points']);w=np.asarray(row['width_px'])
        if flatten_width_px is not None:w=np.full(len(q),flatten_width_px)
        occupied=[];adjusted=w.copy()
        for k,(p,width_at) in enumerate(zip(q,w)):
            x,y=np.rint(p*scale).astype(int);x=np.clip(x,0,counts.shape[1]-1);y=np.clip(y,0,counts.shape[0]-1)
            hit=counts[y,x]>0;occupied.append(bool(hit))
            if relieve_junctions and hit:
                # Desired geometric width is reduced locally, not converted to
                # pressure or claimed as an actual paper response.
                adjusted[k]*=.70
        mask=np.zeros(counts.shape,np.uint8)
        for k in range(len(q)-1):
            a=q[k]*scale;b=q[k+1]*scale;d=b-a;norm=np.linalg.norm(d)
            if norm<1e-9:continue
            normal=np.array([-d[1],d[0]])/norm
            p=np.array([a+normal*adjusted[k]*scale/2,b+normal*adjusted[k+1]*scale/2,
                        b-normal*adjusted[k+1]*scale/2,a-normal*adjusted[k]*scale/2])
            cv2.fillConvexPoly(mask,np.rint(p).astype(np.int32),1)
        for k,p in enumerate(q):
            pt=tuple(np.rint(p*scale).astype(int));radius=max(1,int(round(adjusted[k]*scale/2)))
            cv2.circle(mask,pt,radius,1,-1)
            if not row['observed'][k]:cv2.circle(unknown,pt,max(1,radius),1,-1)
        own=mask>0;image[own]=0;counts[own]+=1
        events.append(dict(curve=index,overlap_sample_fraction=float(np.mean(occupied)),
            width_target_px=adjusted.tolist(),contact_relief_requested=bool(relieve_junctions and any(occupied)),
            desired_width_to_pressure_mapping=None,actual_deposition_measured=False))
    overlay=image.copy();overlay[unknown>0]=[220,145,45]
    return image,counts,events,overlay
