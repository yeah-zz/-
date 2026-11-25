"""Offline per-stroke swept-footprint accounting. No robot or learned ink physics."""
import math
import numpy as np
import cv2


def footprint_audit(paths_mm, width_mm, canvas_mm=(186.8, 186.8), ppm=6., start_factor=1.5):
    """Count each stroke once per pixel, including a hypothetical start footprint.

    Counts are distinct-stroke exposures, not darkness, force, or ink volume.
    Start_factor is an uncalibrated sensitivity assumption, not a fitted value.
    The returned array has a small margin so boundary footprints are not clipped.
    """
    if width_mm <= 0 or ppm <= 0 or start_factor < 1:
        raise ValueError('Positive width/resolution and start_factor >= 1 required')
    thickness=max(1, int(round(width_mm*ppm)))
    pad=int(math.ceil(width_mm*start_factor*ppm))+3
    shape=(int(math.ceil(canvas_mm[1]*ppm))+2*pad+1,
           int(math.ceil(canvas_mm[0]*ppm))+2*pad+1)
    counts=np.zeros(shape,np.uint16);starts=np.zeros(shape,np.uint16);rows=[]
    for ident, points in enumerate(paths_mm):
        p=np.asarray(points,float)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all():
            raise ValueError('Each path must contain >=2 finite XY points')
        if (p.min(axis=0)<-.01).any() or (p.max(axis=0)>np.asarray(canvas_mm)+.01).any():
            raise ValueError('Path outside declared canvas')
        q=np.rint(p*ppm).astype(np.int32)+pad
        margin=int(math.ceil(width_mm*start_factor*ppm))+2
        lo=np.maximum(q.min(axis=0)-margin,0)
        hi=np.minimum(q.max(axis=0)+margin+1,np.array(shape[::-1]))
        sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]))
        mask=np.zeros((hi-lo)[::-1],np.uint8);local=q-lo
        cv2.polylines(mask,[local],False,1,thickness,cv2.LINE_8)
        radius=max(1,int(round(width_mm*ppm/2)))
        for endpoint in [local[0],local[-1]]:
            cv2.circle(mask,tuple(endpoint),radius,1,-1)
        cv2.circle(mask,tuple(local[0]),max(radius,int(round(radius*start_factor))),1,-1)
        own=mask.astype(bool);before=counts[sl]
        area=int(own.sum());new=int(np.count_nonzero(own&(before==0)))
        prior=int(np.count_nonzero(own&(before>0)))
        rows.append(dict(path_id=ident,length_mm=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),
            footprint_mm2=area/ppm**2,new_coverage_mm2=new/ppm**2,
            previously_covered_fraction=prior/max(1,area),
            review_only_high_overlap=bool(prior/max(1,area)>=.8),
            start_xy_mm=p[0].tolist()))
        before[own]+=1
        start_mask=np.zeros_like(mask)
        cv2.circle(start_mask,tuple(local[0]),max(radius,int(round(radius*start_factor))),1,-1)
        starts[sl][start_mask.astype(bool)]+=1
    occupied=counts>0;area=int(occupied.sum());exposures=int(counts.sum())
    summary=dict(paths=len(rows),hypothetical_width_mm=width_mm,assumed_start_diameter_factor=start_factor,
        raster_pixels_per_mm=ppm,union_coverage_mm2=area/ppm**2,
        repeated_coverage_area_mm2=int((counts>=2).sum())/ppm**2,
        repeated_fraction_of_covered_area=float((counts>=2).sum()/max(1,area)),
        excess_exposure_fraction=float((exposures-area)/max(1,exposures)),
        max_distinct_strokes_at_pixel=int(counts.max()),
        max_start_footprints_at_pixel=int(starts.max()),
        high_overlap_review_units=sum(r['review_only_high_overlap'] for r in rows),
        is_measured_brush_width=False,is_ink_simulation=False,ready_for_robot=False,
        no_path_deleted=True,margin_pixels=pad,canvas_mm=list(canvas_mm))
    return summary,rows,counts,starts


def exposure_rgb(counts):
    out=np.full((*counts.shape,3),255,np.uint8)
    out[counts==1]=(104,151,164)
    out[counts==2]=(244,180,72)
    out[(counts>=3)&(counts<5)]=(227,105,54)
    out[counts>=5]=(156,42,65)
    return out
