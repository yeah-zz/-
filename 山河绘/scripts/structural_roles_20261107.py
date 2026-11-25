# -*- coding: utf-8 -*-
"""Source/object-context role inference. Explicit rules, not trained cun recognition."""
import numpy as np,cv2
from module_stroke_core_20261104 import samples,length
from source_stroke_width_20261104 import sample

def wood_support(rgb,q):
    a=rgb.astype(np.float32);brown=(a[:,:,0]-a[:,:,1]>3)&(a[:,:,1]-a[:,:,2]>4)&(a[:,:,0]-a[:,:,2]>12)&(a.mean(2)<225)
    # Dark outline can surround brown wood: inspect the immediate two flanks.
    field=cv2.dilate(brown.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(np.float32)
    return float((sample(field,q)>.5).mean())

def classify(rows,modules,masks,rgb,profiles):
    result=[];features=[]
    for i,r in enumerate(rows):
        p=np.asarray(r['points'],float);q=samples(p,.4);mid=r['module'];mask=masks[mid].astype(np.uint8)
        # Inside AND outside distance avoids treating all outside points as boundary.
        inside=cv2.distanceTransform(mask,cv2.DIST_L2,5);outside=cv2.distanceTransform(1-mask,cv2.DIST_L2,5)
        d=sample(inside+outside,q);boundary=float((d<3).mean());in_object=float((sample(mask.astype(np.float32),q)>.5).mean())
        pp=profiles[i];v=np.asarray(pp['observed']);ww=np.asarray(pp['width_px'])
        width=float(np.median(ww[v])) if v.any() else None
        features.append(dict(arc=length(p),boundary=boundary,in_object=in_object,end_boundary=bool(d[0]<3 and d[-1]<3),
            extent=float(np.linalg.norm(np.ptp(q,axis=0))),center=q.mean(0).tolist(),wood=wood_support(rgb,q),
            width=width,width_known=bool(v.mean()>.5)))
    for mid,m in enumerate(modules):
        ids=[i for i,r in enumerate(rows) if r['module']==mid];kind=m['kind'];kind_features=[features[i] for i in ids]
        if not ids:continue
        if kind=='tree':
            allpoints=np.concatenate([np.asarray(rows[i]['points']) for i in ids]);low,high=np.quantile(allpoints[:,1],[.15,.85]);span=max(high-low,1.)
            for i in ids:
                f=features[i];below=f['center'][1]>low+.48*span
                trunk=f['wood']>=.50 and below and f['arc']>=8 and (f['width'] or 0)>=1.1
                branch=f['wood']>=.28 and f['arc']>=4
                role='trunk' if trunk else 'branch' if branch else 'foliage'
                uncertainty=not trunk and not branch and (f['arc']>14 or f['width'] is None)
                result.append(dict(curve=i,phase=0 if trunk else 1 if branch else 2,role=role,
                    reason='brown-wood flanks + lower tree structure + observed width' if trunk else 'source-supported woody branch' if branch else 'no reliable wood evidence; leaf/needle or uncertain candidate',
                    uncertain=uncertainty,features=f))
        elif kind=='rock':
            bbox=m['bbox'];extent=max(1,np.linalg.norm(np.array(bbox[2:])-bbox[:2]))
            # A SAM color region can stop below a real contour. Supplement its
            # boundary with the envelope of the retained strokes of this object.
            allq=np.concatenate([np.asarray(rows[i]['points'],np.float32) for i in ids])
            envelope=np.zeros(rgb.shape[:2],np.uint8)
            if len(allq)>=3:cv2.fillConvexPoly(envelope,cv2.convexHull(np.rint(allq).astype(np.int32)),1)
            hull_dist=cv2.distanceTransform(envelope,cv2.DIST_L2,5)+cv2.distanceTransform(1-envelope,cv2.DIST_L2,5)
            for i in ids:
                f=features[i];q=samples(np.asarray(rows[i]['points']),.4)
                hull_boundary=float((sample(hull_dist,q)<2.5).mean());f['stroke_envelope_boundary']=hull_boundary
                outline=(f['boundary']>=.45 and f['in_object']>=.35) or hull_boundary>=.5
                structural_fold=f['end_boundary'] and f['extent']>=extent*.35 and f['boundary']>=.18
                if outline or structural_fold:phase=0;role='main_contour' if outline else 'structural_ridge'
                else:phase=1 if f['arc']>=3 else 2;role='cun_candidate' if phase==1 else 'detail'
                result.append(dict(curve=i,phase=phase,role=role,uncertain=(f['in_object']<.5 or (.25<f['boundary']<.45)),features=f,
                    reason='whole-object outer boundary' if outline else 'spanning fold attached to two contour ends' if structural_fold else 'interior stroke, not promoted solely for length'))
        else:
            for i in ids:
                f=features[i];outline=f['boundary']>=.40 or (kind=='architecture' and f['arc']>=14)
                phase=0 if outline else 1 if f['arc']>=4 else 2
                result.append(dict(curve=i,phase=phase,role='structure' if phase==0 else 'interior' if phase==1 else 'detail',uncertain=f['in_object']<.5,features=f,reason='object boundary/architectural support'))
    return sorted(result,key=lambda r:r['curve'])
