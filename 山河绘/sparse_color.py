"""Configurable sparse deposits per connected color patch."""
import cv2
import numpy as np
import watercolor as wc


def plan_paths(work, scale, width=1., gap=.6, max_strokes=2,
               min_area_mm2=12., respect_ink=True, color_analysis=None, color_order=None):
    if isinstance(max_strokes, bool) or not isinstance(max_strokes, (int, np.integer)) or max_strokes < 1:
        raise ValueError('max_strokes must be a positive integer')
    if not np.isfinite([scale,width,gap,min_area_mm2]).all() or scale<=0 or not .3<=width<=10 or not 0<=gap<=10 or min_area_mm2<=0:
        raise ValueError('Invalid sparse color parameters')
    ink, fields=wc.analyze(work,scale) if color_analysis is None else color_analysis
    flow=wc.flow_field(work,scale)
    strokes=[]; regions=[]
    for color in (wc.COLOR_ORDER if color_order is None else color_order):
        evidence,strength=fields[color]
        # A patch is one connected color area BEFORE ink-clearance erosion.
        # Never subdivide it into tiles or tone bands to evade the stroke cap.
        radius=max(1,int(round(1.2/scale)))
        merged=cv2.morphologyEx(evidence.astype(np.uint8),cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2))
        count,labels,stats,_=cv2.connectedComponentsWithStats(merged,8)
        for label in range(1,count):
            area=float(stats[label,cv2.CC_STAT_AREA]*scale*scale)
            if area<min_area_mm2: continue
            region=labels==label
            region_id='%s-patch-%03d'%(color,label)
            occupied=np.zeros(region.shape,np.uint8)
            picked=[]
            for brush_width in sorted({width,max(.3,width*.55)},reverse=True):
                brush_radius=int(np.ceil(brush_width/(2*scale)))
                centers=cv2.erode(region.astype(np.uint8),cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,(2*brush_radius+1,)*2),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
                if respect_ink:
                    ink_radius=brush_radius+int(np.ceil(gap/scale))
                    centers &= cv2.dilate(ink.astype(np.uint8),cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,(2*ink_radius+1,)*2))==0
                distance=cv2.distanceTransform(centers.astype(np.uint8),cv2.DIST_L2,5)
                pitch=max(2,int(round(2./scale)))
                seeds=[(x,y) for y in range(pitch//2,region.shape[0],pitch)
                       for x in range(pitch//2,region.shape[1],pitch) if centers[y,x]]
                seeds.sort(key=lambda p: -float(distance[p[1],p[0]]*(.2+strength[p[1],p[0]])))
                field=flow if color!='red' else np.broadcast_to(np.array([1.,0.]),(*region.shape,2))
                while len(picked)<max_strokes:
                    valid=centers&(occupied==0)
                    candidates=[]
                    for x,y in [p for p in seeds if valid[p[1],p[0]]][:80]:
                        points=wc.trace((x,y),valid,field,max(.75,.5/scale),45./scale)
                        if len(points)<3: continue
                        length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()*scale)
                        if length<3.: continue
                        q=np.rint(points).astype(int)
                        score=length*(.3+float(np.mean(strength[q[:,1],q[:,0]])))
                        candidates.append((score,length,points))
                    if not candidates: break
                    _,length,points=max(candidates,key=lambda c:c[0])
                    alpha,widths,recipe=wc.stroke_profile(points,strength,brush_width)
                    picked.append(dict(color=color,region_id=region_id,recipe=recipe,
                        layer=0,points=points.tolist(),target_opacity=alpha,target_width_mm=widths,
                        brush_width_mm_estimate=brush_width,length_mm=length,
                        path_style='sparse_slope',pressure_mm=None,pigment_recipe_calibrated=False))
                    # Reserve an empty corridor between deposits.
                    cv2.polylines(occupied,[np.rint(points).astype(np.int32)],False,255,
                        max(1,int(np.ceil(2*max(2.5,2*width)/scale))))
                if len(picked)>=max_strokes: break
            strokes.extend(picked)
            x,y,w,h=map(int,stats[label,:4])
            regions.append(dict(id=region_id,color=color,area_mm2=area,bounds_px=[x,y,w,h],
                stroke_count=len(picked),status='planned' if picked else 'too_narrow_skip',
                max_strokes=max_strokes))
    for i,s in enumerate(strokes): s['id']='sparse-color-%04d'%i
    return strokes,regions


def schedule(strokes):
    """One deposition pass; no extra tonal overpainting of the same patch."""
    steps=[]
    for color in wc.COLOR_ORDER:
        ids=[s['id'] for s in strokes if s['color']==color]
        if not ids: continue
        if steps: steps.append(dict(kind='manual_wash_wipe_and_verify_clean',to_color=color))
        steps.extend([dict(kind='manual_load_light_pigment',color=color,water_ratio=None),
                      dict(kind='paint_sparse_regions_once',color=color,stroke_ids=ids)])
    return steps
