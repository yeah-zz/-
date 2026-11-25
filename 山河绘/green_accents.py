"""Sparse short dabs on dark green vegetation, not pale mountain washes."""
import cv2
import numpy as np


def plan_accents(work, scale, ink_strokes=(), max_strokes=180):
    b,g,r=cv2.split(work.astype(np.float32))
    gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
    mask=((g-r>4)&(g-b>2)&(gray>55)&(gray<155)).astype(np.uint8)
    width=.7
    ink=np.zeros_like(mask)
    for s in ink_strokes:
        cv2.polylines(ink,[np.rint(s['points']).astype(np.int32)],False,1,
                      max(1,round((s['width_mm']+width)/scale)))
    mask[ink>0]=0
    radius=max(1,round(.20/scale))
    mask=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2))
    edge=int(np.ceil(20/scale));mask[:edge]=0;mask[-edge:]=0;mask[:,:edge]=0;mask[:,-edge:]=0
    pitch=max(2,round(1.8/scale));half=.2/scale
    strokes=[]
    for y in range(pitch//2,mask.shape[0],pitch):
        for x in range(pitch//2,mask.shape[1],pitch):
            if not mask[y,x]:continue
            points=[[x-half,y],[x,y],[x+half,y]]
            strokes.append(dict(id='green-accent-%04d'%len(strokes),color='green',
                region_id='green-vegetation',points=points,recipe='accent',layer=0,
                target_opacity=[.60,.70,.60],target_width_mm=[width]*3,
                brush_width_mm_estimate=width,length_mm=.4,path_style='green_short_dab',
                pressure_mm=None,pigment_recipe_calibrated=False))
            if len(strokes)>=max_strokes:return strokes
    return strokes
