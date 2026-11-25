"""Fast confidence-weighted structural proposals. These are NOT gold labels."""
import cv2
import numpy as np


def weak_targets(rgb):
    h,w=rgb.shape[:2]
    scale=max(h,w)/1200.
    bgr=rgb[:,:,::-1].copy()
    smooth=cv2.bilateralFilter(bgr,9,30,6)
    hsv=cv2.cvtColor(smooth,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(smooth,cv2.COLOR_BGR2GRAY)
    color=((hsv[:,:,0]>34)&(hsv[:,:,0]<135)&(hsv[:,:,1]>38)&(hsv[:,:,2]>40)).astype(np.uint8)*255
    color=cv2.morphologyEx(color,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    color=cv2.morphologyEx(color,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    n,lab,stats,_=cv2.connectedComponentsWithStats(color,8)
    keep=np.zeros(n,np.uint8)
    keep[1:]=stats[1:,cv2.CC_STAT_AREA]>max(100,500*scale*scale)
    color=keep[lab]*255
    contours,_=cv2.findContours(color,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    structure=np.zeros((h,w),np.uint8)
    for c in contours:
        if cv2.arcLength(c,True)<max(30,50*scale):continue
        p=cv2.approxPolyDP(c,max(1.,1.5*scale),True)
        cv2.drawContours(structure,[p],-1,255,2,cv2.LINE_AA)
    coarse=cv2.GaussianBlur(gray,(0,0),max(.9,1.2*scale))
    local=cv2.morphologyEx(coarse,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
    lab_float=cv2.cvtColor(cv2.GaussianBlur(smooth,(0,0),1.1),cv2.COLOR_BGR2LAB).astype(np.float32)
    dx=cv2.Sobel(lab_float,cv2.CV_32F,1,0,ksize=3)/8.
    dy=cv2.Sobel(lab_float,cv2.CV_32F,0,1,ksize=3)/8.
    edge_strength=np.sqrt((dx*dx+dy*dy).sum(axis=2))
    # A threshold in a broad color fade is not evidence of a painted line.
    supported=(edge_strength>2.2)|(local>8)
    structure[~supported]=0
    detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected=detector.detect(coarse)[0]
    if detected is not None:
        for line in detected[:,0,:]:
            x0,y0,x1,y1=map(float,line)
            length=np.hypot(x1-x0,y1-y0)
            if length<max(14,22*scale):continue
            xs=np.clip(np.rint(np.linspace(x0,x1,24)).astype(int),0,w-1)
            ys=np.clip(np.rint(np.linspace(y0,y1,24)).astype(int),0,h-1)
            normal=np.array([-(y1-y0),x1-x0])/max(length,1.)
            qx=np.clip(np.rint(xs+normal[0]*3).astype(int),0,w-1)
            qy=np.clip(np.rint(ys+normal[1]*3).astype(int),0,h-1)
            context=float(np.mean(color[ys,xs]>0))
            dark=float(np.mean(gray[ys,xs]<125))
            contrast=float(np.mean(np.maximum(local[ys,xs],local[qy,qx])))
            if contrast<8 or (context<.25 and dark<.4):continue
            cv2.line(structure,(round(x0),round(y0)),(round(x1),round(y1)),255,2,cv2.LINE_AA)
    fine=cv2.morphologyEx(cv2.GaussianBlur(gray,(0,0),.7),cv2.MORPH_BLACKHAT,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    detail=((fine>18)&(gray<175)&((color>0)|(gray<110))).astype(np.uint8)*255
    targets=np.stack([structure,detail],axis=0).astype(np.float32)/255.
    weights=np.full_like(targets,.18)
    weights[targets>.3]=.55
    # High-frequency uncolored silk/paper is a useful background constraint.
    background=(color==0)&(fine<6)
    weights[:,background]=.45
    # Seals: exclude the red pictorial lettering rather than label it as ink.
    red=((hsv[:,:,0]<12)|(hsv[:,:,0]>170))&(hsv[:,:,1]>95)
    red=cv2.dilate(red.astype(np.uint8),np.ones((5,5),np.uint8))>0
    targets[:,red]=0;weights[:,red]=.65
    return targets,weights,color


def annotation_regions(record_id,shape,targets,weights):
    h,w=shape[:2]
    # Explicit, coarse content exclusions; no structural paths are hand traced.
    text_boxes={
      'user_qianli_01':[(.32,0,1.,.43)],
      'user_qianli_03':[(.065,.02,.14,.25),(.29,0,.46,.12),(.80,0,.98,.13)],
      'user_qianli_07':[(.65,0,.74,.07)],
    }
    for a,b,c,d in text_boxes.get(record_id,[]):
        ys=slice(round(b*h),round(d*h));xs=slice(round(a*w),round(c*w))
        targets[:,ys,xs]=0;weights[:,ys,xs]=1.
    if record_id=='user_qianli_04':
        # Unknown scene under social watermark: ignore, not invented negatives.
        weights[:,round(.925*h):,:]=0
    return targets,weights
