"""Simple blue-green rock geometry and sparse weak contour targets."""
import cv2
import numpy as np


def make_simple_sample(seed,size=320):
    rng=np.random.default_rng(seed);s=size
    yy,xx=np.mgrid[:s,:s].astype(np.float32)
    paper=np.array([rng.uniform(175,238),rng.uniform(150,216),rng.uniform(95,175)])
    rgb=np.broadcast_to(paper,(s,s,3)).copy()
    weave=(np.sin(xx*2.1)+np.cos(yy*1.8))*rng.uniform(.5,2)
    rgb+=weave[:,:,None]+rng.normal(0,rng.uniform(.6,2),(s,s,1))
    rgb=np.clip(rgb,0,255).astype(np.uint8);target=np.zeros((2,s,s),np.uint8)
    if rng.random()<.06:return rgb,target.astype(np.float32)/255.
    ink=tuple(int(v) for v in [rng.uniform(40,85),rng.uniform(48,88),rng.uniform(30,65)])
    count=1 if rng.random()<.8 else 2
    for number in range(count):
        cx=s*(.5 if count==1 else .32+number*.36)+rng.uniform(-.06,.06)*s
        cy=rng.uniform(.13,.29)*s;half=rng.uniform(.23,.39)*s/(1 if count==1 else 1.4)
        bottom=rng.uniform(.78,.93)*s
        x=np.linspace(cx-half,cx+half,150);u=(x-cx)/half
        y=cy+(bottom-cy)*(1-np.exp(-2.3*u*u))/(1-np.exp(-2.3))
        y+=np.sin(u*8+rng.random()*4)*rng.uniform(1,3)
        ridge=np.column_stack([x,y])
        polygon=np.vstack([ridge,[x[-1],bottom],[x[0],bottom]]).round().astype(np.int32)
        mask=np.zeros((s,s),np.uint8);cv2.fillPoly(mask,[polygon],255)
        ratio=np.clip((yy-cy)/(bottom-cy),0,1)[:,:,None]
        blue=np.array([rng.uniform(30,90),rng.uniform(110,170),rng.uniform(145,205)])
        green=np.array([rng.uniform(70,125),rng.uniform(130,188),rng.uniform(90,145)])
        tone=blue*(1-ratio)+green*ratio
        alpha=(mask/255.)[:,:,None]*rng.uniform(.65,.95)
        rgb[:]=np.clip(rgb*(1-alpha)+tone*alpha,0,255).astype(np.uint8)
        targets_keep=1-alpha[:,:,0]
        target[:]=np.rint(target*targets_keep[None]).astype(np.uint8)
        ridge=np.rint(ridge).astype(np.int32)
        cv2.polylines(rgb,[ridge],False,ink,2,cv2.LINE_AA);cv2.polylines(target[0],[ridge],False,255,2,cv2.LINE_AA)
        for _ in range(int(rng.integers(1,4))):
            t=np.linspace(0,1,65)
            start=np.array([cx+rng.uniform(-.35,.35)*half,cy+rng.uniform(.10,.30)*(bottom-cy)])
            end=np.array([start[0]+rng.uniform(-.25,.25)*half,bottom-rng.uniform(.06,.20)*s])
            p=start[None]+(end-start)[None]*t[:,None];p[:,0]+=np.sin(t*4)*rng.uniform(2,9)
            p=np.rint(p).astype(np.int32)
            cv2.polylines(rgb,[p],False,ink,1,cv2.LINE_AA);cv2.polylines(target[0],[p],False,255,2,cv2.LINE_AA)
        for _ in range(int(rng.integers(3,11))):
            x0=int(rng.integers(0,s));y0=int(rng.integers(0,s))
            if not mask[y0,x0]:continue
            p=np.array([[x0,y0],[x0+rng.integers(-4,5),y0+rng.integers(5,15)]],np.int32)
            cv2.polylines(rgb,[p],False,ink,1,cv2.LINE_AA);cv2.polylines(target[1],[p],False,255,2,cv2.LINE_AA)
    # Empty paper, seals and sparse waterlines stay outside the primary target.
    if rng.random()<.5:
        for i in range(3):
            x=int(rng.uniform(.02,.15)*s);y=int((.2+i*.08)*s)
            cv2.putText(rgb,'A',(x,y),cv2.FONT_HERSHEY_SIMPLEX,.3,ink,1,cv2.LINE_AA)
    if rng.random()<.35:cv2.rectangle(rgb,(s-35,20),(s-18,42),(150,48,33),1)
    return rgb,target.astype(np.float32)/255.


def simple_weak_targets(rgb):
    """Only supported outer contours and up to three long folds, not all edges."""
    h,w=rgb.shape[:2]
    smooth=cv2.bilateralFilter(rgb,9,38,6)
    hsv=cv2.cvtColor(smooth,cv2.COLOR_RGB2HSV)
    color=(((hsv[:,:,0]>34)&(hsv[:,:,0]<135)&(hsv[:,:,1]>42)&(hsv[:,:,2]>40))).astype(np.uint8)*255
    color=cv2.morphologyEx(color,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
    color=cv2.morphologyEx(color,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    n,cc,stats,_=cv2.connectedComponentsWithStats(color,8)
    keep=np.zeros(n,np.uint8);keep[1:]=(stats[1:,cv2.CC_STAT_AREA]>max(160,.018*h*w))
    color=keep[cc]*255
    lab=cv2.cvtColor(cv2.GaussianBlur(smooth,(0,0),1.2),cv2.COLOR_RGB2LAB).astype(np.float32)
    gradient=np.sqrt((np.square(cv2.Sobel(lab,cv2.CV_32F,1,0)/8)+np.square(cv2.Sobel(lab,cv2.CV_32F,0,1)/8)).sum(2))
    primary=np.zeros((h,w),np.uint8)
    contours,_=cv2.findContours(color,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if cv2.arcLength(c,True)<70:continue
        c=cv2.approxPolyDP(c,2.5,True)
        cv2.polylines(primary,[c],True,255,2,cv2.LINE_AA)
    support=cv2.dilate((gradient>=2.5).astype(np.uint8),np.ones((7,7),np.uint8))>0
    primary[~support]=0
    primary=cv2.morphologyEx(primary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    gray=cv2.cvtColor(smooth,cv2.COLOR_RGB2GRAY)
    coarse=cv2.GaussianBlur(gray,(0,0),1.4)
    dark=cv2.morphologyEx(coarse,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
    # Curved line extraction is deliberately limited; short marks go to detail.
    from original_d_tracer import LandscapeBrushGenerator as Trace
    candidate=((dark>11)&(color>0)).astype(np.uint8)*255
    skeleton=Trace._thin_binary(candidate)
    folds=[]
    for path in Trace._trace_skeleton_paths(skeleton):
        p=np.asarray(path,np.int32);span=p.max(0)-p.min(0)
        length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        if length<55 or np.linalg.norm(span)<38 or length>np.linalg.norm(span)*2.5:continue
        if span[0]>4*max(1,span[1]):continue
        folds.append((length,p))
    for _,p in sorted(folds,key=lambda z:-z[0])[:3]:cv2.polylines(primary,[p],False,220,2,cv2.LINE_AA)
    detail=((dark>10)&(color>0)&(primary<80)).astype(np.uint8)*255
    targets=np.stack([primary,detail]).astype(np.float32)/255.
    weights=np.full_like(targets,.08);weights[targets>.3]=.6
    background=(color==0)&(dark<5);weights[:,background]=.55
    # Fine texture is explicitly secondary for this simple contour task.
    weights[0,detail>0]=.25
    red=(((hsv[:,:,0]<12)|(hsv[:,:,0]>170))&(hsv[:,:,1]>100)).astype(np.uint8)
    red=cv2.dilate(red,np.ones((5,5),np.uint8))>0
    targets[:,red]=0;weights[:,red]=.75
    border=5;weights[:,:border,:]=0;weights[:,-border:,:]=0;weights[:,:,:border]=0;weights[:,:,-border:]=0
    return targets,weights
