"""Procedural blue-green paintings with known structure/detail masks.

Known raster geometry is genuine synthetic supervision, never real-art GT.
"""
import cv2
import numpy as np


def make_sample(seed,size=320):
    rng=np.random.default_rng(seed)
    s=size
    yy,xx=np.mgrid[:s,:s].astype(np.float32)
    base=np.array([rng.uniform(150,230),rng.uniform(130,205),rng.uniform(65,150)])
    paper=np.broadcast_to(base,(s,s,3)).copy().astype(np.float32)
    grain=rng.normal(0,rng.uniform(1,5),(s,s))
    weave=(np.sin(xx*rng.uniform(1.5,3.2))+np.cos(yy*rng.uniform(1.5,3.2)))*rng.uniform(.4,2.4)
    stains=cv2.GaussianBlur(rng.normal(0,1,(s,s)).astype(np.float32),(0,0),rng.uniform(10,30))*rng.uniform(30,100)
    paper+= (grain+weave+stains)[:,:,None]
    image=np.clip(paper,0,255).astype(np.uint8)
    targets=np.zeros((2,s,s),np.uint8)
    ink=tuple(int(v) for v in np.array([50,55,35])*rng.uniform(.55,1.9))

    def stroke(p,kind=0,width=None,color=None):
        p=np.rint(p).astype(np.int32)
        width=int(width or rng.choice([1,1,2,2,3]))
        cv2.polylines(image,[p],False,color or ink,width,cv2.LINE_AA)
        cv2.polylines(targets[kind],[p],False,255,max(2,width),cv2.LINE_AA)

    # Layered mountains: color fills have texture but only explicit structural
    # contours are the primary target. Internal fine brushwork is auxiliary.
    for _ in range(int(rng.integers(2,6))):
        cx=rng.uniform(-.05,1.05)*s
        cy=rng.uniform(.10,.62)*s
        spread=rng.uniform(.12,.5)*s
        height=rng.uniform(.18,.65)*s
        x=np.linspace(cx-spread,cx+spread,90)
        t=(x-cx)/spread
        y=cy+height*(1-np.exp(-2.3*t*t))
        y+=np.sin(t*rng.uniform(5,12))*rng.uniform(.5,3.)
        ridge=np.column_stack([x,y])
        polygon=np.vstack([ridge,[x[-1],s],[x[0],s]]).round().astype(np.int32)
        mask=np.zeros((s,s),np.uint8);cv2.fillPoly(mask,[polygon],255)
        blue=np.array([rng.uniform(30,100),rng.uniform(110,185),rng.uniform(140,215)])
        green=np.array([rng.uniform(70,145),rng.uniform(140,195),rng.uniform(100,170)])
        alpha=rng.uniform(.35,.85)*(mask.astype(float)/255.)
        gradient=np.clip((yy-cy)/max(height,1.),0,1.)[:,:,None]
        tone=blue[None,None,:]*(1-gradient)+green[None,None,:]*gradient
        mottles=cv2.GaussianBlur(rng.normal(0,1,(s,s)).astype(np.float32),(0,0),rng.uniform(1,5))
        tone=tone+mottles[:,:,None]*rng.uniform(10,40)
        image[:]=np.clip(image*(1-alpha[:,:,None])+tone*alpha[:,:,None],0,255).astype(np.uint8)
        targets[:]=np.rint(targets.astype(float)*(1-alpha[None,:,:])).astype(np.uint8)
        stroke(ridge,0,width=int(rng.choice([1,2,3])),color=tuple(int(v) for v in np.array(ink)*rng.uniform(.8,1.7)))
        for _ in range(int(rng.integers(1,4))):
            start=rng.uniform(-.6,.6)
            idx=int(np.clip((start+1)*.5*89,0,89))
            a=ridge[idx]
            end=a+[rng.uniform(-.15,.15)*s,rng.uniform(.12,.5)*s]
            t2=np.linspace(0,1,35)
            p=a+(end-a)*t2[:,None]
            p[:,0]+=np.sin(t2*4)*rng.uniform(3,15)
            stroke(p,0,width=1)
        # Small texture marks explicitly excluded from primary structure.
        for _ in range(int(rng.integers(8,40))):
            px=int(rng.integers(0,s));py=int(rng.integers(0,s))
            if mask[py,px]:stroke([[px,py],[px+rng.uniform(-5,5),py+rng.uniform(3,12)]],1,width=1)
    # Sparse trunks/branches, with dot foliage auxiliary only.
    for _ in range(int(rng.integers(2,9))):
        x=rng.uniform(.02,.98)*s;y=rng.uniform(.5,.98)*s;ht=rng.uniform(.05,.27)*s
        trunk=np.array([[x,y],[x+rng.uniform(-8,8),y-ht*.6],[x+rng.uniform(-10,10),y-ht]])
        stroke(trunk,0,width=int(rng.choice([1,2,3])))
        for z in (.4,.65,.8):
            bx=x;by=y-ht*z
            stroke([[bx,by],[bx+rng.uniform(-.5,.5)*ht,by-rng.uniform(.1,.3)*ht]],0,width=1)
        for _ in range(int(rng.integers(10,35))):
            px=int(np.clip(x+rng.normal(0,ht*.25),0,s-1));py=int(np.clip(y-ht+rng.normal(0,ht*.18),0,s-1))
            cv2.circle(image,(px,py),int(rng.integers(1,3)),ink,-1,cv2.LINE_AA)
            cv2.circle(targets[1],(px,py),2,255,-1)
    # Fine-line architecture/bridges.
    if rng.random()<.8:
        x=rng.uniform(.1,.7)*s;y=rng.uniform(.55,.9)*s;ww=rng.uniform(.10,.25)*s;hh=ww*rng.uniform(.35,.7)
        stroke([[x-ww*.12,y-hh],[x+ww*.5,y-hh*1.6],[x+ww*1.12,y-hh]],0,width=1)
        stroke([[x,y-hh],[x,y],[x+ww,y],[x+ww,y-hh]],0,width=1)
        stroke([[x+ww*.5,y-hh*1.5],[x+ww*.5,y-hh]],0,width=1)
        stroke([[x-ww*.1,y+3],[x+ww*1.1,y+3]],0,width=1)
    # Water ripples, textile lines, paper cracks and text-like distractors.
    for _ in range(int(rng.integers(8,30))):
        x=rng.uniform(0,s);y=rng.uniform(.7,1)*s
        stroke([[x,y],[x+rng.uniform(4,35),y+rng.uniform(-1,1)]],1,width=1,color=tuple(int(v) for v in base*.65))
    if rng.random()<.5:
        x=int(rng.uniform(.65,.94)*s)
        for row in range(int(rng.integers(3,8))):
            cv2.putText(image,chr(int(rng.integers(65,91))),(x,15+row*12),cv2.FONT_HERSHEY_SIMPLEX,.27,ink,1,cv2.LINE_AA)
            targets[:,max(0,5+row*12):min(s,20+row*12),max(0,x-2):min(s,x+15)]=0
    if rng.random()<.5:
        pos=(int(rng.uniform(.05,.9)*s),int(rng.uniform(.02,.25)*s))
        cv2.rectangle(image,pos,(pos[0]+15,pos[1]+22),(145,55,35),1,cv2.LINE_AA)
        targets[:,max(0,pos[1]-2):min(s,pos[1]+25),max(0,pos[0]-2):min(s,pos[0]+18)]=0
    image=np.clip(image.astype(float)+rng.normal(0,1.2,image.shape),0,255).astype(np.uint8)
    target=targets.astype(np.float32)/255.
    return image,target
