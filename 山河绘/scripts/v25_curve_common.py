# -*- coding: utf-8 -*-
"""Reference-free runtime helpers for local direct curve proposals."""
from rock_v15_common import LAB,read,jsave,digest,FONT,SMALL
import v21_stroke_learning as v21
import numpy as np,cv2,math
from pathlib import Path

ROOT=LAB/'continuous_curves_v25_20261026'
SIZE=96
POINTS=17
SPAN=32.

def axes_from_image(rgb):
    features,alpha,_=v21.v16.photometry(rgb);gx=cv2.Scharr(alpha,cv2.CV_32F,1,0);gy=cv2.Scharr(alpha,cv2.CV_32F,0,1);a=cv2.GaussianBlur(gx*gx,(0,0),3);b=cv2.GaussianBlur(gy*gy,(0,0),3);c=cv2.GaussianBlur(gx*gy,(0,0),3);theta=.5*np.arctan2(2*c,a-b)+math.pi/2
    return features,alpha,theta

def automatic_seeds(pred,max_count=180):
    center,ink=pred[:2];h,w=center.shape;theta=.5*np.arctan2(pred[3],pred[2]);yy,xx=np.mgrid[:h,:w].astype(np.float32);nx=-np.sin(theta);ny=np.cos(theta);g=cv2.GaussianBlur(center,(0,0),.65);a=cv2.remap(g,xx+1.5*nx,yy+1.5*ny,cv2.INTER_LINEAR);b=cv2.remap(g,xx-1.5*nx,yy-1.5*ny,cv2.INTER_LINEAR);candidate=(center>=.65)&(ink>=.5)&(g>=a)&(g>=b);candidate[:4]=False;candidate[-4:]=False;candidate[:,:4]=False;candidate[:,-4:]=False;options=[]
    for y in range(0,h,10):
        for x in range(0,w,10):
            score=center[y:y+10,x:x+10]*candidate[y:y+10,x:x+10]
            if not score.size or score.max()<.65:continue
            i,j=np.unravel_index(score.argmax(),score.shape);options.append((float(score[i,j]),np.array([x+j,y+i],np.float32)))
    out=[]
    for score,p in sorted(options,key=lambda z:-z[0]):
        if any(np.linalg.norm(p-q)<9 for q in out):continue
        out.append(p)
        if len(out)>=max_count:break
    return np.asarray(out,np.float32).reshape(-1,2)

def extract_patch(features,seed,theta):
    yy,xx=np.mgrid[:SIZE,:SIZE].astype(np.float32);x=xx-(SIZE-1)*.5;y=yy-(SIZE-1)*.5;c=math.cos(theta);s=math.sin(theta);mx=seed[0]+c*x-s*y;my=seed[1]+s*x+c*y
    return np.stack([cv2.remap(a,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101) for a in features])

def to_local(points,seed,theta):
    delta=np.asarray(points,np.float32)-seed;c=math.cos(theta);s=math.sin(theta);return np.stack([delta[:,0]*c+delta[:,1]*s,-delta[:,0]*s+delta[:,1]*c],1)/48.

def to_world(points,seed,theta):
    q=np.asarray(points,np.float32)*48;c=math.cos(theta);s=math.sin(theta);return np.stack([q[:,0]*c-q[:,1]*s,q[:,0]*s+q[:,1]*c],1)+seed
