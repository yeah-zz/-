# -*- coding: utf-8 -*-
"""Whole-curve/multiscale image classifier. No annotation access at inference."""
import cv2,numpy as np,torch
from torch import nn
from module_stroke_core_20261104 import samples,length

SIZE=48
class ContextCurveNet(nn.Module):
    def __init__(self,classes=2):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(5,16,5,stride=2,padding=2),nn.SiLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(32,48,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(48,64,3,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
        self.head=nn.Sequential(nn.Linear(192,96),nn.SiLU(),nn.Dropout(.1),nn.Linear(96,classes))
    def forward(self,x):
        b,v,c,h,w=x.shape;z=self.encoder(x.reshape(b*v,c,h,w)).reshape(b,v,64)
        return self.head(torch.cat([z[:,0],z[:,1:].mean(1),z[:,1:].amax(1)],1))

def raster(shape,paths):
    a=np.zeros(shape,np.uint8)
    for p in paths:
        if len(p)>=2:cv2.polylines(a,[np.rint(p).astype(np.int32)],False,255,1,cv2.LINE_AA)
    return a

def pack(rgb,path,others):
    """Whole bbox plus three native local views; all include candidate/neighbor masks."""
    p=np.asarray(path,float);h,w=rgb.shape[:2];scale=max(1.,max(h,w)/496)
    own=raster((h,w),[p]);other=raster((h,w),others) if isinstance(others,list) else others
    image=np.concatenate([rgb,own[:,:,None],other[:,:,None]],axis=2)
    lo=p.min(0);hi=p.max(0);pad=max(7*scale,float(np.ptp(p,axis=0).max())*.10)
    center=(lo+hi)/2;extent=np.maximum(hi-lo+2*pad,16*scale)
    q=samples(p,max(.4,length(p)/80));centers=[q[int((len(q)-1)*v)] for v in [.15,.5,.85]]
    views=[]
    for c,ex in [(center,extent)]+[(c,np.array([24,24])*scale) for c in centers]:
        xx=np.linspace(c[0]-ex[0]/2,c[0]+ex[0]/2,SIZE,dtype=np.float32)
        yy=np.linspace(c[1]-ex[1]/2,c[1]+ex[1]/2,SIZE,dtype=np.float32)
        x,y=np.meshgrid(xx,yy)
        view=cv2.remap(image,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
        views.append(view.transpose(2,0,1))
    return np.stack(views).astype(np.uint8)

@torch.no_grad()
def predict(model,packed):
    out=[];model.eval();device=next(model.parameters()).device
    for i in range(0,len(packed),96):
        x=torch.from_numpy(packed[i:i+96]).to(device).float()/255
        out.append(model(x).softmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,2),np.float32)
