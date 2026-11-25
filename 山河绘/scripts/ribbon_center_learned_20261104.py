# -*- coding: utf-8 -*-
"""Narrow, newly trained cross-section center locator. No robot interfaces."""
import numpy as np,cv2
import torch
from torch import nn
OFFSETS=np.linspace(-24,24,97,dtype=np.float32)

class CenterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(18,32,7,padding=3),nn.SiLU(),nn.Conv1d(32,48,9,padding=4),nn.SiLU(),
            nn.Conv1d(48,32,9,padding=4),nn.SiLU(),nn.Conv1d(32,16,5,padding=2),nn.SiLU(),nn.Conv1d(16,2,1))
    def forward(self,x):return self.net(x)

def features(raw):
    # Raw intensity plus a robust sloping-background residual; no old extraction.
    raw=np.asarray(raw,np.float32)
    left=np.quantile(raw[...,:8],.8,axis=-1,keepdims=True);right=np.quantile(raw[...,-8:],.8,axis=-1,keepdims=True)
    base=left+(right-left)*np.linspace(0,1,97,dtype=np.float32)
    return np.concatenate([raw-.5,np.clip((base-raw)*5,-1,3)],axis=-2).astype(np.float32)

def extract(rgb,points,normals,scale=1.):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    p=np.asarray(points,np.float32);n=np.asarray(normals,np.float32);t=np.column_stack([-n[:,1],n[:,0]])
    coords=p[:,None,None,:]+OFFSETS[None,None,:,None]*scale*n[:,None,None,:]+np.arange(-4,5,dtype=np.float32)[None,:,None,None]*scale*t[:,None,None,:]
    x=coords[...,0].reshape(-1,97);y=coords[...,1].reshape(-1,97)
    raw=cv2.remap(gray,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101).reshape(-1,9,97)
    return raw

def analytic_center(raw):
    av=np.asarray(raw,np.float32).mean(-2)
    base=np.quantile(av[...,:8],.8,axis=-1)[...,None]+(np.quantile(av[...,-8:],.8,axis=-1)-np.quantile(av[...,:8],.8,axis=-1))[...,None]*np.linspace(0,1,97)
    signal=base-av;pred=[]
    for v in signal:
        use=np.flatnonzero(abs(OFFSETS)<=8);k=int(use[np.argmax(v[use])]);height=v[k]
        lo=hi=k
        while lo>0 and v[lo-1]>=height*.5:lo-=1
        while hi<96 and v[hi+1]>=height*.5:hi+=1
        w=np.maximum(v[lo:hi+1],0)
        pred.append(float((w*OFFSETS[lo:hi+1]).sum()/max(w.sum(),1e-8)))
    return np.array(pred)

@torch.no_grad()
def predict(model,raw,device='cuda'):
    f=features(raw);out=[];model.eval()
    for i in range(0,len(f),512):out.append(model(torch.from_numpy(f[i:i+512]).to(device)).sigmoid().cpu().numpy())
    return np.concatenate(out)

def centers(prob,radius=8.):
    mask=abs(OFFSETS)<=radius;ix=np.flatnonzero(mask)
    k=ix[prob[:,0,mask].argmax(1)]
    return OFFSETS[k],prob[np.arange(len(k)),0,k]
