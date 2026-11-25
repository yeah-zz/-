# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from faint_diagnostics_20261101 import ROOT,BASE,PRIOR,read,jsave,digest
import numpy as np,cv2,torch
from torch import nn
from torch.nn import functional as F

def features(rgb):
    a=rgb.astype(np.float32)/255;g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);local=[np.clip((cv2.GaussianBlur(g,(0,0),s)-g)*6,-1,1) for s in [1.,3.,7.]];x=np.concatenate([a,g[:,:,None],np.stack(local,2),(a[:,:,2]-a[:,:,0])[:,:,None]],2);return x.transpose(2,0,1).copy()

class FaintNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
        self.a=block(8,12);self.b=block(12,24);self.c=block(24,48);self.d=block(72,24);self.e=block(36,12);self.head=nn.Conv2d(12,5,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1));e=self.e(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1));return self.head(e)

@torch.no_grad()
def predict(rgb,model):
    model.eval();h,w=rgb.shape[:2];tile=256;pad=24;bottom=max(pad,tile-h-pad);right=max(pad,tile-w-pad);padded=cv2.copyMakeBorder(rgb,pad,bottom,pad,right,cv2.BORDER_REFLECT101);ph,pw=padded.shape[:2];f=features(padded);starts=lambda n:sorted(set(list(range(0,n-tile+1,192))+[n-tile]));coords=[(y,x) for y in starts(ph) for x in starts(pw)];total=np.zeros((5,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32);window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32)
    for start in range(0,len(coords),8):
        group=coords[start:start+8];batch=torch.from_numpy(np.stack([f[:,y:y+tile,x:x+tile] for y,x in group])).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model(batch).float();o[:,[0,1,4]]=o[:,[0,1,4]].sigmoid();o[:,2:4]=F.normalize(o[:,2:4],dim=1)
        for (y,x),p in zip(group,o.cpu().numpy()):total[:,y:y+tile,x:x+tile]+=p*window;den[y:y+tile,x:x+tile]+=window
    return (total/np.maximum(den[None],1e-6))[:,pad:pad+h,pad:pad+w]
