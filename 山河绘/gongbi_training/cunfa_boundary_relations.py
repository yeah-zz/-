"""Learn sparse boundary direction and side relations, never oracle input."""
import cv2,numpy as np,torch
from torch import nn
import torch.nn.functional as F

def canvas_points(points,box,size=56,input_size=224):
    x0,y0,x1,y1=box;w=x1-x0;h=y1-y0;scale=input_size/max(w,h)
    sw=max(1,round(w*scale));sh=max(1,round(h*scale))
    return np.array([[(x-x0)*sw/w+(input_size-sw)//2,(y-y0)*sh/h+(input_size-sh)//2] for x,y in points],dtype='float32').reshape(-1,2)*(size/input_size)

def sparse_targets(annotation,nearbox,size=56):
    edge=np.full((size,size),-100,dtype='float32');side=np.full((size,size),-100,dtype='int64');angle=np.zeros((2,size,size),dtype='float32');orient=np.zeros((size,size),dtype='bool')
    if annotation is None:return edge,side,angle,orient
    for key,label in [('rock',0),('open',1)]:
        for x,y in canvas_points(annotation[key],nearbox,size):
            xx,yy=round(float(x)),round(float(y));mask=np.zeros((size,size),dtype='uint8');cv2.circle(mask,(xx,yy),2,1,-1);side[mask>0]=label;edge[mask>0]=0
    points=canvas_points(annotation['line'],nearbox,size)
    for a,b in zip(points[:-1],points[1:]):
        mask=np.zeros((size,size),dtype='uint8');cv2.line(mask,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),1,3);active=mask>0;theta=np.arctan2(*(b-a)[::-1]);angle[0,active]=np.cos(2*theta);angle[1,active]=np.sin(2*theta);orient|=active;edge[active]=1
    # Marked boundary corridor never simultaneously supplies either side label.
    side[orient]=-100
    return edge,side,angle,orient

def roi_mask(box,nearbox,size=56):
    points=canvas_points([[box[0],box[1]],[box[2],box[3]]],nearbox,size);mask=np.zeros((size,size),dtype='float32');a,b=np.clip(np.floor(points[0]).astype(int),0,size-1),np.clip(np.ceil(points[1]).astype(int),1,size);mask[a[1]:b[1],a[0]:b[0]]=1
    assert mask.sum()>0;return mask

class BoundaryRoleHead(nn.Module):
    def __init__(self,channels=19):
        super().__init__();self.shared=nn.Sequential(nn.Conv2d(channels,24,3,padding=1),nn.GroupNorm(4,24),nn.SiLU(),nn.Conv2d(24,24,3,padding=1),nn.GroupNorm(4,24),nn.SiLU())
        self.role=nn.Sequential(nn.Linear(48,24),nn.SiLU(),nn.Linear(24,4));self.edge=nn.Conv2d(24,1,1);self.direction=nn.Conv2d(24,2,1);self.side=nn.Conv2d(24,2,1)
    def forward(self,x,roi):
        z=self.shared(x);centre=(z*roi[:,None]).sum((2,3))/roi.sum((1,2))[:,None].clamp_min(1)
        return self.role(torch.cat([centre,z.mean((2,3))],1)),self.edge(z)[:,0],F.normalize(self.direction(z),dim=1),self.side(z)

def auxiliary_loss(outputs,edge,side,angle,orient):
    _,pe,pa,ps=outputs;valid=edge!=-100
    e=F.binary_cross_entropy_with_logits(pe[valid],edge[valid]) if valid.any() else pe.sum()*0
    valid_side=side!=-100;s=F.cross_entropy(ps.permute(0,2,3,1)[valid_side],side[valid_side]) if valid_side.any() else ps.sum()*0
    a=F.smooth_l1_loss(pa.permute(0,2,3,1)[orient],angle.permute(0,2,3,1)[orient]) if orient.any() else pa.sum()*0
    return e+s+.25*a
