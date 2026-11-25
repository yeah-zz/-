# -*- coding: utf-8 -*-
"""Whole-rock context, learned stroke anchors and direct complete-curve heads."""
import torch
from torch import nn
from torch.nn import functional as F
from .rock_context_data import SIZE,GRID,POINTS
def block(a,b,stride=1):return nn.Sequential(nn.Conv2d(a,b,3,stride,1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,1,1),nn.GroupNorm(4,b),nn.SiLU())
class RockContextNet(nn.Module):
 def __init__(self,project_curve=False):
  super().__init__();self.e1=block(3,16,2);self.e2=block(16,32,2);self.e3=block(32,64,2);self.e4=block(64,96,2)
  self.pos=nn.Linear(2,96);self.context=nn.TransformerEncoder(nn.TransformerEncoderLayer(96,4,192,dropout=0.,batch_first=True,norm_first=True),1,enable_nested_tensor=False)
  self.d3=block(160,64);self.d2=block(96,40);self.d1=block(56,24)
  self.anchor=nn.Conv2d(24,1,1);self.curves=nn.Conv2d(24,POINTS*2,1);self.roles=nn.Conv2d(24,2,1)
  self.semantic=nn.Conv2d(24,3,1);self.endpoints=nn.Conv2d(24,3,1);self.groups=nn.Conv2d(24,8,1)
  self.instance=nn.Conv2d(24,8,1);self.pixel=nn.Conv2d(24,8,1);self.mask_bias=nn.Conv2d(24,1,1)
  nn.init.constant_(self.anchor.bias,-3.0);nn.init.constant_(self.endpoints.bias,-3.0)
  self.project_curve=project_curve
  if project_curve:
   import numpy as np
   from scipy.interpolate import BSpline
   controls=10;degree=3;knots=np.r_[np.zeros(degree+1),np.linspace(0,1,controls-degree+1)[1:-1],np.ones(degree+1)];basis=BSpline(knots,np.eye(controls),degree)(np.linspace(0,1,POINTS));projection=basis@np.linalg.pinv(basis);projection[0]=0;projection[0,0]=1;projection[-1]=0;projection[-1,-1]=1
   self.register_buffer('curve_projection',torch.tensor(projection,dtype=torch.float32),persistent=False)
 def forward(self,x):
  a=self.e1(x);b=self.e2(a);c=self.e3(b);d=self.e4(c);h,w=d.shape[-2:];yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=x.device),torch.linspace(-1,1,w,device=x.device),indexing='ij');pos=torch.stack([xx,yy],-1).reshape(1,h*w,2)
  tokens=d.flatten(2).transpose(1,2);d=self.context(tokens+self.pos(pos)).transpose(1,2).reshape(-1,96,h,w)
  d=self.d3(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1));d=self.d2(torch.cat([F.interpolate(d,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1));d=self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
  curves=self.curves(d).tanh()*.8
  if self.project_curve:
   b,_,h,w=curves.shape;curves=torch.einsum('pq,bqchw->bpchw',self.curve_projection,curves.reshape(b,POINTS,2,h,w)).reshape(b,POINTS*2,h,w)
  return dict(anchor=self.anchor(d),curves=curves,roles=self.roles(d),semantic=self.semantic(d),endpoints=self.endpoints(d),groups=F.normalize(self.groups(d),dim=1),instance=self.instance(d),pixel=self.pixel(d),mask_bias=self.mask_bias(d))

def masked_focal(logits,target,valid):
 p=logits.sigmoid();positive=target.eq(1);negative=target.lt(1);weight=(1-target)**4
 pos=-(p.clamp_min(1e-6).log()*(1-p)**2*positive*valid).sum();neg=-((1-p).clamp_min(1e-6).log()*p**2*weight*negative*valid).sum()
 return (pos+neg)/((positive*valid).sum().clamp_min(1))

def joint_loss(out,targets,dense_curves=True):
 device=out['anchor'].device;to=lambda a,dtype=torch.float32:torch.as_tensor(a,dtype=dtype,device=device)
 import numpy as np
 heat=to(np.stack([t['heat'] for t in targets]));valid=to(np.stack([t['heat_valid'] for t in targets]));anchor=masked_focal(out['anchor'][:,0],heat,valid)
 # Endpoint maps use masked BCE; ambiguous directions are never supplied as
 # directional ground truth. The union map includes unoriented segment ends.
 endtarget=to(np.stack([t['endpoints'] for t in targets]));endvalid=to(np.stack([t['endpoint_valid'] for t in targets]));ends=(F.binary_cross_entropy_with_logits(out['endpoints'],endtarget,reduction='none')*endvalid*(1+endtarget*12)).sum()/endvalid.sum().clamp_min(1)
 semtarget=to(np.stack([t['roles'] for t in targets]),torch.long);sem=F.cross_entropy(out['semantic'],semtarget,ignore_index=-1,weight=to([.3,1.,1.]),reduction='sum')/(semtarget>=0).sum().clamp_min(1)
 curve_losses=[];role_losses=[];mask_losses=[];group_losses=[];smooth=[]
 for bi,t in enumerate(targets):
  groupvec=[];groupid=[]
  for a in t['anchors']:
   y,x=a['y'],a['x'];pred=out['curves'][bi,:,y,x].reshape(POINTS,2);cv=to(a['curve_valid']);locations=[]
   # Decode small anchor localization errors consistently. Ambiguous cells
   # nearer another labelled stroke do not get contradictory curve targets.
   for dy in ([-1,0,1] if dense_curves else [0]):
    for dx in ([-1,0,1] if dense_curves else [0]):
     xx,yy=x+dx,y+dy
     if not (0<=xx<GRID and 0<=yy<GRID):continue
     if any((z['x']-xx)**2+(z['y']-yy)**2<dx*dx+dy*dy for z in t['anchors'] if z is not a):continue
     locations.append((xx,yy))
   xx=to([z[0] for z in locations],torch.long);yy=to([z[1] for z in locations],torch.long);near=out['curves'][bi,:,yy,xx].T.reshape(-1,POINTS,2);centers=torch.stack([(xx+.5)/GRID,(yy+.5)/GRID],-1);truth=to(a['curve'])[None]-centers[:,None,:]
   forward=(F.smooth_l1_loss(near*10,truth*10,reduction='none').mean(-1)*cv[None]).sum(-1)/cv.sum().clamp_min(1)
   if not a['oriented']:forward=torch.minimum(forward,(F.smooth_l1_loss(near*10,truth.flip(1)*10,reduction='none').mean(-1)*cv.flip(0)[None]).sum(-1)/cv.sum().clamp_min(1))
   curve_losses.append(forward.mean());smooth.append((pred[2:]-2*pred[1:-1]+pred[:-2]).abs().mean());role_losses.append(F.cross_entropy(out['roles'][bi,:,y,x][None],to([a['role']],torch.long)))
   logits=(out['instance'][bi,:,y,x,None,None]*out['pixel'][bi]).sum(0)/(8**.5)+out['mask_bias'][bi,0,y,x];mv=to(a['mask_valid']);mt=to(a['mask']);weights=mv*(1+mt*8)
   mask_losses.append((F.binary_cross_entropy_with_logits(logits,mt,reduction='none')*weights).sum()/weights.sum().clamp_min(1))
   if a['group']>=0:groupvec.append(out['groups'][bi,:,y,x]);groupid.append(a['group'])
  if len(groupvec)>1:
   g=torch.stack(groupvec);sim=g@g.T;ids=to(groupid,torch.long);same=ids[:,None].eq(ids[None,:]);sel=torch.triu(torch.ones_like(same),diagonal=1).bool();group_losses.append(torch.where(same,1-sim,F.relu(sim-.2))[sel].mean())
 mean=lambda arr:torch.stack(arr).mean() if arr else out['anchor'].sum()*0
 parts=dict(anchor=anchor,curve=mean(curve_losses),role=mean(role_losses),semantic=sem,endpoints=ends,instance=mean(mask_losses),groups=mean(group_losses),smoothness=mean(smooth))
 total=parts['anchor']+parts['curve']*4+parts['role']*.35+parts['semantic']*.7+parts['endpoints']*.5+parts['instance']*.6+parts['groups']*.25+parts['smoothness']*.05
 return total,parts
