# -*- coding: utf-8 -*-
"""Query-conditioned ink ownership. Predicted geometry is input, not a mask.

The image encoder and curve predictor stay frozen for controlled comparison.
No reference curve, width or instance mask enters production inference.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .rock_context_model import block
from .rock_context_data import GRID,POINTS,SIZE

def condition_features(old,image,anchors):
 """anchors is N x (image-index,x,y); supplied/oracle status is caller metadata."""
 device=image.device;n=len(anchors);idx=anchors[:,0].long();xx=anchors[:,1].long();yy=anchors[:,2].long()
 gray=F.interpolate(image.mean(1,keepdim=True),(GRID,GRID),mode='bilinear',align_corners=False);local=(F.avg_pool2d(gray,11,1,5)-gray).clamp(0,.4)/.4
 base=torch.cat([old['pixel'].tanh(),old['semantic'].softmax(1),gray,local],1)[idx]
 gy,gx=torch.meshgrid((torch.arange(GRID,device=device)+.5)/GRID,(torch.arange(GRID,device=device)+.5)/GRID,indexing='ij');grid=torch.stack([gx,gy],-1)
 center=torch.stack([(xx+.5)/GRID,(yy+.5)/GRID],-1);relative=(grid[None]-center[:,None,None]).permute(0,3,1,2)
 curve=old['curves'][idx,:,yy,xx].reshape(n,POINTS,2)+center[:,None]
 start=curve[:,:-1,None,None];segment=(curve[:,1:]-curve[:,:-1])[:,:,None,None];delta=grid[None,None]-start;length=(segment**2).sum(-1).clamp_min(1e-8);t=(delta*segment).sum(-1)/length;t=t.clamp(0,1);distance=((delta-t[...,None]*segment)**2).sum(-1);minimum,closest=distance.min(1)
 vector=(curve[:,1:]-curve[:,:-1]);unit=F.normalize(vector,dim=-1);tangent=unit[torch.arange(n,device=device)[:,None,None],closest];along=(closest+t.gather(1,closest[:,None]).squeeze(1))/(POINTS-1)
 geometry=torch.cat([relative,minimum.sqrt()[:,None].clamp(0,.25)*4,tangent.permute(0,3,1,2),along[:,None]],1)
 query=old['instance'][idx,:,yy,xx].tanh();role=old['roles'][idx,:,yy,xx].softmax(-1);query=torch.cat([query,role],-1)[:,:,None,None].expand(-1,-1,GRID,GRID)
 return torch.cat([base,geometry,query],1),minimum.sqrt()*SIZE,curve

class InkOwnershipDecoder(nn.Module):
 def __init__(self):
  super().__init__();self.e1=block(29,24);self.e2=block(24,40,2);self.e3=block(40,64,2);self.d2=block(104,40);self.d1=block(64,24);self.mask=nn.Conv2d(24,1,1);self.quality=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,1))
 def forward(self,x):
  a=self.e1(x);b=self.e2(a);c=self.e3(b);d=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1));d=self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1));return self.mask(d)[:,0],self.quality(c)[:,0]

class InkOwnershipDecoderHR(InkOwnershipDecoder):
 """Predict full-resolution ownership using original pixels, not resized masks."""
 high_resolution=True
 def __init__(self):
  super().__init__();self.detail=block(28,16);self.residual=nn.Conv2d(16,1,1);nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
 def forward(self,x,image_detail):
  a=self.e1(x);b=self.e2(a);c=self.e3(b);d=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1));d=self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1));up=F.interpolate(d,(SIZE,SIZE),mode='bilinear',align_corners=False);coarse=F.interpolate(self.mask(d),(SIZE,SIZE),mode='bilinear',align_corners=False)
  rgb=image_detail;gray=rgb.mean(1,keepdim=True);local=(F.avg_pool2d(gray,11,1,5)-gray).clamp(0,.4)/.4;delta=self.residual(self.detail(torch.cat([up,rgb,local],1)));return (coarse+delta)[:,0],self.quality(c)[:,0]

def ownership_loss(logits,target,valid,known_ink,quality_logits,nonempty):
 """Balanced positives, other confirmed ink, paper, plus masked region Dice.

Unknown real pixels stay zero-weight. Real crossings are not forced apart.
 """
 bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none');positive=valid*(target>.5);negative=valid*(target<=.5);other=negative*known_ink;paper=negative*(1-known_ink)
 mean=lambda a,w:(a*w).flatten(1).sum(1)/w.flatten(1).sum(1).clamp_min(1)
 pixel=mean(bce,positive)+mean(bce,other)+.35*mean(bce,paper);prob=logits.sigmoid();intersection=(prob*target*valid).flatten(1).sum(1);denom=((prob+target)*valid).flatten(1).sum(1);dice=1-(2*intersection+1)/(denom+1)
 quality=F.binary_cross_entropy_with_logits(quality_logits,nonempty)
 return pixel.mean()+dice.mean()+quality*.25,dict(pixel=pixel.mean(),dice=dice.mean(),quality=quality)
