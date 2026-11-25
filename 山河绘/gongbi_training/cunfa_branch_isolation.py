"""Controlled task-gradient isolation; identical role capacity in every arm."""
import copy
import torch
import torch.nn.functional as F
from .cunfa_boundary_relations import BoundaryRoleHead

class IsolatedRoleHead(BoundaryRoleHead):
    def __init__(self,mode='shared'):
        if mode not in {'shared','detached_heads','separate_trunks'}:raise ValueError(mode)
        super().__init__();self.mode=mode
        if mode=='separate_trunks':self.auxiliary_shared=copy.deepcopy(self.shared)
    def forward(self,x,roi):
        z=self.shared(x);centre=(z*roi[:,None]).sum((2,3))/roi.sum((1,2))[:,None].clamp_min(1)
        role=self.role(torch.cat([centre,z.mean((2,3))],1))
        aux=z if self.mode=='shared' else z.detach() if self.mode=='detached_heads' else self.auxiliary_shared(x)
        return role,self.edge(aux)[:,0],F.normalize(self.direction(aux),dim=1),self.side(aux)

def gradient_comparison(model,x,roi,targets,auxiliary,role_weight,auxiliary_loss):
    role,edge,direction,side=model(x,roi);lr=F.cross_entropy(role,targets,weight=role_weight);la=.3*auxiliary_loss((role,edge,direction,side),*auxiliary,balanced=True)
    parameters=list(model.shared.parameters());gr=torch.autograd.grad(lr,parameters,retain_graph=True);ga=torch.autograd.grad(la,parameters,allow_unused=True)
    r=torch.cat([g.flatten() for g in gr]);a=torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten() for p,g in zip(parameters,ga)]);nr=r.norm();na=a.norm()
    return dict(role_gradient_norm=float(nr),auxiliary_gradient_norm=float(na),cosine=None if nr*na<1e-12 else float(torch.dot(r,a)/(nr*na)),auxiliary_reaches_shared=bool(na>1e-12))
