"""Balanced sparse contour loss; explicit non-contour points stay distinct from paper."""
import numpy as np,cv2,torch
import torch.nn.functional as F
from .cunfa_boundary_relations import sparse_targets,canvas_points

def targets_with_negatives(annotation,nearbox):
    edge,side,angle,orient=sparse_targets(annotation,nearbox)
    if annotation is None:return edge,side,angle,orient
    for x,y in canvas_points(annotation.get('negative_points',[]),nearbox):
        mask=np.zeros(edge.shape,dtype='uint8');cv2.circle(mask,(round(float(x)),round(float(y))),2,1,-1)
        if np.any(edge[mask>0]==1):raise ValueError('Explicit negative conflicts with contour corridor')
        edge[mask>0]=0
    return edge,side,angle,orient

def balanced_binary_loss(logits,targets):
    terms=[]
    if (targets==1).any():terms.append(F.softplus(-logits[targets==1]).mean())
    if (targets==0).any():terms.append(F.softplus(logits[targets==0]).mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0

def supervised_auxiliary(outputs,edge,side,angle,orient,balanced):
    _,pe,pa,ps=outputs;v=edge!=-100
    e=balanced_binary_loss(pe,edge) if balanced else F.binary_cross_entropy_with_logits(pe[v],edge[v]) if v.any() else pe.sum()*0
    v=side!=-100;s=F.cross_entropy(ps.permute(0,2,3,1)[v],side[v]) if v.any() else ps.sum()*0
    a=F.smooth_l1_loss(pa.permute(0,2,3,1)[orient],angle.permute(0,2,3,1)[orient]) if orient.any() else pa.sum()*0
    return e+s+.25*a

def sparse_scores(outputs,targets,mask):
    edge,side,angle,orient=targets;ev=edge[mask];sv=side[mask];ov=orient[mask];av=angle[mask]
    pe,pa,ps=(outputs[k][mask] for k in ['edge','direction','side']);pred=pe>=.5;positive=ev==1;negative=ev==0
    recall=float(pred[positive].mean()) if positive.any() else None;specificity=float((~pred[negative]).mean()) if negative.any() else None
    sidepred=ps.argmax(1);valid=[float((sidepred[sv==c]==c).mean()) for c in [0,1] if (sv==c).any()]
    err=np.arccos(np.clip((pa*av).sum(1),-1,1))/2*180/np.pi
    return dict(positive_pixels=int(positive.sum()),negative_pixels=int(negative.sum()),positive_recall=recall,negative_specificity=specificity,balanced_boundary_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,side_balanced_accuracy=float(np.mean(valid)) if valid else None,axial_angle_mean_degrees=float(err[ov].mean()) if ov.any() else None)
