"""Keep object weak labels separate from source regions and uncertain areas."""
import numpy as np
from PIL import Image,ImageDraw
def reference_masks(case):
    size=tuple(case['size']);unknown=Image.new('L',size,0);sparse=Image.new('L',size,255)
    for r in case['regions']:
        if r['object']=='unknown':ImageDraw.Draw(unknown).polygon([tuple(p) for p in r['points']],fill=255)
        elif r['kind']=='core_polygon':ImageDraw.Draw(sparse).polygon([tuple(p) for p in r['points']],fill=1 if r['object']=='rock' else 2)
    u=np.array(unknown)>0;s=np.array(sparse);u[s!=255]=False
    return s,u

def masked_teacher_kl(logits,teacher,valid):
    import torch
    logp=logits.log_softmax(1);t=teacher.detach().softmax(1)
    loss=(t*(t.clamp_min(1e-8).log()-logp)).sum(1)
    if not bool(valid.any()):return logits.sum()*0
    return loss[valid].mean()

def sparse_class_loss(logits,target):
    import torch
    from .cunfa_supervision import partial_role_loss
    parts=[]
    for cls in (1,2):
        w=target.eq(cls)
        if bool(w.any()):parts.append(partial_role_loss(logits,target,w))
    return torch.stack(parts).mean() if parts else logits.sum()*0
