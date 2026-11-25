"""Training-only conservative cores from coarse labels; never inference masks."""
import cv2
import numpy as np

def core_reference(target, radius=2):
    """Ignore a narrow rock/tree interface; unknown remains unknown.

    Cores are source-label erosion, not expert verified boundaries or ink masks.
    Distances are measured on the model's 256-pixel training view.
    """
    k=np.ones((2*radius+1,2*radius+1),np.uint8)
    rock=(target==1).astype('uint8');tree=(target==2).astype('uint8')
    interface=(cv2.dilate(rock,k)>0)&(cv2.dilate(tree,k)>0)&((target==1)|(target==2))
    revised=target.copy();revised[interface]=255
    cores=np.full_like(target,255)
    for cls in (1,2):
        inside=cv2.erode((target==cls).astype('uint8'),k,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        cores[inside]=cls
    return revised,cores,interface

def core_balanced_loss(logits,targets,cores,weights,strength=.5):
    import torch
    from .cunfa_supervision import partial_role_loss
    loss=partial_role_loss(logits,targets,weights)
    # Equal class means: a small tree core cannot vanish under a large rock region.
    class_losses=[]
    for cls in (1,2):
        w=cores.eq(cls).to(logits.dtype)
        if bool(w.any()):class_losses.append(partial_role_loss(logits,cores,w))
    if class_losses:loss=loss+strength*torch.stack(class_losses).mean()
    return loss
