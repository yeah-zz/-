"""Weak object presence on reviewed regions; no dense ink/role target implied."""
import math

def regional_presence_loss(logits, regions, top_fraction=.2):
    """Average positive bag losses by image and object class.

    Each region is (batch_index, rock/tree class_id, bool support mask).
    Unknown intersections must already be removed from that mask. Top-k is
    intentionally partial: it does not declare every pixel of a box positive.
    """
    import torch
    if not 0<top_fraction<=1:raise ValueError('Invalid support fraction')
    probability=logits.softmax(1);groups={}
    for batch,category,support in regions:
        if category not in {1,2}:raise ValueError('Only reviewed rock/tree presence is supported')
        if support.dtype!=torch.bool or support.shape!=logits.shape[-2:]:raise ValueError('Invalid region support')
        values=probability[batch,category][support]
        if not values.numel():continue
        k=max(1,math.ceil(values.numel()*top_fraction))
        loss=-values.topk(k).values.mean().clamp_min(1e-8).log()
        groups.setdefault((batch,category),[]).append(loss)
    if not groups:return logits.sum()*0
    return torch.stack([torch.stack(v).mean() for v in groups.values()]).mean()
