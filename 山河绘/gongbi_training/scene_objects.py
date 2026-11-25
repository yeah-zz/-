"""Four scene classes, transferred without confusing legacy-other and background."""
import torch
from torch import nn
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from guohua_objects_train_20261112 import ContextObjects
class SceneObjects(ContextObjects):
    def __init__(self,old_state=None):
        super().__init__()
        if old_state is not None:self.load_state_dict(old_state)
        old=self.head[-1];new=nn.Conv2d(old.in_channels,4,old.kernel_size,padding=old.padding)
        with torch.no_grad():
            new.weight.zero_();new.bias.fill_(-2.)
            for target,source in [(1,1),(2,2),(3,0)]:new.weight[target].copy_(old.weight[source]);new.bias[target].copy_(old.bias[source])
        self.head[-1]=new

def scene_metrics(reference,prediction):
    import numpy as np
    valid=reference!=255;c=np.bincount((reference[valid].astype('int64')*4+prediction[valid]).ravel(),minlength=16).reshape(4,4);support=c.sum(1);union=c.sum(0)+support-c.diagonal();ious=[float(c[k,k]/union[k]) if union[k] else None for k in range(4)]
    # Source coarse labels have no open-scene reference: do not insert an artificial
    # background IoU=0 into the old three-foreground-class comparison.
    return dict(confusion=c.tolist(),support=support.tolist(),iou=ious,known_foreground_mean_iou=float(np.mean([ious[k] for k in [1,2,3] if ious[k] is not None])),recall=[float(c[k,k]/support[k]) if support[k] else None for k in range(4)],tree_as_rock=float(c[2,1]/support[2]) if support[2] else None,rock_as_tree=float(c[1,2]/support[1]) if support[1] else None)
