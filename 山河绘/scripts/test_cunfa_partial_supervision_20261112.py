# -*- coding: utf-8 -*-
"""Regression checks for unknown-as-background and work-label propagation."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import torch
from gongbi_training.cunfa_supervision import *
def main():
 work=dict(local_style='斧劈皴',label_scope='work_level',source_url='https://example.invalid/source',source_evidence='Named for the whole painting')
 assert style_target(work)==-100
 example=dict(work,label_scope='named_teaching_example');assert style_target(example)==1
 assert style_target(dict(example,automatic_prediction_as_label=True))==-100
 assert style_target(dict(example,conflicting_style_labels=True))==-100
 assert style_target(dict(example,local_style=None))==-100
 assert not role_targets_checked(dict(role_mask_path='candidate.png',role_reference_checked=False))
 torch.manual_seed(111204);a=torch.randn(1,4,8,8,requires_grad=True);target=torch.full((1,8,8),255);target[0,2,3]=1
 loss=partial_role_loss(a,target);loss.backward();g=a.grad.clone();assert g[:,:,2,3].abs().sum()>0;g[:,:,2,3]=0;assert torch.count_nonzero(g)==0
 changed=a.detach().clone();changed[:,:,0,:]+=1000;assert torch.allclose(partial_role_loss(changed,target),loss.detach())
 b=torch.randn(1,4,8,8,requires_grad=True);empty=partial_role_loss(b,torch.full((1,8,8),255));empty.backward();assert empty.item()==0 and torch.count_nonzero(b.grad)==0
 c=torch.randn(3,2,requires_grad=True);sl=partial_style_loss(c,torch.tensor([0,-100,1]));sl.backward();assert torch.count_nonzero(c.grad[1])==0 and torch.count_nonzero(c.grad[[0,2]])>0
 out=dict(work_label_not_propagated=True,explicit_local_source_accepted=True,automatic_prediction_rejected=True,mixed_label_ignored=True,unverified_role_mask_rejected=True,unknown_pixel_gradients_zero=True,altering_unknown_pixels_no_loss_change=True,empty_mask_safe_zero_loss=True,unknown_style_gradient_zero=True,robot_connected=False)
 root=LAB/'cunfa_role_reference_20261112';root.mkdir(exist_ok=True);(root/'supervision_checks.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
