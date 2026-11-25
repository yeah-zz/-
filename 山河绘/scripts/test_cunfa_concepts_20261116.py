import sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import torch
from gongbi_training.cunfa_concept_supervision import *

def reject(call):
 try: call()
 except ValueError: return
 raise AssertionError('Expected invalid reference to be rejected')

def main():
 r=dict(object_type='rock',object_reference_checked=True,concept_reference_checked=True,
  source_url='https://example.invalid/teaching',reviewer='test',visual_evidence='Visible mark',
  label_scope='observable_region_morphology',concept_origin='visual_region_review',
  concepts=dict(elongated_mark=1,angular_broad_mark=None,curved_parallel_group=0))
 assert concept_targets(r).tolist()==[1,IGNORE,0]
 for obj in ['tree','paper','unknown']:
  assert (concept_targets(dict(r,object_type=obj))==IGNORE).all()
 assert (concept_targets(dict(r,object_type='teaching_mark'))==IGNORE).all()
 assert concept_targets(dict(r,object_type='teaching_mark',reference_kind='named_brush_mark_exercise',source_evidence='Source caption names this practice')).tolist()==[1,IGNORE,0]
 assert permitted_cunfa_output('teaching_mark',True,'cun','pima')=='unknown'
 for key in ['object_reference_checked','concept_reference_checked']:
  assert (concept_targets(dict(r,**{key:False}))==IGNORE).all()
 for origin in ['style_name','work_title','model_prediction']:
  reject(lambda:concept_targets(dict(r,concept_origin=origin)))
 reject(lambda:concept_targets(dict(r,automatic_prediction_as_label=True)))
 reject(lambda:concept_targets(dict(r,concepts={'elongated_mark':2})))
 a=torch.tensor([[.3,-.7,.2]],requires_grad=True);target=torch.tensor([[1,IGNORE,0]])
 loss=partial_concept_loss(a,target);loss.backward();assert a.grad[0,1].item()==0
 changed=a.detach().clone();changed[0,1]=99;assert torch.allclose(loss,partial_concept_loss(changed,target))
 u=torch.randn(2,3,requires_grad=True);partial_concept_loss(u,torch.full((2,3),IGNORE)).backward();assert torch.count_nonzero(u.grad)==0
 base=dict(source_group='sourceA',reference_unit='unitA',source_sha256='hashA',split='train')
 reject(lambda:validate_concept_split([base,dict(base,split='test')]))
 assert permitted_cunfa_output('tree',True,'foliage','pima')=='unknown'
 assert permitted_cunfa_output('rock',False,'cun','pima')=='unknown'
 assert permitted_cunfa_output('rock',True,'outline','pima')=='unknown'
 assert permitted_cunfa_output('rock',True,'cun','pima')=='pima'
 out=dict(unknown_targets_ignored=True,unknown_gradient_zero=True,all_unknown_safe=True,
  style_names_not_concept_targets=True,predictions_not_labels=True,tree_and_unknown_not_rock_supervision=True,
  same_source_crops_cannot_leak=True,named_teaching_marks_not_rock_objects=True,reference_guard_is_not_object_learning=True,robot_connected=False)
 root=LAB/'cunfa_concept_supervision_20261116';root.mkdir(exist_ok=True)
 (root/'contract_checks.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out),flush=True)

if __name__=='__main__':main()
