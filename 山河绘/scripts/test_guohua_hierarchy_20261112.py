"""Regressions for foliage/cun confusion, leakage, and premature skill training."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.guohua_hierarchy import *
def reject(call):
 try:call()
 except ValueError:return
 raise AssertionError('Expected rejected invalid annotation or plan')
def main():
 reject(lambda:validate_region(dict(object_type='tree',role='cun',cunfa='pima')))
 reject(lambda:validate_region(dict(object_type='tree',role='foliage',cunfa='fupi')))
 reject(lambda:validate_region(dict(object_type='rock',role='cun',prediction_as_label=True)))
 r=dict(object_type='rock',object_verified=True,role='cun',cunfa='pima',label_scope='local_cun_group',source_url='https://example.invalid/reference',label_evidence='Checked local teaching demonstration',reviewer='test',label_status='source_checked')
 assert cunfa_training_target(r)=='pima';assert cunfa_training_target(dict(r,label_scope='work_title')) is None;assert cunfa_training_target(dict(r,object_verified=False)) is None;assert cunfa_training_target(dict(r,cunfa='mixed')) is None
 reject(lambda:verify_split([dict(work_id='a',source_group='s',split='train'),dict(work_id='a',source_group='s',split='test')]))
 reject(lambda:verify_split([dict(work_id='a',source_group='s1',artist_id='artist',split='train'),dict(work_id='b',source_group='s2',artist_id='artist',split='test')]))
 assert not verify_split([dict(work_id='a',source_group='s',split='train')])['author_independence_verified']
 assert not allowed_phases(CapabilityState())['tree_skill_training'];assert not allowed_phases(CapabilityState(object_independent_validation=True))['tree_skill_training'];assert allowed_phases(CapabilityState(True,True))['tree_skill_training'];assert not allowed_phases(CapabilityState(True,True))['physical_creation_validation']
 steps=[dict(id='r1',object_id='rock1',object_type='rock',role='outline'),dict(id='r2',object_id='rock1',object_type='rock',role='cun',depends_on=['r1']),dict(id='t1',object_id='tree1',object_type='tree',role='trunk'),dict(id='t2',object_id='tree1',object_type='tree',role='main_branch',depends_on=['t1']),dict(id='t3',object_id='tree1',object_type='tree',role='foliage',depends_on=['t2'])]
 reject(lambda:validate_skill_order(steps));steps[1]['object_complete']=True
 assert validate_skill_order(steps);reject(lambda:validate_skill_order([steps[1]]));reject(lambda:validate_skill_order([steps[-1]]));reject(lambda:validate_skill_order([dict(id='u',object_id='u',object_type='unknown',role='unknown')]))
 out=dict(foliage_cannot_be_cun=True,method_requires_rock_group=True,work_titles_not_local_targets=True,unknown_objects_ignored=True,prediction_labels_rejected=True,work_author_leakage_rejected=True,missing_author_not_claimed_independent=True,tree_training_waits_for_first_two_phases=True,physical_generation_gate_closed=True,pilot_order_dependencies_checked=True,model_trained=False,robot_connected=False)
 root=LAB/'guohua_hierarchy_20261112';root.mkdir(exist_ok=True);(root/'contract_checks.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
