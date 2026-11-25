import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
from gongbi_training.cunfa_spatial_pairs import pyramid_vector,verify_group_separation
from gongbi_training.cunfa_context_roles import context_role_target
ROOT=LAB/'cunfa_spatial_pairs_20261117'
def main():
 p=ROOT/'reference_manifest.json';d=json.loads(p.read_text('utf8'));rows=d['records']
 # Inherited added_this_round flags refer to the old run, not this run.
 if not (ROOT/'protocol.json').exists():
  for i,r in enumerate(rows):r['added_this_round']=i>=112
  d['metadata_correction_before_fit']='Reset inherited added_this_round flags; no targets changed'
  p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
 old=json.loads((LAB/'cunfa_relational_roles_20261117/reference_manifest.json').read_text('utf8'))['records']
 assert [context_role_target(r) for r in rows[:112]]==[context_role_target(r) for r in old]
 a=torch.zeros(1,2,8,8);a[0,0,:4,:4]=1;a[0,1,4:,4:]=1;b=torch.flip(a,[3])
 assert torch.allclose(pyramid_vector(a,(1,)),pyramid_vector(b,(1,)))
 assert not torch.allclose(pyramid_vector(a),pyramid_vector(b))
 assert pyramid_vector(a).shape==(1,10)
 tr=np.array([r['split']=='train' for r in rows]);te=np.array([r['split']=='fresh_check' for r in rows]);assert verify_group_separation(rows,tr,te)
 try:verify_group_separation([dict(role_fold_group='same')]*2,np.array([True,False]),np.array([False,True]))
 except ValueError:pass
 else:raise AssertionError('Group leakage not blocked')
 tree=next(r for r in rows if r['id']=='shen_attributed_7');assert tree['object_type']=='tree' and context_role_target(tree)==-100
 assert sum(context_role_target(r)!=-100 for r in rows[112:] if r['split']=='train')==11
 assert sum(context_role_target(r)!=-100 for r in rows[112:] if r['split']=='fresh_check')==6
 result=dict(spatial_positions_retained=True,group_leakage_blocked=True,old_targets_unchanged=True,tree_unknown_ignored=True,added_training11_fresh6_unknown8=True,robot_connected=False)
 (ROOT/'contract_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
