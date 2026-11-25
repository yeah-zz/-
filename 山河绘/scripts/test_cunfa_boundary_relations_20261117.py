import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
from gongbi_training.cunfa_boundary_relations import *
from gongbi_training.cunfa_context_roles import context_role_target
ROOT=LAB/'cunfa_boundary_relations_20261117'
def main():
 d=json.loads((ROOT/'reference_manifest.json').read_text('utf8'));rows=d['records'];old=json.loads((LAB/'cunfa_spatial_pairs_20261117/reference_manifest.json').read_text('utf8'))['records']
 assert [context_role_target(r) for r in rows[:137]]==[context_role_target(r) for r in old]
 a=dict(line=[[10,20],[40,20]],rock=[[25,30]],open=[[25,10]])
 e,s,g,o=sparse_targets(a,[0,0,56,56]);assert np.allclose(g[0,o],1) and np.allclose(g[1,o],0,atol=1e-6);assert (s==0).any() and (s==1).any();assert (s[o]==-100).all();assert (e==-100).any()
 a2=dict(line=[[20,10],[20,40]],rock=[],open=[]);_,_,g2,o2=sparse_targets(a2,[0,0,56,56]);assert np.allclose(g2[0,o2],-1)
 model=BoundaryRoleHead();model.eval();x=torch.randn(2,19,56,56);roi=torch.ones(2,56,56);out=model(x,roi);loss=auxiliary_loss(out,torch.full((2,56,56),-100.),torch.full((2,56,56),-100,dtype=torch.long),torch.zeros(2,2,56,56),torch.zeros(2,56,56,dtype=torch.bool));assert loss.item()==0;loss.backward()
 assert context_role_target(rows[-1])==-100 and rows[-1]['object_type']=='tree'
 assert all(c['checked'] for c in d['curves']);assert not next(c for c in d['curves'] if c['id']=='qingpu_26_new_0')['open']
 checks=dict(old137_role_targets_unchanged=True,sparse_unknown_not_background=True,orientation_encoding_valid=True,side_excludes_boundary_corridor=True,no_annotations_forward_input=True,tree_excluded=True,robot_connected=False)
 (ROOT/'contract_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8');print(json.dumps(checks))
if __name__=='__main__':main()
