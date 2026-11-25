import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.cunfa_context_roles import *
from gongbi_training.cunfa_concept_supervision import concept_targets,validate_concept_split
def main():
 r=dict(context_role_target='outline_context',object_type='rock',object_reference_checked=True,
  role_reference_checked=True,source_url='https://example.invalid/step',visual_evidence='Visible structure edge',reviewer='test',
  role_reference_scope='region_context_not_pixel_or_stroke_truth')
 assert context_role_target(r)==0
 assert context_role_target(dict(r,object_type='tree'))==-100
 assert context_role_target(dict(r,object_type='unknown'))==-100
 assert context_role_target(dict(r,object_type='paper'))==-100
 assert context_role_target(dict(r,context_role_target=None))==-100
 assert context_role_target(dict(r,object_type='rock',context_role_target='unmarked_surface'))==3
 assert context_role_target(dict(r,object_type='paper',context_role_target='unmarked_surface'))==3
 assert context_box([0,0,10,10],(30,30))==[0,0,17,17]
 root=LAB/'cunfa_cross_source_20261117';data=json.loads((root/'reference_manifest.json').read_text('utf8'));validate_concept_split(data['records'])
 for record in data['records']:
  if record['object_type']=='tree':
   assert context_role_target(record)==-100 and (concept_targets(record)==-100).all()
  if record['context_role_target'] is None:assert context_role_target(record)==-100
 assert all(r['split']=='test' for r in data['records'] if r['source_group'].startswith('ntmofa_'))
 out=dict(source_holdout_not_development=True,tree_not_rock_or_cun=True,
  unknown_role_ignored=True,unmarked_rock_surface_does_not_become_background_object=True,
  context_clipped_to_source=True,no_pixel_or_stroke_truth_claimed=True,robot_connected=False)
 (root/'contract_checks.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
