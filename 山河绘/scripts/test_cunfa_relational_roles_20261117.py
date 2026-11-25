import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image
from gongbi_training.cunfa_relational_roles import *
from gongbi_training.cunfa_context_roles import context_role_target
def main():
 roi=normalized_roi([10,20,30,40],[0,0,100,100]);r=roi.copy()
 for _ in range(4):r=rotate_roi_ccw(r)
 assert np.allclose(r,roi)
 image=Image.fromarray(np.full((60,80),200,'uint8'));f=spatial_features(image,image,roi)
 assert f.shape==(31,) and np.isfinite(f).all()
 root=LAB/'cunfa_relational_roles_20261117';d=json.loads((root/'reference_manifest.json').read_text('utf8'));rows=d['records']
 vids=[r for r in rows if r.get('source_group')=='mixiaolu_short_pima_work'];assert len({r['role_fold_group'] for r in vids})==1
 bad=next(r for r in rows if r['id']=='T_cun_group1_new_3');assert context_role_target(bad)==-100
 assert all(context_role_target(r)==-100 for r in rows if r['object_type']=='tree')
 assert all(r['split']=='train' for r in rows if r.get('added_this_round'))
 out=dict(rotated_roi_alignment_checked=True,finite_spatial_features=True,
  same_video_phases_and_crops_one_fold=True,ambiguous_paper_candidate_ignored=True,
  tree_targets_excluded=True,new_references_development_only=True,robot_connected=False)
 (root/'contract_checks.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
