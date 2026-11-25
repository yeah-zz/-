"""Checks for partial supervision and unchanged historical evaluation."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
from gongbi_training.cunfa_supervision import partial_role_loss
ROOT=LAB/'cunfa_temporal_roles_20261112'
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ds=read(ROOT/'dataset_manifest.json');protocol=read(ROOT/'role_probe_protocol.json');result=read(ROOT/'role_probe_results.json');verification=read(ROOT/'verification.json')
 assert sha(ROOT/'dataset_manifest.json')==protocol['dataset_sha256'];assert sha(ROOT/'role_probe.pt')==result['model_sha256'];assert sha(Path(__file__).with_name('cunfa_temporal_probe_20261112.py'))==protocol['script_sha256']
 train={r['source_family'] for r in ds['records'] if r['split']=='train'};test={r['source_family'] for r in ds['records'] if r['split']!='train'};assert len(train)==len(test)==1 and not train&test
 for r in ds['records']:
  assert sha(Path(r['path']))==r['sha256'];z=np.load(r['path']);t=z['targets'];assert set(np.unique(t))<={0,1,2,255};assert int((t==255).sum())==r['unknown_pixels']
  for k in [0,1,2]:assert int((t==k).sum())==r['confirmed_partial_pixel_counts'][str(k)]
  if r['split']!='train':assert (t==1).sum()==0
  assert not r['expert_verified'] and r['style_target'] is None and not r['complete_instance_ownership']
  tensor=torch.randn(1,3,*t.shape,requires_grad=True);target=torch.from_numpy(t.astype('int64'))[None];weight=torch.ones_like(target,dtype=torch.float32);weight[target==2]=7
  loss=partial_role_loss(tensor,target,weight);loss.backward();g=tensor.grad[0];assert torch.count_nonzero(g[:,target[0]==255])==0;assert torch.count_nonzero(g[:,target[0]!=255])>0
 transfer=read(ROOT/'structure_transfer.json');assert transfer['geometry_transfer_only'] and not transfer['cross_cut_pixel_difference'] and transfer['rms_px']<1
 old=LAB/'cunfa_fresh_validation_20261111';policy=read(LAB/'cunfa_role_reference_20261112/legacy_policy_audit.json');assert sha(old/'test_manifest.json')==policy['legacy_manifest_sha256'];assert read(old/'summary.json')['metrics']['local_plus_context']['confusion']==[[6,2],[4,1]]
 verification.update(unknown_gradient_checks_passed=True,probe_code_hash_matches_protocol=True,old_13_test_manifest_unchanged=True,old_13_confusion_unchanged=True,source_group_and_reference_coverage_checks_passed=True)
 (ROOT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(verification,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
