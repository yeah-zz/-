"""Audit unknown gradients, group separation, provenance and unchanged baselines."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
from gongbi_training.cunfa_supervision import partial_role_loss,role_targets_checked
ROOT=LAB/'cunfa_roles_v2_20261112'
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ds=read(ROOT/'dataset_manifest.json');result=read(ROOT/'results.json');p=result['protocol'];assert sha(ROOT/'dataset_manifest.json')==p['dataset_sha256'];assert sha(Path(__file__).with_name('cunfa_roles_v2_train_20261112.py'))==p['script_sha256']
 assert sha(LAB/'cunfa_temporal_roles_20261112/role_probe.pt')==p['old_checkpoint_sha256'];freeze=read(ROOT/'checkpoint_freeze.json')
 for arm,hashval in result['checkpoint_hashes'].items():assert hashval==sha(ROOT/(arm+'.pt'))==freeze['checkpoints'][arm]
 train={r['source_family'] for r in ds['records'] if r['split']=='train'};held={r['source_family'] for r in ds['records'] if r['split']!='train'};assert len(train)==2 and len(held)==1 and not train&held
 for r in ds['records']:
  assert sha(r['path'])==r['sha256'];assert role_targets_checked(dict(r,role_mask_path=r['path']));assert not r['automatic_predictions_as_label'] and not r['expert_verified'] and not r['complete_instance_ownership']
  z=np.load(r['path']);t=z['targets'];assert set(np.unique(t))<={0,1,2,255};assert (t==255).sum()==r['unknown_pixels'];assert all((t==k).sum()==r['reference_counts'][str(k)] for k in range(3));assert all(r['reference_counts'][str(k)]>0 for k in range(3))
  # Exercise the split loss on the actual sparse mask. No branch may supervise
  # unknown pixels; conditional role loss may not supervise known paper either.
  target=torch.from_numpy(t.astype('int64'))[None];known=target!=255;fg=target.gt(0)&known;ft=torch.where(known,fg.long(),255);ct=torch.where(fg,target-1,255);ink=torch.randn(1,2,*t.shape,requires_grad=True);role=torch.randn(1,2,*t.shape,requires_grad=True);loss=partial_role_loss(ink,ft)+partial_role_loss(role,ct);loss.backward()
  assert torch.count_nonzero(ink.grad[0,:,t==255])==0;assert torch.count_nonzero(role.grad[0,:,t==255])==0;assert torch.count_nonzero(role.grad[0,:,t==0])==0;assert role.grad[0,:,t==2].abs().sum()>0
 old=LAB/'cunfa_fresh_validation_20261111';policy=read(LAB/'cunfa_role_reference_20261112/legacy_policy_audit.json');assert sha(old/'test_manifest.json')==policy['legacy_manifest_sha256'];assert read(old/'summary.json')['metrics']['local_plus_context']['confusion']==[[6,2],[4,1]]
 out=dict(dataset_hash_matches_protocol=True,training_code_hash_matches_protocol=True,all_reference_hashes_match=True,all_three_roles_present_per_observation=True,train_heldout_groups_disjoint=True,unknown_gradient_zero_both_branches=True,paper_no_conditional_role_gradient=True,checkpoints_match_freeze=True,old_checkpoint_unchanged=True,old_13_test_manifest_unchanged=True,old_13_confusion_unchanged=True,artist_independence_not_claimed=True,test_work_not_claimed_blind=True,raw_output_not_reference_masked=True,model_adopted=False,robot_connected=False)
 (ROOT/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
