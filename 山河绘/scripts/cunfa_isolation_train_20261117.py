# -*- coding: utf-8 -*-
import sys,json,os,random,hashlib
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
import torch.nn.functional as F
from gongbi_training.cunfa_branch_isolation import IsolatedRoleHead,gradient_comparison
from gongbi_training.cunfa_boundary_balance import supervised_auxiliary,sparse_scores
from gongbi_training.cunfa_context_roles import context_role_target,ROLE_NAMES
from cunfa_photometry_train_20261117 import role_metric
ROOT=LAB/'cunfa_branch_isolation_20261117';OLD=LAB/'cunfa_boundary_balance_20261117';BASE=LAB/'cunfa_boundary_relations_20261117'
ARMS=['shared','detached_heads','separate_trunks']
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 assert not (ROOT/'protocol.json').exists();d=read(OLD/'reference_manifest.json');write('reference_manifest.json',d);rows=d['records'];cache=np.load(BASE/'features.npz');auxcache=np.load(OLD/'reference_targets.npz');targets=tuple(auxcache['expanded_'+str(j)] for j in range(4));y=cache['targets'];assert [context_role_target(r) for r in rows]==y.tolist()
 fit=np.array([r['split']=='train' for r in rows])&(y!=-100);groups=np.array(['qingpu_book_all' if r['role_fold_group'].startswith('qingpu') else r['role_fold_group'] for r in rows]);assert not set(groups[fit])&set(groups[(~fit)&(y!=-100)])
 hashes={str(p):sha(p) for p in [OLD/'balance_models.pt',OLD/'reference_targets.npz',OLD/'reference_manifest.json',OLD/'results.json',BASE/'features.npz',BASE/'boundary_models.pt']}
 protocol=dict(script_sha256=sha(__file__),module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/cunfa_branch_isolation.py'),old_artifacts=hashes,manifest_sha256=sha(ROOT/'reference_manifest.json'),arms=ARMS,
  data='exact previous146 input features/ROI,49role references,6positive boundary windows,6negative windows18points; all original targets unchanged',
  change='shared: previous expanded-balanced baseline; detached_heads: auxiliary head sees detached shared features; separate_trunks: independent identically initialized auxiliary conv trunk',
  fixed='seed0,100epochs,AdamW .001 weight_decay.01,auxiliary.3,balanced binary loss,angle.25,threshold.5; no augmentation or threshold search',
  capacity='same role trunk and role head in all arms; separate_trunks additionally copies auxiliary trunk, increases parameters, no ensemble',
  selection='same previous criterion: mean trainingOOF role macro recall and sparse boundary balanced accuracy; tie role macro recall',
  folds='four: whole Qingpu book,whole video,each Met development object; all checks excluded from training and model selection',
  check_scope='all25role checks and2NiZan sparse windows already inspected; known regressions only, no new independent works',
  gradient_diagnostic='role/auxiliary gradient cosine on shared trunk at initialization and fitted shared model; exploratory, not proof alone',
  production_replaced=False,robot_connected=False)
 write('protocol.json',protocol);print('FROZEN',flush=True);torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
 x=torch.from_numpy(cache['features']).cuda();roi=torch.from_numpy(cache['roi']).cuda();yy=torch.from_numpy(y).long().cuda();aux=tuple(torch.from_numpy(v).cuda() for v in targets)
 def train(mask,arm):
  torch.manual_seed(0);np.random.seed(0);random.seed(0);m=IsolatedRoleHead(arm).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);idx=torch.from_numpy(np.flatnonzero(mask)).cuda();n=torch.bincount(yy[idx],minlength=4).float();assert (n>0).all();w=n.sum()/n/4
  for epoch in range(100):
   m.train();opt.zero_grad(set_to_none=True);o=m(x[idx],roi[idx]);loss=F.cross_entropy(o[0],yy[idx],weight=w)+.3*supervised_auxiliary(o,*(v[idx] for v in aux),balanced=True);assert torch.isfinite(loss);loss.backward();opt.step()
  return m.eval()
 def infer(m,mask=None):
  with torch.inference_mode():
   o=m(x if mask is None else x[mask],roi if mask is None else roi[mask]);return o[0].argmax(1).cpu().numpy(),dict(edge=o[1].sigmoid().cpu().numpy(),direction=o[2].cpu().numpy(),side=o[3].softmax(1).cpu().numpy())
 models={};preds={};dense={};metrics={};boundary={};scores={};gradients={};counts=torch.bincount(yy[fit],minlength=4).float();weight=counts.sum()/counts/4
 def audit(m):
  answer={}
  for group in ['all_training']+sorted(set(groups[fit])):
   mask=fit if group=='all_training' else fit&(groups==group)
   answer[group]=gradient_comparison(m,x[mask],roi[mask],yy[mask],tuple(v[mask] for v in aux),weight,supervised_auxiliary)
  return answer
 torch.manual_seed(0);gradients['shared_initial']=audit(IsolatedRoleHead('shared').cuda().eval())
 for arm in ARMS:
  op=np.full(len(rows),-100);od=dict(edge=np.zeros((len(rows),56,56),dtype='float32'),direction=np.zeros((len(rows),2,56,56),dtype='float32'),side=np.zeros((len(rows),2,56,56),dtype='float32'))
  for group in sorted(set(groups[fit])):
   tr=fit&(groups!=group);te=fit&(groups==group);m=train(tr,arm);pp,dd=infer(m,te);op[te]=pp
   for k in od:od[k][te]=dd[k]
   print('FOLD',arm,group,flush=True)
  metrics[arm]=role_metric(y[fit],op[fit]);boundary[arm]=sparse_scores(od,targets,fit);scores[arm]=(metrics[arm]['macro_recall']+boundary[arm]['balanced_boundary_accuracy'])/2
  m=train(fit,arm);models[arm]={k:v.cpu() for k,v in m.state_dict().items()};preds[arm],dense[arm]=infer(m)
  if arm=='shared':gradients['shared_fitted']=audit(m)
  if arm=='detached_heads':gradients['detached_fitted']=audit(m)
  print('OOF',arm,json.dumps(dict(role=metrics[arm],boundary=boundary[arm],score=scores[arm])),flush=True)
 selected=max(ARMS,key=lambda a:(scores[a],metrics[a]['macro_recall']));write('training_selection.json',dict(selected_arm=selected,role=metrics,boundary=boundary,scores=scores,saved_before_check_scoring=True));write('gradient_diagnostics.json',gradients)
 prev=read(OLD/'results.json');oldout=np.load(OLD/'dense_outputs.npz');assert all(ROLE_NAMES[int(preds['shared'][i])]==r['predictions']['negative_refs_balanced'] for i,r in enumerate(prev['records']));assert np.allclose(dense['shared']['edge'],oldout['negative_refs_balanced_edge'],atol=1e-6)
 roleonly=read(BASE/'results.json');role_preserved={a:all(ROLE_NAMES[int(preds[a][i])]==r['predictions']['role_only'] for i,r in enumerate(roleonly['records'])) for a in ['detached_heads','separate_trunks']};assert all(role_preserved.values())
 preds['frozen_selected']=np.array([ROLE_NAMES.index(r['selected_prediction']) for r in prev['records']]);a=prev['selected_arm'];dense['frozen_selected']={k:oldout[a+'_'+k] for k in ['edge','direction','side']}
 masks={'known12':np.array([r['id'].startswith('ntm_') for r in rows])&(y!=-100),'seen_shen6':np.array([r['id'].startswith('shen_attributed_') for r in rows])&(y!=-100),'seen_nizan7':np.array([r['id'].startswith('nizan_') for r in rows])&(y!=-100)}
 checks={s:{a:role_metric(y[mask],p[mask]) for a,p in preds.items()} for s,mask in masks.items()};auxiliary={s:{a:sparse_scores(v,targets,mask) for a,v in dense.items()} for s,mask in [('train_resubstitution',fit),('seen_nizan_sparse2',np.array([r['id'].startswith('nizan_') and r['id'] in {c['id'] for c in d['curves']} for r in rows]))]}
 result=dict(selected_arm=selected,training_oof_metrics=metrics,training_oof_auxiliary=boundary,check_metrics=checks,auxiliary_metrics=auxiliary,role_only_predictions_exactly_preserved=role_preserved,records=[dict(id=r['id'],split=r['split'],role_reference=None if y[i]==-100 else ROLE_NAMES[y[i]],predictions={a:ROLE_NAMES[int(p[i])] for a,p in preds.items()},selected_prediction=ROLE_NAMES[int(preds[selected][i])]) for i,r in enumerate(rows)],model_adopted=False,independent_validation_passed=False,robot_connected=False)
 write('results.json',result);torch.save(dict(states=models,selected_arm=selected,protocol=protocol),ROOT/'isolation_models.pt');np.savez_compressed(ROOT/'dense_outputs.npz',**{a+'_'+k:v for a,vals in dense.items() for k,v in vals.items()});assert all(sha(p)==h for p,h in hashes.items());write('verification.json',dict(old_artifacts_unchanged=True,shared_baseline_exactly_replayed=True,isolated_roles_exactly_match_role_only=True,no_check_training=True,unknown_not_background=True,no_new_independent_claim=True,threshold_fixed=True,robot_connected=False));print('SELECTED',selected,flush=True);print('AUX',json.dumps(auxiliary['seen_nizan_sparse2']),flush=True);print('CHECKS',json.dumps(checks),flush=True)
if __name__=='__main__':main()
