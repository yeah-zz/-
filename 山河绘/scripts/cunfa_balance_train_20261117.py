# -*- coding: utf-8 -*-
import sys,json,hashlib,os,random
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from gongbi_training.cunfa_boundary_relations import BoundaryRoleHead
from gongbi_training.cunfa_boundary_balance import *
from gongbi_training.cunfa_context_roles import context_box,context_role_target,ROLE_NAMES
from cunfa_photometry_train_20261117 import role_metric
ROOT=LAB/'cunfa_boundary_balance_20261117';OLD=LAB/'cunfa_boundary_relations_20261117'
ARMS=['old_refs_balanced','negative_refs_unbalanced','negative_refs_balanced']
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 assert not (ROOT/'protocol.json').exists();d=read(ROOT/'reference_manifest.json');rows=d['records'];cache=np.load(OLD/'features.npz');y=cache['targets'];assert [context_role_target(r) for r in rows]==y.tolist()
 fit=np.array([r['split']=='train' for r in rows])&(y!=-100);groups=np.array(['qingpu_book_all' if r['role_fold_group'].startswith('qingpu') else r['role_fold_group'] for r in rows]);test=(~fit)&(y!=-100);assert not set(groups[fit])&set(groups[test])
 original={a['id']:a for a in d['curves']};expanded={**original,**{a['id']:a for a in d['negative_references']}};arrays={}
 for key,annotations in [('original',original),('expanded',expanded)]:
  v=[]
  for r in rows:
   size=Image.open(r['reference_image']).size;v.append(targets_with_negatives(annotations.get(r['id']),context_box(r['box'],size,2.5)))
  arrays[key]=tuple(np.stack([a[j] for a in v]) for j in range(4))
 hashes={str(p):sha(p) for p in [OLD/'boundary_models.pt',OLD/'features.npz',OLD/'dense_outputs.npz',OLD/'results.json',OLD/'reference_manifest.json']}
 protocol=dict(script_sha256=sha(__file__),module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/cunfa_boundary_balance.py'),base_module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/cunfa_boundary_relations.py'),manifest_sha256=sha(ROOT/'reference_manifest.json'),old_artifacts=hashes,
  cached_features='exact previous146 early-feature maps and ROI; no encoder/photometry change',role_training=49,new_negative_windows=6,new_negative_points=18,reference_scope='assistant sparse non-contour points, not non-ink/paper masks; unknown pixels ignored',
  training_counts={k:dict(positive=int(((a[0]==1)&fit[:,None,None]).sum()),negative=int(((a[0]==0)&fit[:,None,None]).sum())) for k,a in arrays.items()},
  arms=ARMS,arm_definitions=['same old references, class-balanced boundary BCE','expanded negative references, previous pixel-mean BCE','expanded negative references, class-balanced boundary BCE'],
  balance='equal mean positive softplus(-logit) and negative softplus(logit); no extra multiplier for missing class',
  fixed_parameters='same shared conv architecture,seed0,100epochs,AdamW lr.001 weight_decay.01,aux.3,angle.25; side/angle loss unchanged; threshold.5 unchanged',
  groups='all Qingpu book one, all video one, each Met work one; four training leave-group-out folds',
  selection='highest mean of trainingOOF role macro recall and sparse boundary balanced accuracy; ties choose role macro recall',
  check_scope='all25 check references and NiZan2 sparse windows already inspected in previous turn; known regression only; no new independent data',
  production_replaced=False,robot_connected=False)
 write('protocol.json',protocol);print('FROZEN',json.dumps(protocol['training_counts']),flush=True)
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
 x=torch.from_numpy(cache['features']).cuda();roi=torch.from_numpy(cache['roi']).cuda();yy=torch.from_numpy(y).long().cuda();gpu={k:tuple(torch.from_numpy(v).cuda() for v in a) for k,a in arrays.items()}
 def train(mask,arm):
  torch.manual_seed(0);np.random.seed(0);random.seed(0);m=BoundaryRoleHead().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);idx=torch.from_numpy(np.flatnonzero(mask)).cuda();counts=torch.bincount(yy[idx],minlength=4).float();assert (counts>0).all();w=counts.sum()/counts/4;aux=gpu['original' if arm=='old_refs_balanced' else 'expanded']
  for epoch in range(100):
   m.train();opt.zero_grad(set_to_none=True);o=m(x[idx],roi[idx]);loss=F.cross_entropy(o[0],yy[idx],weight=w)+.3*supervised_auxiliary(o,*(v[idx] for v in aux),balanced=arm!='negative_refs_unbalanced');assert torch.isfinite(loss);loss.backward();opt.step()
  return m.eval()
 def infer(model,mask=None):
  with torch.inference_mode():
   o=model(x if mask is None else x[mask],roi if mask is None else roi[mask]);return o[0].argmax(1).cpu().numpy(),dict(edge=o[1].sigmoid().cpu().numpy(),direction=o[2].cpu().numpy(),side=o[3].softmax(1).cpu().numpy())
 models={};preds={};dense={};oofmetrics={};oofaux={};scores={};selection_records={}
 for arm in ARMS:
  op=np.full(len(rows),-100);od=dict(edge=np.zeros((len(rows),56,56),dtype='float32'),direction=np.zeros((len(rows),2,56,56),dtype='float32'),side=np.zeros((len(rows),2,56,56),dtype='float32'))
  for group in sorted(set(groups[fit])):
   tr=fit&(groups!=group);te=fit&(groups==group);m=train(tr,arm);pp,dd=infer(m,te);op[te]=pp
   for k in od:od[k][te]=dd[k]
   print('FOLD',arm,group,flush=True)
  oofmetrics[arm]=role_metric(y[fit],op[fit]);oofaux[arm]=sparse_scores(od,arrays['expanded'],fit);scores[arm]=(oofmetrics[arm]['macro_recall']+oofaux[arm]['balanced_boundary_accuracy'])/2
  m=train(fit,arm);models[arm]={k:v.cpu() for k,v in m.state_dict().items()};preds[arm],dense[arm]=infer(m)
  selection_records[arm]=dict(score=scores[arm],role=oofmetrics[arm],boundary=oofaux[arm]);print('OOF',arm,json.dumps(selection_records[arm]),flush=True)
 selected=max(ARMS,key=lambda a:(scores[a],oofmetrics[a]['macro_recall']));write('training_selection.json',dict(selected_arm=selected,records=selection_records,criterion=protocol['selection'],saved_before_check_scoring=True))
 oldresult=read(OLD/'results.json');oldout=np.load(OLD/'dense_outputs.npz');preds['frozen_previous']=np.array([ROLE_NAMES.index(r['predictions']['boundary_relations']) for r in oldresult['records']]);dense['frozen_previous']={k:oldout['boundary_relations_'+k] for k in ['edge','direction','side']}
 masks={'known12':np.array([r['id'].startswith('ntm_') for r in rows])&(y!=-100),'seen_shen6':np.array([r['id'].startswith('shen_attributed_') for r in rows])&(y!=-100),'seen_nizan7':np.array([r['id'].startswith('nizan_') for r in rows])&(y!=-100)}
 checkmetrics={s:{a:role_metric(y[mask],p[mask]) for a,p in preds.items()} for s,mask in masks.items()}
 auxiliary={s:{a:sparse_scores(v,arrays['expanded'],mask) for a,v in dense.items()} for s,mask in [('train_resubstitution',fit),('seen_nizan_sparse2',np.array([r['id'] in original and r['id'].startswith('nizan_') for r in rows]))]}
 result=dict(selected_arm=selected,training_oof_metrics=oofmetrics,training_oof_auxiliary=oofaux,check_metrics=checkmetrics,auxiliary_metrics=auxiliary,records=[dict(id=r['id'],split=r['split'],role_reference=None if y[i]==-100 else ROLE_NAMES[y[i]],predictions={a:ROLE_NAMES[int(p[i])] for a,p in preds.items()},selected_prediction=ROLE_NAMES[int(preds[selected][i])]) for i,r in enumerate(rows)],model_adopted=False,independent_validation_passed=False,robot_connected=False)
 write('results.json',result);torch.save(dict(states=models,selected_arm=selected,protocol=protocol),ROOT/'balance_models.pt');np.savez_compressed(ROOT/'dense_outputs.npz',**{a+'_'+k:v for a,vals in dense.items() for k,v in vals.items()});np.savez_compressed(ROOT/'reference_targets.npz',**{k+'_'+str(j):v for k,a in arrays.items() for j,v in enumerate(a)})
 assert all(sha(p)==h for p,h in hashes.items());write('verification.json',dict(old_artifacts_unchanged=True,all146_old_role_targets_unchanged=True,all_checks_previously_opened=True,additional_negatives_training_only=True,noncontour_not_paper=True,unknown_not_background=True,selection_oof_only=True,model_uses_no_reference_input=True,threshold_not_tuned=True,robot_connected=False));print('SELECTED',selected,flush=True);print('AUX',json.dumps(auxiliary['seen_nizan_sparse2']),flush=True);print('CHECKS',json.dumps(checkmetrics),flush=True)
if __name__=='__main__':main()
