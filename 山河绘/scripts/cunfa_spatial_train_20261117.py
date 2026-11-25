# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,joblib
from PIL import Image
from torchvision.models import mobilenet_v3_small
from gongbi_training.cunfa_photometry import appearance_tensor,contrast_features
from gongbi_training.cunfa_context_roles import context_box,context_role_target,ROLE_NAMES
from gongbi_training.cunfa_relational_roles import normalized_roi,spatial_features,predict_role_probe
from gongbi_training.cunfa_spatial_pairs import pyramid_vector,make_probe,verify_group_separation
from cunfa_photometry_train_20261117 import role_metric
ROOT=LAB/'cunfa_spatial_pairs_20261117';OLD=LAB/'cunfa_relational_roles_20261117';PREV=LAB/'cunfa_photometry_20261117'
WEIGHT=LAB/'cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 if '--resume-feature-check' in sys.argv:
  assert (ROOT/'protocol.json').exists() and not (ROOT/'results.json').exists()
  (ROOT/'protocol.json').rename(ROOT/'protocol_aborted_feature_check.json')
 assert not (ROOT/'protocol.json').exists(),'Experiment already frozen'
 rows=read(ROOT/'reference_manifest.json')['records'];y=np.array([context_role_target(r) for r in rows]);groups=np.array([r['role_fold_group'] for r in rows]);fit=np.array([r['split']=='train' for r in rows])&(y!=-100);known=np.array([r['split']=='test' for r in rows])&(y!=-100);fresh=np.array([r['split']=='fresh_check' for r in rows])&(y!=-100)
 # Historic manifests use test-like split names; derive the original12 directly.
 oldids={r['id'] for r in read(OLD/'results.json')['records'] if r['split']!='train' and r['role_reference'] is not None}
 known=np.array([r['id'] in oldids for r in rows]);verify_group_separation(rows,fit,known|fresh)
 hashes={str(p):sha(p) for p in [OLD/'relational_models.joblib',OLD/'results.json',PREV/'photometry_scoped_models.joblib']}
 protocol=dict(reference_manifest_sha256=sha(ROOT/'reference_manifest.json'),old_artifacts=hashes,encoder_sha256=sha(WEIGHT),script_sha256=sha(__file__),module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/cunfa_spatial_pairs.py'),
   roles=list(ROLE_NAMES),training_regions=int(fit.sum()),known_regions=int(known.sum()),fresh_regions=int(fresh.sum()),
   arms=['expanded_global','expanded_spatial_pyramid'],
   representation='Frozen MobileNetV3-small; given local and2.5x views; global mean vs1x1+2x2 spatial feature-map pooling; no automatic object detection',
   classifier='train-only StandardScaler + PCA up to12 dimensions + balanced LogisticRegression C=.1',
   selection='work-grouped OOF macro recall, tie structure recall; fixed before first fresh predictions',
   fresh_scope='one new attributed museum work with6 assistant region references, no fresh tone reference, same museum as added training; no expert or author-independent claim',
   role_reference_unit='given region context, not pixels or real strokes',unknown_ignored=True,previous_unknown_rejector_not_retrained=True,production_replaced=False,robot_connected=False)
 protocol['aborted_pre_fit_attempt']='Initial run stopped at feature replay assertion; matched historical TF32 settings before any fitting' if (ROOT/'protocol_aborted_feature_check.json').exists() else None
 write('protocol.json',protocol);torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
 net=mobilenet_v3_small(weights=None);net.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True));net=net.features.cuda().eval()
 arrays=[];features=[];reference={};oldnear=[];oldspatial=[];app=[]
 (ROOT/'views').mkdir(exist_ok=True)
 for r in rows:
  assert sha(r['image'])==r['image_sha256']
  if r['reference_image'] not in reference:reference[r['reference_image']]=Image.open(r['reference_image']).convert('RGB')
  ref=reference[r['reference_image']];local=Image.open(r['image']).convert('RGB');near=ref.crop(context_box(r['box'],ref.size,2.5));farbox=context_box(r['box'],ref.size,6);far=ref.crop(farbox)
  arrays.extend([appearance_tensor(v,ref,'reference_linear') for v in [local,near,far]])
  stats=np.concatenate([contrast_features(v,ref) for v in [local,near,far]]);grid=spatial_features(far,ref,normalized_roi(r['box'],farbox));app.append(np.concatenate([stats,grid]));features.append(stats[:28])
  local.save(ROOT/'views'/(r['id']+'_local.png'));near.save(ROOT/'views'/(r['id']+'_near.png'))
 means=[];spatial=[]
 with torch.inference_mode():
  for i in range(0,len(arrays),12):
   fm=net(torch.from_numpy(np.stack(arrays[i:i+12])).cuda());means.append(torch.nn.functional.normalize(fm.mean((2,3)),dim=1).cpu().numpy());spatial.append(pyramid_vector(fm).cpu().numpy())
 means=np.concatenate(means).reshape(len(rows),3,-1);spatial=np.concatenate(spatial).reshape(len(rows),3,-1);features=np.stack(features);app=np.stack(app)
 near=np.concatenate([means[:,:2].reshape(len(rows),-1)/np.sqrt(2),features],1)
 large=np.concatenate([means.reshape(len(rows),-1)/np.sqrt(3),app],1)
 cache=np.load(OLD/'features.npz');write('feature_replay.json',dict(near_max_error=float(abs(near[:112]-cache['near']).max()),large_max_error=float(abs(large[:112]-cache['spatial']).max())))
 assert np.allclose(near[:112],cache['near'],atol=1e-5);assert np.allclose(large[:112],cache['spatial'],atol=1e-5)
 x={'expanded_global':near,'expanded_spatial_pyramid':np.concatenate([spatial[:,:2].reshape(len(rows),-1)/np.sqrt(2),features],1)}
 metrics={};models={};preds={}
 for arm,xx in x.items():
  oof=np.full(len(rows),-100)
  for g in sorted(set(groups[fit])):
   tr=fit&(groups!=g);te=fit&(groups==g);model=make_probe(int(tr.sum())).fit(xx[tr],y[tr]);oof[te]=model.predict(xx[te])
  metrics[arm]=role_metric(y[fit],oof[fit]);models[arm]=make_probe(int(fit.sum())).fit(xx[fit],y[fit]);print('TRAIN_OOF',arm,json.dumps(metrics[arm]),flush=True)
 selected=max(x,key=lambda a:(metrics[a]['macro_recall'],metrics[a]['per_role']['outline_context']['agreement']))
 write('training_selection.json',dict(selected_arm=selected,training_oof_metrics=metrics,before_fresh_predictions=True))
 for arm,xx in x.items():preds[arm]=models[arm].predict(xx).astype('int64')
 frozen=joblib.load(PREV/'photometry_scoped_models.joblib')['role_models']['linear_plus_contrast'];preds['frozen_photometry']=frozen.predict(near).astype('int64')
 prior=joblib.load(OLD/'relational_models.joblib');preds['frozen_relational']=predict_role_probe(prior['models'][prior['selected_arm']],large,app)
 checkscores={split:{a:role_metric(y[mask],p[mask]) for a,p in preds.items()} for split,mask in [('known12',known),('fresh_work',fresh)]}
 result=dict(selected_arm=selected,training_oof_metrics=metrics,check_metrics=checkscores,records=[dict(id=r['id'],split=r['split'],role_reference=None if y[i]==-100 else ROLE_NAMES[y[i]],predictions={a:ROLE_NAMES[int(p[i])] for a,p in preds.items()},selected_prediction=ROLE_NAMES[int(preds[selected][i])]) for i,r in enumerate(rows)],model_adopted=False,independent_expert_validation_passed=False,robot_connected=False)
 write('results.json',result);joblib.dump(dict(models=models,selected_arm=selected,protocol=protocol),ROOT/'spatial_models.joblib');np.savez_compressed(ROOT/'features.npz',**x,near=near,large=large,appearance=app,targets=y)
 assert all(sha(p)==h for p,h in hashes.items());write('verification.json',dict(old_artifacts_unchanged=True,old112_feature_replay=True,no_work_overlap=True,selection_training_only=True,unknown_not_negative=True,fresh_work_not_used_in_fit=True,robot_connected=False))
 print('SELECTED',selected,flush=True);print('CHECKS',json.dumps(checkscores),flush=True)
if __name__=='__main__':main()
