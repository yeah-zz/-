# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,joblib
from PIL import Image
from torchvision.models import mobilenet_v3_small
from gongbi_training.cunfa_photometry import appearance_tensor,appearance_canvas,contrast_features
from gongbi_training.cunfa_context_roles import context_box,context_role_target,ROLE_NAMES
from gongbi_training.cunfa_relational_roles import *
from cunfa_photometry_train_20261117 import role_metric
ROOT=LAB/'cunfa_relational_roles_20261117';OLD=LAB/'cunfa_photometry_20261117';SOURCE=LAB/'cunfa_cross_source_20261117'
WEIGHT=LAB/'cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
ARMS=['expanded_flat_near','expanded_flat_spatial','expanded_hierarchical','expanded_hierarchical_rotated']
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')

def main():
 assert not (ROOT/'protocol.json').exists(),'Frozen experiment'
 data=read(ROOT/'reference_manifest.json');rows=data['records'];oldrows=read(SOURCE/'reference_manifest.json')['records']
 assert [r['id'] for r in rows[:93]]==[r['id'] for r in oldrows]
 for r in rows:assert sha(r['image'])==r['image_sha256']
 checks=read(ROOT/'contract_checks.json');assert all(v for k,v in checks.items() if k!='robot_connected')
 y=np.array([context_role_target(r) for r in rows]);train=np.array([r['split']=='train' for r in rows]);fit=train&(y!=-100);test=~train&(y!=-100)
 groups=np.array([r['role_fold_group'] for r in rows]);oldfiles={n:sha(OLD/n) for n in ['photometry_scoped_models.joblib','features.npz','results.json']}
 protocol=dict(prepared_before_fit=True,reference_manifest_sha256=sha(ROOT/'reference_manifest.json'),old_artifacts=oldfiles,
  script_sha256=sha(__file__),module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/cunfa_relational_roles.py'),encoder_sha256=sha(WEIGHT),
  old_roles_unchanged=True,training_original_regions=int(fit.sum()),known_check_regions=int(test.sum()),
  added_views=19,added_role_references=15,new_independent_works=0,
  grouped_folds='whole book illustration or whole reused video work; all video timestamps/crops one group; not author-independent',
  input_views='given crop,2.5x window,6x window; all source-clipped; no automatic rock-instance detection',
  spatial_features='3x3 relative darkness/variation grid plus supplied ROI location in larger window',
  photometry='same source-wide linear reference as previous run, no local equalization',
  role_arms=ARMS,classifier='balanced LogisticRegression C=1 with train-only StandardScaler',
  hierarchy='appearance: marked/tone/unmarked; marked role: structure/interior; exploratory probability product, not calibrated',
  rotations='original plus90/180/270 degree views; all views of a reference stay in one fold; sample weights.25 sum to one per original',
  selection='training grouped OOF: require every class recall>=.5, then highest macro recall, then structure recall; if none qualify no selected candidate',
  known_checks_previously_opened=True,selection_saved_before_known_check_metrics=True,
  no_source_targets_as_model_input=True,no_unknown_rejector_retraining=True,production_replaced=False,robot_connected=False)
 write('protocol.json',protocol);print('FROZEN',int(fit.sum()),int(test.sum()),flush=True)
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
 net=mobilenet_v3_small(weights=None);net.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True));net=net.features.cuda().eval()
 cache=np.load(OLD/'features.npz');arrays=[];indices={};stats={};grids={};positions={};references={}
 transposes=[None,Image.Transpose.ROTATE_90,Image.Transpose.ROTATE_180,Image.Transpose.ROTATE_270]
 for i,r in enumerate(rows):
  if r['reference_image'] not in references:references[r['reference_image']]=Image.open(r['reference_image']).convert('RGB')
  ref=references[r['reference_image']];local=Image.open(r['image']).convert('RGB');nearbox=context_box(r['box'],ref.size,2.5);farbox=context_box(r['box'],ref.size,6.)
  pair=[local,ref.crop(nearbox),ref.crop(farbox)];roi=normalized_roi(r['box'],farbox);positions[i]=roi.tolist()
  for k in range(4 if fit[i] else 1):
   views=pair if k==0 else [v.transpose(transposes[k]) for v in pair];roi1=roi.copy()
   for _ in range(k):roi1=rotate_roi_ccw(roi1)
   stats[i,k]=np.concatenate([contrast_features(v,ref) for v in views]);grids[i,k]=spatial_features(views[2],ref,roi1)
   index=[]
   for j,v in enumerate(views):
    if k==0 and i<93 and j<2:index.append(None)
    else:index.append(len(arrays));arrays.append(appearance_tensor(v,ref,'reference_linear'))
   indices[i,k]=index
  for v,name in zip(pair,['local','near','far']):appearance_canvas(v,ref,'reference_linear').save(ROOT/'views'/(r['id']+'_'+name+'.png'))
 encoded=[]
 with torch.inference_mode():
  for i in range(0,len(arrays),12):
   z=net(torch.from_numpy(np.stack(arrays[i:i+12])).cuda()).mean((2,3));encoded.append(torch.nn.functional.normalize(z,dim=1).cpu().numpy())
 encoded=np.concatenate(encoded);near_features=[];spatial_deep=[];appearance=[];aug_deep={};aug_app={}
 for i in range(len(rows)):
  for k in range(4 if fit[i] else 1):
   idx=indices[i,k]
   if k==0 and i<93:
    # Recover the exact existing local/near normalized vectors from their cache.
    oldnear=cache['reference_linear_context'][i]*np.sqrt(2);parts=[oldnear[:576],oldnear[576:],encoded[idx[2]]]
   else:parts=[encoded[j] for j in idx]
   pstats=np.concatenate([stats[i,k],grids[i,k]])
   deep=np.concatenate([np.concatenate(parts)/np.sqrt(3),pstats])
   if k==0:
    near_features.append(np.concatenate([np.concatenate(parts[:2])/np.sqrt(2),stats[i,0][:28]]));spatial_deep.append(deep);appearance.append(pstats)
   else:aug_deep[i,k]=deep;aug_app[i,k]=pstats
 near_features=np.stack(near_features);spatial_deep=np.stack(spatial_deep);appearance=np.stack(appearance)
 assert np.allclose(near_features[:93],cache['linear_plus_contrast'],atol=1e-12)
 features={'expanded_flat_near':near_features,'expanded_flat_spatial':spatial_deep,'expanded_hierarchical':spatial_deep,'expanded_hierarchical_rotated':spatial_deep}
 models={};oofmetrics={};allpred={}
 def training_arrays(mask,arm):
  ids=list(np.flatnonzero(mask));xd=features[arm][mask];xa=appearance[mask];yy=y[mask];weights=None
  if arm=='expanded_hierarchical_rotated':
   keys=[(i,k) for i in ids for k in [1,2,3]]
   xd=np.concatenate([xd,np.stack([aug_deep[key] for key in keys])]);xa=np.concatenate([xa,np.stack([aug_app[key] for key in keys])]);yy=np.concatenate([yy,np.array([y[i] for i,k in keys])]);weights=np.full(len(yy),.25)
   assert np.isclose(weights.sum(),len(ids))
  return xd,xa,yy,weights
 for arm in ARMS:
  pred=np.full(len(rows),-100,dtype='int64');hier=arm.startswith('expanded_hierarchical')
  for group in sorted(set(groups[fit])):
   tr=fit&(groups!=group);te=fit&(groups==group);xd,xa,yy,weights=training_arrays(tr,arm)
   m=fit_role_probe(xd,xa,yy,hier,weights);pred[te]=predict_role_probe(m,features[arm][te],appearance[te])
  oofmetrics[arm]=role_metric(y[fit],pred[fit]);xd,xa,yy,weights=training_arrays(fit,arm)
  models[arm]=fit_role_probe(xd,xa,yy,hier,weights);allpred[arm]=predict_role_probe(models[arm],features[arm],appearance)
  print('TRAIN_OOF',arm,oofmetrics[arm]['agreement'],oofmetrics[arm]['macro_recall'],flush=True)
 def eligible(a):return all(v['agreement']/v['count']>=.5 for v in oofmetrics[a]['per_role'].values() if v['count'])
 qualifying=[a for a in ARMS if eligible(a)]
 selected=max(qualifying,key=lambda a:(oofmetrics[a]['macro_recall'],oofmetrics[a]['per_role']['outline_context']['agreement'])) if qualifying else None
 write('training_selection.json',dict(selected_arm=selected,eligible=qualifying,metrics=oofmetrics,criterion=protocol['selection'],saved_before_known_check_scoring=True))
 frozen=joblib.load(OLD/'photometry_scoped_models.joblib')['role_models']['linear_plus_contrast']
 # Only frozen previous cases form the old-model comparison; new development
 # cases have old-model predictions for diagnosis without entering old training.
 allpred['frozen_previous']=frozen.predict(near_features).astype('int64')
 metrics={a:role_metric(y[test],pred[test]) for a,pred in allpred.items()}
 records=[dict(id=r['id'],split=r['split'],role_reference=None if y[i]==-100 else ROLE_NAMES[y[i]],
  predictions={a:ROLE_NAMES[int(pred[i])] for a,pred in allpred.items()},
  selected_prediction=None if selected is None else ROLE_NAMES[int(allpred[selected][i])]) for i,r in enumerate(rows)]
 result=dict(protocol=protocol,selected_arm=selected,training_oof_metrics=oofmetrics,known_check_metrics=metrics,records=records,
  independent_validation_passed=False,full_rock_detection_learned=False,model_adopted=False,robot_connected=False)
 write('results.json',result);joblib.dump(dict(models=models,selected_arm=selected,protocol=protocol,production_use=False),ROOT/'relational_models.joblib')
 np.savez_compressed(ROOT/'features.npz',near=near_features,spatial=spatial_deep,appearance=appearance,role_targets=y)
 assert all(sha(OLD/n)==h for n,h in oldfiles.items())
 write('verification.json',dict(previous_models_features_and_targets_unchanged=True,original93_near_features_replayed=True,
  video_phases_one_fold=True,no_before_frame_or_reference_mask_in_model_input=True,
  no_unknown_targets_as_negative=True,rotation_total_weight_per_original_one=True,
  larger_window_not_called_automatic_whole_rock=True,selection_training_only=True,
  no_new_independent_test_claimed=True,production_replaced=False,robot_connected=False))
 print('SELECTED',selected,flush=True);print('KNOWN_CHECKS',json.dumps(metrics),flush=True)
if __name__=='__main__':main()
