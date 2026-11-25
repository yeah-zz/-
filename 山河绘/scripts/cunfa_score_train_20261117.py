# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import torch,numpy as np,joblib
from PIL import Image
from torchvision.models import mobilenet_v3_small
from gongbi_training.cunfa_input_scale import preprocess
from gongbi_training.cunfa_concept_supervision import CONCEPTS,IGNORE,concept_targets,validate_concept_split
from gongbi_training.cunfa_score_alignment import *
ROOT=LAB/'cunfa_score_alignment_20261117';OLD=LAB/'cunfa_concept_supervision_20261116'
REF=LAB/'cunfa_role_reference_20261112';WEIGHT=LAB/'cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')

def metric(y,scores,scope):
 out={}
 for j,k in enumerate(CONCEPTS):
  valid=scope&(y[:,j]!=IGNORE);truth=y[valid,j];scores0=scores[valid,j];pred=(scores0>=.5).astype('int64')
  tp=int(((truth==1)&(pred==1)).sum());tn=int(((truth==0)&(pred==0)).sum());fp=int(((truth==0)&(pred==1)).sum());fn=int(((truth==1)&(pred==0)).sum())
  out[k]=dict(known=int(valid.sum()),positive=int((truth==1).sum()),negative=int((truth==0).sum()),
   agreement=tp+tn,tp=tp,tn=tn,fp=fp,fn=fn,
   majority=max(int((truth==1).sum()),int((truth==0).sum())),
   balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
   brier=float(np.mean((scores0-truth)**2)) if len(truth) else None)
 return out

def main():
 assert not (ROOT/'protocol.json').exists(),'Frozen experiment'
 ds=read(ROOT/'reference_manifest.json');rows=ds['records'];validate_concept_split(rows)
 oldrows=read(OLD/'reference_manifest_corrected.json')['records'];oldresult=read(OLD/'results.json')
 assert [r['id'] for r in rows[:25]]==[r['id'] for r in oldrows]
 for r in rows:assert sha(r['image'])==r['image_sha256']
 checks=read(ROOT/'contract_checks.json');assert all(v for k,v in checks.items() if k!='robot_connected')
 old_artifacts={p.name:sha(p) for p in [OLD/'concept_heads.joblib',OLD/'concept_style_head.joblib',OLD/'features_and_scores.npz',OLD/'reference_manifest_corrected.json',OLD/'results.json']}
 protocol=dict(prepared_before_fit=True,reference_manifest_sha256=sha(ROOT/'reference_manifest.json'),
  code_sha256=sha(__file__),module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/cunfa_score_alignment.py'),
  old_artifacts=old_artifacts,encoder_sha256=sha(WEIGHT),old_feature_cache_reused=True,
  base_heads='same balanced LogisticRegression C=1, partial labels; encoder frozen',
  arms=['legacy_groundtruth_to_soft','oof_original_raw','oof_expanded_raw','oof_expanded_sigmoid'],
  threshold=.5,no_test_tuning=True,style_head='StandardScaler fit training unit means + linear SVM C=1',
  oof='Base attribute heads leave an entire training illustration out; all AAM images remain one illustrator family',
  calibration='StandardScaler + balanced logistic sigmoid fitted to known TRAINING OOF decision scores only',
  internal_calibrated_oof_not_reported_as_independent_accuracy=True,
  style_training_input='Means of TRAINING OOF attribute predictions; no manual test attributes and no image-feature bypass',
  style_training_named_units=4,new_training_images_only_supervise_morphology=True,
  original_test_scope='Same known HKMA7 regions and2 named units from1 photo',
  added_test_scope='4 newly fixed shape regions from same previously reviewed Xia_Gui work; not new independent work',
  mixed_unknown_recognition_trained=False,object_or_stroke_model_trained=False,
  new_independent_validation=False,robot_connected=False)
 write('protocol.json',protocol)
 # Existing 25 features are reused exactly; only new23 regions are encoded.
 cached=np.load(OLD/'features_and_scores.npz');assert sha(WEIGHT)==read(OLD/'protocol.json')['encoder_weights_sha256']
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.use_deterministic_algorithms(True);net=mobilenet_v3_small(weights=None)
 net.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True));net=net.features.cuda().eval()
 arrays=[preprocess(Image.open(r['image']).convert('RGB'),True) for r in rows[25:]];features=[]
 with torch.inference_mode():
  for i in range(0,len(arrays),12):
   z=net(torch.from_numpy(np.stack(arrays[i:i+12])).cuda()).mean((2,3));features.append(torch.nn.functional.normalize(z,dim=1).cpu().numpy())
 x=np.concatenate([cached['features'],np.concatenate(features)]);y=np.stack([concept_targets(r) for r in rows])
 train=np.array([r['split']=='train' for r in rows]);test=~train;oldtrain=train&np.array([not r['new_region_reference'] for r in rows])
 units=np.array([r['reference_unit'] for r in rows]);original=fit_training_concepts(x[oldtrain],y[oldtrain],units[oldtrain])
 expanded=fit_training_concepts(x[train],y[train],units[train]);oldheads=joblib.load(OLD/'concept_heads.joblib')['heads']
 legacy=np.stack([oldheads[k].predict_proba(x)[:,1] for k in CONCEPTS],axis=1)
 assert np.allclose(legacy[:25],cached['scores'],atol=1e-12)
 original_raw=predict_concepts(original,x,False);assert np.allclose(original_raw,legacy,atol=1e-12)
 expanded_raw=predict_concepts(expanded,x,False);expanded_cal=predict_concepts(expanded,x,True)
 scores={'legacy_groundtruth_to_soft':legacy,'oof_original_raw':original_raw,'oof_expanded_raw':expanded_raw,'oof_expanded_sigmoid':expanded_cal}
 scopes={'known_HKMA_regression':test&np.array([r['evaluation_scope']=='known_HKMA_regression' for r in rows]),
         'known_work_new_shape_probe':test&np.array([r['evaluation_scope']=='known_work_new_shape_probe' for r in rows]),'all_known_shape_probes':test}
 metrics={arm:{name:metric(y,s,mask) for name,mask in scopes.items()} for arm,s in scores.items()}
 refs=[r for r in read(REF/'source_references.json')['records'] if r['publisher'] in {'AAM','HKMA'}]
 named_train=[r['id'] for r in refs if r['publisher']=='AAM'];styley=np.array([r['style_target'] for r in refs if r['publisher']=='AAM'])
 named_test=[r['id'] for r in refs if r['publisher']=='HKMA'];stylepred={};head_artifacts={}
 # Legacy head replay only. It was trained on manual training concept means.
 legacy_style=joblib.load(OLD/'concept_style_head.joblib')['model']
 stylepred['legacy_groundtruth_to_soft']=legacy_style.predict(aggregate_regions(legacy,units,named_test)).astype('int64')
 for arm,bundle,mask,cal in [('oof_original_raw',original,oldtrain,False),('oof_expanded_raw',expanded,train,False),('oof_expanded_sigmoid',expanded,train,True)]:
  oof_scores=scores_from_decisions(bundle,bundle['oof_decision'],cal)
  head=fit_style_from_oof(oof_scores,units[mask],named_train,styley)
  head_artifacts[arm]=head;stylepred[arm]=head.predict(aggregate_regions(scores[arm],units,named_test)).astype('int64')
 # The direct-image baseline is retained from the previous frozen run, not
 # retrained on added pictures of other named methods.
 styles=[]
 for i,r in enumerate([r for r in refs if r['publisher']=='HKMA']):
  previous=next(z for z in oldresult['style_probe'] if z['id']==r['id'])
  assert int(stylepred['legacy_groundtruth_to_soft'][i])==previous['concept_bottleneck']
  styles.append(dict(id=r['id'],source_label=r['local_style'],target=int(r['style_target']),direct_previous=previous['direct'],predictions={k:int(v[i]) for k,v in stylepred.items()}))
 records=[]
 for i,r in enumerate(rows):
  records.append(dict(id=r['id'],split=r['split'],evaluation_scope=r['evaluation_scope'],
   targets=[None if v==IGNORE else int(v) for v in y[i]],
   outputs={arm:dict(scores=s[i].tolist(),predicted=(s[i]>=.5).astype('int64').tolist(),
    errors=[CONCEPTS[j] for j in range(3) if y[i,j]!=IGNORE and int(s[i,j]>=.5)!=y[i,j]]) for arm,s in scores.items()}))
 joblib.dump(dict(original=original,expanded=expanded,style_heads=head_artifacts,protocol=protocol,production_use=False),ROOT/'score_alignment_models.joblib')
 np.savez_compressed(ROOT/'features_and_scores.npz',features=x,targets=y,**scores)
 result=dict(protocol=protocol,shape_metrics=metrics,style_probe=styles,
  style_summary={arm:sum(r['predictions'][arm]==r['target'] for r in styles) for arm in stylepred},
  direct_previous_style_agreement=2,style_probe_units=2,style_probe_source_photos=1,
  records=records,training_folds={'original':original['folds'],'expanded':expanded['folds']},
  calibration_parameters=[dict(mean=float(m.named_steps['standardscaler'].mean_[0]),scale=float(m.named_steps['standardscaler'].scale_[0]),coef=float(m.named_steps['logisticregression'].coef_[0,0]),intercept=float(m.named_steps['logisticregression'].intercept_[0])) for m in expanded['calibrators']],
  expert_accuracy=False,model_adopted=False,independent_validation_passed=False,tree_skill_training_started=False,robot_connected=False)
 write('results.json',result)
 unchanged=all(sha(OLD/name)==value for name,value in old_artifacts.items());assert unchanged
 write('verification.json',dict(old_artifacts_unchanged=unchanged,original25_features_and_targets_preserved=True,
  legacy_predictions_replayed=True,original_head_refit_matches_legacy=True,
  illustrationwise_base_oof=True,train_only_sigmoid_fit=True,train_only_style_scaler=True,
  external_reference_labels_not_used_for_fit=True,new_regions_not_given_style_targets=True,
  no_claim_full_pipeline_oof_unbiased_validation=True,single_training_illustrator_disclosed=True,
  known_work_new_crops_not_independent_works=True,full_stroke_labels_added=0,
  production_replaced=False,robot_connected=False))
 print('STYLE',json.dumps(result['style_summary']),flush=True)
 for arm in ['legacy_groundtruth_to_soft','oof_expanded_raw','oof_expanded_sigmoid']:
  print(arm,json.dumps(metrics[arm],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
