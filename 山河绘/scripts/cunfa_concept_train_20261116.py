# -*- coding: utf-8 -*-
"""First region-concept supervision probe, not a validated cunfa recognizer."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,joblib
from PIL import Image
from torchvision.models import mobilenet_v3_small
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from gongbi_training.cunfa_input_scale import preprocess
from gongbi_training.cunfa_concept_supervision import CONCEPTS,IGNORE,concept_targets,validate_concept_split
ROOT=LAB/'cunfa_concept_supervision_20261116';OLD=LAB/'cunfa_role_reference_20261112'
WEIGHTS=LAB/'cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')

def main():
 assert not (ROOT/'protocol.json').exists(),'Frozen run; new settings require another directory'
 rows=read(ROOT/'reference_manifest.json')['records'];validate_concept_split(rows)
 refs=[r for r in read(OLD/'source_references.json')['records'] if r['publisher'] in {'AAM','HKMA'}]
 checks=read(ROOT/'contract_checks.json');assert all(v for k,v in checks.items() if k!='robot_connected')
 for r in rows:assert sha(r['image'])==r['image_sha256'] and sha(r['reference_image'])==next(x['crop_sha256'] for x in refs if x['id']==r['reference_unit'])
 y=np.stack([concept_targets(r) for r in rows]);train=np.array([r['split']=='train' for r in rows]);test=~train
 assert len({r['reference_unit'] for r in rows if r['split']=='train'})==4
 assert len({r['reference_unit'] for r in rows if r['split']=='test'})==2
 protocol=dict(prepared_before_fit=True,reference_manifest_sha256=sha(ROOT/'reference_manifest.json'),
  definitions_sha256=sha(ROOT/'definitions.json'),module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/cunfa_concept_supervision.py'),
  script_sha256=sha(__file__),encoder_weights_sha256=sha(WEIGHTS),encoder='frozen ImageNet MobileNetV3-small',
  concept_heads='3 independently masked binary LogisticRegression heads, balanced class weights, C=1',
  threshold=.5,random_state=0,max_iter=2000,concepts=list(CONCEPTS),
  split='AAM4 illustrations train / HKMA2 examples from one source photograph test',
  patches_train=int(train.sum()),patches_test=int(test.sum()),independent_train_units=4,independent_test_source_images=1,
  concepts_labelled_from_visual_regions=True,concepts_not_from_style_names=True,
  role='unknown for every shape patch; shape does not imply outline/cun role',
  unknown_targets=IGNORE,no_test_tuning=True,new_independent_validation=False,
  style_probe='AAM4 named units train / HKMA2 named units test: direct whole-image SVM vs concept-only bottleneck; given review windows required',
  bottleneck_training='Mean of available assistant concept references in each training unit; no raw-image features in style head',
  bottleneck_test='Mean of model concept scores in same predefined review windows; no test concept labels supplied',
  unknown_recognition_not_trained=True,full_rock_instance_detection_not_trained=True,robot_connected=False)
 write('protocol.json',protocol)
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.use_deterministic_algorithms(True);net=mobilenet_v3_small(weights=None)
 net.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True));net=net.features.cuda().eval()
 for p in net.parameters():p.requires_grad_(False)
 paths=[r['image'] for r in rows]+[r['image'] for r in refs]
 arrays=[preprocess(Image.open(p).convert('RGB'),True) for p in paths];encoded=[]
 with torch.inference_mode():
  for i in range(0,len(arrays),12):
   z=net(torch.from_numpy(np.stack(arrays[i:i+12])).cuda()).mean((2,3))
   encoded.append(torch.nn.functional.normalize(z,dim=1).cpu().numpy())
 features=np.concatenate(encoded);x=features[:len(rows)];full=features[len(rows):]
 scores=np.zeros_like(y,dtype='float64');heads={};metrics={}
 for j,k in enumerate(CONCEPTS):
  fit=train&(y[:,j]!=IGNORE);assert set(y[fit,j])=={0,1},k
  model=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',random_state=0,max_iter=2000)
  model.fit(x[fit],y[fit,j]);scores[:,j]=model.predict_proba(x)[:,list(model.classes_).index(1)];heads[k]=model
  check=test&(y[:,j]!=IGNORE);truth=y[check,j];pred=(scores[check,j]>=.5).astype('int64')
  tp=int(((truth==1)&(pred==1)).sum());tn=int(((truth==0)&(pred==0)).sum())
  fp=int(((truth==0)&(pred==1)).sum());fn=int(((truth==1)&(pred==0)).sum())
  majority=max(int((truth==0).sum()),int((truth==1).sum()))
  metrics[k]=dict(known_train=int(fit.sum()),known_test=int(check.sum()),positive_test=int((truth==1).sum()),negative_test=int((truth==0).sum()),
   agreement_count=tp+tn,tp=tp,tn=tn,fp=fp,fn=fn,majority_agreement_count=majority,
   both_polarities_tested=bool((truth==0).any() and (truth==1).any()),
   balanced_accuracy=(tp/max(1,tp+fn)+tn/max(1,tn+fp))/2 if (truth==0).any() and (truth==1).any() else None)
  print(k,json.dumps(metrics[k]),flush=True)
 joblib.dump(dict(heads=heads,concepts=CONCEPTS,protocol=protocol,production_use=False),ROOT/'concept_heads.joblib')
 loaded=joblib.load(ROOT/'concept_heads.joblib')
 for j,k in enumerate(CONCEPTS):assert np.allclose(loaded['heads'][k].predict_proba(x)[:,1],scores[:,j])
 # Illustrative concept bottleneck: only visible morphology means, NOT a
 # hard-coded style rule and NOT direct final-style training on cropped marks.
 train_units=[i for i,r in enumerate(refs) if r['publisher']=='AAM'];test_units=[i for i,r in enumerate(refs) if r['publisher']=='HKMA']
 target=np.array([r['style_target'] for r in refs]);means=[];observed=[]
 for r in refs:
  indices=[i for i,z in enumerate(rows) if z['reference_unit']==r['id']]
  means.append(scores[indices].mean(0));gt=y[indices].astype('float64');gt[gt==IGNORE]=np.nan
  observed.append(np.nanmean(gt,axis=0))
 means=np.stack(means);observed=np.stack(observed)
 assert np.isfinite(observed).all()
 style_head=SVC(C=1,kernel='linear',class_weight='balanced').fit(observed[train_units],target[train_units])
 direct=SVC(C=1,kernel='linear',class_weight='balanced').fit(full[train_units],target[train_units])
 bottleneck=style_head.predict(means[test_units]);base=direct.predict(full[test_units])
 # This oracle probe diagnoses how much inferred concept error matters. It is
 # explicitly reference-assisted and never part of the normal model pipeline.
 oracle=style_head.predict(observed[test_units])
 joblib.dump(dict(model=style_head,concepts=CONCEPTS,trained_unit_ids=[refs[i]['id'] for i in train_units],production_use=False),ROOT/'concept_style_head.joblib')
 styles=[dict(id=refs[i]['id'],source_label=refs[i]['local_style'],target=int(target[i]),
  direct=int(base[j]),concept_bottleneck=int(bottleneck[j]),reference_oracle=int(oracle[j]),
  predicted_concepts=means[i].tolist(),reference_concepts=observed[i].tolist(),
  oracle_is_not_model_inference=True,provided_region_windows_required=True) for j,i in enumerate(test_units)]
 records=[dict(id=r['id'],split=r['split'],reference_unit=r['reference_unit'],source_group=r['source_group'],
  targets=[None if v==IGNORE else int(v) for v in y[i]],scores=scores[i].tolist(),
  predicted=(scores[i]>=.5).astype('int64').tolist(),
  errors=[CONCEPTS[j] for j in range(3) if y[i,j]!=IGNORE and int(scores[i,j]>=.5)!=y[i,j]],
  visual_evidence=r['visual_evidence'],role='unknown') for i,r in enumerate(rows)]
 result=dict(protocol=protocol,concept_metrics=metrics,records=records,style_probe=styles,
  style_summary={k:sum(r[k]==r['target'] for r in styles) for k in ['direct','concept_bottleneck','reference_oracle']},
  style_probe_units=2,style_probe_source_images=1,expert_accuracy=False,
  previous_63_and_13_experiment_not_replaced=True,independent_validation_passed=False,
  tree_skill_training_started=False,model_adopted=False,robot_connected=False)
 write('results.json',result);np.savez_compressed(ROOT/'features_and_scores.npz',features=x,full_image_features=full,targets=y,scores=scores)
 write('verification.json',dict(source_group_disjoint=True,source_image_disjoint=True,illustration_crops_grouped=True,
  train_target_mask_matches_contract=True,unknown_not_negative=True,no_final_style_targets_on_mark_crops=True,
  style_bottleneck_no_image_feature_bypass=True,only_training_concept_targets_used_by_normal_pipeline=True,
  concept_head_roundtrip=True,parameters_fixed_before_fit=True,reference_oracle_disclosed=True,
  old_63_and_13_weights_untouched=True,object_gate_is_reference_constraint_not_detector=True,
  fully_resolved_roles=0,complete_stroke_labels=0,unknown_recognition_trained=False,
  production_replaced=False,robot_connected=False))
 print('STYLE_PROBE',json.dumps(result['style_summary']),flush=True)

if __name__=='__main__':main()
