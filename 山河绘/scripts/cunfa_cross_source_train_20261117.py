# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import torch,numpy as np,joblib
from PIL import Image
from torchvision.models import mobilenet_v3_small
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from gongbi_training.cunfa_input_scale import preprocess
from gongbi_training.cunfa_concept_supervision import CONCEPTS,concept_targets,validate_concept_split
from gongbi_training.cunfa_score_alignment import *
from gongbi_training.cunfa_context_roles import ROLE_NAMES,context_role_target,context_box
from cunfa_score_train_20261117 import metric
ROOT=LAB/'cunfa_cross_source_20261117';OLD=LAB/'cunfa_score_alignment_20261117'
WEIGHT=LAB/'cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 assert not (ROOT/'protocol.json').exists(),'Frozen run'
 ds=read(ROOT/'reference_manifest.json');rows=ds['records'];validate_concept_split(rows)
 for r in rows:assert sha(r['image'])==r['image_sha256']
 checks=read(ROOT/'contract_checks.json');assert all(v for k,v in checks.items() if k!='robot_connected')
 oldfiles={n:sha(OLD/n) for n in ['score_alignment_models.joblib','reference_manifest.json','results.json','features_and_scores.npz']}
 protocol=dict(prepared_before_fit=True,reference_manifest_sha256=sha(ROOT/'reference_manifest.json'),
  source_inventory_sha256=sha(ROOT/'source_inventory.json'),new_test_units_sha256=sha(ROOT/'new_test_units.json'),
  old_artifacts=oldfiles,encoder_sha256=sha(WEIGHT),script_sha256=sha(__file__),
  train_sources=['existing AAM illustrations','new Qingpu teaching book, all figures grouped as one unresolved-illustrator source family'],
  new_holdout_sources='NTMOFA printed36 examples, 3works; no holdout target used for fitting or parameter choice',
  known_regressions='HKMA7 and XueTang4 kept as checks, not new independent works',
  attributes='Same C=1 balanced partial LogisticRegression; illustration-wise base OOF + training-only sigmoid',
  style_heads='old frozen4 AAM examples; new attributes with4 examples; new attributes with6 named examples',
  supplied_windows=True,source_caption_not_local_mark_role_truth=True,threshold=.5,no_test_tuning=True,
  contextual_roles=dict(classes=list(ROLE_NAMES),model='StandardScaler+balanced LogisticRegression C=1',
   arms=['local_only','local_plus_2.5_context'],task='given-region contextual appearance, not per-pixel roles or stroke instances'),
  Mi_dot_not_in_two_class_training=True,unknown_recognition_not_trained=True,
  expert_local_truth=False,verified_author_disjointness=False,production_replaced=False,robot_connected=False)
 write('protocol.json',protocol);print('FROZEN',len(rows),flush=True)
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.use_deterministic_algorithms(True);net=mobilenet_v3_small(weights=None)
 net.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True));net=net.features.cuda().eval()
 arrays=[];local_indices={};context_indices={};style_indices={}
 # Old local features are reused exactly; all context views are independently
 # generated from unmodified reference images, without target-colour overlays.
 for i,r in enumerate(rows):
  if i>=48:local_indices[i]=len(arrays);arrays.append(preprocess(Image.open(r['image']),True))
  ref=Image.open(r['reference_image']).convert('RGB');box=context_box(r['box'],ref.size)
  rview=ref.crop(box);rview.save(ROOT/'views'/(r['id']+'_context.png'))
  context_indices[i]=len(arrays);arrays.append(preprocess(rview,True))
 for s in ds['style_units']:
  style_indices[s['id']]=len(arrays);arrays.append(preprocess(Image.open(s['image']),True))
 features=[]
 with torch.inference_mode():
  for i in range(0,len(arrays),12):
   z=net(torch.from_numpy(np.stack(arrays[i:i+12])).cuda()).mean((2,3));features.append(torch.nn.functional.normalize(z,dim=1).cpu().numpy())
 encoded=np.concatenate(features);oldcache=np.load(OLD/'features_and_scores.npz')
 x=np.concatenate([oldcache['features'],np.stack([encoded[local_indices[i]] for i in range(48,len(rows))])]);ctx=np.stack([encoded[context_indices[i]] for i in range(len(rows))]);both=np.concatenate([x,ctx],1)/np.sqrt(2)
 y=np.stack([concept_targets(r) for r in rows]);train=np.array([r['split']=='train' for r in rows]);test=~train;units=np.array([r['reference_unit'] for r in rows]);objects=np.array([r['object_type'] for r in rows])
 oldmodel=joblib.load(OLD/'score_alignment_models.joblib');frozen=predict_concepts(oldmodel['expanded'],x,True)
 assert np.allclose(frozen[:48],oldcache['oof_expanded_sigmoid'],atol=1e-12)
 bundle=fit_training_concepts(x[train],y[train],units[train]);candidate=predict_concepts(bundle,x,True)
 scopes={'new_NT_source_probe':test&np.array([r['evaluation_scope']=='new_NT_source_probe' for r in rows]),
  'known_HKMA_regression':test&np.array([r['evaluation_scope']=='known_HKMA_regression' for r in rows]),
  'known_work_new_shape_probe':test&np.array([r['evaluation_scope']=='known_work_new_shape_probe' for r in rows])}
 attrmetrics={a:{n:metric(y,s,m) for n,m in scopes.items()} for a,s in [('frozen_previous',frozen),('new_source_candidate',candidate)]}
 # Role probes use only checked Qingpu region references. Old attribute-only
 # regions and every uncertain/new tree region have no role loss.
 role_y=np.array([context_role_target(r) for r in rows]);role_fit=train&(role_y!=-100);role_test=test&(role_y!=-100)
 assert set(role_y[role_fit])==set(range(4));role_heads={};role_outputs={};role_metrics={}
 for name,f in [('local_only',x),('local_plus_context',both)]:
  model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',random_state=0,max_iter=2000))
  model.fit(f[role_fit],role_y[role_fit]);pred=model.predict(f).astype('int64');prob=model.predict_proba(f);role_heads[name]=model;role_outputs[name]=(pred,prob)
  matrix=confusion_matrix(role_y[role_test],pred[role_test],labels=list(range(4)));role_metrics[name]=dict(agreement=int(np.trace(matrix)),regions=int(role_test.sum()),confusion=matrix.tolist(),
   per_role={k:dict(reference_count=int((role_y[role_test]==j).sum()),agreement=int(matrix[j,j])) for j,k in enumerate(ROLE_NAMES)},
   interpretation='assistant region-context agreement, not pixel/artist-stroke accuracy')
 # Style comparisons only use provided rock/teaching region windows. No source
 # role truth, test concepts, or test method labels become input to the heads.
 aggregate_allowed=np.isin(objects,['rock','teaching_mark']);styles=ds['style_units'];named4=[r['id'] for r in styles if r['split']=='train' and r['id'].startswith('aam_')];named6=[r['id'] for r in styles if r['split']=='train'];test_names=[r['id'] for r in styles if r['split']=='test'];targetbyid={r['id']:r['target'] for r in styles}
 oof=scores_from_decisions(bundle,bundle['oof_decision'],True);aggregate_train=aggregate_allowed[train]
 style_heads={};style_predictions={}
 style_predictions['frozen_previous']=oldmodel['style_heads']['oof_expanded_sigmoid'].predict(aggregate_regions(frozen[aggregate_allowed],units[aggregate_allowed],test_names)).astype('int64')
 for name,named in [('new_attributes_4_named',named4),('new_attributes_6_named',named6)]:
  h=fit_style_from_oof(oof[aggregate_train],units[train][aggregate_train],named,[targetbyid[i] for i in named]);style_heads[name]=h
  style_predictions[name]=h.predict(aggregate_regions(candidate[aggregate_allowed],units[aggregate_allowed],test_names)).astype('int64')
 style_results=[dict(id=unit,target=targetbyid[unit],outside_two_class_scope=targetbyid[unit] is None,
  predictions={a:int(p[i]) for a,p in style_predictions.items()},unknown_rejection_verified=False) for i,unit in enumerate(test_names)]
 records=[]
 for i,r in enumerate(rows):
  records.append(dict(id=r['id'],split=r['split'],scope=r['evaluation_scope'],object_reference=r['object_type'],
   targets=[None if v==-100 else int(v) for v in y[i]],
   outputs={a:dict(scores=s[i].tolist(),predicted=(s[i]>=.5).astype('int64').tolist(),errors=[CONCEPTS[j] for j in range(3) if y[i,j]!=-100 and int(s[i,j]>=.5)!=y[i,j]]) for a,s in [('frozen_previous',frozen),('new_source_candidate',candidate)]},
   role_reference=None if role_y[i]==-100 else ROLE_NAMES[role_y[i]],
   role_predictions={a:dict(predicted=ROLE_NAMES[int(p[i])],scores=prob[i].tolist()) for a,(p,prob) in role_outputs.items()},
   role_is_reference_context_not_pixel_truth=True,tree_reference_guard_only_not_learned=True))
 result=dict(protocol=protocol,attribute_metrics=attrmetrics,role_metrics=role_metrics,role_training_regions=int(role_fit.sum()),role_test_regions=int(role_test.sum()),
  style_probe=style_results,style_summary={a:sum(r['predictions'][a]==r['target'] for r in style_results if r['target'] is not None) for a in style_predictions},
  known_method_test_units=2,outside_scope_units=1,records=records,training_folds=bundle['folds'],
  independent_validation_passed=False,expert_accuracy=False,model_adopted=False,robot_connected=False)
 write('results.json',result);joblib.dump(dict(concepts=bundle,roles=role_heads,style_heads=style_heads,protocol=protocol,production_use=False),ROOT/'cross_source_models.joblib')
 np.savez_compressed(ROOT/'features_and_scores.npz',local=x,context=ctx,targets=y,role_targets=role_y,frozen_previous=frozen,new_source_candidate=candidate)
 assert all(sha(OLD/n)==v for n,v in oldfiles.items())
 write('verification.json',dict(old_models_features_and_references_unchanged=True,original48_feature_predictions_replayed=True,
  new_source_never_used_as_training=True,unknown_concepts_and_roles_ignored=True,tree_targets_excluded=True,
  style_units_not_propagated_to_local_roles=True,provided_region_context_only=True,full_stroke_truth_added=0,
  no_unknown_rejection_capability_claimed=True,author_independence_not_claimed=True,production_replaced=False,robot_connected=False))
 print('ATTRIBUTES',json.dumps(attrmetrics,ensure_ascii=False),flush=True);print('ROLES',json.dumps(role_metrics,ensure_ascii=False),flush=True);print('STYLES',json.dumps(style_results),flush=True)
if __name__=='__main__':main()
