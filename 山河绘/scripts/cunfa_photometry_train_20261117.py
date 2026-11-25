# -*- coding: utf-8 -*-
"""Known-case photometric ablations and a small supervised known-set rejector."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,torch,joblib
from PIL import Image
from torchvision.models import mobilenet_v3_small
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from gongbi_training.cunfa_photometry import *
from gongbi_training.cunfa_input_scale import input_canvas
from gongbi_training.cunfa_context_roles import ROLE_NAMES,context_role_target,context_box
ROOT=LAB/'cunfa_photometry_20261117';OLD=LAB/'cunfa_cross_source_20261117'
WEIGHT=LAB/'cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
ARMS=['raw_gray_context','reference_linear_context','linear_plus_contrast','contrast_only','linear_augmented']
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def classifier():return make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',random_state=0,max_iter=2000))
def role_metric(truth,pred):
 m=confusion_matrix(truth,pred,labels=range(4));counts=m.sum(1)
 recall=np.divide(m.diagonal(),counts,out=np.zeros(4,dtype=float),where=counts>0)
 return dict(agreement=int(m.trace()),regions=int(m.sum()),macro_recall=float(recall[counts>0].mean()),confusion=m.tolist(),
  per_role={name:dict(count=int(counts[j]),agreement=int(m[j,j])) for j,name in enumerate(ROLE_NAMES)})

def rejection_units(data):
 refs=read(LAB/'cunfa_role_reference_20261112/source_references.json')['records']
 units=[dict(id=r['id'],image=r['image'],known=1,split='train',source_group=r['source_group'],source_url=r['source_url'],basis='Explicitly named pima/fupi teaching unit') for r in refs if r['publisher']=='AAM']
 aamurl='https://education.asianart.org/wp-content/uploads/sites/6/2019/09/Chinese-Brushstrokes-Vocabulary-AAM_0.pdf'
 for file,name in [(5,'mi_dot'),(9,'raveled_rope'),(11,'bands_dragged_in_mud'),(15,'cloud_head'),(17,'nail_head')]:
  units.append(dict(id='aam_outside_'+name,image=str(LAB/'cunfa_gabor_baseline_20261110/sources'/('aam_2_img'+str(file)+'.png')),
   known=0,split='train',source_group='Lampo_Leong_all_examples',source_url=aamurl,
   basis='Explicit separately named teaching illustration outside pima/fupi; not a universal unknown class'))
 for r in data['style_units']:
  if r['split']=='train' and r['id'].startswith('qingpu_'):
   units.append(dict(id=r['id'],image=r['image'],known=1,split='train',source_group='Qingpu_book_illustrator_unverified',source_url=r['source_url'],basis='Explicitly named pima/fupi teaching unit'))
 inv={r['id']:r for r in read(OLD/'source_image_inventory.json')}
 for ident,name,box in [('qingpu_22_0','cloud_head',[40,50,445,445]),('qingpu_23_0','big_mi_dot',[300,65,645,330])]:
  im=Image.open(inv[ident]['path']);dest=ROOT/'views'/(ident+'_outside.png');im.crop(box).save(dest)
  units.append(dict(id=ident+'_outside',image=str(dest),known=0,split='train',source_group='Qingpu_book_illustrator_unverified',
   source_url='https://m.qplll.net/File/Course/QP_COURSE_PDF/中国山水画法基础.pdf',basis='Separate named '+name+' illustration, crop excludes inscriptions; not a local-stroke label'))
 for r in data['style_units']:
  if r['split']=='test':units.append(dict(id=r['id'],image=r['image'],known=int(r['target'] is not None),split='test',
   source_group='NTM_'+r['id'],source_url=r['source_url'],basis='Known3work regression from preceding run, not new independent test'))
 for r in refs:
  if r['publisher']=='HKMA':units.append(dict(id=r['id'],image=r['image'],known=1,split='test',source_group=r['source_group'],source_url=r['source_url'],basis='Previously seen named unit regression; same HKMA source photo'))
 for r in units:r['sha256']=sha(r['image'])
 assert len([r for r in units if r['split']=='train'])==13
 assert not ({r['source_group'] for r in units if r['split']=='train'}&{r['source_group'] for r in units if r['split']=='test'})
 return units

def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'views').mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Frozen run'
 data=read(OLD/'reference_manifest.json');rows=data['records'];oldresult=read(OLD/'results.json');cache=np.load(OLD/'features_and_scores.npz')
 checks=read(ROOT/'contract_checks.json');assert all(v for k,v in checks.items() if k!='robot_connected')
 for r in rows:assert sha(r['image'])==r['image_sha256']
 y=np.array([context_role_target(r) for r in rows]);train=np.array([r['split']=='train' for r in rows]);fit=train&(y!=-100);test=(~train)&(y!=-100);groups=np.array([r['reference_unit'] for r in rows])
 reject_units=rejection_units(data);write('rejection_units.json',dict(units=reject_units,
  task='binary membership in the two named pima/fupi methods, not identifying every outside method',
  captions_as_unit_labels_only=True,independent_unknown_validation=False,threshold=.5))
 oldfiles={n:sha(OLD/n) for n in ['cross_source_models.joblib','reference_manifest.json','features_and_scores.npz','results.json']}
 protocol=dict(prepared_before_fit=True,reference_manifest_sha256=sha(OLD/'reference_manifest.json'),old_artifacts=oldfiles,
  code_sha256=sha(__file__),module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/cunfa_photometry.py'),encoder_sha256=sha(WEIGHT),
  role_training_regions=int(fit.sum()),known_role_regression_regions=int(test.sum()),role_arms=ARMS,
  matched_legacy_control='Frozen local+context histogram-equalized role model replay',
  raw_gray='no per-patch equalization; same aspect resize and ImageNet normalization',
  reference_linear='one reference-image gray90th-percentile anchor, output reference240; does not use paper masks or targets',
  contrast_features=list(CONTRAST_NAMES),augmentation='one full-reference warm(.94,.84),darken.82,shared Gaussian grain sigma1.5 variant per train region; source-hash seed',
  augmentation_not_new_independent_data=True,role_classifier='StandardScaler + balanced LogisticRegression C=1, no parameter search',
  selection='highest training leave-whole-reference-unit-out macro recall; tie chooses earlier declared arm; selection saved before external metrics',
  training_oof_not_independent_source_test=True,all_external_cases_previously_opened=True,
  rejection=dict(train_named_units=13,known=6,outside=7,test_previously_seen_units=5,raw_gray_encoder=True,
   classifier='StandardScaler + balanced LogisticRegression C=1',threshold=.5,not_general_unknown_or_mixed_recognition=True),
  source_objects_and_targets_unchanged=True,production_replaced=False,robot_connected=False)
 write('protocol.json',protocol);print('FROZEN',int(fit.sum()),int(test.sum()),flush=True)
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
 net=mobilenet_v3_small(weights=None);net.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True));net=net.features.cuda().eval()
 arrays=[];indices={};aux=[];augindices={};augaux={};references={};augrefs={};audit=[]
 for i,r in enumerate(rows):
  ref=references.setdefault(r['reference_image'],Image.open(r['reference_image']).convert('RGB'))
  local=Image.open(r['image']).convert('RGB');ctx=ref.crop(context_box(r['box'],ref.size));pair=[local,ctx]
  aux.append(np.concatenate([contrast_features(v,ref) for v in pair]))
  for mode in ['raw_gray','reference_linear']:
   indices[i,mode]=[len(arrays),len(arrays)+1];arrays.extend(appearance_tensor(v,ref,mode) for v in pair)
   appearance_canvas(local,ref,mode).save(ROOT/'views'/(r['id']+'_'+mode+'.png'))
  legacy,_=input_canvas(local,fill_extent=True);legacy.save(ROOT/'views'/(r['id']+'_legacy.png'))
  g=np.asarray(local.convert('L'));eq=cv2.equalizeHist(g);lin=np.clip(g.astype(float)*240/reference_white(ref),0,255)
  spread=lambda a:float(np.percentile(a,95)-np.percentile(a,5))
  audit.append(dict(id=r['id'],role_reference=r.get('context_role_target'),split=r['split'],raw_q95_q05=spread(g),equalized_q95_q05=spread(eq),linear_q95_q05=spread(lin),reference_anchor=reference_white(ref)))
  if fit[i]:
   if r['reference_image'] not in augrefs:
    seed=int(hashlib.sha256(r['reference_image'].encode('utf8')).hexdigest()[:8],16)
    augrefs[r['reference_image']]=substrate_variant(ref,seed)
   aref=augrefs[r['reference_image']];apair=[aref.crop(r['box']),aref.crop(context_box(r['box'],aref.size))]
   augindices[i]=[len(arrays),len(arrays)+1];arrays.extend(appearance_tensor(v,aref,'reference_linear') for v in apair)
   augaux[i]=np.concatenate([contrast_features(v,aref) for v in apair]);apair[0].save(ROOT/'views'/(r['id']+'_training_variant.png'))
 reject_indices=[]
 for u in reject_units:
  im=Image.open(u['image']).convert('RGB');reject_indices.append(len(arrays));arrays.append(appearance_tensor(im,im,'raw_gray'))
 encoded=[]
 with torch.inference_mode():
  for i in range(0,len(arrays),12):
   v=net(torch.from_numpy(np.stack(arrays[i:i+12])).cuda()).mean((2,3));encoded.append(torch.nn.functional.normalize(v,dim=1).cpu().numpy())
 encoded=np.concatenate(encoded);aux=np.stack(aux);features={}
 for mode,arm in [('raw_gray','raw_gray_context'),('reference_linear','reference_linear_context')]:
  features[arm]=np.stack([np.concatenate([encoded[j] for j in indices[i,mode]])/np.sqrt(2) for i in range(len(rows))])
 features['linear_plus_contrast']=np.concatenate([features['reference_linear_context'],aux],1)
 features['contrast_only']=aux;features['linear_augmented']=features['linear_plus_contrast']
 augfeatures={i:np.concatenate([np.concatenate([encoded[j] for j in augindices[i]])/np.sqrt(2),augaux[i]]) for i in augindices}
 models={};allpred={};oofmetrics={};oofpred={}
 for arm in ARMS:
  f=features[arm];p=np.full(len(rows),-100,dtype='int64')
  for unit in sorted(set(groups[fit])):
   tr=fit&(groups!=unit);te=fit&(groups==unit);xx=f[tr];yy=y[tr]
   if arm=='linear_augmented':
    add=[i for i in np.flatnonzero(tr)];xx=np.concatenate([xx,np.stack([augfeatures[i] for i in add])]);yy=np.concatenate([yy,y[add]])
   model=classifier().fit(xx,yy);p[te]=model.predict(f[te])
  oofmetrics[arm]=role_metric(y[fit],p[fit]);oofpred[arm]=p
  xx=f[fit];yy=y[fit]
  if arm=='linear_augmented':
   add=list(np.flatnonzero(fit));xx=np.concatenate([xx,np.stack([augfeatures[i] for i in add])]);yy=np.concatenate([yy,y[add]])
  models[arm]=classifier().fit(xx,yy);allpred[arm]=models[arm].predict(f).astype('int64')
  print('TRAIN_OOF',arm,oofmetrics[arm]['agreement'],oofmetrics[arm]['macro_recall'],flush=True)
 selected=max(ARMS,key=lambda a:oofmetrics[a]['macro_recall'])
 write('training_selection.json',dict(selected_arm=selected,criterion=protocol['selection'],metrics=oofmetrics,saved_before_external_scoring=True,
  not_source_independent_validation=True))
 # Frozen legacy is the exact preceding feature/model pair, with no refitting.
 frozen=joblib.load(OLD/'cross_source_models.joblib')['roles']['local_plus_context'];legacyf=np.concatenate([cache['local'],cache['context']],1)/np.sqrt(2)
 allpred['legacy_hist_context']=frozen.predict(legacyf).astype('int64')
 assert role_metric(y[test],allpred['legacy_hist_context'][test])['confusion']==oldresult['role_metrics']['local_plus_context']['confusion']
 metrics={a:role_metric(y[test],p[test]) for a,p in allpred.items()}
 # The rejector knows only explicit illustration-level membership examples.
 rx=encoded[reject_indices];rt=np.array([u['split']=='train' for u in reject_units]);ry=np.array([u['known'] for u in reject_units]);rejector=classifier().fit(rx[rt],ry[rt]);rs=rejector.predict_proba(rx)[:,1];rp=(rs>=.5).astype('int64')
 reject_records=[dict(id=u['id'],split=u['split'],reference_known=u['known'],known_score=float(rs[i]),accept_known=bool(rp[i]),
  final_action='two_class_candidate' if rp[i] else 'outside_known_set_or_uncertain',source_url=u['source_url']) for i,u in enumerate(reject_units)]
 reject_metrics=dict(test_units=int((~rt).sum()),known_test=int(((~rt)&(ry==1)).sum()),outside_test=int(((~rt)&(ry==0)).sum()),
  known_accepted=int(((~rt)&(ry==1)&(rp==1)).sum()),known_falsely_rejected=int(((~rt)&(ry==1)&(rp==0)).sum()),
  outside_rejected=int(((~rt)&(ry==0)&(rp==0)).sum()),outside_falsely_accepted=int(((~rt)&(ry==0)&(rp==1)).sum()))
 records=[dict(id=r['id'],split=r['split'],scope=r['evaluation_scope'],role_reference=None if y[i]==-100 else ROLE_NAMES[y[i]],
  predictions={a:ROLE_NAMES[int(p[i])] for a,p in allpred.items()},selected_prediction=ROLE_NAMES[int(allpred[selected][i])]) for i,r in enumerate(rows)]
 result=dict(protocol=protocol,selected_arm=selected,training_oof_metrics=oofmetrics,known_role_regression_metrics=metrics,
  records=records,rejection_records=reject_records,rejection_metrics=reject_metrics,
  independent_validation_passed=False,model_adopted=False,full_stroke_learning=False,robot_connected=False)
 write('results.json',result);write('contrast_audit.json',dict(records=audit,measurements_not_ink_masks=True,cause_of_all_errors_not_established=True))
 joblib.dump(dict(role_models=models,selected_arm=selected,rejector=rejector,protocol=protocol,production_use=False),ROOT/'photometry_models.joblib')
 np.savez_compressed(ROOT/'features.npz',**features,rejection=rx,role_targets=y)
 assert all(sha(OLD/n)==v for n,v in oldfiles.items())
 write('verification.json',dict(old_models_features_and_labels_unchanged=True,legacy_confusion_replayed=True,
  source_anchor_no_labels=True,no_independent_per_patch_histogram_in_candidates=True,selection_uses_only_training_oof=True,
  augmentation_only_on_training_references=True,no_augmented_work_count_inflation=True,
  unknown_not_added_as_pima_or_fupi=True,rejector_not_general_unknown_recognition=True,
  previously_opened_checks_not_new_independent_tests=True,production_replaced=False,robot_connected=False))
 print('SELECTED',selected,flush=True);print('ROLE_METRICS',json.dumps(metrics),flush=True);print('REJECTION',json.dumps(reject_metrics),json.dumps([r for r in reject_records if r['split']=='test']),flush=True)
if __name__=='__main__':main()
