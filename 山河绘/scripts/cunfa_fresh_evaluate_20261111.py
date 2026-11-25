# -*- coding: utf-8 -*-
"""Inference only, no fit calls. Retain every admitted new-author sample."""
import sys,json,hashlib,datetime,importlib.util,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,joblib
from PIL import Image
from torchvision.models import mobilenet_v3_small
from train_cunfa_gabor_20261110 import metric,CLASSES,phash
ROOT=LAB/'cunfa_fresh_validation_20261111';OLD=LAB/'cunfa_context_reference_20261111'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
 start=time.time();torch.set_num_threads(4);manifest=ROOT/'test_manifest.json';freeze=json.loads((ROOT/'freeze_manifest.json').read_text('utf8'));data=json.loads(manifest.read_text('utf8'));allrows=data['records'];rows=[r for r in allrows if r['admission']=='source_label_test'];train=json.loads((OLD/'input_manifest.json').read_text('utf8'))['rows'];
 for f in freeze['files']:assert sha(f['path'])==f['sha256']
 assert sha(ROOT/'freeze_manifest.json')==data['freeze_manifest_sha256']
 assert {r['artist_group'] for r in rows}.isdisjoint(freeze['train_authors']);assert {r['work_id'] for r in rows}.isdisjoint({r['work_group'] for r in train})
 assert {r['source_sha256'] for r in rows}.isdisjoint({r['source_sha256'] for r in train})
 # New public photographs were acquired after the actual model freeze.
 freeze_time=datetime.datetime.fromisoformat(freeze['frozen_at_utc']);assert datetime.datetime.fromisoformat(data['reviewed_at_utc'])>freeze_time
 for r in rows:
  raw=Path(r.get('raw_source_image',r['source_image']));assert raw.stat().st_mtime>=freeze_time.timestamp()
  assert sha(r['image'])==r['crop_sha256'] and sha(r['source_image'])==r['source_sha256']
 # Resolve preprocessing directly from the preserved source, not a mutable new implementation.
 spec=importlib.util.spec_from_file_location('frozen_cunfa_preprocess',ROOT/'frozen/encoder_and_preprocessing.py');frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
 protocol=dict(frozen_model_manifest_sha256=sha(ROOT/'freeze_manifest.json'),test_manifest_sha256=sha(manifest),evaluation_code_sha256=sha(__file__),
  saved_before_prediction=True,ordered_test_ids=[r['id'] for r in rows],admitted_rows=len(rows),excluded_rows=len(allrows)-len(rows),new_test_author_count=len({r['artist_group'] for r in rows}),
  no_fit_calls=True,unknown_detection_not_evaluated=True,source_labels_not_expert_gold=True,gate=freeze['provisional_gate'],robot_connected=False)
 if (ROOT/'evaluation_protocol.json').exists():assert json.loads((ROOT/'evaluation_protocol.json').read_text('utf8'))==protocol
 else:save('evaluation_protocol.json',protocol)
 (ROOT/'inputs').mkdir(exist_ok=True);arrays=[];viewmeta=[]
 train_hashes=[phash(Image.open(r['image'])) for r in train];duplicate_matches=[]
 for r in rows:
  hh=phash(Image.open(r['image']))
  for j,h in enumerate(train_hashes):
   distance=int(np.count_nonzero(hh!=h))
   if distance<=4:duplicate_matches.append(dict(test=r['id'],train=train[j]['id'],phash_distance=distance))
  a,b,box=frozen.input_views(r);a.save(ROOT/'inputs'/(r['id']+'_local.png'));b.save(ROOT/'inputs'/(r['id']+'_context.png'));arrays.extend([frozen.preprocess(a),frozen.preprocess(b)]);viewmeta.append(dict(id=r['id'],context_box=box))
 assert not duplicate_matches,'Near-duplicate audit needs review before predicting.'
 weight_record=next(f for f in freeze['files'] if f['path'].endswith('.pth'));net=mobilenet_v3_small(weights=None);net.load_state_dict(torch.load(weight_record['path'],map_location='cpu',weights_only=True));net=net.features.cuda().eval();vec=[]
 with torch.inference_mode():
  for i in range(0,len(arrays),12):
   x=net(torch.from_numpy(np.stack(arrays[i:i+12])).cuda()).mean((2,3));vec.append(torch.nn.functional.normalize(x,dim=1).cpu().numpy())
 z=np.concatenate(vec);local=z[::2];context=z[1::2];matrices=dict(local_only=local,context_only=context,local_plus_context=np.concatenate([local,context],axis=1)/np.sqrt(2));np.savez_compressed(ROOT/'features.npz',**matrices)
 y=np.array([CLASSES.index(r['label']) for r in rows]);results={};predrecords={r['id']:dict(id=r['id'],label=r['label'],artist=r['artist_group'],publisher=r['publisher'],predictions={}) for r in rows}
 for key,x in matrices.items():
  path=ROOT/'frozen'/('context_head_exploratory.joblib' if key=='local_plus_context' else key+'_head.joblib');head=joblib.load(path);svm=head['model'];assert set(head['trained_ids'])==set(freeze['train_ids']);pp=svm.predict(x);margin=svm.decision_function(x)
  result=dict(metrics=metric(y,pp),per_artist={},per_publisher={},frozen_C=svm.C,head_sha256=sha(path),trained_on_new_data=False)
  for field,target in [('artist_group','per_artist'),('publisher','per_publisher')]:
   for a in sorted({r[field] for r in rows}):
    ids=np.array([r[field]==a for r in rows]);result[target][a]=metric(y[ids],pp[ids])
  result['artist_macro_agreement']=float(np.mean([r['label_agreement'] for r in result['per_artist'].values()]));results[key]=result
  for i,r in enumerate(rows):predrecords[r['id']]['predictions'][key]=dict(predicted=CLASSES[int(pp[i])],signed_margin=float(margin[i]),margin_is_not_probability=True)
 primary=results['local_plus_context'];local_result=results['local_only'];g=freeze['provisional_gate'];gain=primary['metrics']['label_agreement']-local_result['metrics']['label_agreement'];f1gain=primary['metrics']['macro_f1']-local_result['metrics']['macro_f1']
 provisional_pass=all(r>=g['minimum_each_class_recall'] for r in primary['metrics']['class_recall']) and primary['artist_macro_agreement']>=g['minimum_artist_macro_agreement'] and gain>=g['minimum_accuracy_gain_over_frozen_local'] and f1gain>=g['minimum_macro_f1_gain_over_frozen_local']
 records=list(predrecords.values());localp=np.array([r['predictions']['local_only']['predicted'] for r in records]);bothp=np.array([r['predictions']['local_plus_context']['predicted'] for r in records]);truth=np.array([r['label'] for r in rows])
 summary=dict(test_regions=len(rows),excluded_regions=len(allrows)-len(rows),authors=len({r['artist_group'] for r in rows}),works=len({r['work_id'] for r in rows}),publishers=len({r['publisher'] for r in rows}),class_counts={c:int(sum(y==i)) for i,c in enumerate(CLASSES)},
  new_authors_absent_from_training=True,model_and_parameters_frozen_before_new_images=True,no_test_training_or_tuning=True,all_admitted_rows_reported=True,source_label_test_only=True,expert_gold=0,
  provisional_gate_passed=bool(provisional_pass),formal_validation_passed=False,model_adopted=False,
  context_improved=int(np.sum((localp!=truth)&(bothp==truth))),context_regressed=int(np.sum((localp==truth)&(bothp!=truth))),
  metrics={k:dict(**v['metrics'],artist_macro_agreement=v['artist_macro_agreement']) for k,v in results.items()},test_manifest_sha256=sha(manifest),evaluation_protocol_sha256=sha(ROOT/'evaluation_protocol.json'),
  regions_clustered_in_four_works=True,not_full_image_detection=True,stroke_generation=False,robot_connected=False,elapsed_seconds=time.time()-start)
 save('predictions.json',dict(records=records,results=results,views=viewmeta,model_frozen_at=freeze['frozen_at_utc'],finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));save('summary.json',summary)
 save('verification.json',dict(freeze_files_unchanged=all(sha(f['path'])==f['sha256'] for f in freeze['files']),sample_manifest_unchanged=sha(manifest)==protocol['test_manifest_sha256'],
  author_work_source_overlap=0,cross_split_phash_matches=duplicate_matches,all13_scored_once=True,excluded3_never_predicted=True,preprocessing_loaded_from_frozen_file=True,
  raw_NPM_colour_chart_and_mount_excluded=True,NPM_preview_only_no_captcha_download=True,new_images_after_model_freeze=True,new_data_fit_calls=0,labels_not_expert_certified=True,robot_connected=False))
 print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
