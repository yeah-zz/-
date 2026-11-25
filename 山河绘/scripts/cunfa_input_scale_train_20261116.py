# -*- coding: utf-8 -*-
import sys,json,hashlib,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import torch,numpy as np,joblib
from PIL import Image
from torchvision.models import mobilenet_v3_small
from sklearn.svm import SVC
from gongbi_training.cunfa_input_scale import input_canvas,preprocess
ROOT=LAB/'cunfa_input_scale_20261116';TRAIN=LAB/'cunfa_context_reference_20261111';OLD=LAB/'cunfa_fresh_validation_20261111';CONTEXT=LAB/'cunfa_resolution_context_20261116';CLASSES=['披麻皴','斧劈皴'];ARMS=['native_extent','fixed_extent']
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Frozen run';train=read(TRAIN/'input_manifest.json')['rows'];test=[r for r in read(OLD/'test_manifest.json')['records'] if r['admission']=='source_label_test'];freeze=read(OLD/'freeze_manifest.json')
 for f in freeze['files']:assert sha(f['path'])==f['sha256']
 oldhead=joblib.load(OLD/'frozen/context_head_exploratory.joblib');assert {r['id'] for r in train}==set(oldhead['trained_ids']);assert not ({r['artist_group'] for r in train}&{r['artist_group'] for r in test});assert not ({r['source_sha256'] for r in train}&{r['source_sha256'] for r in test});assert len(train)==63 and len(test)==13
 spec=importlib.util.spec_from_file_location('frozen_input_source',OLD/'frozen/encoder_and_preprocessing.py');frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen);weight=next(f['path'] for f in freeze['files'] if f['path'].endswith('.pth'))
 toy=Image.fromarray(np.arange(75*80,dtype='uint8').reshape(80,75));oldcanvas,oldmeta=input_canvas(toy,fill_extent=False);newcanvas,newmeta=input_canvas(toy,fill_extent=True);assert max(newmeta['content_size'])==224 and oldmeta['content_size']==[75,80];assert newmeta['content_rectangle_fraction']>oldmeta['content_rectangle_fraction'];assert np.array_equal(preprocess(toy,False),frozen.preprocess(toy))
 p=dict(arms=ARMS,prepared_before_fit=True,train_rows=63,test_rows=13,train_manifest_sha256=sha(TRAIN/'input_manifest.json'),test_manifest_sha256=sha(OLD/'test_manifest.json'),old_head_sha256=sha(OLD/'frozen/context_head_exploratory.joblib'),encoder_sha256=sha(weight),script_sha256=sha(__file__),input_module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/cunfa_input_scale.py'),classifier='Frozen ImageNet encoder, separate linear SVMs trained from same63 rows; same C/kernel/class_weight as frozen head',C=float(oldhead['model'].C),difference='native_extent shrinks only; fixed_extent scales both small and large views to long side224 preserving aspect, same equalization/median letterbox/normalization',common_input='Given local crop plus2.5x context; not automatic whole-rock detection',runtime='Both arms reencoded in the same process, CUDA TF32 off, batch12, no external-test parameter selection',data_status='All known cases reused for controlled regression; provisional work-description labels and assistant crops, no expert local truth',class_scope='Two provisional styles only; mixed/unknown not trained',no_new_native_resolution=True,source_quality_probe=str(CONTEXT/'larger_variant_probe.json'),new_independent_validation=False,robot_connected=False);write('protocol.json',p);write('train_test_manifest.json',dict(train=train,test=test))
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);net=mobilenet_v3_small(weights=None);net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));net=net.features.cuda().eval();models={};results={};meta={};(ROOT/'views').mkdir(exist_ok=True)
 for arm in ARMS:
  fill=arm=='fixed_extent';arrays=[];mm=[]
  for scope,rows in [('train',train),('test',test)]:
   for r in rows:
    assert sha(r['image'])==r['crop_sha256'] and sha(r['source_image'])==r['source_sha256'];a,b,box=frozen.input_views(r);arrays.extend([preprocess(a,fill),preprocess(b,fill)]);_,am=input_canvas(a,fill_extent=fill);_,bm=input_canvas(b,fill_extent=fill);mm.append(dict(id=r['id'],scope=scope,local=am,context=bm))
    if scope=='test':
     for kind,im in [('local',a),('context',b)]:canvas,_=input_canvas(im,fill_extent=fill);canvas.save(ROOT/'views'/(r['id']+'_'+arm+'_'+kind+'.png'))
  enc=[]
  with torch.inference_mode():
   for i in range(0,len(arrays),12):
    z=net(torch.from_numpy(np.stack(arrays[i:i+12])).cuda()).mean((2,3));enc.append(torch.nn.functional.normalize(z,dim=1).cpu().numpy())
  z=np.concatenate(enc);features=np.concatenate([z[::2],z[1::2]],axis=1)/np.sqrt(2);xtrain=features[:len(train)];xtest=features[len(train):];y=np.array([CLASSES.index(r['label']) for r in train]);model=SVC(C=p['C'],kernel='linear',class_weight='balanced');model.fit(xtrain,y);pred=model.predict(xtest);margins=model.decision_function(xtest);joblib.dump(dict(model=model,protocol=p,trained_ids=[r['id'] for r in train]),ROOT/(arm+'.joblib'));np.savez_compressed(ROOT/(arm+'_features.npz'),train=xtrain,test=xtest)
  records=[dict(id=r['id'],source_label=r['label'],prediction=CLASSES[int(pred[i])],margin=float(margins[i]),agreement=CLASSES[int(pred[i])]==r['label'],artist=r['artist_group'],work=r['work_id']) for i,r in enumerate(test)];results[arm]=dict(agreement_count=sum(r['agreement'] for r in records),regions=len(test),source_label_agreement=sum(r['agreement'] for r in records)/len(test),per_work={w:dict(regions=sum(r['work']==w for r in records),agreement_count=sum(r['work']==w and r['agreement'] for r in records)) for w in sorted({r['work'] for r in records})},records=records);meta[arm]=mm;models[arm]=model
  print(arm,results[arm]['agreement_count'],'/',len(test),flush=True)
 oldrecords={r['id']:r for r in read(OLD/'predictions.json')['records']};baseline_changes=[r['id'] for r in results['native_extent']['records'] if r['prediction']!=oldrecords[r['id']]['predictions']['local_plus_context']['predicted']]
 write('results.json',dict(protocol=p,results=results,input_metadata=meta,head_hashes={a:sha(ROOT/(a+'.joblib')) for a in ARMS},historical_baseline_prediction_changes=baseline_changes,old_artifacts_unchanged=sha(OLD/'frozen/context_head_exploratory.joblib')==p['old_head_sha256'],model_adopted=False,new_understanding_claimed=False,expert_accuracy=False,new_independent_validation=False,robot_connected=False));write('verification.json',dict(same_train_ids_as_frozen_head=True,test_authors_and_sources_absent_from_training=True,native_preprocessing_matches_frozen=True,aspect_ratio_preserved=True,small_crops_fill_extent=True,same_C_no_test_tuning=True,upscale_not_claimed_new_detail=True,old_weights_unchanged=True,unknown_recognition_not_trained=True,robot_connected=False));print(json.dumps({a:dict(count=results[a]['agreement_count'],total=13) for a in ARMS}),flush=True)
if __name__=='__main__':main()
