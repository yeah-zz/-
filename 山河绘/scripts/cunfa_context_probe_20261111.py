# -*- coding: utf-8 -*-
"""Frozen ImageNet encoder + small supervised heads. Exploratory reused data only."""
import sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2,torch,joblib
from PIL import Image
from torchvision.models import mobilenet_v3_small
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold
from train_cunfa_gabor_20261110 import metric,CLASSES,phash
ROOT=LAB/'cunfa_context_reference_20261111';OLD=LAB/'cunfa_gabor_baseline_20261110';PREV=LAB/'cunfa_structure_20261111'
WEIGHTS=LAB/'cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def preprocess(im):
 g=np.asarray(im.convert('L'));g=cv2.equalizeHist(g)
 rgb=Image.fromarray(g).convert('RGB');rgb.thumbnail((224,224),Image.Resampling.BILINEAR)
 # Preserve aspect ratio: letterbox in the image median rather than stretch geometry.
 bg=Image.new('RGB',(224,224),(int(np.median(g)),)*3);bg.paste(rgb,((224-rgb.width)//2,(224-rgb.height)//2))
 x=np.asarray(bg).astype('float32')/255;x=(x-np.array([.485,.456,.406],'float32'))/np.array([.229,.224,.225],'float32')
 return x.transpose(2,0,1)
def input_views(r):
 im=Image.open(r['source_image']).convert('RGB');x0,y0,x1,y1=r['source_box'];w,h=im.size;cx=(x0+x1)/2;cy=(y0+y1)/2
 b=[max(0,int(cx-(x1-x0)*1.25)),max(0,int(cy-(y1-y0)*1.25)),min(w,int(cx+(x1-x0)*1.25)),min(h,int(cy+(y1-y0)*1.25))]
 return Image.open(r['image']).convert('RGB'),im.crop(b),b
def select_c(x,y,groups):
 scores={};splits=list(GroupKFold(n_splits=min(3,len(set(groups)))).split(x,y,groups))
 for c in [.1,1.,10.]:
  yy=[];pp=[]
  for tr,va in splits:
   assert set(groups[tr]).isdisjoint(groups[va]);svm=SVC(C=c,kernel='linear',class_weight='balanced');svm.fit(x[tr],y[tr]);yy.extend(y[va].tolist());pp.extend(svm.predict(x[va]).tolist())
  scores[str(c)]=metric(np.array(yy),np.array(pp))['macro_f1']
 return max([.1,1.,10.],key=lambda c:scores[str(c)]),scores
def evaluate(x,y,groups,rows):
 pred=np.full(len(rows),-1);folds=[]
 for tr,te in LeaveOneGroupOut().split(x,y,groups):
  assert {rows[i]['work_group'] for i in tr}.isdisjoint({rows[i]['work_group'] for i in te})
  assert {rows[i]['source_sha256'] for i in tr}.isdisjoint({rows[i]['source_sha256'] for i in te})
  c,scores=select_c(x[tr],y[tr],groups[tr]);svm=SVC(C=c,kernel='linear',class_weight='balanced');svm.fit(x[tr],y[tr]);pred[te]=svm.predict(x[te]);folds.append(dict(held_artist=str(groups[te][0]),selected_C=c,inner_scores=scores,train_ids=[rows[i]['id'] for i in tr],test_ids=[rows[i]['id'] for i in te],metrics=metric(y[te],pred[te])))
 out=dict(metrics=metric(y,pred),folds=folds,artist_macro_agreement=float(np.mean([f['metrics']['label_agreement'] for f in folds])),predictions=[dict(id=r['id'],label=r['label'],predicted=CLASSES[int(pred[i])],artist=r['artist_group']) for i,r in enumerate(rows)])
 return out
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'inputs').mkdir(exist_ok=True);cv2.setNumThreads(2);torch.set_num_threads(4);start=time.time()
 d=json.loads((PREV/'dataset_manifest.json').read_text('utf8'));rows=d['development_records']+d['external_holdout'];y=np.array([CLASSES.index(r['label']) for r in rows]);groups=np.array([r['artist_group'] for r in rows])
 protocol=dict(data_sha256=sha(PREV/'dataset_manifest.json'),code_sha256=sha(__file__),weights_sha256=sha(WEIGHTS),primary='local_plus_context',ablations=['local_only','context_only'],
  input='Source local crop plus 2.5x neighbouring window; not a predicted whole-object segmentation',normalisation='Per-image grayscale histogram equalisation,aspect-preserving224 letterbox,ImageNet channel normalisation; differs from stock central crop',
  frozen_encoder=True,pretraining='ImageNet1K, not cunfa-specific',trainable='Linear SVM classification head only',nested_author_cv=True,
  selected_C=[.1,1,10],seed=111112,expert_gold=0,all63_data_previously_examined=True,new_independent_test=False,
  previous23_test_is_now_reused_for_exploration=True,reference_groups_not_training_labels=True,robot_connected=False)
 if (ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text('utf8'))==protocol
 else:save('protocol.json',protocol)
 arrays=[];viewmeta=[]
 for r in rows:
  a,b,box=input_views(r);a.save(ROOT/'inputs'/(r['id']+'_local.png'));b.save(ROOT/'inputs'/(r['id']+'_context.png'));arrays.extend([preprocess(a),preprocess(b)]);viewmeta.append(dict(id=r['id'],context_box=box,source_box=r['source_box']))
 net=mobilenet_v3_small(weights=None);net.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True));net=net.features.cuda().eval()
 for p in net.parameters():p.requires_grad_(False)
 feats=[]
 with torch.inference_mode():
  for i in range(0,len(arrays),12):
   v=net(torch.from_numpy(np.stack(arrays[i:i+12])).cuda()).mean((2,3));v=torch.nn.functional.normalize(v,dim=1);feats.append(v.cpu().numpy())
 z=np.concatenate(feats);local=z[::2];context=z[1::2];both=np.concatenate([local,context],axis=1)/np.sqrt(2);matrices=dict(local_only=local,context_only=context,local_plus_context=both);np.savez_compressed(ROOT/'features.npz',**matrices)
 results={}
 for kind,x in matrices.items():
  results[kind]=evaluate(x,y,groups,rows);save(kind+'_evaluation.json',results[kind]);print(kind,results[kind]['metrics'],flush=True)
 # Export exploratory head; this is not used to claim in-sample performance.
 c,inner=select_c(both,y,groups);svm=SVC(C=c,kernel='linear',class_weight='balanced');svm.fit(both,y);joblib.dump(dict(model=svm,encoder_weights_sha256=protocol['weights_sha256'],trained_ids=[r['id'] for r in rows],feature='concat(normalised local,normalised2.5xcontext)/sqrt2',formal_validation=False,robot_execution=False),ROOT/'context_head_exploratory.joblib')
 loaded=joblib.load(ROOT/'context_head_exploratory.joblib');assert np.array_equal(svm.predict(both),loaded['model'].predict(both))
 # Pairwise changes compare predictions for exactly the same rows and same grouped protocol.
 p0=np.array([r['predicted'] for r in results['local_only']['predictions']]);p1=np.array([r['predicted'] for r in results['local_plus_context']['predictions']]);truth=np.array([r['label'] for r in rows])
 summary=dict(regions=len(rows),artists=len(set(groups)),class_counts={c:int(sum(truth==c)) for c in CLASSES},expert_labels=0,frozen_backbone=True,cunfa_head_trained=True,stroke_groups_supervised=False,
  morphology_reference_not_model_output=True,primary='local_plus_context',exploratory_reused_data=True,independent_test_count=0,
  metrics={k:dict(**v['metrics'],artist_macro_agreement=v['artist_macro_agreement']) for k,v in results.items()},context_improved=int(np.sum((p0!=truth)&(p1==truth))),context_regressed=int(np.sum((p0==truth)&(p1!=truth))),
  adopted=False,robot_connected=False,weights_downloaded=False,elapsed_seconds=time.time()-start)
 save('summary.json',summary);save('input_manifest.json',dict(rows=rows,views=viewmeta));save('verification.json',dict(known63_predicted_once_per_method=True,artist_work_source_disjoint_in_each_fold=True,unknown_probe_not_training=True,new_dependency_installed=False,weights_sha256=sha(WEIGHTS),model_reload_equivalent=True,original_data_manifest_unchanged=sha(PREV/'dataset_manifest.json')==protocol['data_sha256'],source_crop_hashes_match=all(sha(r['image'])==r['crop_sha256'] and sha(r['source_image'])==r['source_sha256'] for r in rows),model_adopted=False,robot_connected=False))
 print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
