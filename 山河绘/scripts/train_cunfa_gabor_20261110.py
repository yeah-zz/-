# -*- coding: utf-8 -*-
"""Gabor/PCA/SVM method adaptation with nested artist-group validation.

Not reproduction of the 2014 paper's dataset/result (coarse dot/line/plane).
This pilot uses provisional, source-evidenced named-type labels, not expert gold.
"""
import sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2,joblib
from PIL import Image
from scipy.signal import fftconvolve
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold
from sklearn.metrics import confusion_matrix,f1_score,balanced_accuracy_score
from contextlib import nullcontext
ROOT=LAB/'cunfa_gabor_baseline_20261110'
CLASSES=['披麻皴','斧劈皴'];CS=[.1,1.,10.]

def save(name,v):(ROOT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def phash(im):
 a=cv2.resize(np.array(im.convert('L')),(32,32)).astype('float32');v=cv2.dct(a)[:8,:8].ravel()[1:];return v>np.median(v)

def features(im):
 a=np.array(im.convert('RGB'));g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
 g=cv2.resize(g,(128,128),interpolation=cv2.INTER_AREA);g=cv2.equalizeHist(g).astype('float32')/255
 full=[];pooled=[];sigma=2*np.pi
 for nu in range(5):
  k=(np.pi/2)/(np.sqrt(2)**nu);radius=int(np.ceil(3*sigma/k));yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
  for mu in range(8):
   theta=np.pi*mu/8;kernel=(k*k/(sigma*sigma))*np.exp(-k*k*(xx*xx+yy*yy)/(2*sigma*sigma))*(np.exp(1j*k*(xx*np.cos(theta)+yy*np.sin(theta)))-np.exp(-sigma*sigma/2))
   padded=np.pad(g,radius,mode='reflect');out=fftconvolve(padded,kernel,mode='same')[radius:-radius,radius:-radius]
   mag=np.abs(out).astype('float32');full.append(mag.ravel());pooled.extend([float(mag.mean()),float(mag.std())])
 # Nuisance baseline: brightness and colour in brighter parts, no Gabor curves.
 gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(float)/255;rgb=a.astype(float)/255;bright=gray>=np.quantile(gray,.70)
 nuisance=np.r_[rgb[bright].mean(0),rgb[bright].std(0),np.quantile(gray,[.1,.25,.5,.75,.9]),gray.std()]
 return np.concatenate(full),np.array(pooled,'float32'),nuisance.astype('float32')

def fit_projection(x,kind):
 mean=x.mean(0);z=x-mean
 if kind=='full_gabor':
  # Exact dual PCA: sample covariance eigendecomposition, avoiding giant d*d.
  gram=(z@z.T)/max(1,len(z)-1);ev,u=np.linalg.eigh(gram.astype('float64'));idx=np.argsort(ev)[::-1]
  idx=[i for i in idx if ev[i]>max(float(ev.max())*1e-8,1e-12)][:min(8,len(z)-1)]
  if not idx:axes=np.zeros((1,x.shape[1]),'float32')
  else:axes=((u[:,idx].T@z)/np.sqrt(ev[idx,None]*max(1,len(z)-1))).astype('float32')
  zz=z@axes.T
 else:axes=None;zz=z
 avg=zz.mean(0);std=np.maximum(zz.std(0),1e-6)
 return dict(mean=mean,axes=axes,avg=avg,std=std)
def project(x,p):
 z=x-p['mean'];z=z@p['axes'].T if p['axes'] is not None else z
 return (z-p['avg'])/p['std']
def metric(y,p):
 cm=confusion_matrix(y,p,labels=[0,1]);recall=[float(cm[k,k]/cm[k].sum()) if cm[k].sum() else None for k in [0,1]]
 return dict(samples=len(y),label_agreement=float(np.mean(y==p)),macro_f1=float(f1_score(y,p,labels=[0,1],average='macro',zero_division=0)),confusion=cm.tolist(),class_recall=recall)

def inner_select(x,y,groups,kind):
 n=min(3,len(set(groups)));splits=list(GroupKFold(n_splits=n).split(x,y,groups));pred={c:[] for c in CS};truth=[];margins={c:[] for c in CS}
 for tr,va in splits:
  assert set(groups[tr]).isdisjoint(groups[va]);assert len(set(y[tr]))==2
  p=fit_projection(x[tr],kind);a,b=project(x[tr],p),project(x[va],p)
  truth.extend(y[va].tolist())
  for c in CS:
   svm=SVC(C=c,kernel='rbf',gamma='scale',class_weight='balanced');svm.fit(a,y[tr]);pred[c].extend(svm.predict(b).tolist());margins[c].extend(abs(svm.decision_function(b)).tolist())
 scores={c:metric(np.array(truth),np.array(pred[c]))['macro_f1'] for c in CS};best=max(CS,key=lambda c:scores[c]);threshold=float(np.quantile(margins[best],.20))
 return best,threshold,{str(c):v for c,v in scores.items()}

def grouped_cv(x,y,groups,kind,rows,probe_x=None):
 pred=np.full(len(y),-1,int);margin=np.zeros(len(y));threshold=np.zeros(len(y));folds=[];probe_result=None
 for number,(tr,te) in enumerate(LeaveOneGroupOut().split(x,y,groups)):
  held=str(groups[te][0]);assert set(groups[tr]).isdisjoint(groups[te]);c,thr,inner=inner_select(x[tr],y[tr],groups[tr],kind)
  p=fit_projection(x[tr],kind);svm=SVC(C=c,kernel='rbf',gamma='scale',class_weight='balanced');svm.fit(project(x[tr],p),y[tr]);z=project(x[te],p)
  pred[te]=svm.predict(z);margin[te]=abs(svm.decision_function(z));threshold[te]=thr
  fold=dict(held_artist_group=held,train_ids=[rows[i]['id'] for i in tr],test_ids=[rows[i]['id'] for i in te],selected_C=c,
   pca_components=int(len(p['axes'])) if p['axes'] is not None else None,inner_group_macro_f1=inner,review_margin_threshold=thr,metrics=metric(y[te],pred[te]))
  folds.append(fold)
  if held=='Lampo_Leong' and probe_x is not None:
   pp=project(probe_x,p);v=svm.decision_function(pp);probe_result=dict(predicted=svm.predict(pp).tolist(),absolute_margin=abs(v).tolist(),review_threshold=thr,
    flagged_for_review=(abs(v)<thr).tolist(),probe_artist_absent_from_training=True,threshold_uses_known_training_groups_only=True)
  print('FOLD',kind,number+1,held,fold['metrics']['label_agreement'],flush=True)
 assert (pred>=0).all()
 result=dict(**metric(y,pred),artist_groups=len(set(groups)),nested_artist_group_validation=True,expert_accuracy=False,
  probabilities_calibrated=False,folds=folds,predictions=[dict(id=rows[i]['id'],artist_group=rows[i]['artist_group'],label=CLASSES[y[i]],predicted=CLASSES[pred[i]],
   absolute_margin=float(margin[i]),review_threshold=float(threshold[i]),candidate='需复核' if margin[i]<threshold[i] else CLASSES[pred[i]],
   label_scope=rows[i]['label_scope']) for i in range(len(y))],unknown_probe=probe_result)
 return result

def main():
 cv2.setNumThreads(2);start=time.time();manifest_path=ROOT/'dataset_manifest.json';manifest=json.loads(manifest_path.read_text('utf8'));allrows=manifest['records']
 rows=[r for r in allrows if r['admission']=='pilot_only'];probes=[r for r in allrows if r['admission']=='unknown_probe'];allused=rows+probes
 # Conservative duplicate union: all crops of any linked artist remain together.
 parent={r['artist_group']:r['artist_group'] for r in rows}
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 ims=[Image.open(r['image']).convert('RGB') for r in allused];hs=[phash(im) for im in ims];near=[]
 for i in range(len(rows)):
  for j in range(i):
   if rows[i]['artist_group']==rows[j]['artist_group']:continue
   distance=int(np.count_nonzero(hs[i]!=hs[j]))
   if rows[i]['crop_sha256']==rows[j]['crop_sha256'] or distance<=4:
    a,b=find(rows[i]['artist_group']),find(rows[j]['artist_group']);parent[a]=b;near.append(dict(a=rows[i]['id'],b=rows[j]['id'],phash_distance=distance))
 groups=np.array([find(r['artist_group']) for r in rows]);y=np.array([CLASSES.index(r['label']) for r in rows])
 for k in [0,1]:assert len(set(groups[y==k]))>=3
 full=[];pooled=[];colour=[]
 for i,im in enumerate(ims):
  a,b,c=features(im);full.append(a);pooled.append(b);colour.append(c)
  if (i+1)%10==0:print('FEATURES',i+1,len(ims),flush=True)
 full=np.array(full,'float32');pooled=np.array(pooled,'float32');colour=np.array(colour,'float32')
 np.save(ROOT/'gabor_magnitude_features.npy',full);np.savez_compressed(ROOT/'compact_features.npz',pooled=pooled,colour=colour)
 n=len(rows);results={}
 # This existing environment's old threadpoolctl crashes enumerating OpenBLAS.
 # Thread limits are set in the launch environment before numpy imports instead.
 with nullcontext():
  for kind,x in [('full_gabor',full),('pooled_gabor',pooled),('colour_control',colour)]:
   result=grouped_cv(x[:n],y,groups,kind,rows,x[n:]);results[kind]=result;save(kind+'_evaluation.json',result)
  c,thr,inner=inner_select(full[:n],y,groups,'full_gabor');p=fit_projection(full[:n],'full_gabor');svm=SVC(C=c,kernel='rbf',gamma='scale',class_weight='balanced');svm.fit(project(full[:n],p),y)
 joblib.dump(dict(projection=p,svm=svm,classes=CLASSES,review_margin_threshold=thr,feature='40 Gabor magnitude maps,128x128 histogram-equalised grayscale,dual PCA up to8,projected standardisation,RBF SVM',
  input_scope='manually selected mountain region',label_quality='source-evidenced provisional',formal_validation_passed=False,robot_execution=False),ROOT/'gabor_pca_svm_pilot.joblib')
 # Serialize/reload equivalence on all inputs, including held-out other types.
 loaded=joblib.load(ROOT/'gabor_pca_svm_pilot.joblib');assert np.array_equal(loaded['svm'].predict(project(full,loaded['projection'])),svm.predict(project(full,p)))
 summary=dict(classes=CLASSES,known_regions=n,unknown_probes=len(probes),counts={CLASSES[k]:int((y==k).sum()) for k in [0,1]},
  artists=len(set(r['artist_group'] for r in rows)),validation_groups=len(set(groups)),works=len(set(r['work_group'] for r in rows)),publishers=len(set(r['publisher_group'] for r in rows)),
  expert_local_annotations=0,formal_test_samples=0,target_30_to_50_per_class_reached=False,manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),near_duplicates_merged=near,
  method_adaptation='2014 method family, not exact paper reproduction: new named classes/dataset,8 PCA components,reflected finite kernels,nested group CV',
  feature_shape=list(full.shape),pca_and_scaler_fit_on_training_only=True,hyperparameters_selected_in_inner_groups=True,
  all_regions_of_same_artist_kept_together=True,probe_types_not_a_trained_third_class=True,unknown_decisions_not_validated=True,
  paper_nuisance_control_included=True,model_reload_equivalent=True,accepted_painting_model_replaced=False,robot_connected=False,
  metrics={k:{z:v for z,v in r.items() if z in ['label_agreement','macro_f1','confusion','class_recall']} for k,r in results.items()},
  unresolved='Partial source labels and limited works; ROI/attribution uncertainty; no full-image detection, stroke generation or robot execution',elapsed_seconds=time.time()-start)
 save('summary.json',summary);print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
