"""Fixed source-held-out experiment. Structural statistics are proxies, not stroke labels."""
import sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2,joblib
from PIL import Image
from scipy.ndimage import convolve
from skimage.morphology import skeletonize
from sklearn.svm import SVC
from train_cunfa_gabor_20261110 import features,grouped_cv,inner_select,fit_projection,project,metric,CLASSES,phash
ROOT=LAB/'cunfa_structure_20261111';OLD=LAB/'cunfa_gabor_baseline_20261110'
def save(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hist(v,w,n=12,ran=(0,np.pi)):
 h=np.histogram(v,bins=n,range=ran,weights=w)[0];return h/(h.sum()+1e-9)
def quant(x):return np.quantile(x,[.1,.25,.5,.75,.9]).tolist() if x.size else [0.]*5
def structure(im,diagnostic=False):
 a=cv2.cvtColor(np.array(im.convert('RGB')),cv2.COLOR_RGB2GRAY).astype('float32')/255
 a=cv2.resize(a,(128,128),interpolation=cv2.INTER_AREA)
 bg=cv2.GaussianBlur(a,(0,0),11);ink=np.maximum(0,(bg-a)/np.maximum(bg,.04));ink=np.clip(ink/max(float(np.quantile(ink,.95)),.03),0,1)
 out=[];diag=[]
 # Spatial orientation and coherence at three scales, no colour feature.
 for scale in [.7,1.5,3.]:
  sm=cv2.GaussianBlur(ink,(0,0),scale);gx=cv2.Sobel(sm,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(sm,cv2.CV_32F,0,1,ksize=3)
  mag=np.hypot(gx,gy);angle=np.mod(np.arctan2(gy,gx),np.pi)
  for yy in range(0,128,32):
   for xx in range(0,128,32):out.extend(hist(angle[yy:yy+32,xx:xx+32],mag[yy:yy+32,xx:xx+32],9))
  jxx=cv2.GaussianBlur(gx*gx,(0,0),3);jyy=cv2.GaussianBlur(gy*gy,(0,0),3);jxy=cv2.GaussianBlur(gx*gy,(0,0),3)
  coherence=np.sqrt((jxx-jyy)**2+4*jxy*jxy)/(jxx+jyy+1e-8);theta=np.mod(.5*np.arctan2(2*jxy,jxx-jyy),np.pi)
  weights=ink*(jxx+jyy);out.extend(hist(theta,weights));out.extend(quant(coherence[ink>.2]));out.extend(quant(mag[ink>.2]))
  if scale==1.5:diag=[theta,coherence]
 # Width, fragmentation and connectivity are measured on several local-contrast masks.
 for threshold in [.20,.40,.60]:
  mask=ink>threshold;sk=skeletonize(mask);degree=convolve(sk.astype('uint8'),np.ones((3,3),'uint8'),mode='constant')-sk.astype('uint8')
  distance=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)
  out.extend(quant(distance[sk]/128));out.extend([float(mask.mean()),float(sk.mean()),float(np.sum(sk&(degree==1)))/max(1,int(sk.sum())),float(np.sum(sk&(degree>=3)))/max(1,int(sk.sum()))])
  n,labels,stats,centres=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
  areas=stats[1:,cv2.CC_STAT_AREA];valid=areas>=3;st=stats[1:][valid]
  out.extend([float(valid.sum())/128]);out.extend(quant(areas[valid]/16384))
  ratios=np.minimum(st[:,2],st[:,3])/np.maximum(st[:,2],st[:,3]) if len(st) else np.array([]);out.extend(quant(ratios))
  # Autocorrelation describes spacing, without pretending to assign painter stroke groups.
  z=mask.astype('float32')-mask.mean();den=float((z*z).mean())+1e-8
  for lag in [2,4,8,16]:out.extend([float((z[:,lag:]*z[:,:-lag]).mean()/den),float((z[lag:]*z[:-lag]).mean()/den)])
 feat=np.asarray(out,'float32');assert np.isfinite(feat).all()
 return (feat,(ink,*diag)) if diagnostic else feat
def predict_records(rows,y,p,margin,threshold):
 return [dict(id=r['id'],label=CLASSES[int(y[i])],predicted=CLASSES[int(p[i])],artist_group=r['artist_group'],absolute_margin=float(margin[i]),review_threshold=threshold,review=bool(margin[i]<threshold)) for i,r in enumerate(rows)]
def main():
 cv2.setNumThreads(2);start=time.time();manifest=ROOT/'dataset_manifest.json';d=json.loads(manifest.read_text('utf8'));train=d['development_records'];test=d['external_holdout'];probes=d['previous_unknown_probes'];rows=train+test+probes;n=len(train);m=len(test)
 assert {r['artist_group'] for r in train}.isdisjoint({r['artist_group'] for r in test})
 # Preregister the candidate, secondary and controls before any new predictions.
 protocol=dict(primary='structure_statistics_SVM',secondary='structure_plus_pooled_Gabor_SVM',controls=['frozen_yesterday_Gabor_PCA_SVM','brightness_colour_SVM'],train_ids=[r['id'] for r in train],external_test_ids=[r['id'] for r in test],
  train_artists=sorted({r['artist_group'] for r in train}),external_artists=sorted({r['artist_group'] for r in test}),manifest_sha256=sha(manifest),code_sha256=sha(__file__),
  selection='All C values .1/1/10 selected using only 3fold author-group CV within old40; primary predetermined, no choice based on external test',
  external_images_used_for_training=False,development_cv_not_new_blind_test=True,unknown_probes_not_independent_new_test=True,
  geometry='Structure statistics measured after128square resizing; not physical brush width or expert stroke ownership',expert_local_gold=0,robot_connected=False)
 if (ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text('utf8'))==protocol,'Protocol changed after initial run; do not silently overwrite.'
 else:save('protocol.json',protocol)
 ims=[Image.open(r['image']).convert('RGB') for r in rows];oldfull=np.load(OLD/'gabor_magnitude_features.npy');oldcompact=np.load(OLD/'compact_features.npz');newfull=[];newpooled=[];newcolour=[]
 # Exact/near duplicates across train and external test are an audit gate.
 near=[];train_hash=[phash(im) for im in ims[:n]]
 for i,im in enumerate(ims[n:n+m]):
  hh=phash(im)
  for j,h in enumerate(train_hash):
   distance=int(np.count_nonzero(hh!=h))
   if distance<=4 or rows[n+i]['crop_sha256']==rows[j]['crop_sha256']:near.append(dict(train=rows[j]['id'],test=rows[n+i]['id'],phash_distance=distance))
 assert not near,'Cross-split possible duplicate: stop and review, no prediction.'
 for i,im in enumerate(ims[n:n+m]):
  a,b,c=features(im);newfull.append(a);newpooled.append(b);newcolour.append(c)
  if (i+1)%8==0:print('NEW_FEATURES',i+1,m,flush=True)
 full=np.concatenate([oldfull[:n],np.asarray(newfull),oldfull[n:]])
 pooled=np.concatenate([oldcompact['pooled'][:n],np.asarray(newpooled),oldcompact['pooled'][n:]])
 colour=np.concatenate([oldcompact['colour'][:n],np.asarray(newcolour),oldcompact['colour'][n:]])
 shape=np.array([structure(im) for im in ims]);fusion=np.concatenate([shape,pooled],axis=1);np.savez_compressed(ROOT/'features.npz',structure=shape,pooled=pooled,colour=colour)
 y=np.array([CLASSES.index(r['label']) for r in train]);yt=np.array([CLASSES.index(r['label']) for r in test]);groups=np.array([r['artist_group'] for r in train]);results={};models={}
 for kind,x in [('structure_statistics',shape),('structure_plus_gabor',fusion),('colour_control',colour)]:
  cv=grouped_cv(x[:n],y,groups,kind,train,x[n+m:]);save(kind+'_development_cv.json',cv)
  c,threshold,inner=inner_select(x[:n],y,groups,kind);projection=fit_projection(x[:n],kind);svm=SVC(C=c,kernel='rbf',gamma='scale',class_weight='balanced');svm.fit(project(x[:n],projection),y)
  xt=project(x[n:n+m],projection);p=svm.predict(xt);margin=abs(svm.decision_function(xt));pp=svm.predict(project(x[n+m:],projection))
  result=dict(metrics=metric(yt,p),per_artist={a:metric(yt[np.array([r['artist_group']==a for r in test])],p[np.array([r['artist_group']==a for r in test])]) for a in sorted({r['artist_group'] for r in test})},
   selected_C=c,train_only_inner_scores=inner,feature_count=x.shape[1],predictions=predict_records(test,yt,p,margin,threshold),old_unknown_probe_forced_binary=[CLASSES[int(k)] for k in pp],expert_accuracy=False)
  result['artist_macro_label_agreement']=float(np.mean([a['label_agreement'] for a in result['per_artist'].values()]));results[kind]=result;save(kind+'_external.json',result)
  model=dict(projection=projection,svm=svm,classes=CLASSES,feature_count=x.shape[1],review_threshold=threshold,trained_ids=protocol['train_ids'],formal_validation=False)
  models[kind]=model;joblib.dump(model,ROOT/(kind+'_pilot.joblib'));reload=joblib.load(ROOT/(kind+'_pilot.joblib'));assert np.array_equal(p,reload['svm'].predict(project(x[n:n+m],reload['projection'])))
 # Fixed yesterday model: no re-fit and no hyperparameter choice using new data.
 b=joblib.load(OLD/'gabor_pca_svm_pilot.joblib');z=project(full[n:n+m],b['projection']);bp=b['svm'].predict(z);bm=abs(b['svm'].decision_function(z));br=dict(metrics=metric(yt,bp),predictions=predict_records(test,yt,bp,bm,b['review_margin_threshold']),model_sha256=sha(OLD/'gabor_pca_svm_pilot.joblib'),refitted=False)
 br['per_artist']={a:metric(yt[np.array([r['artist_group']==a for r in test])],bp[np.array([r['artist_group']==a for r in test])]) for a in sorted({r['artist_group'] for r in test})};br['artist_macro_label_agreement']=float(np.mean([a['label_agreement'] for a in br['per_artist'].values()]));results['frozen_gabor']=br;save('frozen_gabor_external.json',br)
 # Paired region-level changes retained; crops are clustered, not independent trials.
 newp=np.array([CLASSES.index(r['predicted']) for r in results['structure_statistics']['predictions']]);better=int(np.sum((bp!=yt)&(newp==yt)));worse=int(np.sum((bp==yt)&(newp!=yt)))
 primary=results['structure_statistics'];passed=primary['metrics']['macro_f1']>br['metrics']['macro_f1'] and primary['metrics']['macro_f1']>results['colour_control']['metrics']['macro_f1'] and min(primary['metrics']['class_recall'])>=.7
 summary=dict(train_regions=n,new_test_regions=m,new_test_artists=4,total_known_regions=n+m,known_counts={cl:sum(r['label']==cl for r in train+test) for cl in CLASSES},old_unknown_probes=len(probes),
  total_known_artists=len({r['artist_group'] for r in train+test}),new_test_is_source_labelled_not_expert=True,expert_local_gold=0,primary_passed_provisional_gate=bool(passed),gate='Externally exceed frozen Gabor and colour-control macroF1 and each class recall>=.70; still not formal expert validation',
  model_adopted=False,paired_region_improved=better,paired_region_regressed=worse,crops_not_independent_trials=True,formal_unknown_detection=False,stroke_generation=False,robot_connected=False,
  metrics={k:dict(**v['metrics'],artist_macro_label_agreement=v['artist_macro_label_agreement']) for k,v in results.items()},manifest_sha256=sha(manifest),protocol_sha256=sha(ROOT/'protocol.json'),code_sha256=sha(__file__),elapsed_seconds=time.time()-start)
 save('summary.json',summary)
 verification=dict(protocol_saved_before_external_prediction=True,old_manifest_matches=sha(OLD/'dataset_manifest.json')==d['old_manifest_sha256'],train_external_artist_overlap=0,cross_split_near_duplicate_matches=near,
  old_model_unchanged=sha(OLD/'gabor_pca_svm_pilot.joblib')==br['model_sha256'],all_source_crop_hashes_match=all(sha(r['image'])==r['crop_sha256'] and sha(r['source_image'])==r['source_sha256'] for r in train+test),
  model_reload_equivalent=True,external_test_never_used_for_C_PCA_scaler=True,external_known_predicted_once_each=True,automatic_predictions_not_training_labels=True,robot_connected=False)
 assert verification['old_manifest_matches'] and verification['all_source_crop_hashes_match'];save('verification.json',verification);print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
