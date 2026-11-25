# -*- coding: utf-8 -*-
import sys,json,hashlib,random,os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,joblib
import torch.nn.functional as F
from PIL import Image
from torchvision.models import mobilenet_v3_small
from gongbi_training.cunfa_boundary_relations import *
from gongbi_training.cunfa_spatial_pairs import pyramid_vector
from gongbi_training.cunfa_context_roles import context_role_target,context_box,ROLE_NAMES
from gongbi_training.cunfa_photometry import appearance_tensor,appearance_canvas,contrast_features
from cunfa_photometry_train_20261117 import role_metric
ROOT=LAB/'cunfa_boundary_relations_20261117';OLD=LAB/'cunfa_spatial_pairs_20261117';WEIGHT=LAB/'cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 if '--resume-determinism-fix' in sys.argv:
  assert (ROOT/'protocol.json').exists() and not (ROOT/'training_selection.json').exists()
  (ROOT/'protocol.json').rename(ROOT/'protocol_aborted_cublas_pre_fit.json')
 assert not (ROOT/'protocol.json').exists(),'Frozen experiment'
 d=read(ROOT/'reference_manifest.json');rows=d['records'];annotations={a['id']:a for a in d['curves']};y=np.array([context_role_target(r) for r in rows]);fit=np.array([r['split']=='train' for r in rows])&(y!=-100);groups=np.array(['qingpu_book_all' if r['role_fold_group'].startswith('qingpu') else r['role_fold_group'] for r in rows])
 hashes={str(p):sha(p) for p in [OLD/'spatial_models.joblib',OLD/'features.npz',OLD/'results.json',OLD/'reference_manifest.json']}
 protocol=dict(script_sha256=sha(__file__),module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/cunfa_boundary_relations.py'),manifest_sha256=sha(ROOT/'reference_manifest.json'),encoder_sha256=sha(WEIGHT),old_artifacts=hashes,
  role_training=int(fit.sum()),training_auxiliary_windows=sum(r['split']=='train' for r in d['curves']),training_side_windows=sum(r['split']=='train' and bool(r['open']) for r in d['curves']),fresh_work=45636,
  arms=['role_only','boundary_relations'],architecture='frozen MobileNet first2 blocks16channels56x56 + raw gray andxy differences; shared2layer conv24channels; role ROI/global pooling plus sparse edge,direction,side heads',
  epochs=100,optimizer='AdamW lr.001 weight_decay.01, full batch, seed0 for each identical initialization',loss='class-balanced role CE; joint adds.3*(sparse boundary BCE+side CE+.25 axial direction SmoothL1)',
  reference_scope='assistant sparse boundary corridor and side scribbles, not expert ink edge or stroke; other pixels ignored',
  grouping='all Qingpu book illustrations together; all video frames together; each Met object entire work',
  selection='training leave4wholegroups OOF macro recall, tie structure recall; no choosing on checks',
  inputs='given2.5x neighborhood andROI box only; no curve, side point, stage, label or reference mask at inference',
  checks='original12 and attributedShen6 known regressions; newNiZan7 assistant references (2structure2internal1tone2paper), same museum source; no expert or author-independent certification',
  deterministic_workspace=':4096:8',aborted_attempt='first backward stopped before any model scores; set deterministic cuBLAS workspace' if (ROOT/'protocol_aborted_cublas_pre_fit.json').exists() else None,
  production_replaced=False,robot_connected=False)
 write('protocol.json',protocol);print('FROZEN',int(fit.sum()),len(rows),flush=True)
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
 net=mobilenet_v3_small(weights=None);net.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True));net=net.cuda().eval();tensors=[];gray=[];rois=[];targets=[];refs={};oldfeatures=np.load(OLD/'features.npz');(ROOT/'views').mkdir(exist_ok=True)
 for r in rows:
  assert sha(r['image'])==r['image_sha256']
  if r['reference_image'] not in refs:refs[r['reference_image']]=Image.open(r['reference_image']).convert('RGB')
  ref=refs[r['reference_image']];nb=context_box(r['box'],ref.size,2.5);near=ref.crop(nb);local=Image.open(r['image']).convert('RGB');tensors.append(appearance_tensor(near,ref,'reference_linear'));g=np.asarray(appearance_canvas(near,ref,'reference_linear').convert('L'),dtype='float32')/255;gray.append(g);rois.append(roi_mask(r['box'],nb));targets.append(sparse_targets(annotations.get(r['id']),nb));near.save(ROOT/'views'/(r['id']+'_near.png'));local.save(ROOT/'views'/(r['id']+'_local.png'))
 maps=[]
 with torch.inference_mode():
  for i in range(0,len(rows),12):maps.append(net.features[:2](torch.from_numpy(np.stack(tensors[i:i+12])).cuda()).cpu())
 maps=torch.cat(maps);g=F.interpolate(torch.from_numpy(np.stack(gray))[:,None],size=(56,56),mode='area');dx=F.pad(g[:,:,:,1:]-g[:,:,:,:-1],(0,1,0,0));dy=F.pad(g[:,:,1:,:]-g[:,:,:-1,:],(0,0,0,1));x=torch.cat([maps,g,dx*10,dy*10],1).cuda();roi=torch.from_numpy(np.stack(rois)).cuda();yy=torch.from_numpy(y).long().cuda()
 edge=torch.from_numpy(np.stack([a[0] for a in targets])).cuda();side=torch.from_numpy(np.stack([a[1] for a in targets])).cuda();angle=torch.from_numpy(np.stack([a[2] for a in targets])).cuda();orient=torch.from_numpy(np.stack([a[3] for a in targets])).cuda()
 def train(mask,joint):
  torch.manual_seed(0);np.random.seed(0);random.seed(0);m=BoundaryRoleHead().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);idx=torch.from_numpy(np.flatnonzero(mask)).cuda();count=torch.bincount(yy[idx],minlength=4).float();assert (count>0).all();w=count.sum()/count/4
  for epoch in range(100):
   m.train();opt.zero_grad(set_to_none=True);o=m(x[idx],roi[idx]);loss=F.cross_entropy(o[0],yy[idx],weight=w)
   if joint:loss=loss+.3*auxiliary_loss(o,edge[idx],side[idx],angle[idx],orient[idx])
   assert torch.isfinite(loss);loss.backward();opt.step()
  return m.eval()
 models={};metrics={};preds={};dense={}
 for arm in ['role_only','boundary_relations']:
  oof=np.full(len(rows),-100)
  for group in sorted(set(groups[fit])):
   tr=fit&(groups!=group);te=fit&(groups==group);m=train(tr,arm=='boundary_relations')
   with torch.inference_mode():oof[te]=m(x[te],roi[te])[0].argmax(1).cpu().numpy()
   print('FOLD',arm,group,flush=True)
  metrics[arm]=role_metric(y[fit],oof[fit]);m=train(fit,arm=='boundary_relations');models[arm]={k:v.cpu() for k,v in m.state_dict().items()}
  with torch.inference_mode():
   o=m(x,roi);preds[arm]=o[0].argmax(1).cpu().numpy();dense[arm]=dict(edge=o[1].sigmoid().cpu().numpy(),direction=o[2].cpu().numpy(),side=o[3].softmax(1).cpu().numpy())
  print('TRAIN_OOF',arm,json.dumps(metrics[arm]),flush=True)
 selected=max(metrics,key=lambda a:(metrics[a]['macro_recall'],metrics[a]['per_role']['outline_context']['agreement']));write('training_selection.json',dict(selected_arm=selected,metrics=metrics,before_check_scoring=True))
 # Frozen previous predictions replay on137 known regions, infer new work from
 # pixels only. No new reference geometry is sent into either frozen model.
 previous=joblib.load(OLD/'spatial_models.joblib');newpairs=[];stats=[]
 for r in rows[137:]:
  ref=refs[r['reference_image']];pair=[Image.open(r['image']).convert('RGB'),ref.crop(context_box(r['box'],ref.size,2.5))];newpairs.extend(appearance_tensor(v,ref,'reference_linear') for v in pair);stats.append(np.concatenate([contrast_features(v,ref) for v in pair]))
 with torch.inference_mode():
  fm=net.features(torch.from_numpy(np.stack(newpairs)).cuda());mean=F.normalize(fm.mean((2,3)),dim=1).cpu().numpy().reshape(9,-1)/np.sqrt(2);sp=pyramid_vector(fm).cpu().numpy().reshape(9,-1)/np.sqrt(2)
 feature={a:np.concatenate([oldfeatures[a],np.concatenate([mean if a=='expanded_global' else sp,np.stack(stats)],1)]) for a in ['expanded_global','expanded_spatial_pyramid']}
 for a in feature:preds['frozen_'+a]=previous['models'][a].predict(feature[a]).astype('int64')
 oldresult=read(OLD/'results.json');assert all(ROLE_NAMES[int(preds['frozen_expanded_global'][i])]==r['predictions']['expanded_global'] for i,r in enumerate(oldresult['records']))
 masks={'known12':np.array([r['id'].startswith('ntm_') for r in rows])&(y!=-100),'seen_shen6':np.array([r['id'].startswith('shen_attributed_') for r in rows])&(y!=-100),'fresh_nizan7':np.array([r['id'].startswith('nizan_') for r in rows])&(y!=-100)}
 checks={name:{a:role_metric(y[mask],p[mask]) for a,p in preds.items()} for name,mask in masks.items()}
 auxiliary={}
 for split in ['train','fresh_check']:
  mask=np.array([r['split']==split and r['id'] in annotations for r in rows]);de=dense['boundary_relations'];ev=edge[mask].cpu().numpy();sv=side[mask].cpu().numpy();ov=orient[mask].cpu().numpy();av=angle[mask].cpu().numpy();ep=de['edge'][mask]>=.5;spred=de['side'][mask].argmax(1);ap=de['direction'][mask]
  angle_error=np.arccos(np.clip((ap*av).sum(1),-1,1))/2*180/np.pi
  auxiliary[split]=dict(windows=int(mask.sum()),boundary_positive_recall=float(ep[ev==1].mean()),boundary_negative_specificity=float((~ep[ev==0]).mean()),side_balanced_accuracy=float(np.mean([(spred[sv==c]==c).mean() for c in [0,1]])),axial_angle_mean_degrees=float(angle_error[ov].mean()),scope='sparse assistant corridor/points only, no full image IoU')
 result=dict(selected_arm=selected,training_oof_metrics=metrics,check_metrics=checks,auxiliary_metrics=auxiliary,records=[dict(id=r['id'],split=r['split'],role_reference=None if y[i]==-100 else ROLE_NAMES[y[i]],predictions={a:ROLE_NAMES[int(p[i])] for a,p in preds.items()},selected_prediction=ROLE_NAMES[int(preds[selected][i])]) for i,r in enumerate(rows)],model_adopted=False,robot_connected=False)
 write('results.json',result);torch.save(dict(states=models,selected_arm=selected,protocol=protocol),ROOT/'boundary_models.pt');np.savez_compressed(ROOT/'features.npz',features=x.cpu().numpy(),roi=roi.cpu().numpy(),targets=y);np.savez_compressed(ROOT/'dense_outputs.npz',**{a+'_'+k:v for a,dn in dense.items() for k,v in dn.items()});assert all(sha(p)==h for p,h in hashes.items());write('verification.json',dict(old_artifacts_unchanged=True,old137_frozen_predictions_replayed=True,annotations_loss_and_scoring_only=True,fresh_not_fit=True,whole_book_and_video_grouped=True,tree_unknown_ignored=True,robot_connected=False));print('CHECKS',json.dumps(checks),flush=True);print('AUX',json.dumps(auxiliary),flush=True)
if __name__=='__main__':main()
