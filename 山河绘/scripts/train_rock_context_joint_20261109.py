# -*- coding: utf-8 -*-
"""Whole-image joint training with explicit procedural/partial-real distinction."""
import sys,time,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.rock_context_data import *
from gongbi_training.rock_context_model import *
import torch,numpy as np
from PIL import Image
def infer(net,image,threshold=.25,max_predictions=32):
 with torch.no_grad():o=net(torch.tensor(image.transpose(2,0,1)[None],dtype=torch.float32,device='cuda'))
 probability=o['anchor'][0,0].sigmoid();peaks=(probability==torch.nn.functional.max_pool2d(probability[None,None],3,1,1)[0,0])&(probability>=threshold);ys,xs=peaks.nonzero(as_tuple=True);order=torch.argsort(probability[ys,xs],descending=True)[:max_predictions];records=[]
 for i in order:
  y,x=int(ys[i]),int(xs[i]);anchor=np.array([(x+.5)/GRID,(y+.5)/GRID]);q=(o['curves'][0,:,y,x].reshape(POINTS,2).cpu().numpy()+anchor)*SIZE;roleprob=o['roles'][0,:,y,x].softmax(0).cpu().numpy();group=o['groups'][0,:,y,x].cpu().numpy();ml=(o['instance'][0,:,y,x,None,None]*o['pixel'][0]).sum(0)/(8**.5)+o['mask_bias'][0,0,y,x]
  records.append(dict(curve=q.tolist(),role=int(roleprob.argmax()),role_probability=roleprob.tolist(),confidence=float(probability[y,x]),anchor=[x,y],group_embedding=group.tolist(),mask=ml.sigmoid().cpu().numpy()))
 maps={k:o[k][0].detach().cpu().numpy() for k in ['anchor','semantic','endpoints']};return records,maps
def evaluate_procedural(net,seeds,threshold=.25):
 from scipy.optimize import linear_sum_assignment
 observations=[]
 for seed in seeds:
  r=synthetic(seed);pred,maps=infer(net,r['image'],threshold);truth=r['strokes'];cost=np.zeros((len(truth),len(pred)),float)
  for i,t in enumerate(truth):
   for j,p in enumerate(pred):
    a=np.array(p['curve']);b=t['curve'];cost[i,j]=min(np.linalg.norm(a-b,axis=1).mean(),np.linalg.norm(a-b[::-1],axis=1).mean())
  pairs=list(zip(*linear_sum_assignment(cost))) if len(pred) else [];matched=[(i,j) for i,j in pairs if cost[i,j]<=12];roles=[pred[j]['role']==truth[i]['role'] for i,j in matched]
  observations.append(dict(seed=seed,true_count=len(truth),predicted_count=len(pred),peak_anchor_probability=float(1/(1+np.exp(-maps['anchor'].max()))),matched_within_12px=len(matched),role_accuracy_on_matched=float(np.mean(roles)) if roles else None,mean_matched_curve_error_px=float(np.mean([cost[i,j] for i,j in matched])) if matched else None))
 return observations
def main():
 prepare();torch.manual_seed(110914);torch.set_num_threads(4);net=RockContextNet().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.0001);real=real_example();start=time.time();history=[];counts=dict(procedural_examples=0,real_augmented_views=0,original_real_works=1,original_real_modules=1,real_reference_segments=8)
 # No external real work enters training or checkpoint selection. The held-out
 # procedural seeds are diagnostics only; final budget is fixed in advance.
 for step in range(1001):
  if step:
   rngbase=110914000+step*8;batchdata=[synthetic((110940+i) if step<=100 else (rngbase+i)) for i in range(4 if step<=600 else 3)];counts['procedural_examples']+=len(batchdata)
   if step>600:batchdata.append(augment(real,step+921));counts['real_augmented_views']+=1
   ts=[targets(r) for r in batchdata];x=torch.tensor(np.stack([r['image'].transpose(2,0,1) for r in batchdata]),dtype=torch.float32,device='cuda');net.train();o=net(x);loss,parts=joint_loss(o,ts,dense_curves=False);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
  if step%100==0:
   net.eval();ev=evaluate_procedural(net,[880001,880002],threshold=.25);history.append(dict(step=step,loss_parts={k:float(v.detach()) for k,v in parts.items()} if step else None,procedural_development=ev));write(ROOT/'training_progress.json',dict(step=step,counts=counts,history=history,elapsed_seconds=time.time()-start));print('STEP',step,history[-1],flush=True)
  if step in [600,1000]:torch.save(dict(model={k:v.detach().cpu() for k,v in net.state_dict().items()},step=step,counts=dict(counts),input_size=SIZE,points_per_curve=POINTS,training_source_works=['short_pima_work_only'],checkpoint_selection='fixed budgets; no real external work selection',named_cunfa_classifier=False),ROOT/f'whole_context_step{step}.pt')
 write(ROOT/'training_report.json',dict(counts=counts,history=history,elapsed_seconds=time.time()-start,selected_step=1000,
  model_family='whole-module encoder + global attention + learned stroke anchors + direct complete curves + partial ink masks + roles + endpoints + group embeddings',
  original_image_not_small_local_patches=True,input='completed static RGB module only',manual_seed_at_inference=False,manual_direction_at_inference=False,oracle_endpoints_at_inference=False,
  procedural_is_real_cunfa=False,procedural_unique_scenes=3204,first_100_steps_use_four_fixed_sanity_scenes=True,unlabelled_real_is_negative=False,complete_real_instance_masks_available=False,author_disjointness_verified=False,
  postprocess_geometry_stitching=False,named_cunfa_recognition_trained=False,robot_connected=False,adopted=False))
 print('TRAIN_FINISHED',counts,time.time()-start,flush=True)
if __name__=='__main__':main()
