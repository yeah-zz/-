# -*- coding: utf-8 -*-
"""No oracle seeds at inference; all real evaluations are partial development.

Manual module crops are disclosed. Synthetic labels certify representation
checks only. No automatic candidate becomes a gold label or robot trajectory.
"""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.rock_context_data import *
from gongbi_training.rock_context_model import *
from train_rock_context_joint_20261109 import infer
import torch,numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from scipy.optimize import linear_sum_assignment

def match(pred,refs,tolerance=6.):
 if not refs:return []
 if not pred:return [dict(id=r['id'],matched=False,reference_coverage=0.,mean_reference_distance_px=None,role_correct=None) for r in refs]
 distances=[]
 for r in refs:
  q=r['curve'][r.get('curve_valid',np.ones(len(r['curve'])))>0];row=[]
  for p in pred:
   a=np.array(p['curve']);dist=np.linalg.norm(q[:,None]-a[None,:],axis=2);row.append(float((dist.min(0).mean()+dist.min(1).mean())/2))
  distances.append(row)
 pairs=dict(zip(*linear_sum_assignment(np.array(distances))));out=[]
 for i,r in enumerate(refs):
  if i not in pairs:out.append(dict(id=r['id'],matched=False,reference_coverage=0.,role_correct=None));continue
  j=int(pairs[i]);q=r['curve'][r.get('curve_valid',np.ones(len(r['curve'])))>0];a=np.array(pred[j]['curve']);dist=np.linalg.norm(q[:,None]-a[None,:],axis=2);covered=float((dist.min(1)<=tolerance).mean());out.append(dict(id=r['id'],candidate_index=j,matched=covered>=.7,reference_coverage=covered,mean_reference_distance_px=float(dist.min(1).mean()),role_correct=pred[j]['role']==r['role'],reference_is_partial=True,global_precision=None))
 return out

def fit_module(rgb,box):
 x0,y0,x1,y1=box;scale=min((SIZE-24)/(x1-x0),(SIZE-24)/(y1-y0));origin=np.array([(SIZE-(x1-x0)*scale)/2,(SIZE-(y1-y0)*scale)/2]);m=np.float32([[scale,0,origin[0]-x0*scale],[0,scale,origin[1]-y0*scale]])
 a=cv2.warpAffine(rgb,m,(SIZE,SIZE),borderMode=cv2.BORDER_REFLECT_101);return a.astype(np.float32)/255,m

def external_examples():
 old=LAB/'pima_single_stroke_20261109';rgb=np.array(Image.open(old/'single_reference'/'S03_after.png').convert('RGB'));image,m=fit_module(rgb,[260,185,440,475]);labels=json.loads((old/'single_reference_manifest.json').read_text('utf8'))['records'];refs=[]
 for r in labels:
  if r['status']=='admitted_candidate':refs.append(dict(id=r['id'],curve=sample_curve(cv2.transform(np.array(r['reference_curve'],np.float32)[:,None,:],m)[:,0]),role=1,reference='previous video motion candidate, not expert truth'))
 examples=[dict(id='liupengkai_other_work',image=image,strokes=refs,scope='different work, prior method development; not blind test',original_artist_verified=True)]
 # Same painting, spatially different complete rock, unseen by training crop.
 rgb=np.array(Image.open(PREV/'finished_frame.jpg').convert('RGB'));image,m=fit_module(rgb,[575,235,805,405]);examples.append(dict(id='same_work_right_rock',image=image,strokes=[],scope='same-work untrained rock; qualitative only',original_artist_verified=False))
 root=LAB/'pima_static_discovery_20261109';allrefs=json.loads((root/'external_visible_curve_references.json').read_text('utf8'))
 for filename,box in [('new_howto_frame_00.jpg',[325,175,465,320]),('new_howto_frame_01.jpg',[340,25,650,260])]:
  rgb=np.array(Image.open(root/'external'/filename).convert('RGB'));image,m=fit_module(rgb,box);refs=[]
  for r in allrefs:
   if r['image']==filename:refs.append(dict(id=r['id'],curve=sample_curve(cv2.transform(np.array(r['points'],np.float32)[:,None,:],m)[:,0]),role=0,reference=r['provenance']))
  examples.append(dict(id=Path(filename).stem,image=image,strokes=refs,scope='different work, partial visible contour; prior method development, not blind test',original_artist_verified=False))
 return examples

def procedural_score(net,seeds,threshold):
 rows=[]
 for seed in seeds:
  r=synthetic(seed);p,_=infer(net,r['image'],threshold);cost=np.zeros((len(r['strokes']),len(p)),float)
  for i,s in enumerate(r['strokes']):
   for j,c in enumerate(p):
    a=np.array(c['curve']);b=s['curve'];cost[i,j]=min(np.linalg.norm(a-b,axis=1).mean(),np.linalg.norm(a-b[::-1],axis=1).mean())
  pairs=list(zip(*linear_sum_assignment(cost))) if len(p) else [];good=[(i,j) for i,j in pairs if cost[i,j]<=12];n=len(good);f1=2*n/max(1,len(p)+len(r['strokes']));ious=[]
  for i,j in good:
   gt=cv2.resize(r['strokes'][i]['mask'],(GRID,GRID),interpolation=cv2.INTER_NEAREST)>.5;pm=p[j]['mask']>.5;ious.append(float((gt&pm).sum()/max(1,(gt|pm).sum())))
  rows.append(dict(seed=seed,truth=len(r['strokes']),predicted=len(p),matches_12px=n,f1=f1,mean_instance_iou_on_matched=float(np.mean(ious)) if ious else None))
 return rows

def group_plan(pred):
 # Roles/embeddings are predicted. Contours-before-cun remains an explicit
 # planning rule, not a trained painter policy or recovered artist order.
 groups=[]
 for i,p in enumerate(pred):
  if p['role']==0:p['group_id']='contour';continue
  g=np.array(p['group_embedding']);chosen=None
  for j,group in enumerate(groups):
   if min(float(g@np.array(pred[k]['group_embedding'])) for k in group)>.65:chosen=j;break
  if chosen is None:chosen=len(groups);groups.append([])
  groups[chosen].append(i);p['group_id']=f'cun_group_{chosen+1}'
 ordered=sorted(range(len(pred)),key=lambda i:(pred[i]['role'],pred[i].get('group_id',''),np.array(pred[i]['curve'])[:,1].mean()))
 return ordered

def paint(image,pred,maps,name):
 image=np.uint8(np.clip(image,0,1)*255);base=Image.fromarray(image).resize((512,512));line=base.copy();d=ImageDraw.Draw(line)
 for p in pred:
  q=np.array(p['curve'])*2;col='#ad741b' if p['role']==0 else '#157ca4';d.line([tuple(v) for v in q],fill=col,width=2);d.ellipse((q[0,0]-3,q[0,1]-3,q[0,0]+3,q[0,1]+3),fill=col)
 base.save(ROOT/(name+'_input.png'));line.save(ROOT/(name+'_curves.png'))
 sem=maps['semantic'].argmax(0);palette=np.array([[250,249,242],[191,132,46],[47,143,171]],np.uint8);Image.fromarray(palette[sem]).resize((512,512)).save(ROOT/(name+'_roles.png'))
 maskoverlay=cv2.resize(image,(512,512)).astype(float);colors=np.array([[42,133,172],[191,128,40],[105,154,72],[152,94,161]],float)
 for i,p in enumerate(pred):
  mask=cv2.resize(p['mask'],(512,512))>.5;maskoverlay[mask]=maskoverlay[mask]*.65+colors[i%4]*.35
 Image.fromarray(np.uint8(maskoverlay)).save(ROOT/(name+'_masks.png'))
 heat=1/(1+np.exp(-maps['anchor'][0]));color=cv2.applyColorMap(np.uint8(heat*255),cv2.COLORMAP_VIRIDIS);Image.fromarray(cv2.cvtColor(color,cv2.COLOR_BGR2RGB)).resize((512,512)).save(ROOT/(name+'_anchors.png'))
 ep=1/(1+np.exp(-maps['endpoints']));color=np.stack([ep[1],ep[0]*.6,ep[2]],axis=2);Image.fromarray(np.uint8(color*255)).resize((512,512)).save(ROOT/(name+'_endpoints.png'))

def main():
 torch.set_num_threads(4);ck=torch.load(ROOT/'whole_context_step1250.pt',map_location='cpu',weights_only=False);net=RockContextNet(project_curve=True).cuda();net.load_state_dict(ck['model']);net.eval()
 calibration=[]
 for threshold in [.08,.12,.18,.25,.35]:
  rows=procedural_score(net,[880001,880002,880003,880004],threshold);calibration.append(dict(threshold=threshold,mean_procedural_f1=float(np.mean([r['f1'] for r in rows])),rows=rows))
 selected=max(calibration,key=lambda r:r['mean_procedural_f1']);threshold=selected['threshold'];write(ROOT/'threshold_calibration.json',dict(selected=threshold,source='procedural development only; before real inference',rows=calibration));held=procedural_score(net,[990101,990102,990103,990104,990105,990106],threshold)
 real=real_example();examples=[dict(real,id='real_training_module',scope='same module as sparse real training; fit diagnostic only')]+external_examples();allresults=[]
 for r in examples:
  p,maps=infer(net,r['image'],threshold)
  for c in p:
   q=np.array(c['curve']);c['outside_module_fraction']=float(((q<0)|(q>SIZE)).any(1).mean())
  order=group_plan(p);matches=match(p,r['strokes']);paint(r['image'],p,maps,r['id']);np.savez_compressed(ROOT/(r['id']+'_raw_maps.npz'),**maps,masks=np.array([x['mask'] for x in p]))
  serial=[{k:v for k,v in s.items() if k!='mask'} for s in p];result=dict(id=r['id'],scope=r['scope'],predicted_count=len(p),prediction_cap=32,cap_reached=len(p)==32,manual_module_box_at_input=True,manual_stroke_seeds_at_inference=False,manual_directions_at_inference=False,temporal_difference_at_inference=False,predictions=serial,matches=matches,drawing_order=order,
    planning_rule='predicted contours first, then predicted cun groups; not a trained next-action policy',unknown_regions_not_scored_as_false_positive=True,artist_stroke_truth_complete=False)
  allresults.append(result);print('REAL_EVAL',r['id'],len(p),matches,flush=True)
 # Blank paper and removed-context diagnostics are not new training examples.
 blank=np.full_like(real['image'],np.quantile(real['image'],.94));bp,bm=infer(net,blank,threshold);paint(blank,bp,bm,'blank_control')
 localized=real['image'].copy();keep=np.zeros((SIZE,SIZE),bool);q=real['strokes'][4]['curve'][POINTS//2].astype(int);x,y=q;keep[max(0,y-16):min(SIZE,y+16),max(0,x-16):min(SIZE,x+16)]=True;localized[~keep]=blank[~keep];lp,lm=infer(net,localized,threshold);paint(localized,lp,lm,'local_context_ablation')
 write(ROOT/'evaluation_report.json',dict(threshold=threshold,threshold_calibrated_on='procedural development, no real test tuning',procedural_holdout=held,real=allresults,blank_predictions=len(bp),local_context_ablation_predictions=len(lp),
  original_real_training_works=1,different_real_development_works=2,verified_author_blind_test_works=0,whole_image_object_detection_trained=False,decoder_nms_kernel=3,decoder_projection='differentiable cubic spline subspace, continued joint training',
  named_cunfa_recognition_trained=False,original_painter_order_recovered=False,real_ink_instance_truth_complete=False,posthoc_geometry_stitching_used=False,robot_connected=False,adopted=False))
 old=json.loads((LAB/'pima_single_stroke_20261109'/'verification.json').read_text('utf8'));finger=[]
 for r in old['approved_plans_fingerprints']:
  same=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'];assert same;finger.append(dict(**r,unchanged=same))
 p,m=infer(net,real['image'],threshold);assert len(p)==allresults[0]['predicted_count'];assert all(np.allclose(np.array(a['curve']),np.array(b['curve']),atol=1e-4) for a,b in zip(p,allresults[0]['predictions']))
 write(ROOT/'verification.json',dict(passed=True,saved_checkpoint_replayed=True,input_shape=[3,SIZE,SIZE],learned_anchor_not_skeleton_endpoint=True,complete_curve_prediction_not_geometric_stitch=True,all_joint_heads_trained=True,
  approved_plans_unchanged=True,approved_plans_fingerprints=finger,checkpoint_sha256=hashlib.sha256((ROOT/'whole_context_step1250.pt').read_bytes()).hexdigest(),robot_connected=False))
 print('EVALUATION_FINISHED',threshold,flush=True)
if __name__=='__main__':main()
