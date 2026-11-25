# -*- coding: utf-8 -*-
"""Focused decoder repair; preserves old anchors/curves/roles and baselines."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.rock_context_data import *
from gongbi_training.rock_context_model import RockContextNet
from gongbi_training.rock_instance_decoder import *
import numpy as np,torch,cv2
OUT=LAB/'ink_ownership_fix_20261110'
def load_decoder():
 high=(OUT/'ink_decoder_hr.pt').exists();net=(InkOwnershipDecoderHR() if high else InkOwnershipDecoder()).cuda();net.load_state_dict(torch.load(OUT/('ink_decoder_hr.pt' if high else 'ink_decoder.pt'),map_location='cpu',weights_only=False)['model']);net.eval();return net

def old_model():
 net=RockContextNet(project_curve=True).cuda();ck=torch.load(ROOT/'whole_context_step1250.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);net.eval();net.requires_grad_(False);return net

def make_batch(base,r,seed,max_positive=6,include_negatives=True,full_resolution=False):
 rng=np.random.default_rng(seed);tt=targets(r);image=torch.tensor(r['image'].transpose(2,0,1)[None],dtype=torch.float32,device='cuda')
 with torch.no_grad():old=base(image)
 dim=SIZE if full_resolution else GRID;lookup={s['id']:s for s in r['strokes']};anchors=[];truth=[];valid=[];present=[];allink=np.maximum.reduce([cv2.resize(s['mask']*s['mask_valid'],(dim,dim),interpolation=cv2.INTER_NEAREST) for s in r['strokes']]);allink_grid=cv2.resize(allink,(GRID,GRID),interpolation=cv2.INTER_NEAREST);chosen=rng.choice(len(tt['anchors']),min(max_positive,len(tt['anchors'])),replace=False)
 for k in chosen:
  a=tt['anchors'][int(k)];x,y=a['x'],a['y'];dx,dy=rng.integers(-1,2,2);nx,ny=int(np.clip(x+dx,0,GRID-1)),int(np.clip(y+dy,0,GRID-1))
  if any((z['x']-nx)**2+(z['y']-ny)**2<dx*dx+dy*dy for z in tt['anchors'] if z is not a):nx,ny=x,y
  mask=lookup[a['id']]['mask'] if full_resolution else a['mask'];v=lookup[a['id']]['mask_valid'].copy() if full_resolution else a['mask_valid'].copy()
  if full_resolution and not r['complete']:v=np.maximum(v,r['paper'])
  anchors.append([0,nx,ny]);truth.append(mask)
  if not r['complete']:
   # Confirmed positive pixels from other labelled instances are negatives
   # only where this instance has no positive assignment. Unmarked stays unknown.
   v=np.maximum(v,((allink>.5)&(mask<=.5)).astype(np.float32))
  valid.append(v);present.append(1.)
 if r['complete'] and include_negatives:
  # Fully labelled procedural paper queries teach empty-query rejection.
  for _ in range(2):
   for trial in range(100):
    x,y=rng.integers(6,GRID-6,2)
    if all((a['x']-x)**2+(a['y']-y)**2>64 for a in tt['anchors']) and allink_grid[y,x]<.1:break
   anchors.append([0,int(x),int(y)]);truth.append(np.zeros((dim,dim),np.float32));valid.append(np.ones((dim,dim),np.float32));present.append(0.)
 aa=torch.tensor(anchors,device='cuda');features,dist,q=condition_features(old,image,aa)
 to=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device='cuda')
 return features.detach(),to(truth),to(valid),to([allink]*len(anchors)),to(present),dist,old,image,tt

def main():
 OUT.mkdir(exist_ok=True);torch.manual_seed(1110);torch.set_num_threads(4);base=old_model();net=InkOwnershipDecoder().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.0001);real=real_example();history=[];start=time.time();procedural=real_views=0
 for step in range(1,701):
  # All new procedural seeds are outside previous train/development ranges.
  use_real=step>400 and step%5==0;r=augment(real,3111000+step) if use_real else synthetic(311100000+step)
  procedural+=int(not use_real);real_views+=int(use_real);features,truth,valid,known,present,*_=make_batch(base,r,step);net.train();logits,quality=net(features);loss,parts=ownership_loss(logits,truth,valid,known,quality,present);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
  if step%100==0:
   history.append(dict(step=step,parts={k:float(v.detach()) for k,v in parts.items()}));write(OUT/'progress.json',dict(history=history,seconds=time.time()-start));print('MASK_TRAIN',history[-1],flush=True)
 torch.save(dict(model={k:v.detach().cpu() for k,v in net.state_dict().items()},steps=700,base_checkpoint=str(ROOT/'whole_context_step1250.pt'),base_frozen=True,procedural_seeds=[311100001,311100700],real_source_works=1),OUT/'ink_decoder.pt')
 write(OUT/'training_report.json',dict(steps=700,procedural_scenes=procedural,real_augmented_views=real_views,original_real_modules=1,history=history,seconds=time.time()-start,
  base_encoder_curves_anchors_roles_frozen=True,reference_curve_not_input=True,predicted_curve_is_conditioning_not_rendered_output=True,
  unknown_real_ignored=True,other_confirmed_ink_as_nonownership_negatives=True,objective='balanced instance pixels + masked Dice + empty query rejection',
  full_real_instance_truth=False,named_cunfa_model=False,robot_connected=False))
 print('MASK_TRAIN_FINISHED',time.time()-start,flush=True)
if __name__=='__main__':main()
