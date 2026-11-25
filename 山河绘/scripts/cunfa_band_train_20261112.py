# -*- coding: utf-8 -*-
"""Matched continuation control vs added conservative near-boundary supervision."""
import sys,json,hashlib,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,torch
from gongbi_training.cunfa_supervision import partial_role_loss,role_targets_checked
from cunfa_roles_v2_train_20261112 import Net,features,conf,metrics
ROOT=LAB/'cunfa_band_v3_20261112';OLD=LAB/'cunfa_roles_v2_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def ink_metrics(t,p):
 positive=np.isin(t,[1,2]);negative=t==0;tp=int((positive&p).sum());fp=int((negative&p).sum());return dict(ink_reference_pixels=int(positive.sum()),near_paper_pixels=int(negative.sum()),ink_core_recall=tp/int(positive.sum()) if positive.any() else None,near_paper_false_ink=fp/int(negative.sum()) if negative.any() else None,ink_missed=int(positive.sum())-tp,paper_as_ink=fp)
def widths(mask,guides):
 out=[]
 for g in guides:
  q=np.array(g['points'],'float32');s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))];positions=np.arange(4,s[-1]-4,2.);values=[];centres=[];offsets=np.arange(-12,12.01,.5,dtype='float32')
  for p in positions:
   idx=min(np.searchsorted(s,p,side='right')-1,len(q)-2);v=q[idx+1]-q[idx];v/=np.linalg.norm(v);normal=np.array([-v[1],v[0]]);center=q[idx]+v*(p-s[idx]);samples=center+offsets[:,None]*normal;line=cv2.remap(mask.astype('float32'),samples[:,0,None],samples[:,1,None],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).ravel()>.5;k=len(offsets)//2;centres.append(bool(line[k]));left=right=k
   if line[k]:
    while left>0 and line[left-1]:left-=1
    while right+1<len(line) and line[right+1]:right+=1
    values.append(float((right-left+1)*.5))
   else:values.append(0.)
  a=np.array(values);inside=np.array(centres);out.append(dict(id=g['id'],samples=len(a),centre_retained=float(inside.mean()) if len(a) else None,median_width_px=float(np.median(a[inside])) if inside.any() else None,over_outer_width_fraction=float((a>g['outer']*2+1).mean()) if len(a) else None,reference_outer_width_px=g['outer']*2,not_true_physical_width=True,guide_used_only_for_evaluation=True))
 return out
def main():
 assert not (ROOT/'results.json').exists(),'Create a new experiment for further tuning.'
 ds=json.loads((ROOT/'dataset_manifest.json').read_text('utf8'));records=ds['records'];assert all(sha(r['path'])==r['sha256'] and role_targets_checked(r) for r in records);data={r['id']:dict(np.load(r['path'])) for r in records};train=[r for r in records if r['split']=='train'];held=[r for r in records if r['split']!='train'];assert not {r['source_family'] for r in train}&{r['source_family'] for r in held};seed=121204;steps=240
 protocol=dict(seed=seed,steps_per_arm=steps,arms=['core_replay_control','boundary_supervision'],initial_checkpoint_sha256=sha(OLD/'flat_roles.pt'),dataset_sha256=sha(ROOT/'dataset_manifest.json'),script_sha256=sha(__file__),architecture='Identical v2 flat three-class network and input; no new foreground head',loss='Original class-balanced sparse core CE; boundary arm adds 0.5 x balanced reviewed band CE',model_selection='Last fixed step; no diagnostic tuning or selected checkpoint',optimizer='AdamW lr0.0003 weight_decay0.0001',sampling='Alternate training source works and observations',augmentation='same image/core/band affine +/-12deg,0.55..1.55 scale,+/-8 translation; same seed per arm',diagnostic_work_seen_before=True,diagnostic_artist_unverified=True,manual_intervals_not_exact_masks=True,endcaps_and_occlusion_ignored=True,no_inference_guide_or_threshold_mask=True,robot_connected=False)
 write('protocol.json',protocol);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);initial=torch.load(OLD/'flat_roles.pt',map_location='cuda',weights_only=False)['model'];groups=[[r for r in train if r['source_family']==f] for f in sorted({r['source_family'] for r in train})];cw=np.zeros(3);bw=np.zeros(3)
 for r in train:
  for name,counts in [('core_targets',cw),('band_targets',bw)]:
   t=data[r['id']][name];counts+=np.bincount(t[t!=255],minlength=3)
 weights=[]
 for c in [cw,bw]:
  v=torch.tensor(1/np.maximum(c,1),dtype=torch.float32,device='cuda');v/=v.mean();weights.append(v)
 models={};curves={};started=time.time()
 for arm in protocol['arms']:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);net=Net(False).cuda();net.load_state_dict(initial);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001);losses=[]
  for step in range(steps):
   group=groups[step%len(groups)];r=group[(step//len(groups))%len(group)];z=data[r['id']];image=z['image'];h,w=image.shape[:2];m=cv2.getRotationMatrix2D((w/2,h/2),random.uniform(-12,12),random.uniform(.55,1.55));m[:,2]+=np.array([random.uniform(-8,8),random.uniform(-8,8)]);image=cv2.warpAffine(image,m,(w,h),borderValue=(255,255,255));targets=[cv2.warpAffine(z[n],m,(w,h),flags=cv2.INTER_NEAREST,borderValue=255) for n in ['core_targets','band_targets']]
   if random.random()<.5:image=image[:,::-1].copy();targets=[t[:,::-1].copy() for t in targets]
   image=np.clip(image.astype('float32')*random.uniform(.9,1.1)+random.uniform(-8,8),0,255).astype('uint8');logits=net(torch.from_numpy(features(image))[None].cuda())[1];ts=[torch.from_numpy(t.astype('int64'))[None].cuda() for t in targets];loss=partial_role_loss(logits,ts[0],weights[0][ts[0].clamp(0,2)]*ts[0].ne(255))
   if arm=='boundary_supervision':loss=loss+.5*partial_role_loss(logits,ts[1],weights[1][ts[1].clamp(0,2)]*ts[1].ne(255))
   opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
   if (step+1)%80==0:print(arm,step+1,round(losses[-1],4),flush=True)
  net.eval();models[arm]=net;curves[arm]=losses;torch.save(dict(model=net.state_dict(),protocol=protocol,arm=arm),ROOT/(arm+'.pt'))
 hashes={a:sha(ROOT/(a+'.pt')) for a in models};write('checkpoint_freeze.json',hashes);old=Net(False).cuda();old.load_state_dict(initial);old.eval();models['v2_frozen']=old;(ROOT/'predictions').mkdir(exist_ok=True);results=[]
 for r in records:
  z=data[r['id']];im=z['image'];scores={}
  with torch.inference_mode():
   x=torch.from_numpy(features(im))[None].cuda()
   for arm,net in models.items():
    probability=net.probabilities(x)[0].cpu().numpy();p=probability.argmax(0).astype('uint8');np.savez_compressed(ROOT/'predictions'/(r['id']+'_'+arm+'.npz'),prediction=p,probability=probability);scores[arm]=dict(core_roles=metrics(conf(z['core_targets'],p)),band_ink=ink_metrics(z['band_targets'],p!=0),widths=widths(p!=0,r['guides']))
  # Contrast-only ink detector is a scope control, not a role model.
  threshold=features(im)[1]*32>3;scores['contrast_threshold_only']=dict(band_ink=ink_metrics(z['band_targets'],threshold),widths=widths(threshold,r['guides']),threshold_fixed=3,no_semantic_role_output=True)
  results.append(dict(id=r['id'],split=r['split'],scores=scores))
 heldresults=[r for r in results if r['split']!='train'];assert len(heldresults)==1
 write('results.json',dict(protocol=protocol,checkpoint_hashes=hashes,records=results,diagnostic=heldresults[0],loss_curves=curves,duration_seconds=time.time()-started,adopted=False,robot_connected=False));print(json.dumps(heldresults[0],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
