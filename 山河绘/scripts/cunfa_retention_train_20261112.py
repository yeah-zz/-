# -*- coding: utf-8 -*-
"""Paired experiment: known-ink retention constraint without diagnostic labels."""
import sys,json,hashlib,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,torch
from torch.nn import functional as F
from cunfa_roles_v2_train_20261112 import Net,features,conf,metrics
from cunfa_band_checked_train_20261112 import Refiner
from cunfa_band_train_20261112 import ink_metrics,widths
ROOT=LAB/'cunfa_retention_v4_20261112';PREV=LAB/'cunfa_band_checked_20261112';TEACHER=LAB/'cunfa_roles_v2_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def fg_loss(prob,target):
 valid=target!=255;ink=target.gt(0)&valid;loss=F.binary_cross_entropy(prob.clamp(1e-6,1-1e-6),ink.float(),reduction='none');pieces=[loss[m].mean() for m in [valid&~ink,ink] if m.any()];return sum(pieces)/len(pieces) if pieces else prob.sum()*0
def retain_loss(prob,teacher_fg,target):
 known_ink=(target!=255)&target.gt(0)
 # Teacher supplies a numeric margin only where reviewed references already
 # assert ink. No teacher prediction is converted into a new target label.
 return (teacher_fg[known_ink]-prob[known_ink]).relu().mean() if known_ink.any() else prob.sum()*0
def main():
 assert not (ROOT/'results.json').exists(),'No rerun on reviewed diagnostic.';ROOT.mkdir(exist_ok=True);ds=json.loads((PREV/'dataset_manifest.json').read_text('utf8'));records=ds['records'];train=[r for r in records if r['split']=='train'];data={r['id']:dict(np.load(r['path'])) for r in records};assert all(sha(r['path'])==r['sha256'] for r in records);assert not {r['source_family'] for r in train}&{r['source_family'] for r in records if r['split']!='train'}
 seed=121206;steps=240;protocol=dict(seed=seed,steps_per_arm=steps,arms=['matched_continuation','known_ink_retention'],teacher_sha256=sha(TEACHER/'flat_roles.pt'),initial_refiner_sha256=sha(PREV/'foreground_refiner.pt'),reference_dataset_sha256=sha(PREV/'dataset_manifest.json'),audit_sha256=sha(ROOT/'retention_audit.json'),script_sha256=sha(__file__),architecture='Frozen semantic teacher and same foreground residual network; no new inference heuristics',common_loss='core foreground BCE + interval foreground BCE +0.02 residual squared on known cores',only_loss_difference='Retention arm adds 4.0*mean(relu(teacher_fg-refined_fg)) on reviewed known ink cores',retention_weight_fixed_before_new_evaluation=4.,common_augmentation='same affine +/-12deg,scale0.55..1.55,translation+/-8,flip,brightness; additional local contrast factor0.65..1.10 around sigma12 background',source_sampling='Alternate two source works and observations',optimizer='AdamW lr0.0003 weight_decay0.0001',fixed_budget_final_step=True,diagnostic_repeated_not_blind=True,no_new_expert_stroke_or_style_claim=True,no_diagnostic_gradients_or_parameter_selection=True,robot_connected=False);write('protocol.json',protocol)
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);teacher=Net(False).cuda();teacher.load_state_dict(torch.load(TEACHER/'flat_roles.pt',map_location='cuda',weights_only=False)['model']);teacher.eval().requires_grad_(False);initial=torch.load(PREV/'foreground_refiner.pt',map_location='cuda',weights_only=False)['model'];groups=[[r for r in train if r['source_family']==f] for f in sorted({r['source_family'] for r in train})];models={};curves={};started=time.time()
 for arm in protocol['arms']:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);net=Refiner().cuda();net.load_state_dict(initial);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001);losses=[]
  for step in range(steps):
   group=groups[step%len(groups)];r=group[(step//len(groups))%len(group)];z=data[r['id']];image=z['image'];h,w=image.shape[:2];m=cv2.getRotationMatrix2D((w/2,h/2),random.uniform(-12,12),random.uniform(.55,1.55));m[:,2]+=np.array([random.uniform(-8,8),random.uniform(-8,8)]);image=cv2.warpAffine(image,m,(w,h),borderValue=(255,255,255));targets=[cv2.warpAffine(z[n],m,(w,h),flags=cv2.INTER_NEAREST,borderValue=255) for n in ['core_targets','band_targets']]
   if random.random()<.5:image=image[:,::-1].copy();targets=[t[:,::-1].copy() for t in targets]
   f=image.astype('float32');background=cv2.GaussianBlur(f,(0,0),12);f=background+random.uniform(.65,1.1)*(f-background);image=np.clip(f*random.uniform(.9,1.1)+random.uniform(-8,8),0,255).astype('uint8');x=torch.from_numpy(features(image))[None].cuda();ts=[torch.from_numpy(t.astype('int64'))[None].cuda() for t in targets]
   with torch.no_grad():old=teacher.probabilities(x)
   probability,residual=net(x,old);fg=probability[:,1:].sum(1);known=ts[0]!=255;loss=fg_loss(fg,ts[0])+fg_loss(fg,ts[1])+.02*(residual[:,0][known]**2).mean()
   if arm=='known_ink_retention':loss=loss+4*retain_loss(fg,old[:,1:].sum(1).detach(),ts[0])
   opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
   if (step+1)%80==0:print(arm,step+1,round(losses[-1],4),flush=True)
  net.eval();models[arm]=net;curves[arm]=losses;torch.save(dict(model=net.state_dict(),protocol=protocol,arm=arm),ROOT/(arm+'.pt'))
 hashes={k:sha(ROOT/(k+'.pt')) for k in models};write('checkpoint_freeze.json',hashes);previous=Refiner().cuda();previous.load_state_dict(initial);previous.eval();models['previous_refiner']=previous;(ROOT/'predictions').mkdir(exist_ok=True);results=[]
 for r in records:
  z=data[r['id']];x=torch.from_numpy(features(z['image']))[None].cuda();scores={};semantic_errors={}
  with torch.inference_mode():
   old=teacher.probabilities(x)
   probs={'v2_frozen':old}
   for name,net in models.items():probs[name]=net(x,old)[0]
  original_class=old.argmax(1)[0].cpu().numpy();prior_class=probs['previous_refiner'].argmax(1)[0].cpu().numpy();t=z['core_targets'];suppressed=(t!=255)&(t>0)&(original_class>0)&(prior_class==0)
  for name,prob in probs.items():
   a=old[:,1:]/old[:,1:].sum(1,keepdim=True).clamp_min(1e-8);b=prob[:,1:]/prob[:,1:].sum(1,keepdim=True).clamp_min(1e-8);semantic_errors[name]=float((a-b).abs().max());assert semantic_errors[name]<1e-5;arr=prob[0].cpu().numpy();p=arr.argmax(0).astype('uint8');np.savez_compressed(ROOT/'predictions'/(r['id']+'_'+name+'.npz'),prediction=p,probability=arr)
   scores[name]=dict(core_roles=metrics(conf(t,p)),band_ink=ink_metrics(z['band_targets'],p>0),widths=widths(p>0,r['guides']),previously_suppressed_ink_pixels=int(suppressed.sum()),recovered_previous_suppression=int((suppressed&(p>0)).sum()),newly_suppressed_vs_previous=int(((t!=255)&(t>0)&(prior_class>0)&(p==0)).sum()))
  results.append(dict(id=r['id'],split=r['split'],scores=scores,semantic_conditional_max_error=semantic_errors))
 write('results.json',dict(protocol=protocol,checkpoint_hashes=hashes,records=results,diagnostic=next(r for r in results if r['split']!='train'),loss_curves=curves,duration_seconds=time.time()-started,adopted=False,robot_connected=False));print(json.dumps(results[-1],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
