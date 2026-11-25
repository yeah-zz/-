# -*- coding: utf-8 -*-
"""Freeze semantic conditional probabilities; learn only foreground residual."""
import sys,json,hashlib,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from cunfa_roles_v2_train_20261112 import Net,features,conf,metrics
from cunfa_band_train_20261112 import ink_metrics,widths
ROOT=LAB/'cunfa_band_checked_20261112';REF=LAB/'cunfa_band_checked_20261112';OLD=LAB/'cunfa_roles_v2_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
class Refiner(nn.Module):
 def __init__(self):
  super().__init__();self.layers=nn.Sequential(nn.Conv2d(4,12,3,padding=1),nn.SiLU(),nn.Conv2d(12,12,3,padding=1),nn.SiLU(),nn.Conv2d(12,1,1));nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
 def forward(self,x,teacher):
  fg=teacher[:,1:].sum(1,keepdim=True).clamp(1e-5,1-1e-5);logit=torch.logit(fg);residual=self.layers(torch.cat([x,logit.clamp(-8,8)/8],1));f=torch.sigmoid(logit+residual);role=teacher[:,1:]/teacher[:,1:].sum(1,keepdim=True).clamp_min(1e-8);return torch.cat([1-f,f*role],1),residual
def main():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'results.json').exists(),'Frozen run; do not retune on diagnostic.'
 ds=json.loads((REF/'dataset_manifest.json').read_text('utf8'));records=ds['records'];train=[r for r in records if r['split']=='train'];data={r['id']:dict(np.load(r['path'])) for r in records};assert all(sha(r['path'])==r['sha256'] for r in records);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);seed=121205;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 teacher=Net(False).cuda();teacher.load_state_dict(torch.load(OLD/'flat_roles.pt',map_location='cuda',weights_only=False)['model']);teacher.eval();teacher.requires_grad_(False);refiner=Refiner().cuda();opt=torch.optim.AdamW(refiner.parameters(),lr=.001,weight_decay=.0001);steps=240
 protocol=dict(seed=seed,steps=steps,teacher_sha256=sha(OLD/'flat_roles.pt'),reference_dataset_sha256=sha(REF/'dataset_manifest.json'),invalidated_prior_run_sha256=sha(LAB/'cunfa_band_refiner_20261112/results.json'),reference_audit_sha256=sha(REF/'reference_coordinate_audit.json'),script_sha256=sha(__file__),reason='Reference coordinate errors were corrected and audited first; freeze semantic probabilities and refine foreground only. Repeated diagnostic, not blind validation.',architecture='Three local conv layers 4->12->12->1, zero-initialized residual, frozen v2 role teacher',input='Static local contrast channels and frozen teacher foreground logit; no guides or masks at inference',loss='Known core foreground BCE + reviewed band foreground BCE +0.02 residual squared on known old cores',source_sampling='Alternate two source groups and observations',fixed_budget_and_last_step=True,posthoc_redesign=True,diagnostic_repeated_not_blind=True,no_hyperparameter_search=True,exact_mask_or_endpoint_gold=False,robot_connected=False)
 write('protocol.json',protocol);groups=[[r for r in train if r['source_family']==f] for f in sorted({r['source_family'] for r in train})];losses=[];started=time.time()
 def fg_loss(prob,t):
  valid=t!=255;target=(t>0)&valid;loss=F.binary_cross_entropy(prob.clamp(1e-6,1-1e-6),target.float(),reduction='none');parts=[]
  for mask in [valid&~target,target]:
   if mask.any():parts.append(loss[mask].mean())
  return sum(parts)/len(parts) if parts else prob.sum()*0
 for step in range(steps):
  group=groups[step%len(groups)];r=group[(step//len(groups))%len(group)];z=data[r['id']];image=z['image'];h,w=image.shape[:2];m=cv2.getRotationMatrix2D((w/2,h/2),random.uniform(-12,12),random.uniform(.55,1.55));m[:,2]+=np.array([random.uniform(-8,8),random.uniform(-8,8)]);image=cv2.warpAffine(image,m,(w,h),borderValue=(255,255,255));ts=[cv2.warpAffine(z[n],m,(w,h),flags=cv2.INTER_NEAREST,borderValue=255) for n in ['core_targets','band_targets']]
  if random.random()<.5:image=image[:,::-1].copy();ts=[t[:,::-1].copy() for t in ts]
  image=np.clip(image.astype('float32')*random.uniform(.9,1.1)+random.uniform(-8,8),0,255).astype('uint8');x=torch.from_numpy(features(image))[None].cuda();targets=[torch.from_numpy(t.astype('int64'))[None].cuda() for t in ts]
  with torch.no_grad():old=teacher.probabilities(x)
  prob,residual=refiner(x,old);fg=prob[:,1:].sum(1);known=targets[0]!=255;loss=fg_loss(fg,targets[0])+fg_loss(fg,targets[1])+.02*(residual[:,0][known]**2).mean();opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
  if (step+1)%80==0:print('FROZEN_ROLE_REFINER',step+1,round(losses[-1],4),flush=True)
 checkpoint=ROOT/'foreground_refiner.pt';torch.save(dict(model=refiner.state_dict(),protocol=protocol),checkpoint);write('checkpoint_freeze.json',dict(refiner_sha256=sha(checkpoint),teacher_sha256=sha(OLD/'flat_roles.pt')));refiner.eval();(ROOT/'predictions').mkdir(exist_ok=True);results=[]
 for r in records:
  z=data[r['id']];x=torch.from_numpy(features(z['image']))[None].cuda()
  with torch.inference_mode():old=teacher.probabilities(x);new,_=refiner(x,old)
  # Preserve conditional structure/cun probability, not merely model parameters.
  a=(old[:,1:]/old[:,1:].sum(1,keepdim=True).clamp_min(1e-8));b=new[:,1:]/new[:,1:].sum(1,keepdim=True).clamp_min(1e-8);semantic_error=float((a-b).abs().max());assert semantic_error<1e-5
  scores={}
  for arm,prob in [('v2_frozen',old[0].cpu().numpy()),('frozen_role_refiner',new[0].cpu().numpy())]:
   p=prob.argmax(0).astype('uint8');np.savez_compressed(ROOT/'predictions'/(r['id']+'_'+arm+'.npz'),prediction=p,probability=prob);scores[arm]=dict(core_roles=metrics(conf(z['core_targets'],p)),band_ink=ink_metrics(z['band_targets'],p!=0),widths=widths(p!=0,r['guides']))
  results.append(dict(id=r['id'],split=r['split'],semantic_conditional_max_error=semantic_error,scores=scores))
 write('results.json',dict(protocol=protocol,checkpoint_sha256=sha(checkpoint),records=results,diagnostic=next(r for r in results if r['split']!='train'),loss_curve=losses,duration_seconds=time.time()-started,adopted=False,robot_connected=False));print(json.dumps(results[-1],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
