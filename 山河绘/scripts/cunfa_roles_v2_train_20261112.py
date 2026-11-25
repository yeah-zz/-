# -*- coding: utf-8 -*-
"""Fixed two-arm experiment: flat roles vs explicit ink and conditional roles."""
import sys,json,random,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from gongbi_training.cunfa_supervision import partial_role_loss
from cunfa_temporal_probe_20261112 import Block,RoleNet,features as old_features
ROOT=LAB/'cunfa_roles_v2_20261112';OLD=LAB/'cunfa_temporal_roles_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def features(image):
 g=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype('float32')
 # Local background normalization removes global paper brightness as an input.
 return np.stack([np.clip((cv2.GaussianBlur(g,(0,0),s)-g)/32,-1,1) for s in [2,6,18]])
class Net(nn.Module):
 def __init__(self,split):
  super().__init__();self.split=split;self.a=Block(3,12);self.b=Block(12,24);self.c=Block(24,48);self.d=Block(48,96);self.u=Block(144,48);self.v=Block(72,24);self.w=Block(36,12);self.ink=nn.Conv2d(12,1,1);self.roles=nn.Conv2d(12,2 if split else 3,1)
 def forward(self,x):
  a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(F.avg_pool2d(c,2));up=lambda x,y:F.interpolate(x,size=y.shape[-2:],mode='bilinear',align_corners=False)
  u=self.u(torch.cat([up(d,c),c],1));v=self.v(torch.cat([up(u,b),b],1));w=self.w(torch.cat([up(v,a),a],1));return self.ink(w),self.roles(w)
 def probabilities(self,x):
  ink,role=self(x)
  if not self.split:return role.softmax(1)
  fg=ink.sigmoid();conditional=role.softmax(1);return torch.cat([1-fg,fg*conditional],1)
def conf(t,p):
 k=t!=255;return np.bincount((t[k]*3+p[k]).ravel(),minlength=9).reshape(3,3)
def metrics(c):
 sup=c.sum(1);rec=[float(c[k,k]/sup[k]) if sup[k] else None for k in range(3)];i=c[1:].sum();return dict(confusion=c.tolist(),reference_support=sup.tolist(),recall=rec,balanced_role_recall=float(np.mean(rec[1:])) if all(x is not None for x in rec[1:]) else None,paper_false_ink=float(c[0,1:].sum()/sup[0]) if sup[0] else None,ink_recall=float(c[1:,1:].sum()/i) if i else None,complete_stroke_or_style_accuracy=False)
def main():
 assert torch.cuda.is_available();assert not (ROOT/'results.json').exists(),'No reruns after test evaluation.'
 ds=json.loads((ROOT/'dataset_manifest.json').read_text('utf8'));rows=ds['records'];train=[r for r in rows if r['split']=='train'];held=[r for r in rows if r['split']!='train'];data={r['id']:dict(np.load(r['path'])) for r in rows};assert not {r['source_family'] for r in train}&{r['source_family'] for r in held};assert all(sha(r['path'])==r['sha256'] for r in rows)
 seed=121203;steps=360;protocol=dict(seed=seed,steps_per_arm=steps,arms=['flat_roles','ink_then_role'],architecture='Same small U-Net; explicit ink sigmoid and conditional structure/cun softmax in second arm',input='Three local contrast channels sigma2/6/18, static context crop, no before/delta/guide/phase',optimizer='AdamW lr0.001 weight_decay0.0001',source_sampling='Alternate two source works; alternate each work observation',augmentations='same affine image/target; angle +/-12, scale0.55..1.55, translation +/-8; horizontal flip and brightness perturbation',model_selection='Final fixed step, no heldout checkpoint or threshold selection',foreground_rule='Learned head, no threshold mask or reference mask applied after inference',loss='Balanced known foreground/paper + conditional role cross entropy; 255 ignored',old_liu_test_now_development=True,heldout_work_seen_before=True,heldout_painter_identity_verified=False,expert_gold=False,style_or_stroke_generation_trained=False,dataset_sha256=sha(ROOT/'dataset_manifest.json'),script_sha256=sha(__file__),old_checkpoint_sha256=sha(OLD/'role_probe.pt'),robot_connected=False)
 write('protocol.json',protocol);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);groups=[[r for r in train if r['source_family']==f] for f in sorted({r['source_family'] for r in train})];weights=np.zeros(3)
 for r in train:weights+=np.bincount(data[r['id']]['targets'][data[r['id']]['targets']!=255],minlength=3)
 counts=torch.tensor(weights,dtype=torch.float32,device='cuda');rolew=1/counts;rolew/=rolew.mean();fgw=1/torch.stack([counts[0],counts[1:].sum()]);fgw/=fgw.mean();condw=1/counts[1:];condw/=condw.mean();models={};curves={};started=time.time()
 for arm in protocol['arms']:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Net(arm=='ink_then_role').cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);losses=[]
  for step in range(steps):
   group=groups[step%len(groups)];r=group[(step//len(groups))%len(group)];z=data[r['id']];im=z['image'];target=z['targets'];h,w=target.shape;m=cv2.getRotationMatrix2D((w/2,h/2),random.uniform(-12,12),random.uniform(.55,1.55));m[:,2]+=np.array([random.uniform(-8,8),random.uniform(-8,8)]);im=cv2.warpAffine(im,m,(w,h),borderValue=(255,255,255));target=cv2.warpAffine(target,m,(w,h),flags=cv2.INTER_NEAREST,borderValue=255)
   if random.random()<.5:im=im[:,::-1].copy();target=target[:,::-1].copy()
   im=np.clip(im.astype('float32')*random.uniform(.90,1.10)+random.uniform(-8,8),0,255).astype('uint8');x=torch.from_numpy(features(im))[None].cuda();t=torch.from_numpy(target.astype('int64'))[None].cuda();ink,role=model(x)
   if arm=='flat_roles':loss=partial_role_loss(role,t,rolew[t.clamp(0,2)]*t.ne(255))
   else:
    known=t!=255;fg=t.gt(0)&known;ft=torch.where(known,fg.long(),255);inklog=torch.cat([torch.zeros_like(ink),ink],1);fl=partial_role_loss(inklog,ft,fgw[ft.clamp(0,1)]*known);ct=torch.where(fg,t-1,255);cl=partial_role_loss(role,ct,condw[ct.clamp(0,1)]*fg);loss=fl+cl
   opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
   if (step+1)%90==0:print(arm,step+1,round(losses[-1],4),flush=True)
  path=ROOT/(arm+'.pt');torch.save(dict(model=model.state_dict(),protocol=protocol,arm=arm),path);model.eval();models[arm]=model;curves[arm]=losses
 # Freeze both trained checkpoints before opening the heldout target for scores.
 hashes={arm:sha(ROOT/(arm+'.pt')) for arm in models};write('checkpoint_freeze.json',dict(checkpoints=hashes,evaluation_started=False))
 old=RoleNet().cuda();old.load_state_dict(torch.load(OLD/'role_probe.pt',map_location='cuda',weights_only=False)['model']);old.eval();(ROOT/'predictions').mkdir(exist_ok=True);records=[]
 for r in rows:
  z=data[r['id']];im=z['image'];t=z['targets'];preds={}
  with torch.inference_mode():
   x=torch.from_numpy(features(im))[None].cuda()
   for arm,model in models.items():preds[arm]=model.probabilities(x)[0].cpu().numpy()
   preds['previous_frozen']=old(torch.from_numpy(old_features(im))[None].cuda()).softmax(1)[0].cpu().numpy()
  scored={}
  for arm,prob in preds.items():
   p=prob.argmax(0).astype('uint8');np.savez_compressed(ROOT/'predictions'/(r['id']+'_'+arm+'.npz'),prediction=p,probability=prob);scored[arm]=metrics(conf(t,p))
  # Same learned foreground for constant-role controls; paper is not deliberately
  # labelled ink to make those controls look worse.
  foreground=preds['ink_then_role'][1:].sum(0)>.5
  for k in [1,2]:scored['same_foreground_role_'+str(k)]=metrics(conf(t,np.where(foreground,k,0).astype('uint8')))
  records.append(dict(id=r['id'],split=r['split'],scores=scored))
 totals={}
 for split in ['train','heldout_source_partial_reference']:
  relevant=[r for r in records if r['split']==split];totals[split]={arm:metrics(sum(np.array(r['scores'][arm]['confusion']) for r in relevant)) for arm in records[0]['scores']}
 write('results.json',dict(protocol=protocol,checkpoint_hashes=hashes,records=records,aggregates=totals,loss_curves=curves,duration_seconds=time.time()-started,adopted=False,robot_connected=False));print(json.dumps(totals['heldout_source_partial_reference'],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
