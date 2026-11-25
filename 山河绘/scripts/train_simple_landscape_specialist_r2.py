# -*- coding: utf-8 -*-
"""GPU specialist for declared synthetic stroke pairs; independent checkpoint."""
from simple_landscape_pairs_r2 import ROOT,ROLES
from rock_v15_common import LAB,read,jsave,digest
import ink_center_v16 as prior
import numpy as np,cv2,torch,json,time,math,hashlib,functools
from torch import nn
from torch.nn import functional as F
from PIL import Image
INIT=LAB/'simple_landscape_specialist_20261029'/'specialist.pt'

class Specialist(prior.RibbonNet):
    def __init__(self):
        super().__init__();self.role_head=nn.Conv2d(16,5,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1));e=self.e(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1));return torch.cat((self.head(e),self.role_head(e)),1)

def initialize():
    model=Specialist().cuda();weights=torch.load(INIT,map_location='cpu',weights_only=False)['model'];own=model.state_dict();selected={k:v for k,v in weights.items() if k in own}
    missing=model.load_state_dict(selected,strict=False)
    assert not missing.missing_keys and not missing.unexpected_keys
    return model

@functools.lru_cache(maxsize=32)
def get_record(image,data):
    rgb=np.asarray(Image.open(image).convert('RGB'));z=np.load(data);return rgb,z['target'].astype(np.float32),z['role'].astype(np.int64)

def crop(r,rng,augment=True):
    rgb,t,role=get_record(r['image'],r['data']);h,w=rgb.shape[:2]
    yy,xx=np.where(t[0]>.3)
    if len(xx) and rng.random()<.75:
        i=int(rng.integers(len(xx)));x=int(np.clip(xx[i]-128+rng.uniform(-40,40),0,w-256));y=int(np.clip(yy[i]-128+rng.uniform(-40,40),0,h-256))
    else:x=int(rng.integers(w-255));y=int(rng.integers(h-255))
    rgb=rgb[y:y+256,x:x+256].copy();t=t[:,y:y+256,x:x+256].copy();role=role[y:y+256,x:x+256].copy()
    if augment:
        paper=np.maximum(rgb.max((0,1)),240).astype(np.float32);opacity=np.clip((paper-rgb.astype(np.float32))/255,0,1);gamma=float(rng.uniform(.75,1.35));rgb=np.clip(paper-255*np.power(opacity,gamma),0,255)
        rgb+=rng.normal(0,rng.uniform(0,.7),rgb.shape[:2])[:,:,None];rgb=np.clip(rgb,0,255).astype(np.uint8)
        if rng.random()<.25:rgb=cv2.GaussianBlur(rgb,(0,0),float(rng.uniform(.2,.55)))
        if rng.random()<.5:rgb=rgb[:,::-1].copy();t=t[:,:,::-1].copy();t[3]*=-1;role=role[:,::-1].copy()
    return prior.photometry(rgb)[0],t,role

def loss_fn(o,t,role):
    center=t[:,0];p=o[:,0].float().sigmoid();b=F.binary_cross_entropy_with_logits(o[:,0].float(),center,reduction='none');pos=center>.05;neg=center<.01
    lc=(b*pos).sum()/pos.sum().clamp_min(1)+.4*(b*neg).sum()/neg.sum().clamp_min(1)
    dice=1-(2*(p*center).sum()+1)/(p.sum()+center.sum()+1);lc+=1.5*dice
    ink=t[:,1];bi=F.binary_cross_entropy_with_logits(o[:,1].float(),ink,reduction='none');li=(bi*(1+3*ink)).mean()
    direction=F.normalize(o[:,2:4].float(),dim=1);valid=t[:,5];ld=((1-(direction*t[:,2:4]).sum(1))*valid).sum()/valid.sum().clamp_min(1)
    lw=(F.smooth_l1_loss(o[:,4].float().sigmoid(),t[:,4],reduction='none')*valid).sum()/valid.sum().clamp_min(1)
    lr=F.cross_entropy(o[:,5:10].float(),role,weight=torch.tensor([.15,1,1,1,1.],device=o.device))
    continuity=(F.relu(.40-p)*(center>.5)).sum()/(center>.5).sum().clamp_min(1)
    total=lc+.7*li+.12*ld+.2*lw+.35*lr+.2*continuity
    return total,dict(center=lc,ink=li,direction=ld,width=lw,role=lr)

@torch.no_grad()
def validation(model,rows):
    model.eval();loss=[]
    for start in range(0,len(rows),4):
        batch=[crop(r,np.random.default_rng(r['seed']),False) for r in rows[start:start+4]]
        x,t,role=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(3)]
        with torch.autocast('cuda',dtype=torch.bfloat16):value,_=loss_fn(model(x),t,role)
        loss.append(float(value))
    return float(np.mean(loss))

def train():
    checkpoint=ROOT/'specialist.pt'
    if checkpoint.exists():raise FileExistsError('Preserve completed specialist checkpoint')
    manifest=read(ROOT/'paired_manifest.json');rows=[r for r in manifest['records'] if r['split']=='train'];cal=[r for r in manifest['records'] if r['split']=='calibration'][:12]
    assert len(rows)==512 and not {r['seed'] for r in rows}&{r['seed'] for r in cal}
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(102912);rng=np.random.default_rng(102912);model=initialize();steps=1200
    backbone=list(model.a.parameters())+list(model.b.parameters())+list(model.c.parameters());decoder=list(model.d.parameters())+list(model.e.parameters())+list(model.head.parameters());rates=[2e-5,7e-5,3e-4]
    optimizer=torch.optim.AdamW([dict(params=backbone,lr=rates[0]),dict(params=decoder,lr=rates[1]),dict(params=model.role_head.parameters(),lr=rates[2])],weight_decay=1e-4)
    config=dict(steps=steps,batch_size=6,init=str(INIT),init_sha256=digest(INIT),paired_manifest_sha256=digest(ROOT/'paired_manifest.json'),train_ids=[r['id'] for r in rows],checkpoint_calibration_ids=[r['id'] for r in cal],heldout_test_labels_used=False,imagegen_holdouts_used=False,device=torch.cuda.get_device_name(),torch_version=torch.__version__,roles=ROLES,renderer_is_procedural_not_artist_reference=True,new_dependencies_installed=False,robot_connected=False)
    jsave(ROOT/'training_config.json',config);best=float('inf');best_state=None;best_step=0;start=time.perf_counter();schedule=hashlib.sha256();logs=[];torch.cuda.reset_peak_memory_stats()
    for step in range(1,steps+1):
        indices=rng.integers(len(rows),size=6);schedule.update(indices.astype(np.int32).tobytes());batch=[crop(rows[i],rng) for i in indices]
        x,t,role=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(3)];model.train()
        factor=.25+.75*.5*(1+math.cos(math.pi*step/steps))
        for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate*factor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):value,parts=loss_fn(model(x),t,role)
        if not torch.isfinite(value):raise RuntimeError('Nonfinite training loss')
        value.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        if step%100==0:
            row=dict(step=step,loss=float(value.detach()),elapsed_seconds=time.perf_counter()-start)
            if step%200==0:
                score=validation(model,cal);row['calibration_loss']=score
                if score<best:best=score;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            logs.append(row)
            with (ROOT/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('SPECIALIST_TRAIN',json.dumps(row),flush=True)
    torch.cuda.synchronize();torch.save(dict(model=best_state,config=config,retained_step=best_step,completed_updates=steps),checkpoint)
    jsave(ROOT/'training_summary.json',dict(completed_updates=steps,retained_step=best_step,calibration_loss=best,elapsed_seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(),actual_tensor_device=str(next(model.parameters()).device),checkpoint_sha256=digest(checkpoint),initial_checkpoint_unchanged=digest(INIT)==config['init_sha256'],peak_cuda_bytes=torch.cuda.max_memory_allocated(),schedule_sha256=schedule.hexdigest(),logs=logs))
    print('SPECIALIST_TRAINED',str(checkpoint),flush=True)

if __name__=='__main__':train()
