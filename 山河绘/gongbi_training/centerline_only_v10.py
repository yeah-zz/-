# -*- coding: utf-8 -*-
"""Geometry-only student conditioned on a frozen ink model. No robot imports."""
import json,math,time,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt
from .model import GongbiLineNet
from .rock_multitask_v09 import synthetic as rock_synthetic,soft_skeleton

def input_from_ink(ink):
    coverage,density=ink.astype(np.float32)
    return np.stack((1-density,1-coverage,1-cv2.GaussianBlur(density,(0,0),1.0))).astype(np.float32)

def model_new(parent=None):
    model=GongbiLineNet();model.head=nn.Conv2d(24,5,1)
    with torch.no_grad():model.head.bias.zero_();model.head.bias[0]=-2
    if parent:
        state=torch.load(parent,map_location='cpu',weights_only=False)['model'];model.load_state_dict({k:v for k,v in state.items() if not k.startswith('head.')},strict=False)
        with torch.no_grad():model.head.weight[:3]=state['head.weight'][:3];model.head.bias[:3]=state['head.bias'][:3];model.head.weight[3:].zero_()
    return model

def geometry_target(center,angles):
    h,w=center.shape;dist,idx=distance_transform_edt(center==0,return_indices=True);yy,xx=np.mgrid[:h,:w];a,b=idx
    return np.concatenate((np.exp(-dist[None]**2/(2*.85**2)),angles[:,a,b],np.clip((b-xx)[None]/8,-1,1),np.clip((a-yy)[None]/8,-1,1))).astype(np.float32),dist

def make_pool(teacher,seed_start,count):
    pool=[];seeds=[];seed=seed_start
    with torch.no_grad():
        while len(pool)<count:
            rgb,t,c,ids=rock_synthetic(np.random.default_rng(seed));current=seed;seed+=1
            if (t[0]>.5).sum()<750:continue
            center=t[0]>.99;target,dist=geometry_target(center,t[1:3]);confidence=np.ones_like(target);confidence[1:3]*=dist<2.5;confidence[3:5]*=dist<6
            pad=cv2.copyMakeBorder(rgb,32,32,32,32,cv2.BORDER_REPLICATE);x=torch.from_numpy(pad.transpose(2,0,1).copy())[None].cuda().float()/255;ink=teacher(x).sigmoid()[0,:,32:-32,32:-32].float().cpu().numpy();features=input_from_ink(ink)
            pool.append((features.astype(np.float16),target.astype(np.float16),confidence.astype(np.float16)));seeds.append(current)
            if len(pool)%40==0:print('KNOWN_STROKE_POOL',len(pool),count,flush=True)
    return pool,seeds

def loss_fn(logits,t,c):
    p=logits.float();t=t.float();c=c.float();prob=p[:,0].sigmoid();b=F.binary_cross_entropy_with_logits(p[:,0],t[:,0],pos_weight=p.new_tensor(2.5),reduction='none');b=(b*c[:,0]).sum()/c[:,0].sum().clamp_min(1)
    dice=1-(2*(prob*t[:,0]*c[:,0]).sum()+1)/((prob+t[:,0])*c[:,0]).sum().add(1)
    m=(t[:,0]>.3)*c[:,1];direction=((1-F.cosine_similarity(p[:,1:3],t[:,1:3],dim=1))*m).sum()/m.sum().clamp_min(1)
    offsets=(F.smooth_l1_loss(p[:,3:5].tanh(),t[:,3:5],reduction='none')*c[:,3:5]).sum()/c[:,3:5].sum().clamp_min(1)
    sk=soft_skeleton(prob[:,None],2);truth=t[:,0:1];st=soft_skeleton(truth,2);weight=c[:,0:1];a=(sk*truth*weight).sum()/((sk*weight).sum()+1e-5);z=(st*prob[:,None]*weight).sum()/((st*weight).sum()+1e-5);top=1-2*a*z/(a+z+1e-5)
    return b+.4*dice+.2*direction+.3*offsets+.12*top

def train(root,parent,ink_path):
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(10024);torch.backends.cudnn.benchmark=True
    if (root/'training_summary.json').exists():raise FileExistsError('Preserve existing run')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));training=[r for r in manifest['records'] if r['split']=='train'];validation=[r for r in manifest['records'] if r['split']=='validation']
    teacher=GongbiLineNet().cuda();teacher.load_state_dict(torch.load(ink_path,map_location='cuda',weights_only=False)['model']);teacher.eval();pool,seeds=make_pool(teacher,610000,200);val_pool,val_seeds=make_pool(teacher,620000,20);del teacher;torch.cuda.empty_cache()
    arrays={}
    for r in training+validation:
        z=np.load(r['labels']);ink=np.load(r['ink_cache'])['ink'].astype(np.float32);arrays[r['id']]=(input_from_ink(ink),z['target'].astype(np.float32),z['confidence'].astype(np.float32));z.close()
    size=256;rng=np.random.default_rng(10024)
    def tensor(a):return tuple(torch.from_numpy(v.copy()).float() for v in a)
    def sample_real(r,rr,augment=True):
        x,t,c=arrays[r['id']];s=int(size/rr.choice([.85,1.,1.2])) if augment else size;h,w=x.shape[1:]
        if h<s or w<s:
            bottom=max(0,s-h);right=max(0,s-w);x=np.pad(x,((0,0),(0,bottom),(0,right)),mode='edge');t=np.pad(t,((0,0),(0,bottom),(0,right)));c=np.pad(c,((0,0),(0,bottom),(0,right)));h,w=x.shape[1:]
        yy,xx=np.nonzero((t[0]>.5)&(c[0]>.1));i=int(rr.integers(len(xx)));left=int(np.clip(xx[i]-s*rr.uniform(.2,.8),0,w-s));top=int(np.clip(yy[i]-s*rr.uniform(.2,.8),0,h-s))
        x=np.stack([cv2.resize(v[top:top+s,left:left+s],(size,size)) for v in x]);t=np.stack([cv2.resize(v[top:top+s,left:left+s],(size,size)) for v in t]);t[3:5]=np.clip(t[3:5]*size/s,-1,1);c=np.stack([cv2.resize(v[top:top+s,left:left+s],(size,size)) for v in c])
        if augment and rr.random()<.5:x=x[:,:,::-1].copy();t=t[:,:,::-1].copy();t[2]*=-1;t[3]*=-1;c=c[:,:,::-1].copy()
        return tensor((x,t,c))
    model=model_new(parent).cuda();torch.save(dict(model=model.state_dict(),phase=0,step=0),root/'initial.pt');best=float('inf');selected=None;start=time.perf_counter();used=dict(assistant_trace=0,known_procedural=0)
    config=dict(task='frozen ink -> centerline, tangent and auxiliary projection offsets only',outputs=['centerline','cos2theta','sin2theta','axis_offset_x_over8','axis_offset_y_over8'],initialization=str(parent),initialization_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),frozen_ink_sha256=hashlib.sha256(ink_path.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),input_representation=['1-ink_density','1-ink_coverage','1-smoothed_ink_density'],planner_frozen='stroke_streamline_v08 strict=True',new_assistant_traces_not_expert=True,optimizer_steps_limit=1600,expert_annotations=0,ready_for_robot=False)
    (root/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8');(root/'synthetic_seeds.json').write_text(json.dumps(dict(train=seeds,validation=val_seeds),indent=2),encoding='utf-8')
    def validate():
        model.eval();values=[];rr=np.random.default_rng(10620)
        with torch.no_grad():
            for i in range(12):
                a=tensor(val_pool[i]) if i<6 else sample_real(validation[(i-6)%len(validation)],rr,False);x,t,c=[v[None].cuda() for v in a];values.append(float(loss_fn(model(x),t,c)))
        model.train();return float(np.mean(values))
    phase_summary=[]
    for phase in (1,2):
        run=root/'runs'/f'phase_{phase}';run.mkdir(exist_ok=True);base=[7e-5,3e-5][phase-1];enc=list(model.encoder.parameters());eids={id(p) for p in enc};opt=torch.optim.AdamW([dict(params=enc,lr=base*.2),dict(params=[p for p in model.parameters() if id(p) not in eids],lr=base)],weight_decay=1e-4);phase_best=float('inf')
        for step in range(1,801):
            batch=[]
            for _ in range(6):
                if rng.random()<.55:r=training[int(rng.integers(len(training)))];a=sample_real(r,rng);used['assistant_trace']+=1
                else:a=tensor(pool[int(rng.integers(len(pool)))]);used['known_procedural']+=1
                batch.append(a)
            x,t,c=[torch.stack([a[k] for a in batch]).cuda() for k in range(3)];factor=.2+.8*.5*(1+math.cos(math.pi*step/800))
            for group,mult in zip(opt.param_groups,(.2,1)):group['lr']=base*mult*factor
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):value=loss_fn(model(x),t,c)
            assert torch.isfinite(value);value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);opt.step()
            if step%50==0:
                row=dict(phase=phase,step=step,loss=float(value.detach()),elapsed_seconds=time.perf_counter()-start)
                with (run/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                print('TRAIN',json.dumps(row),flush=True)
            if step%200==0:
                score=validate();ck=dict(model=model.state_dict(),phase=phase,step=step,validation_loss=score,config=config);torch.save(ck,run/'last.pt')
                if score<phase_best:phase_best=score;torch.save(ck,run/'best.pt')
                if score<best:best=score;selected=dict(phase=phase,step=step);torch.save(ck,root/'best.pt')
                print('VALIDATION',phase,step,score,flush=True)
        phase_summary.append(dict(phase=phase,successful_updates=800,best_validation_loss=phase_best));model.load_state_dict(torch.load(run/'best.pt',map_location='cuda',weights_only=False)['model'])
    summary=dict(successful_updates=1600,skipped_updates=0,selected=selected,best_validation_loss=best,elapsed_seconds=time.perf_counter()-start,phases=phase_summary,consumed_samples=used,real_training_images=len(training),expert_annotations=0,frozen_ink_unchanged=True,planner_unchanged=True,actual_drawing=False)
    (root/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('TRAINING_COMPLETE',json.dumps(summary),flush=True)

def infer(model,ink,tile=256,stride=192):
    features=input_from_ink(ink);h,w=features.shape[1:];features=np.pad(features,((0,0),(20,max(20,tile-h-20)),(20,max(20,tile-w-20))),mode='edge');ph,pw=features.shape[1:];starts=lambda n:sorted(set(list(range(0,n-tile+1,stride))+[n-tile]));loc=[(y,x) for y in starts(ph) for x in starts(pw)];window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32);total=np.zeros((5,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32)
    with torch.no_grad():
        for start in range(0,len(loc),8):
            group=loc[start:start+8];batch=np.stack([features[:,y:y+tile,x:x+tile] for y,x in group]);p=model(torch.from_numpy(batch).cuda().float()).float();p[:,0]=p[:,0].sigmoid();p[:,1:3]=F.normalize(p[:,1:3],dim=1);p[:,3:5]=p[:,3:5].tanh()
            for (y,x),v in zip(group,p.cpu().numpy()):total[:,y:y+tile,x:x+tile]+=v*window;den[y:y+tile,x:x+tile]+=window
    return (total/den)[:,20:20+h,20:20+w]
