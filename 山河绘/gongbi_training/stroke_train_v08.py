# -*- coding: utf-8 -*-
"""Partial assistant traces + known synthetic rock strokes. Offline only."""
import json,math,time,hashlib
from pathlib import Path
import numpy as np,cv2,torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt
from .stroke_direction_v07 import model_new as previous_model

def model_new(parent=None):
    m=previous_model();old=m.head;m.head=torch.nn.Conv2d(24,6,1)
    with torch.no_grad():m.head.bias.fill_(-2)
    if parent:
        state=torch.load(parent,map_location='cpu',weights_only=False)['model']
        m.load_state_dict({k:v for k,v in state.items() if not k.startswith('head.')},strict=False)
        with torch.no_grad():m.head.weight[:4]=state['head.weight'];m.head.bias[:4]=state['head.bias']
    return m

def synthetic_rock(rng,size=256):
    h=w=size;center=np.zeros((h,w),np.uint8);ink=np.zeros((h,w),np.float32);angle=np.zeros((2,h,w),np.float32);width=np.zeros((h,w),np.float32);roles=np.zeros((2,h,w),np.float32)
    left=rng.uniform(15,40);right=rng.uniform(size-40,size-15);peak=rng.uniform(size*.35,size*.63);top=rng.uniform(18,65);base=rng.uniform(size-50,size-15)
    # Known contour and internal-stroke identity; no historical cunfa names.
    t=np.linspace(0,1,180);upper=np.stack((left+(right-left)*t,top+(base-top)*np.abs((t-(peak-left)/(right-left))/.7)**.8),1);upper[:,1]=np.clip(upper[:,1],top,base)
    lower=np.stack((np.linspace(left,right,140),base+np.sin(np.linspace(0,np.pi,140))*rng.uniform(-3,3)),1)
    strokes=[(upper,0),(lower,0)]
    for k in range(int(rng.integers(5,10))):
        x=left+(right-left)*rng.uniform(.1,.85);y=top+(base-top)*rng.uniform(.15,.55);end=min(right-4,x+rng.uniform(8,45));u=np.linspace(0,1,100)
        p=np.stack((x+(end-x)*u**.75+np.sin(u*np.pi)*rng.uniform(-10,10),y+(base-y)*u),1);strokes.append((p,1))
    for p,role in strokes:
        p=np.clip(p,3,size-4);dp=np.gradient(p,axis=0);theta=np.arctan2(dp[:,1],dp[:,0]);xy=np.rint(p).astype(int);xs,ys=xy.T;u=np.linspace(0,1,len(p));wi=rng.uniform(2,7)*(.25+.75*np.sin(np.minimum(u/.8,1)*np.pi/2));wi*=np.interp(u,[0,.1,.9,1],[.5,1,1,.5]);opacity=rng.uniform(.12,.8)
        layer=np.zeros_like(ink)
        gap_start=int(rng.integers(20,len(p)-20));gap_width=int(rng.integers(2,7)) if rng.random()<.65 else 0
        for i,((x,y),th,ww) in enumerate(zip(xy,theta,wi)):
            center[y,x]=1;angle[:,y,x]=[np.cos(2*th),np.sin(2*th)];width[y,x]=ww/12;roles[:,y,x]=0;roles[role,y,x]=1
            if not gap_start<=i<gap_start+gap_width:cv2.circle(layer,(int(x),int(y)),max(0,round((ww-1)/2)),float(opacity),-1)
        if rng.random()<.5:layer*=rng.uniform(.4,1,layer.shape).astype(np.float32)
        ink=np.maximum(ink,layer)
    dist,idx=distance_transform_edt(center==0,return_indices=True);a,b=idx;target=np.concatenate((np.exp(-dist[None]**2/(2*1.05**2)),angle[:,a,b],width[a,b][None],roles[:,a,b])).astype(np.float32)
    conf=np.ones((size,size),np.float32);role_conf=(dist<2.5).astype(np.float32)
    paper=np.ones((size,size,3),np.float32)*rng.uniform(195,254,(1,1,3));paper+=cv2.resize(rng.normal(0,1,(5,5)).astype(np.float32),(size,size))[:,:,None]*rng.uniform(1,8)
    rgb=np.clip(paper*(1-ink[:,:,None])+rng.normal(0,1.2,(size,size,1)),0,255).astype(np.uint8)
    return rgb,target,conf,role_conf

def loss_fn(logits,target,confidence,role_conf):
    p=logits.float();t=target.float();c=confidence.float();rc=role_conf.float();prob=p[:,0].sigmoid()
    bce=F.binary_cross_entropy_with_logits(p[:,0],t[:,0],pos_weight=p.new_tensor(3),reduction='none');bce=(bce*c).sum()/c.sum().clamp_min(1)
    dice=1-(2*(prob*t[:,0]*c).sum()+1)/((prob+t[:,0])*c).sum().add(1)
    positive=(t[:,0]>.4)*c;angle=((1-F.cosine_similarity(p[:,1:3],t[:,1:3],dim=1))*positive).sum()/positive.sum().clamp_min(1)
    width=(F.smooth_l1_loss(p[:,3].sigmoid(),t[:,3],reduction='none')*positive).sum()/positive.sum().clamp_min(1)
    semantic=F.binary_cross_entropy_with_logits(p[:,4:6],t[:,4:6],reduction='none');semantic=(semantic*rc[:,None]).sum()/(2*rc.sum()).clamp_min(1)
    # Continuity supervision samples every annotated center point. This never
    # labels an unmarked dark region as an absent stroke.
    continuity=(-torch.log(prob.clamp_min(1e-5))*positive).sum()/positive.sum().clamp_min(1)
    return bce+.45*dice+.20*angle+.6*width+.25*semantic+.12*continuity

def train(root,parent,old_root):
    from collections import OrderedDict
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(8023);torch.backends.cudnn.benchmark=True
    if (root/'training_summary.json').exists():raise FileExistsError('Preserve completed experiment')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));train_rows=[r for r in manifest['records'] if r['split']=='train'];validation=[r for r in manifest['records'] if r['split']=='validation']
    old=json.loads((old_root/'manifest.json').read_text(encoding='utf-8'));weak=[r for r in old['real'] if r['split']=='train'];cache=OrderedDict();size=256;rng=np.random.default_rng(8023)
    print('BUILDING_SYNTHETIC_ROCK_POOL',flush=True);pool=[synthetic_rock(np.random.default_rng(100000+i),size) for i in range(180)];val_pool=[synthetic_rock(np.random.default_rng(200000+i),size) for i in range(16)]
    def real_arrays(r,is_trace):
        key=r['id']+str(is_trace)
        if key not in cache:
            from PIL import Image
            rgb=np.asarray(Image.open(r['image']).convert('RGB'));z=np.load(r['labels'] if is_trace else r['direction_labels']);t=z['target'].astype(np.float32);c=z['confidence'].astype(np.float32)
            rc=z['roles_confidence'].astype(np.float32) if is_trace else np.zeros_like(c)
            if not is_trace:t=np.concatenate((t,np.zeros((2,*c.shape),np.float32)));c*=.35
            cache[key]=(rgb,t,c,rc);z.close()
            if len(cache)>60:cache.popitem(last=False)
        return cache[key]
    def tensor(a):
        rgb,t,c,rc=a
        return torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255,torch.from_numpy(t.copy()),torch.from_numpy(c.copy()),torch.from_numpy(rc.copy())
    def crop(a,rr,augment=True,anchor=None):
        rgb,t,c,rc=a;h,w=rgb.shape[:2];s=int(size/rr.choice([.85,1,1.2])) if augment else size
        if h<s or w<s:
            b=max(0,s-h);r=max(0,s-w);rgb=cv2.copyMakeBorder(rgb,0,b,0,r,cv2.BORDER_REFLECT_101);t=np.pad(t,((0,0),(0,b),(0,r)));c=np.pad(c,((0,b),(0,r)));rc=np.pad(rc,((0,b),(0,r)));h,w=rgb.shape[:2]
        if anchor is None:
            ys,xs=np.nonzero((t[0]>.55)&(c>.15))
            if len(xs) and rr.random()<.9:
                i=int(rr.integers(len(xs)));x=int(np.clip(xs[i]-rr.uniform(.2,.8)*s,0,w-s));y=int(np.clip(ys[i]-rr.uniform(.2,.8)*s,0,h-s))
            else:x=int(rr.integers(w-s+1));y=int(rr.integers(h-s+1))
        else:x=int(np.clip(anchor[0],0,w-s));y=int(np.clip(anchor[1],0,h-s))
        rgb=cv2.resize(rgb[y:y+s,x:x+s],(size,size));t=np.stack([cv2.resize(v[y:y+s,x:x+s],(size,size)) for v in t]);t[3]=np.clip(t[3]*size/s,0,1);c=cv2.resize(c[y:y+s,x:x+s],(size,size));rc=cv2.resize(rc[y:y+s,x:x+s],(size,size))
        if augment:
            if rr.random()<.5:rgb=rgb[:,::-1].copy();t=t[:,:,::-1].copy();t[2]*=-1;c=c[:,::-1].copy();rc=rc[:,::-1].copy()
            rgb=np.clip(rgb.astype(float)*rr.uniform(.82,1.1)+rr.uniform(-10,10),0,255).astype(np.uint8)
        return tensor((rgb,t,c,rc))
    model=model_new(parent).cuda();start=time.perf_counter();summaries=[];used=dict(assistant_trace=0,procedural_rock=0,real_weak=0);global_best=float('inf');selected=None
    cfg=dict(initialization=str(parent),initialization_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),channels=['centerline','cos2theta','sin2theta','relative_width','outline_candidate','internal_line_candidate'],annotation_provenance='assistant initial visual traces; not human expert gold',steps_per_round=800,rounds=3,physical_z=None,ready_for_robot=False)
    (root/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    def validate():
        model.eval();vals=[]
        with torch.no_grad():
            vrng=np.random.default_rng(9908)
            for i in range(12):
                a=tensor(val_pool[i]) if i<8 else crop(real_arrays(validation[0],True),vrng,False,anchor=[(0,0),(180,0),(0,105),(180,105)][i-8])
                x,t,c,rc=[v[None].cuda() for v in a];vals.append(float(loss_fn(model(x),t,c,rc)))
        model.train();return float(np.mean(vals))
    for ri in (1,2,3):
        folder=root/'runs'/f'round_{ri}';folder.mkdir(exist_ok=True);best=float('inf');lr=[8e-5,4e-5,2e-5][ri-1]
        enc=list(model.encoder.parameters());ids={id(p) for p in enc};opt=torch.optim.AdamW([dict(params=enc,lr=lr*.15),dict(params=[p for p in model.parameters() if id(p) not in ids],lr=lr)],weight_decay=1e-4);head_start=model.head.weight.detach().clone()
        for step in range(1,801):
            batch=[]
            for k in range(6):
                chance=rng.random()
                if chance<.5:r=train_rows[int(rng.integers(len(train_rows)))];a=crop(real_arrays(r,True),rng);used['assistant_trace']+=1
                elif chance<.85:a=tensor(pool[int(rng.integers(len(pool)))]);used['procedural_rock']+=1
                else:r=weak[int(rng.integers(len(weak)))];a=crop(real_arrays(r,False),rng);used['real_weak']+=1
                batch.append(a)
            x,t,c,rc=[torch.stack([a[k] for a in batch]).cuda() for k in range(4)];factor=.25+.75*.5*(1+math.cos(math.pi*step/800))
            for g,mult in zip(opt.param_groups,(.15,1)):g['lr']=lr*mult*factor
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x);loss=loss_fn(out,t,c,rc)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradients; no update applied')
            opt.step()
            if step%50==0:
                log=dict(round=ri,step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start)
                with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(log)+'\n')
                print('TRAIN',json.dumps(log),flush=True)
            if step%200==0:
                value=validate();payload=dict(model=model.state_dict(),round=ri,step=step,validation_loss=value,config=cfg)
                torch.save(payload,folder/'last.pt')
                if value<best:best=value;torch.save(payload,folder/'best.pt')
                if value<global_best:global_best=value;selected=dict(round=ri,step=step);torch.save(payload,root/'best.pt')
                print('VALIDATION',ri,step,value,flush=True)
        delta=float((model.head.weight.detach()-head_start).abs().mean());assert delta>1e-7
        summaries.append(dict(round=ri,successful_updates=800,best_validation_loss=best,head_mean_absolute_update=delta));model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=False)['model'])
    result=dict(rounds=summaries,successful_updates=2400,elapsed_seconds=time.perf_counter()-start,selected=selected,best_validation_loss=global_best,consumed_samples=used,human_expert_annotations=0,assistant_traced_training_images=len(train_rows),actual_drawing=False,ready_for_robot=False)
    (root/'training_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('TRAINING_COMPLETE',json.dumps(result),flush=True)

def infer(model,rgb,tile=256,stride=192):
    h,w=rgb.shape[:2];pad=cv2.copyMakeBorder(rgb,20,max(20,tile-h-20),20,max(20,tile-w-20),cv2.BORDER_REFLECT_101);ph,pw=pad.shape[:2];starts=lambda n:sorted(set(list(range(0,n-tile+1,stride))+[n-tile]));loc=[(y,x) for y in starts(ph) for x in starts(pw)];window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32);total=np.zeros((6,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32)
    with torch.no_grad():
        for start in range(0,len(loc),8):
            group=loc[start:start+8];arr=np.stack([pad[y:y+tile,x:x+tile].transpose(2,0,1) for y,x in group]);out=model(torch.from_numpy(arr).cuda().float()/255).float();out[:,[0,3,4,5]]=out[:,[0,3,4,5]].sigmoid();out[:,1:3]=F.normalize(out[:,1:3],dim=1)
            for (y,x),p in zip(group,out.cpu().numpy()):total[:,y:y+tile,x:x+tile]+=p*window;den[y:y+tile,x:x+tile]+=window
    return (total/den)[:,20:20+h,20:20+w]
