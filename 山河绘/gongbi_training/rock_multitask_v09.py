# -*- coding: utf-8 -*-
"""Stroke geometry, appearance and instance supervision; not expert cunfa gold."""
import json,math,time,hashlib
from pathlib import Path
from collections import OrderedDict
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt
from .model import GongbiLineNet

CHANNELS=['centerline','cos2theta','sin2theta','relative_width','outline_candidate','internal_candidate','optical_ink_density','visible_ink_coverage','known_stroke_endpoints','synthetic_dry_appearance','instance_embedding_0','instance_embedding_1','instance_embedding_2','instance_embedding_3']

def model_new(parent=None):
    model=GongbiLineNet();model.head=nn.Conv2d(24,14,1)
    with torch.no_grad():model.head.bias.fill_(-2);model.head.bias[1:3].zero_();model.head.bias[10:].zero_()
    if parent:
        state=torch.load(parent,map_location='cpu',weights_only=False)['model'];model.load_state_dict({k:v for k,v in state.items() if not k.startswith('head.')},strict=False)
        with torch.no_grad():model.head.weight[:6]=state['head.weight'][:6];model.head.bias[:6]=state['head.bias'][:6]
    return model

def synthetic(rng,size=256):
    s=size;center=np.zeros((s,s),np.uint8);angles=np.zeros((2,s,s),np.float32);widthmap=np.zeros((s,s),np.float32);roles=np.zeros((2,s,s),np.float32);instance=np.zeros((s,s),np.int32);alpha=np.zeros((s,s),np.float32);expected=np.zeros((s,s),np.float32);ends=np.zeros((s,s),np.float32)
    left=rng.uniform(15,35);right=rng.uniform(s-35,s-15);top=rng.uniform(20,60);base=rng.uniform(s-45,s-18);peak=rng.uniform(.35,.65);t=np.linspace(0,1,180)
    upper=np.stack((left+(right-left)*t,top+(base-top)*np.minimum(1,np.abs((t-peak)/max(peak,1-peak))**.85)),1)
    lower=np.stack((np.linspace(left,right,130),base+np.sin(np.linspace(0,np.pi,130))*rng.uniform(-4,4)),1);paths=[(upper,0),(lower,0)]
    family=int(rng.integers(4))
    for k in range(int(rng.integers(5,13))):
        x=rng.uniform(left+5,right-15);y=top+(base-top)*rng.uniform(.12,.6);u=np.linspace(0,1,100)
        if family==0:p=np.stack((x+u*rng.uniform(8,35)+np.sin(u*np.pi)*rng.uniform(-9,9),y+(base-y)*u),1)
        elif family==1:p=np.stack((x+np.sin(u*np.pi*2)*rng.uniform(3,12)+u*rng.uniform(0,25),y+(base-y)*u),1)
        elif family==2:p=np.stack((x+u*rng.uniform(15,40),y+np.maximum(0,u-.5)*rng.uniform(20,50)),1)
        else:
            y=rng.uniform(top+15,base-5);p=np.stack((x+u*rng.uniform(1,5),y+u*rng.uniform(0,2)),1)
        paths.append((p,1))
    for index,(p,role) in enumerate(paths,1):
        p=np.clip(p,3,s-4);xy=np.rint(p).astype(int);dp=np.gradient(p,axis=0);a=np.arctan2(dp[:,1],dp[:,0]);u=np.linspace(0,1,len(p));wi=rng.uniform(1.6,9)*(.3+.7*np.sin(np.minimum(u/.85,1)*np.pi/2));wi*=np.interp(u,[0,.08,.92,1],[.6,1,1,.5]);opacity=np.clip(rng.uniform(.1,.85)*(1+u*rng.uniform(-.4,.4)),.05,.95)
        layer=np.zeros_like(alpha);full=np.zeros_like(alpha);is_dry=rng.random()<.5;gap=int(rng.integers(20,len(p)-15))
        for j,((x,y),angle,width,ink) in enumerate(zip(xy,a,wi,opacity)):
            center[y,x]=1;angles[:,y,x]=[np.cos(2*angle),np.sin(2*angle)];widthmap[y,x]=width/12;roles[:,y,x]=0;roles[role,y,x]=1
            if instance[y,x] and instance[y,x]!=index:instance[y,x]=-1
            else:instance[y,x]=index
            radius=max(0,round((width-1)/2));cv2.circle(full,(int(x),int(y)),radius,1,-1)
            if not (is_dry and gap<=j<gap+4):cv2.circle(layer,(int(x),int(y)),radius,float(ink),-1)
        if is_dry:layer*= (rng.random(layer.shape)>.18).astype(np.float32)
        alpha=1-(1-alpha)*(1-layer);expected=np.maximum(expected,full)
        for x,y in (xy[0],xy[-1]):ends[y,x]=1
    instance[instance<0]=0;dist,idx=distance_transform_edt(center==0,return_indices=True);a,b=idx
    dry=cv2.boxFilter(np.maximum(expected-(alpha>.025),0),-1,(9,9))/np.maximum(cv2.boxFilter(expected,-1,(9,9)),.01);dry=np.clip(dry,0,1)
    endpoints=cv2.GaussianBlur(ends,(0,0),1.4);endpoints/=max(1e-6,endpoints.max())
    target=np.concatenate((np.exp(-dist[None]**2/(2*1.05**2)),angles[:,a,b],widthmap[a,b][None],roles[:,a,b],alpha[None],(alpha>.025)[None].astype(float),endpoints[None],dry[None])).astype(np.float32)
    conf=np.ones_like(target);conf[1:4]*=(dist<2.5);conf[4:6]*=(dist<2.5);conf[9]*=expected;ids=instance[a,b];ids[dist>2]=0
    paper=np.ones((s,s,3),np.float32)*rng.uniform(195,254,(1,1,3));paper+=cv2.resize(rng.normal(0,1,(6,6)).astype(np.float32),(s,s))[:,:,None]*rng.uniform(0,6)
    # Known paper creases are nuisance background, not painting strokes.
    if rng.random()<.4:
        x=int(rng.integers(s));cv2.line(paper,(x,0),(x,s-1),tuple(float(v) for v in rng.uniform(175,220,3)),1)
    rgb=np.clip(paper*(1-alpha[:,:,None])+rng.normal(0,1.2,(s,s,1)),0,255).astype(np.uint8)
    return rgb,target,conf,ids

def soft_skeleton(x,iterations=3):
    def erode(y):return torch.minimum(-F.max_pool2d(-y,(3,1),1,(1,0)),-F.max_pool2d(-y,(1,3),1,(0,1)))
    opening=F.max_pool2d(erode(x),3,1,1);skel=F.relu(x-opening)
    for _ in range(iterations):
        x=erode(x);delta=F.relu(x-F.max_pool2d(erode(x),3,1,1));skel=skel+F.relu(delta-skel*delta)
    return skel

def loss_fn(logits,target,confidence,instances,topology=True):
    p=logits.float();t=target.float();c=confidence.float();prob=p[:,0].sigmoid();bce=F.binary_cross_entropy_with_logits(p[:,0],t[:,0],pos_weight=p.new_tensor(3),reduction='none');bce=(bce*c[:,0]).sum()/c[:,0].sum().clamp_min(1)
    dice=1-(2*(prob*t[:,0]*c[:,0]).sum()+1)/((prob+t[:,0])*c[:,0]).sum().add(1)
    mask=(t[:,0]>.4)*c[:,0];angle=((1-F.cosine_similarity(p[:,1:3],t[:,1:3],dim=1))*mask).sum()/mask.sum().clamp_min(1)
    regression=0
    for channel,factor in ((3,.35),(6,1.2),(9,.15)):
        weight=c[:,channel]*(1+4*t[:,0])
        if channel==3:weight*=t[:,0]>.3
        v=F.smooth_l1_loss(p[:,channel].sigmoid(),t[:,channel],reduction='none');regression+=factor*(v*weight).sum()/weight.sum().clamp_min(1)
    other=0
    for ch,factor in ((4,.10),(5,.10),(7,.25),(8,.08)):
        weight=c[:,ch];v=F.binary_cross_entropy_with_logits(p[:,ch],t[:,ch],reduction='none');other+=factor*(v*weight).sum()/weight.sum().clamp_min(1)
    cl=p.new_tensor(0.)
    if topology:
        # Only explicitly supported regions participate. This is not a command
        # to connect arbitrary neighboring strokes into a single component.
        q=prob[:,None];truth=t[:,0:1];weight=c[:,0:1];sk=soft_skeleton(q);st=soft_skeleton(truth)
        precision=(sk*truth*weight).sum()/((sk*weight).sum()+1e-5);sensitivity=(st*q*weight).sum()/((st*weight).sum()+1e-5);cl=1-2*precision*sensitivity/(precision+sensitivity+1e-5)
    embedding=F.normalize(p[:,10:14],dim=1);pull=p.new_tensor(0.);push=p.new_tensor(0.);groups=0
    for i in range(len(p)):
        ids=torch.unique(instances[i]);means=[]
        for identity in ids[ids>0][:18]:
            pix=(instances[i]==identity)&(t[i,0]>.4)
            if pix.sum()<5:continue
            vectors=embedding[i,:,pix].T[::3];mean=F.normalize(vectors.mean(0),dim=0);pull+=(1-vectors@mean).mean();means.append(mean);groups+=1
        if len(means)>1:
            m=torch.stack(means);sim=m@m.T;off=~torch.eye(len(m),device=p.device,dtype=torch.bool);push+=F.relu(sim[off]-.4).mean()
    instance_loss=pull/max(1,groups)+push/max(1,len(p))
    total=bce+.4*dice+.15*angle+regression+other+.2*cl+.12*instance_loss
    return total,dict(center_bce=float(bce.detach()),topology_loss=float(cl.detach()),instance_loss=float(instance_loss.detach()))

def train(root,parent,old_root,trace_root):
    from PIL import Image
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(9024);torch.backends.cudnn.benchmark=True
    if (root/'training_summary.json').exists():raise FileExistsError('Preserve completed run')
    frozen=json.loads((root/'frozen_manifest.json').read_text(encoding='utf-8'));weak=[r for r in json.loads((old_root/'manifest.json').read_text(encoding='utf-8'))['real'] if r['split']=='train'];trace_manifest=json.loads((trace_root/'manifest.json').read_text(encoding='utf-8'));traces=[r for r in trace_manifest['records'] if r['split']=='train'];validation_trace=[r for r in trace_manifest['records'] if r['split']=='validation']
    def compact(a):
        rgb,t,c,ids=a;return rgb,t.astype(np.float16),c.astype(np.float16),ids.astype(np.int16)
    print('PRECOMPUTING_KNOWN_GEOMETRY',flush=True);pool=[compact(synthetic(np.random.default_rng(300000+i))) for i in range(240)];val_pool=[compact(synthetic(np.random.default_rng(400000+i))) for i in range(20)];cache=OrderedDict();size=256;rng=np.random.default_rng(9024)
    def load(r,traced=False):
        key=r['id']+str(traced)
        if key in cache:cache.move_to_end(key);return cache[key]
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));h,w=rgb.shape[:2];target=np.zeros((10,h,w),np.float32);conf=np.zeros_like(target);ids=np.zeros((h,w),np.int32)
        if traced:
            with np.load(r['labels']) as z:
                target[:6]=z['target'];conf[:4]=z['confidence'];conf[4:6]=z['roles_confidence'];target[6]=z['density'];target[7]=np.minimum(target[6]*3,1);conf[6:8]=.20
            from scipy.interpolate import PchipInterpolator
            for number,stroke in enumerate(r['dense_paths'],1):
                xy=np.rint(stroke['points']).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);ids[xy[:,1],xy[:,0]]=number
            dist,idx=distance_transform_edt(ids==0,return_indices=True);ids=ids[tuple(idx)];ids[dist>2]=0
        else:
            with np.load(r['direction_labels']) as z:target[:4]=z['target'];conf[:4]=z['confidence']*.3
            with np.load(r['labels']) as z:target[6]=z['targets'][1]/255.;target[7]=z['targets'][0]/255.;conf[6:8]=z['weights']*.25/255 if 'weights' in z else .2
        cache[key]=(rgb,target.astype(np.float16),conf.astype(np.float16),ids.astype(np.int16))
        if len(cache)>20:cache.popitem(last=False)
        return cache[key]
    def tensor(a):
        rgb,t,c,ids=a;return torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255,torch.from_numpy(t.copy()),torch.from_numpy(c.copy()),torch.from_numpy(ids.copy()).long()
    def crop(a,rr,augment=True,anchor=None):
        rgb,t,c,ids=a;t=t.astype(np.float32);c=c.astype(np.float32);h,w=rgb.shape[:2];s=int(size/rr.choice([.85,1,1.2])) if augment else size
        if h<s or w<s:
            b=max(0,s-h);r=max(0,s-w);rgb=cv2.copyMakeBorder(rgb,0,b,0,r,cv2.BORDER_REFLECT_101);t=np.pad(t,((0,0),(0,b),(0,r)));c=np.pad(c,((0,0),(0,b),(0,r)));ids=np.pad(ids,((0,b),(0,r)));h,w=rgb.shape[:2]
        if anchor is None:
            yy,xx=np.nonzero((t[0]>.5)&(c[0]>.05))
            if len(xx) and rr.random()<.85:i=int(rr.integers(len(xx)));x=int(np.clip(xx[i]-rr.uniform(.2,.8)*s,0,w-s));y=int(np.clip(yy[i]-rr.uniform(.2,.8)*s,0,h-s))
            else:x=int(rr.integers(w-s+1));y=int(rr.integers(h-s+1))
        else:x=int(np.clip(anchor[0],0,w-s));y=int(np.clip(anchor[1],0,h-s))
        rgb=cv2.resize(rgb[y:y+s,x:x+s],(size,size));t=np.stack([cv2.resize(v[y:y+s,x:x+s],(size,size)) for v in t]);t[3]=np.clip(t[3]*size/s,0,1);c=np.stack([cv2.resize(v[y:y+s,x:x+s],(size,size)) for v in c]);ids=cv2.resize(ids[y:y+s,x:x+s],(size,size),interpolation=cv2.INTER_NEAREST)
        if augment:
            if rr.random()<.5:rgb=rgb[:,::-1].copy();t=t[:,:,::-1].copy();t[2]*=-1;c=c[:,:,::-1].copy();ids=ids[:,::-1].copy()
            # Density supervision is photometric, so do not alter darkness
            # without transforming its target. Paper tint changes preserve alpha.
            rgb=np.clip(rgb.astype(float)*rr.uniform(.88,1.05,(1,1,3)),0,255).astype(np.uint8)
        return tensor((rgb,t,c,ids))
    model=model_new(parent).cuda();start=time.perf_counter();best=float('inf');selected=None;summary=[];used=dict(assistant_trace=0,real_weak=0,known_procedural=0)
    cfg=dict(channels=CHANNELS,initialization=str(parent),initialization_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),frozen_manifest_sha256=hashlib.sha256((root/'frozen_manifest.json').read_bytes()).hexdigest(),expert_annotations=0,source_snapshot_note='source-only new probe images never sampled during training',real_weak_provenance='automatic paper/ridge labels',dryness_provenance='known procedural holes only, not physical water',geometry_phases=[800,800,800],ready_for_robot=False)
    (root/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    def validate():
        model.eval();values=[]
        with torch.no_grad():
            rr=np.random.default_rng(9090)
            for i in range(12):
                a=tensor(val_pool[i]) if i<8 else crop(load(validation_trace[0],True),rr,False,[(0,0),(180,0),(0,105),(180,105)][i-8]);x,t,c,ids=[v[None].cuda() for v in a];value,_=loss_fn(model(x),t,c,ids,topology=True);values.append(float(value))
        model.train();return float(np.mean(values))
    for phase in (1,2,3):
        folder=root/'runs'/f'phase_{phase}';folder.mkdir(exist_ok=True);lr=[8e-5,4e-5,2e-5][phase-1];encoder=list(model.encoder.parameters());eid={id(p) for p in encoder};opt=torch.optim.AdamW([dict(params=encoder,lr=lr*.15),dict(params=[p for p in model.parameters() if id(p) not in eid],lr=lr)],weight_decay=1e-4);phase_best=float('inf')
        for step in range(1,801):
            batch=[]
            for _ in range(6):
                value=rng.random()
                if value<.40:a=tensor(pool[int(rng.integers(len(pool)))]);used['known_procedural']+=1
                elif value<.70:a=crop(load(traces[int(rng.integers(len(traces)))],True),rng);used['assistant_trace']+=1
                else:a=crop(load(weak[int(rng.integers(len(weak)))]),rng);used['real_weak']+=1
                batch.append(a)
            x,t,c,ids=[torch.stack([a[k] for a in batch]).cuda() for k in range(4)];factor=.25+.75*.5*(1+math.cos(math.pi*step/800))
            for g,mult in zip(opt.param_groups,(.15,1)):g['lr']=lr*mult*factor
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):value,parts=loss_fn(model(x),t,c,ids,topology=(phase>1 or step>400))
            if not torch.isfinite(value):raise RuntimeError('Nonfinite loss')
            value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient; refusing update')
            opt.step()
            if step%50==0:
                row=dict(phase=phase,step=step,loss=float(value.detach()),elapsed_seconds=time.perf_counter()-start,**parts)
                with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                print('TRAIN',json.dumps(row),flush=True)
            if step%200==0:
                val=validate();checkpoint=dict(model=model.state_dict(),phase=phase,step=step,validation_loss=val,config=cfg);torch.save(checkpoint,folder/'last.pt')
                if val<phase_best:phase_best=val;torch.save(checkpoint,folder/'best.pt')
                if val<best:best=val;selected=dict(phase=phase,step=step);torch.save(checkpoint,root/'best.pt')
                print('VALIDATION',phase,step,val,flush=True)
        summary.append(dict(phase=phase,successful_updates=800,best_validation_loss=phase_best));model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=False)['model'])
    result=dict(phases=summary,successful_updates=2400,skipped_updates=0,selected=selected,best_validation_loss=best,elapsed_seconds=time.perf_counter()-start,consumed_samples=used,synthetic_training_geometries=240,expert_annotations=0,actual_drawing=False,ready_for_robot=False)
    (root/'training_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('TRAINING_COMPLETE',json.dumps(result),flush=True)

def infer(model,rgb,tile=256,stride=192):
    h,w=rgb.shape[:2];pad=cv2.copyMakeBorder(rgb,20,max(20,tile-h-20),20,max(20,tile-w-20),cv2.BORDER_REFLECT_101);ph,pw=pad.shape[:2];starts=lambda n:sorted(set(list(range(0,n-tile+1,stride))+[n-tile]));loc=[(y,x) for y in starts(ph) for x in starts(pw)];window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32);total=np.zeros((14,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32)
    with torch.no_grad():
        for start in range(0,len(loc),8):
            group=loc[start:start+8];batch=np.stack([pad[y:y+tile,x:x+tile].transpose(2,0,1) for y,x in group]);p=model(torch.from_numpy(batch).cuda().float()/255).float();p[:,[0,3,4,5,6,7,8,9]]=p[:,[0,3,4,5,6,7,8,9]].sigmoid();p[:,1:3]=F.normalize(p[:,1:3],dim=1);p[:,10:14]=F.normalize(p[:,10:14],dim=1)
            for (y,x),v in zip(group,p.cpu().numpy()):total[:,y:y+tile,x:x+tile]+=v*window;den[y:y+tile,x:x+tile]+=window
    return (total/den)[:,20:20+h,20:20+w]
