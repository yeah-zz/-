# -*- coding: utf-8 -*-
"""Matched control/aligned training: shared data schedule, initialization and model."""
from aligned_targets_v17 import *
import time
import torch
from torch import nn
from torch.nn import functional as F
import ink_center_v16 as v16

def loss_fn(o,t,m):
    o=o.float();pc=o[:,0].sigmoid();pi=o[:,1].sigmoid();cp=(t[:,0]>.5)*m[:,0];cn=(t[:,0]<=.5)*m[:,0];b=F.binary_cross_entropy_with_logits(o[:,0],t[:,0],reduction='none');center=(b*cp).sum()/cp.sum().clamp_min(1)+(b*cn).sum()/cn.sum().clamp_min(1);ip=(t[:,1]>.5)*m[:,1];ib=(t[:,1]<=.5)*m[:,1];b=F.binary_cross_entropy_with_logits(o[:,1],t[:,1],reduction='none');ink=(b*ip).sum()/ip.sum().clamp_min(1)+(b*ib).sum()/ib.sum().clamp_min(1);dice=1-(2*(pc*t[:,0]*m[:,0]).sum()+1)/(((pc+t[:,0])*m[:,0]).sum()+1);axis=((1-F.cosine_similarity(o[:,2:4],t[:,2:4],dim=1))*m[:,2]).sum()/m[:,2].sum().clamp_min(1);width=(F.smooth_l1_loss(o[:,4].sigmoid(),t[:,4],reduction='none')*m[:,3]).sum()/m[:,3].sum().clamp_min(1);consistency=(F.relu(pc-pi)*m[:,0]).sum()/m[:,0].sum().clamp_min(1);return center+.7*ink+.25*dice+.12*axis+.15*width+.12*consistency

def train(variant):
    folder=ROOT/variant;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_summary.json').exists():raise FileExistsError('preserve completed model')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(172510);rng=np.random.default_rng(172510);manifest=read(ROOT/'manifest.json');rows=[r for r in manifest['records'] if r['split']=='train'];arrays=[]
    for r in rows:
        z=np.load(r['data']);prefix='' if variant=='aligned' else 'control_';arrays.append(dict(id=r['id'],rgb=z['rgb'].copy(),target=z[prefix+'target'].copy(),mask=z[prefix+'mask'].copy(),bg=z['background'].copy()))
    user=[a for a in arrays if a['id'].startswith('R')];assistant=[a for a in arrays if not a['id'].startswith('R')];replay=[v16.generated(171000+i)[:3] for i in range(40)]
    for _,t,_ in replay:t[4]*=.5
    model=v16.RibbonNet().cuda();model.load_state_dict(torch.load(manifest['init_weights'],map_location='cuda',weights_only=False)['model']);encoder_params=list(model.a.parameters())+list(model.b.parameters())+list(model.c.parameters());decoder_params=list(model.d.parameters())+list(model.e.parameters())+list(model.head.parameters());opt=torch.optim.AdamW([dict(params=encoder_params,lr=.00006),dict(params=decoder_params,lr=.0003)],weight_decay=.0001);steps=1200;start=time.perf_counter();usage=dict(user_real=0,assistant_real=0,explicit_background_focused=0,procedural_replay=0)
    config=dict(variant=variant,steps=steps,seed=172510,init_weights=manifest['init_weights'],init_sha256=manifest['init_sha256'],training_ids=[r['id'] for r in rows],calibration_ids=manifest['calibration_ids'],matching_schedule_and_optimizer=True,network_and_preprocessing_unchanged_from_v16=True,global_unanchored_weak_targets_used=False,real_domain_priority=True,synthetic_replay_seeds=[171000,171039],robot_connected=False);jsave(folder/'config.json',config);shutil.copy2(Path(__file__),folder/'source_at_start.py')
    for step in range(1,steps+1):
        batch=[]
        for _ in range(6):
            chance=rng.random()
            if chance<.10:
                rgb,t,m=replay[int(rng.integers(len(replay)))];usage['procedural_replay']+=1
            else:
                group=user if chance<.6 else assistant;a=group[int(rng.integers(len(group)))];rgb=a['rgb'];t=a['target'];m=a['mask'];usage['user_real' if group is user else 'assistant_real']+=1
                # Actual reviewed paper gets sampled directly; it is not
                # replaced by a procedural noise approximation.
                if rng.random()<.25 and a['bg'].any():
                    ys,xs=np.nonzero(a['bg']);pick=int(rng.integers(len(xs)));h,w=rgb.shape[:2];cx=int(np.clip(xs[pick]-128,0,max(0,w-256)));cy=int(np.clip(ys[pick]-128,0,max(0,h-256)));rgb=rgb[cy:cy+256,cx:cx+256];t=t[:,cy:cy+256,cx:cx+256];m=m[:,cy:cy+256,cx:cx+256];usage['explicit_background_focused']+=1
            batch.append(v16.crop(rgb,t,m,rng,fade=True))
        x,t,m=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(3)];opt.zero_grad(set_to_none=True);factor=.3+.7*.5*(1+math.cos(math.pi*step/steps))
        for g,lr in zip(opt.param_groups,[.00006,.0003]):g['lr']=lr*factor
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=loss_fn(model(x),t,m)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(grad):raise RuntimeError('nonfinite gradient')
        opt.step()
        if step%100==0:
            row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start)
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(__import__('json').dumps(row)+'\n')
            print('V17_TRAIN',variant,__import__('json').dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),config=config),folder/'final.pt');jsave(folder/'training_summary.json',dict(successful_updates=steps,elapsed_seconds=time.perf_counter()-start,consumed_samples=usage,stable_parent_unchanged=digest(PARENT)==manifest['stable_parent_sha256'],robot_connected=False));print('V17_TRAINED',variant,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('variant',choices=['control','aligned']);a=p.parse_args();train(a.variant)
