# -*- coding: utf-8 -*-
"""Two undirected axes + endpoint/junction supervision, offline experiment."""
import sys,json,math,time,shutil
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2,torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt
from PIL import Image
from gongbi_training.model import GongbiLineNet
from human_mountain_v11 import fields,digest,jsave
from stroke_link_v12 import dense,synthetic
ROOT=LAB/'stroke_topology_v13_20261024'
OLD=LAB/'clean_strokes_v12_20261024'
PARENT=OLD/'final.pt'

def model_new():
    m=GongbiLineNet();m.head=nn.Conv2d(24,8,1)
    with torch.no_grad():m.head.bias.fill_(-3)
    old=torch.load(PARENT,map_location='cpu',weights_only=False)['model'];m.load_state_dict({k:v for k,v in old.items() if not k.startswith('head.')},strict=False)
    with torch.no_grad():
        m.head.weight[:3]=old['head.weight'][:3];m.head.bias[:3]=old['head.bias'][:3]
        # Break permutation symmetry: second initial axis is perpendicular.
        m.head.weight[3:5]=-old['head.weight'][1:3];m.head.bias[3:5]=-old['head.bias'][1:3]
    return m

def heatmap(points,shape,sigma):
    a=np.zeros(shape,np.uint8)
    for p in points:
        x,y=np.rint(p).astype(int)
        if 0<=x<shape[1] and 0<=y<shape[0]:a[y,x]=1
    return np.exp(-distance_transform_edt(a==0)**2/(2*sigma*sigma)).astype(np.float32) if a.any() else a.astype(np.float32)

def targets(doc,complete=False):
    h,w=doc['height'],doc['width'];center,_,base_mask,dist=fields(doc);shape=(h,w)
    d1=np.full(shape,1e6,np.float32);d2=d1.copy();a1=np.zeros((2,h,w),np.float32);a2=a1.copy();ends=[]
    for s in doc['strokes']:
        if s['role'] not in {'contour','cun','unknown'}:continue
        p=dense(s['points'],.6);v=np.gradient(p,axis=0);theta=np.arctan2(v[:,1],v[:,0]);xy=np.rint(p).astype(int);xx=np.clip(xy[:,0],0,w-1);yy=np.clip(xy[:,1],0,h-1);binary=np.zeros(shape,np.uint8);binary[yy,xx]=1;ax=np.zeros((2,h,w),np.float32);ax[:,yy,xx]=np.stack((np.cos(2*theta),np.sin(2*theta)))
        distance,idx=distance_transform_edt(binary==0,return_indices=True);axis=ax[:,idx[0],idx[1]];closer=distance<d1
        d2[closer]=d1[closer];a2[:,closer]=a1[:,closer];d1[closer]=distance[closer];a1[:,closer]=axis[:,closer]
        second=(~closer)&(distance<d2);d2[second]=distance[second];a2[:,second]=axis[:,second]
        if np.linalg.norm(p[0]-p[-1])>3:ends.extend((p[0],p[-1]))
    dot=(a1*a2).sum(0);multi=(d1<=2.3)&(d2<=2.3)&(dot<.85);junction=(d1<=1.1)&(d2<=1.1)&(dot<.70)
    if junction.any():junction_heat=np.exp(-distance_transform_edt(~junction)**2/(2*2.0**2)).astype(np.float32)
    else:junction_heat=np.zeros(shape,np.float32)
    endpoint=heatmap(ends,shape,2.0);valid=base_mask[3]
    if complete:
        pos=(dist<=1.8)*valid;neg=(dist>=3.0)*valid;key_valid=valid;second_conf=(dist<=2.3)*valid
    else:
        pos=base_mask[0];neg=base_mask[1];key_valid=None;second_conf=(multi.astype(np.float32)+neg).clip(0,1)*valid
    # Missing labels in partial real pictures never become endpoint/junction
    # negatives, nor do they remove a possible second crossing direction.
    ep_mask=key_valid if complete else ((endpoint>.03)| (neg>0))*valid
    ju_mask=key_valid if complete else ((junction_heat>.03)|(neg>0))*valid
    target=np.concatenate((np.exp(-dist[None]**2/(2*1.1**2)),a1,a2,endpoint[None],junction_heat[None],multi[None].astype(np.float32))).astype(np.float32)
    mask=np.stack((pos,neg,(dist<=2.3)*valid,multi*valid,ep_mask,ju_mask,second_conf,valid)).astype(np.float32)
    return target,mask

def procedural(seed,size=256):
    rgb,paths,p=synthetic(seed,320);factor=size/320.;rgb=cv2.resize(rgb,(size,size));paths=[a*factor for a in paths]
    doc=dict(width=size,height=size,sample_id=str(seed),strokes=[dict(role='contour',points=a.tolist()) for a in paths],regions=[])
    target,mask=targets(doc,True);return rgb,target,mask,paths

def heat_loss(logit,target,mask):
    prob=logit.float().sigmoid().clamp(1e-5,1-1e-5);positive=(target>.99)*mask;negative=(target<.99)*mask
    return (-(prob.log())*(1-prob)**2*positive-(1-prob).log()*prob**2*(1-target)**4*negative).sum()/positive.sum().clamp_min(1)

def loss_fn(logits,t,m):
    o=logits.float();p=o[:,0].sigmoid();bce=F.binary_cross_entropy_with_logits(o[:,0],t[:,0],reduction='none');center=(bce*m[:,0]).sum()/m[:,0].sum().clamp_min(1)+(F.softplus(o[:,0])*m[:,1]).sum()/m[:,1].sum().clamp_min(1)
    one=F.normalize(o[:,1:3],dim=1);two=F.normalize(o[:,3:5],dim=1);a=t[:,1:3];b=t[:,3:5];aa=1-(one*a).sum(1);ab=1-(one*b).sum(1);ba=1-(two*a).sum(1);bb=1-(two*b).sum(1)
    # Unordered axes: a partial single-axis annotation supervises the nearer
    # predicted axis only; a known crossing supervises both assignments.
    angle=torch.minimum(aa,ba)*(1-m[:,3])+torch.minimum(aa+bb,ab+ba)*m[:,3]*.5
    axis=(angle*m[:,2]).sum()/m[:,2].sum().clamp_min(1)
    ep=heat_loss(o[:,5],t[:,5],m[:,4]);junction=heat_loss(o[:,6],t[:,6],m[:,5]);b=F.binary_cross_entropy_with_logits(o[:,7],t[:,7],reduction='none');multi=(b*m[:,6]).sum()/m[:,6].sum().clamp_min(1)
    prior=o.new_tensor(0.)
    if t.shape[1]>8:
        b=F.binary_cross_entropy_with_logits(o[:,0],t[:,8],reduction='none');prior=(b*m[:,8]).sum()/m[:,8].sum().clamp_min(1)
    return center+.2*axis+.04*ep+.04*junction+.12*multi+.1*prior,dict(center=float(center.detach()),axes=float(axis.detach()),endpoints=float(ep.detach()),junctions=float(junction.detach()),multi=float(multi.detach()),parent_prior=float(prior.detach()))

@torch.no_grad()
def infer(model,rgb,tile=256,stride=192):
    h,w=rgb.shape[:2];pad=cv2.copyMakeBorder(rgb,24,max(24,tile-h-24),24,max(24,tile-w-24),cv2.BORDER_REFLECT_101);ph,pw=pad.shape[:2];starts=lambda n:sorted(set(list(range(0,n-tile+1,stride))+[n-tile]));loc=[(y,x) for y in starts(ph) for x in starts(pw)];window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32);total=np.zeros((8,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32)
    for start in range(0,len(loc),6):
        group=loc[start:start+6];x=np.stack([pad[y:y+tile,x:x+tile].transpose(2,0,1) for y,x in group]);out=model(torch.from_numpy(x).cuda().float()/255).float();out[:,[0,5,6,7]]=out[:,[0,5,6,7]].sigmoid();out[:,1:3]=F.normalize(out[:,1:3],dim=1);out[:,3:5]=F.normalize(out[:,3:5],dim=1)
        # Align axis permutation in overlaps before averaging.
        for (y,x),v in zip(group,out.cpu().numpy()):
            existing=total[:,y:y+tile,x:x+tile];weight=den[y:y+tile,x:x+tile];direct=(existing[1:3]*v[1:3]).sum(0)+(existing[3:5]*v[3:5]).sum(0);swap=(existing[1:3]*v[3:5]).sum(0)+(existing[3:5]*v[1:3]).sum(0);choose=(weight>0)&(swap>direct);first=v[1:3].copy();v[1:3,choose]=v[3:5,choose];v[3:5,choose]=first[:,choose];total[:,y:y+tile,x:x+tile]+=v*window;den[y:y+tile,x:x+tile]+=window
    return (total/den)[:,24:24+h,24:24+w]

def optimize(model,arrays,steps,folder,small=False):
    folder.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(132410 if small else 132411);lr=1.5e-4 if small else 4e-5;opt=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=lr*.2),dict(params=[p for n,p in model.named_parameters() if not n.startswith('encoder.')],lr=lr)],weight_decay=0 if small else 1e-4);start=time.perf_counter()
    for step in range(1,steps+1):
        selection=list(range(len(arrays))) if small else rng.integers(0,len(arrays),6);batch=[]
        for index in selection:
            rgb,t,m=arrays[index][:3];h,w=rgb.shape[:2]
            if small:x=rgb.astype(np.float32).transpose(2,0,1)/255
            else:
                size=256
                if min(h,w)<size:
                    bottom=max(0,size-h);right=max(0,size-w);rgb=cv2.copyMakeBorder(rgb,0,bottom,0,right,cv2.BORDER_REFLECT_101);t=np.pad(t,((0,0),(0,bottom),(0,right)));m=np.pad(m,((0,0),(0,bottom),(0,right)));h,w=rgb.shape[:2]
                yy,xx=np.nonzero((t[0]>.7)&(m[0]>.1));k=int(rng.integers(len(xx)));left=int(np.clip(xx[k]-rng.uniform(.2,.8)*size,0,w-size));top=int(np.clip(yy[k]-rng.uniform(.2,.8)*size,0,h-size));rgb=rgb[top:top+size,left:left+size];t=t[:,top:top+size,left:left+size];m=m[:,top:top+size,left:left+size];x=rgb.astype(np.float32).transpose(2,0,1)/255
            batch.append((x,t,m))
        x,t,m=[torch.from_numpy(np.stack([b[k] for b in batch])).cuda().float() for k in range(3)];model.train()
        for layer in model.encoder.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
        factor=.25+.75*.5*(1+math.cos(math.pi*step/steps))
        for group,mult in zip(opt.param_groups,[.2,1]):group['lr']=lr*mult*factor
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=loss_fn(model(x),t,m)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(grad):raise RuntimeError('nonfinite gradients')
        opt.step()
        if step%50==0:
            row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start,**parts)
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('SMALL_FIT' if small else 'TOPO_TRAIN',json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),steps=steps,outputs=['center','axis1_cos2','axis1_sin2','axis2_cos2','axis2_sin2','endpoint','junction','multiple_axes']),folder/'final.pt');return dict(successful_updates=steps,elapsed_seconds=time.perf_counter()-start)

def run_small():
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(132410);folder=ROOT/'small_fit'
    if (folder/'training_summary.json').exists():raise FileExistsError('Preserve small fit')
    examples=[procedural(s) for s in (710002,710001,710003)];folder.mkdir(parents=True,exist_ok=True)
    for i,(rgb,t,m,paths) in enumerate(examples):Image.fromarray(rgb).save(folder/(f'known_{i}.png'));np.savez_compressed(folder/(f'known_{i}.npz'),target=t,mask=m)
    model=model_new().cuda();summary=optimize(model,examples,600,folder,True);summary.update(real_artworks=0,known_procedural_training_cases=3,diagnostic_only_not_generalization=True,baseline_sha256=digest(PARENT));jsave(folder/'training_summary.json',summary);print('SMALL_FIT_COMPLETE',flush=True)

def run_train():
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(132411);folder=ROOT/'main'
    if (folder/'training_summary.json').exists():raise FileExistsError('Preserve trained run')
    gate=json.loads((ROOT/'small_fit'/'diagnostic.json').read_text())
    if not gate['localization_gate_passed']:raise RuntimeError('Small-fit localization gate failed; do not expand training')
    folder.mkdir(parents=True,exist_ok=True);records=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))['records'];real=[];rows=[]
    for r in records:
        if r['split']!='train':continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=json.loads(Path(r['labels']).read_text(encoding='utf-8'));t,m=targets(doc,False)
        with np.load(OLD/'evaluation'/r['id']/'probabilities.npz') as z:prior=z['after'][0].copy()
        unknown=(1-m[0])*(1-m[1])*m[7];t=np.concatenate((t,prior[None]));m=np.concatenate((m,unknown[None]));real.append((rgb,t,m));rows.append(dict(id=r['id'],source_group=r['source_group'],image=r['image'],labels=r['labels'],image_sha256=digest(r['image']),label_sha256=digest(r['labels']),provenance=r['provenance'],complete=False))
    pool=[]
    for i in range(120):
        rgb,t,m,paths=procedural(720000+i);pool.append((rgb,np.concatenate((t,t[:1])),np.concatenate((m,np.zeros_like(m[:1]))),paths))
    arrays=real*8+pool
    jsave(folder/'manifest.json',dict(real=rows,known_procedural_training_geometries=120,test_images_not_trained=True,new_real_labels_not_expert=True,unmarked_real_regions_unknown=True,weak_parent_regularization_not_user_labels=True,purple_loss_excluded=True,source_work_independence_not_verified=True))
    model=model_new().cuda();summary=optimize(model,arrays,1000,folder,False);summary.update(real_training_crops=len(real),known_procedural_training_geometries=120,initialized_from=str(PARENT),parent_unchanged=digest(PARENT)==gate['baseline_sha256'],small_fit_weights_not_used_for_main=True,robot_connected=False,semantic_cunfa_classification=False);jsave(folder/'training_summary.json',summary);print('TOPOLOGY_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['small','train']);a=p.parse_args();run_small() if a.mode=='small' else run_train()
