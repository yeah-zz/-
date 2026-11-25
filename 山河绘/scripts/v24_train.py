# -*- coding: utf-8 -*-
"""Matched raster control and joint raster/turn/instance-affinity training."""
from v24_sources import ROOT
import v21_stroke_learning as v21
from rock_v15_common import read,jsave,digest
from pathlib import Path
import numpy as np,cv2,torch,math,time,json,shutil
from torch import nn
from torch.nn import functional as F

V21=v21.ROOT

def sample(im,q):
    q=np.asarray(q,np.float32);return cv2.remap(im.astype(np.float32),q[:,0][None],q[:,1][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]

class JointNet(v21.v16.RibbonNet):
    def __init__(self):
        super().__init__();self.node_head=nn.Conv2d(16,1,1);self.embedding_head=nn.Conv2d(16,8,1)
        nn.init.constant_(self.node_head.bias,-2.)
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1));e=self.e(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1));return torch.cat((self.head(e),self.node_head(e),self.embedding_head(e)),1)

def load_pool():
    manifest=read(ROOT/'training_manifest.json');pairs=read(ROOT/'local_relations.json')['records'];groups={k:[] for k in ['new_visible_reference','v21_clear_replay','reviewed_replay']}
    for r in manifest['records']:
        if r['split']!='train':continue
        z=np.load(r['data']);features,alpha,_=v21.v16.photometry(z['rgb']);row=dict(id=r['id'],kind=r['kind'],x=features,t=z['target'],m=z['mask'],node=z['node'],teacher=z['teacher'].astype(np.float32),valid=z['valid'][None],alpha=alpha,pairs=[p for p in pairs if p['split']=='train' and p['image_id']==r['id']]);groups[r['kind']].append(row)
    return manifest,groups

def training_crop(row,rng,pair=None):
    # Every mutation is applied to all raster targets and all point coordinates.
    x,t,m,node,teacher,valid=[row[k].copy() for k in ['x','t','m','node','teacher','valid']];h,w=x.shape[1:];ph=max(0,256-h);pw=max(0,256-w)
    if ph or pw:
        x=np.pad(x,((0,0),(0,ph),(0,pw)),mode='reflect');t=np.pad(t,((0,0),(0,ph),(0,pw)));m=np.pad(m,((0,0),(0,ph),(0,pw)));node=np.pad(node,((0,0),(0,ph),(0,pw)));teacher=np.pad(teacher,((0,0),(0,ph),(0,pw)));valid=np.pad(valid,((0,0),(0,ph),(0,pw)))
    h,w=x.shape[1:]
    if pair is not None:
        focus=np.asarray(pair['center'])+rng.uniform(-35,35,2);left=int(np.clip(focus[0]-128,0,w-256));top=int(np.clip(focus[1]-128,0,h-256));points=np.asarray([pair['p'],pair['q']],np.float32)-[left,top];support=np.zeros((12,2),np.float32);weights=np.zeros(12,np.float32);q=np.asarray(pair['support'],np.float32)
        if len(q):
            n=min(12,len(q));support[:n]=q[:n]-[left,top];weights[:n]=(sample(row['valid'][0],q[:n])>.9)&(sample(row['alpha'],q[:n])>.006)
        label=float(pair['label']);pair_weight=float(((points>=0)&(points<=255)).all())
    else:
        left=int(rng.integers(w-255));top=int(rng.integers(h-255));points=np.zeros((2,2),np.float32);support=np.zeros((12,2),np.float32);weights=np.zeros(12,np.float32);label=0.;pair_weight=0.
    x,t,m,node,teacher,valid=[a[:,top:top+256,left:left+256].copy() for a in [x,t,m,node,teacher,valid]]
    if rng.random()<.5:
        x,t,m,node,teacher,valid=[a[:,:,::-1].copy() for a in [x,t,m,node,teacher,valid]];t[3]*=-1;points[:,0]=255-points[:,0];support[:,0]=255-support[:,0]
    rot=int(rng.integers(4))
    if rot:
        x,t,m,node,teacher,valid=[np.rot90(a,rot,axes=(1,2)).copy() for a in [x,t,m,node,teacher,valid]]
        if rot%2:t[2:4]*=-1
        for _ in range(rot):points=np.c_[points[:,1],255-points[:,0]];support=np.c_[support[:,1],255-support[:,0]]
    return x,t,m,node,teacher,valid,points.astype(np.float32),support.astype(np.float32),weights,np.float32(label),np.float32(pair_weight)

def sample_features(field,points):
    grid=points[:,None]/127.5-1
    return F.grid_sample(field,grid,mode='bilinear',padding_mode='border',align_corners=True)[:,:,0]

def loss_components(o,t,m,node,teacher,valid,points,support,weights,label,pair_weight,joint):
    base=v21.train17.loss_fn(o[:,:5],t,m);unknown=(m[:,:2]==0)*valid;distill=(((o[:,:2].float().sigmoid()-teacher)**2)*unknown).sum()/unknown.sum().clamp_min(1)
    zero=base.new_zeros(());node_loss=zero;affinity=zero;continuity=zero
    if joint:
        b=F.binary_cross_entropy_with_logits(o[:,5].float(),node[:,0],reduction='none');pos=(node[:,0]>.5)*node[:,1];neg=(node[:,0]<=.5)*node[:,1];node_loss=(b*pos).sum()/pos.sum().clamp_min(1)+(b*neg).sum()/neg.sum().clamp_min(1)
        emb=F.normalize(sample_features(o[:,6:14].float(),points),dim=1);cosine=(emb[:,:,0]*emb[:,:,1]).sum(1);positive=(label>.5)*pair_weight;negative=(label<=.5)*pair_weight;affinity=((1-cosine)*positive).sum()/positive.sum().clamp_min(1)+(F.relu(cosine-.2)*negative).sum()/negative.sum().clamp_min(1)
        logits=sample_features(o[:,0:1].float(),support)[:,0];ww=weights*(label*pair_weight)[:,None];continuity=(F.softplus(-logits)*ww).sum()/ww.sum().clamp_min(1)
    total=base+.15*distill+.15*node_loss+.12*affinity+.15*continuity
    return total,dict(raster=base,distillation=distill,node=node_loss,affinity=affinity,continuity=continuity)

def train(variant):
    folder=ROOT/variant;folder.mkdir(exist_ok=True)
    if (folder/'final.pt').exists():raise FileExistsError('Preserve finished model')
    torch.set_num_threads(4);cv2.setNumThreads(2);seed=242610;torch.manual_seed(seed);rng=np.random.default_rng(seed);manifest,pool=load_pool();model=JointNet().cuda();missing=model.load_state_dict(torch.load(manifest['init_weights'],map_location='cuda',weights_only=False)['model'],strict=False)
    if set(missing.missing_keys)!={'node_head.weight','node_head.bias','embedding_head.weight','embedding_head.bias'} or missing.unexpected_keys:raise ValueError('Unexpected parent mapping')
    enc=list(model.a.parameters())+list(model.b.parameters())+list(model.c.parameters());dec=list(model.d.parameters())+list(model.e.parameters())+list(model.head.parameters());aux=list(model.node_head.parameters())+list(model.embedding_head.parameters());lr=[1e-5,4e-5,2e-4];opt=torch.optim.AdamW([dict(params=enc,lr=lr[0]),dict(params=dec,lr=lr[1]),dict(params=aux,lr=lr[2])],weight_decay=1e-4);steps=1200
    config=dict(variant=variant,steps=steps,seed=seed,init_weights=manifest['init_weights'],init_sha256=manifest['init_sha256'],training_manifest_sha256=digest(ROOT/'training_manifest.json'),local_relations_sha256=digest(ROOT/'local_relations.json'),training_ids=[r['id'] for r in manifest['records'] if r['split']=='train'],calibration_ids=[r['id'] for r in manifest['records'] if r['split']=='calibration'],same_sampling_and_raster_loss_as_control=True,new_source_probability=.30,clear_replay_probability=.35,reviewed_replay_probability=.35,pair_focused_crop_probability=.75,rotation_and_flip_applied_to_all_targets=True,loss_weights=dict(distill=.15,node=.15,affinity=.12,continuity=.15),physical_width_calibrated=False,robot_connected=False)
    jsave(folder/'config.json',config);shutil.copy2(Path(__file__),folder/'source_at_start.py');usage={};start=time.perf_counter()
    for step in range(1,steps+1):
        batch=[]
        for _ in range(6):
            chance=rng.random();kind='new_visible_reference' if chance<.30 else 'v21_clear_replay' if chance<.65 else 'reviewed_replay';rows=pool[kind];row=rows[int(rng.integers(len(rows)))];pair=row['pairs'][int(rng.integers(len(row['pairs'])))] if row['pairs'] and rng.random()<.75 else None;batch.append(training_crop(row,rng,pair));usage[row['id']]=usage.get(row['id'],0)+1
        x,t,m,node,teacher,valid,points,support,weights,label,pair_weight=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(11)];factor=.3+.7*.5*(1+math.cos(math.pi*step/steps))
        for g,r in zip(opt.param_groups,lr):g['lr']=r*factor
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=loss_components(model(x),t,m,node,teacher,valid,points,support,weights,label,pair_weight,variant=='joint')
        if not torch.isfinite(loss):raise RuntimeError('Non-finite joint loss')
        loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(grad):raise RuntimeError('Non-finite gradient')
        opt.step()
        if step%100==0:
            row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start,**{k:float(v.detach()) for k,v in parts.items()})
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('V24_TRAIN',variant,json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),config=config),folder/'final.pt');jsave(folder/'training_summary.json',dict(updates=steps,usage=usage,elapsed_seconds=time.perf_counter()-start,checkpoint_sha256=digest(folder/'final.pt'),robot_connected=False));print('V24_TRAINED',variant,flush=True)

def load_model(variant):
    m=JointNet().cuda();m.load_state_dict(torch.load(ROOT/variant/'final.pt',map_location='cuda',weights_only=False)['model']);return m.eval()

@torch.no_grad()
def predict(model,rgb,tile=256,stride=192):
    h,w=rgb.shape[:2];pad=cv2.copyMakeBorder(rgb,20,max(20,tile-h-20),20,max(20,tile-w-20),cv2.BORDER_REFLECT_101);ph,pw=pad.shape[:2];starts=lambda n:sorted(set(list(range(0,n-tile+1,stride))+[n-tile]));loc=[(y,x) for y in starts(ph) for x in starts(pw)];window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32);total=np.zeros((14,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32);features=v21.v16.photometry(pad)[0]
    for start in range(0,len(loc),8):
        group=loc[start:start+8];arr=np.stack([features[:,y:y+tile,x:x+tile] for y,x in group]);o=model(torch.from_numpy(arr).cuda()).float();o[:,[0,1,4,5]]=o[:,[0,1,4,5]].sigmoid();o[:,2:4]=F.normalize(o[:,2:4],dim=1);o[:,6:14]=F.normalize(o[:,6:14],dim=1)
        for (y,x),p in zip(group,o.cpu().numpy()):total[:,y:y+tile,x:x+tile]+=p*window;den[y:y+tile,x:x+tile]+=window
    out=(total/den)[:,20:20+h,20:20+w];out[6:14]/=np.maximum(np.linalg.norm(out[6:14],axis=0,keepdims=True),1e-6);return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('variant',choices=['control','joint']);a=p.parse_args();train(a.variant)
