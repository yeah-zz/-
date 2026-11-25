# -*- coding: utf-8 -*-
"""Real reviewed-stroke pair CNN. References are forbidden at inference."""
from rock_v15_common import *
import math,time
import numpy as np,torch,cv2
from torch import nn
from scipy.spatial import cKDTree
from PIL import Image
from stroke_link_v12 import dense,features,density,tangent
from gongbi_training.stroke_train_v08 import model_new,infer
from human_mountain_v11 import fields
from human_mountain_eval_v11 import extract
from topology_trace_v13 import graph_paths,adapt_old

class PairNet(nn.Module):
    def __init__(self):
        super().__init__();self.image=nn.Sequential(nn.Conv2d(4,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,24,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(24,32,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d((2,2)),nn.Flatten());self.geometry=nn.Sequential(nn.Linear(14,32),nn.SiLU());self.head=nn.Sequential(nn.Linear(160,48),nn.SiLU(),nn.Linear(48,1))
    def forward(self,patch,feat):return self.head(torch.cat((self.image(patch),self.geometry(feat)),1))[:,0]

def pair_inputs(a,ea,b,eb,p,ink):
    feat=features(a,ea,b,eb,p,ink);aa=a[0 if ea==0 else -1];bb=b[0 if eb==0 else -1];mid=(aa+bb)/2
    # Canonical local coordinates along the gap, 40 native pixels wide.
    delta=bb-aa;d=np.linalg.norm(delta);u=delta/max(d,1e-6)
    if d<.5:u=tangent(a,ea==0)
    v=np.array([-u[1],u[0]]);yy,xx=np.mgrid[:32,:32].astype(np.float32);xx=(xx-15.5)*1.25;yy=(yy-15.5)*1.25;q=mid[None,None]+xx[:,:,None]*u+yy[:,:,None]*v;mx=q[:,:,0].astype(np.float32);my=q[:,:,1].astype(np.float32)
    p0=cv2.remap(p.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);i0=cv2.remap(ink.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    def mark(pt):return np.exp(-((q[:,:,0]-pt[0])**2+(q[:,:,1]-pt[1])**2)/8)
    return np.stack((p0,i0,mark(aa),mark(bb))).astype(np.float32),feat

def probability(rgb,path):
    if path.exists():return np.load(path)['prediction']
    model=model_new().cuda();model.load_state_dict(torch.load(PARENT,map_location='cuda',weights_only=False)['model']);model.eval();p=infer(model,rgb);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,prediction=p);del model;return p

def reference_records():
    out=[]
    for r in read(PREVIOUS/'manifest.json')['records']:
        out.append(dict(id=r['id'],image=r['image'],labels=r['labels'],split='train' if r['id']=='R01' else 'calibration' if r['id']=='R02' else 'development',family='heye' if r['id'] in ('R01','R02') else 'pima',native=r['native_content'],label_provenance='user_reviewed_replacement_corrected'))
    for r in read(LAB/'human_mountain_v11_20261024'/'manifest.json')['records']:
        if r['id'] not in ['A01','A02','A03','A04','A05']:continue
        out.append(dict(id=r['id'],image=r['snapshot_image'],labels=r['snapshot_labels'],split='calibration' if r['id']=='A05' else 'train',family='qianli_jiangshan',label_provenance='user_reviewed_partial'))
    return out

def examples(paths,p,rgb,seed,cuts=12):
    rng=np.random.default_rng(seed);ink=density(rgb);pieces=[];ids=[];positive=[];negative=[]
    for identity,path in enumerate(paths):
        q=dense(path)
        if len(q)<26:continue
        for _ in range(cuts):
            k=int(rng.integers(9,len(q)-10));gap=int(rng.integers(1,min(13,len(q)-k-2)));a=q[max(0,k-36):k+1];b=q[k+gap:min(len(q),k+gap+37)]
            if len(a)<4 or len(b)<4:continue
            patch,f=pair_inputs(a,1,b,0,p,ink);positive.append((patch,f,1));pieces.extend((a,b));ids.extend((identity,identity))
    if pieces:
        ep=np.array([q[k] for q in pieces for k in (0,-1)]);pairs=list(cKDTree(ep).query_pairs(16));rng.shuffle(pairs)
        for ai,bi in pairs:
            ia,ea=divmod(ai,2);ib,eb=divmod(bi,2)
            if ids[ia]==ids[ib]:continue
            patch,f=pair_inputs(pieces[ia],ea,pieces[ib],eb,p,ink)
            # Include geometrically tempting neighboring strokes, not only
            # trivial opposite-direction negatives. Identity comes from user.
            if f[1]>.15 and min(f[2],f[3])>-.15:negative.append((patch,f,0))
            if len(negative)>=max(200,len(positive)*3):break
    return positive+negative

def prepare():
    folder=ROOT/'pairs';folder.mkdir(parents=True,exist_ok=True);records=reference_records();data={};audit=[]
    for i,r in enumerate(records):
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=read(r['labels']);p=probability(rgb,ROOT/'probabilities'/(r['id']+'.npz'));rows=examples([s['points'] for s in doc['strokes']],p[0],rgb,152500+i,18 if r['split']=='train' else 10)
        if not rows:continue
        patch,feat,y=zip(*rows);path=folder/(r['id']+'.npz');np.savez_compressed(path,patch=np.asarray(patch,np.float16),features=np.asarray(feat,np.float32),targets=np.asarray(y,np.float32));audit.append(dict(**r,image_sha256=digest(r['image']),label_sha256=digest(r['labels']),pairs=len(y),positive=int(sum(y)),negative=int(len(y)-sum(y)),reference_strokes=len(doc['strokes']),data=str(path)))
        print('REVIEW_PAIR_DATA',r['id'],len(y),int(sum(y)),flush=True)
    jsave(folder/'manifest.json',dict(records=audit,unmarked_never_negative=True,stroke_ids_are_desired_robot_strokes_not_historical_pen_lifts=True,shared_artwork_calibration_not_independent=True));return audit

def load_data(records,split):
    sets=[np.load(r['data']) for r in records if r['split']==split];return [np.concatenate([z[k] for z in sets]) for k in ['patch','features','targets']]

@torch.no_grad()
def scores(model,arrays):
    out=[]
    for start in range(0,len(arrays[2]),256):out.append(model(torch.from_numpy(arrays[0][start:start+256]).cuda().float(),torch.from_numpy(arrays[1][start:start+256]).cuda()).sigmoid().cpu().numpy())
    return np.concatenate(out)

def calibration(model,arrays):
    score=scores(model,arrays);y=arrays[2];f=arrays[1];admissible=(f[:,0]<=1.34)&(f[:,1]>.25)&(np.minimum(f[:,2],f[:,3])>-.05);pos=(y==1)&admissible;neg=(y==0)&admissible
    if int(neg.sum())<100:threshold=1.001
    else:
        choices=[]
        for t in np.arange(.7,.999,.003):
            fp=float((score[neg]>=t).mean());tp=float((score[pos]>=t).mean())
            # Conservative per-pair operating point, not an artwork accuracy.
            if fp<=.01:choices.append((tp,-t,t))
        threshold=max(choices)[2] if choices else 1.001
    return float(threshold),dict(admissible_positive=int(pos.sum()),admissible_negative=int(neg.sum()),same_cut_recall=float((score[pos]>=threshold).mean()) if pos.any() else None,different_stroke_false_join_fraction=float((score[neg]>=threshold).mean()) if neg.any() else None,disabled_if_insufficient_calibration=bool(threshold>1))

def train(round_index):
    folder=ROOT/f'round_{round_index:02d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_summary.json').exists():raise FileExistsError('Preserve completed round')
    records=read(ROOT/'pairs'/'manifest.json')['records'];train=load_data(records,'train');valid=load_data(records,'calibration');torch.set_num_threads(4);torch.manual_seed(152500+round_index);rng=np.random.default_rng(152500+round_index);model=PairNet().cuda()
    if (folder/'pair_net.pt').exists():
        checkpoint=torch.load(folder/'pair_net.pt',map_location='cuda',weights_only=False);model.load_state_dict(checkpoint['model']);model.eval();threshold,cal=calibration(model,valid);assert abs(threshold-checkpoint['threshold'])<1e-6
        summary=dict(round=round_index,successful_updates=checkpoint['steps'],reference_pair_samples=len(train[2]),auto_pseudo_pair_samples=0,threshold=threshold,calibration=cal,center_network_frozen=True,parent_sha256=digest(PARENT),recovered_completed_checkpoint_after_report_serialization_error=True,independent_artwork_accuracy=False);jsave(folder/'training_summary.json',summary);print('RECOVERED_COMPLETED_PAIR_ROUND',json.dumps(summary),flush=True);return
    if round_index>1:model.load_state_dict(torch.load(ROOT/f'round_{2 if round_index==4 else round_index-1:02d}'/'pair_net.pt',map_location='cuda',weights_only=False)['model'])
    pseudo_samples=0
    additional_real=0
    if round_index>=3 and (ROOT/'additional_real_pairs.npz').exists():
        z=np.load(ROOT/'additional_real_pairs.npz');additional_real=len(z['targets']);train=[np.concatenate((a,z[k])) for a,k in zip(train,['patch','features','targets'])]
    if round_index>=3 and (ROOT/'pseudo_pairs.npz').exists():
        z=np.load(ROOT/'pseudo_pairs.npz');pseudo_samples=len(z['targets']);train=[np.concatenate((a,z[k])) for a,k in zip(train,['patch','features','targets'])]
    opt=torch.optim.AdamW(model.parameters(),lr=.0007 if round_index==1 else .00025,weight_decay=.0001);start=time.perf_counter();n=len(train[2]);positive=np.flatnonzero(train[2]==1);negative=np.flatnonzero(train[2]==0)
    if min(len(positive),len(negative))<100:raise RuntimeError('insufficient real positive/negative samples')
    hard_negative=negative
    if round_index>1:
        ss=scores(model,train);hard_negative=negative[np.argsort(ss[negative])[-max(100,len(negative)//3):]]
    steps=650
    for step in range(1,steps+1):
        model.train();idx=np.concatenate((rng.choice(positive,64),rng.choice(negative,32),rng.choice(hard_negative,32)));rng.shuffle(idx);patch=torch.from_numpy(train[0][idx]).cuda().float();feat=torch.from_numpy(train[1][idx]).cuda();target=torch.from_numpy(train[2][idx]).cuda()
        if round_index>1:
            patch[:,1]*=float(rng.uniform(.8,1.2));patch[:,:2]=(patch[:,:2]+torch.randn_like(patch[:,:2])*.012).clamp(0,1)
        opt.zero_grad(set_to_none=True);out=model(patch,feat);loss=nn.functional.binary_cross_entropy_with_logits(out,target)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite pair loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%100==0:
            row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start)
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('PAIR_CNN_ROUND',round_index,json.dumps(row),flush=True)
    model.eval();threshold,cal=calibration(model,valid);torch.save(dict(model=model.state_dict(),threshold=threshold,round=round_index,steps=steps,architecture='three-convolution local patch + geometry MLP',raster_parent_sha256=digest(PARENT)),folder/'pair_net.pt');summary=dict(round=round_index,successful_updates=steps,reference_pair_samples=n-pseudo_samples,additional_actual_arcs_and_assistant_pairs=additional_real,auto_pseudo_pair_samples=pseudo_samples,pseudo_dataset_sha256=digest(ROOT/'pseudo_pairs.npz') if pseudo_samples else None,initialization_round=2 if round_index==4 else round_index-1,threshold=threshold,calibration=cal,center_network_frozen=True,parent_sha256=digest(PARENT),elapsed_seconds=time.perf_counter()-start,independent_artwork_accuracy=False);jsave(folder/'training_summary.json',summary);print('PAIR_ROUND_COMPLETE',json.dumps(summary),flush=True)

def load(round_index):
    z=torch.load(ROOT/f'round_{round_index:02d}'/'pair_net.pt',map_location='cuda',weights_only=False);m=PairNet().cuda();m.load_state_dict(z['model']);return m.eval(),z['threshold']

def connect(paths,p,rgb,model,threshold,max_gap=13):
    paths=[dense(q) for q in paths if len(q)>1];ink=density(rgb);events=[];uncertain=[]
    for iteration in range(5):
        if len(paths)<2:break
        ep=np.array([q[k] for q in paths for k in (0,-1)]);pairs=[];pp=[];ff=[]
        for ai,bi in sorted(cKDTree(ep).query_pairs(max_gap)):
            ia,ea=divmod(ai,2);ib,eb=divmod(bi,2)
            if ia==ib or np.linalg.norm(paths[ia][0]-paths[ia][-1])<2 or np.linalg.norm(paths[ib][0]-paths[ib][-1])<2:continue
            patch,f=pair_inputs(paths[ia],ea,paths[ib],eb,p,ink)
            if f[1]<.25 or min(f[2],f[3])<-.05:continue
            pairs.append((ai,bi,ia,ea,ib,eb));pp.append(patch);ff.append(f)
        if not pairs:break
        with torch.no_grad():s=model(torch.from_numpy(np.asarray(pp)).cuda(),torch.from_numpy(np.asarray(ff)).cuda()).sigmoid().cpu().numpy()
        by={}
        for i,(ai,bi,*_) in enumerate(pairs):
            by.setdefault(ai,[]).append((float(s[i]),i));by.setdefault(bi,[]).append((float(s[i]),i))
        candidates=[]
        for i,(ai,bi,*_) in enumerate(pairs):
            aa=sorted(by[ai],reverse=True);bb=sorted(by[bi],reverse=True);mutual=aa[0][1]==i and bb[0][1]==i;margin=min(aa[0][0]-aa[1][0] if len(aa)>1 else 1,bb[0][0]-bb[1][0] if len(bb)>1 else 1)
            if s[i]>=threshold and mutual and margin>=.08:candidates.append((float(s[i]),i))
            elif s[i]>.35 and iteration==0:uncertain.append(dict(endpoint_a=ep[ai].round(2).tolist(),endpoint_b=ep[bi].round(2).tolist(),score=float(s[i]),reason='abstain_or_competing_join'))
        used=set();merged=[]
        for score,i in sorted(candidates,reverse=True):
            _,_,ia,ea,ib,eb=pairs[i]
            if ia in used or ib in used:continue
            a=paths[ia][::-1] if ea==0 else paths[ia];b=paths[ib][::-1] if eb==1 else paths[ib];bridge=np.linspace(a[-1],b[0],max(2,math.ceil(np.linalg.norm(a[-1]-b[0])*2)+1));merged.append(np.concatenate((a,bridge[1:-1],b)));used.update([ia,ib]);events.append(dict(iteration=iteration,score=score,bridge=bridge.round(2).tolist(),status='automatic_connection_hypothesis'))
        if not used:break
        paths=merged+[q for i,q in enumerate(paths) if i not in used]
    return paths,events,uncertain

def frozen_paths(pred):
    sk,_=extract(pred);paths,info=graph_paths(sk>0,adapt_old(pred),False,min_length=3);return paths

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','1','2','3','4']);args=parser.parse_args();prepare() if args.mode=='prepare' else train(int(args.mode))
