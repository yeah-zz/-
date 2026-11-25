# -*- coding: utf-8 -*-
"""Learn endpoint affinity; no reference path identities are used at inference."""
import sys,math,json,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
import cv2,numpy as np,torch
from torch import nn
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from PIL import Image
from clean_strokes_v12 import ROOT,jsave

def dense(p,step=1):
    p=np.asarray(p,float);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];keep=np.r_[True,np.diff(arc)>.01];p=p[keep];arc=arc[keep]
    if len(p)<2:return p
    u=np.linspace(0,arc[-1],max(2,math.ceil(arc[-1]/step)+1));return np.stack([np.interp(u,arc,p[:,k]) for k in (0,1)],1)
def tangent(p,start=False,span=6):
    q=p if start else p[::-1];dist=np.linalg.norm(q-q[0],axis=1);k=min(len(q)-1,max(1,int(np.searchsorted(dist,span))));v=q[0]-q[k];return v/max(1e-6,np.linalg.norm(v))
def density(rgb):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;return np.clip((cv2.GaussianBlur(g,(0,0),4)-g)*5,0,1)
def sample(im,q):
    return cv2.remap(im.astype(np.float32),q[:,0].astype(np.float32)[None],q[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]
def features(a,ea,b,eb,prob,ink):
    aa=a[0 if ea==0 else -1];bb=b[0 if eb==0 else -1];va=tangent(a,ea==0);vb=tangent(b,eb==0);delta=bb-aa;d=float(np.linalg.norm(delta));u=delta/max(d,1e-6)
    if d<.25:u=(va-vb)/max(1e-6,np.linalg.norm(va-vb))
    q=aa[None]+np.linspace(0,1,11)[:,None]*delta;ev=sample(ink,q);pp=sample(prob,q);la=np.linalg.norm(np.diff(a,axis=0),axis=1).sum();lb=np.linalg.norm(np.diff(b,axis=0),axis=1).sum()
    return np.array([d/12,float(va@(-vb)),float(va@u),float((-vb)@u),abs(va[0]*u[1]-va[1]*u[0]),abs(vb[0]*u[1]-vb[1]*u[0]),float(va@tangent(a,ea==0,12)),float(vb@tangent(b,eb==0,12)),math.log1p(min(la,lb))/6,math.log1p(max(la,lb))/6,float(ev.mean()),float((ev>.025).mean()),float(pp.mean()),float(np.quantile(pp,.15))],np.float32)
def net():return nn.Sequential(nn.Linear(14,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,1))

def synthetic(seed,size=320):
    rng=np.random.default_rng(seed);u=np.linspace(0,1,220);paths=[];kind=seed%5
    if kind==0:
        for k in range(5):paths.append(np.stack((50+k*43+np.sin(u*2.5)*rng.uniform(5,16),35+u*245),1))
    elif kind==1:
        paths=[np.stack((30+u*260,90+u*130+np.sin(u*math.pi)*15),1),np.stack((30+u*260,230-u*150+np.sin(u*math.pi)*12),1)]
    elif kind==2:
        paths=[np.stack((40+u*245,75+np.sin(u*2.6)*20),1),np.stack((75+u*180,150+np.sin(u*2.5)*25),1),np.stack((110+u*140,230+np.sin(u*2)*15),1)]
    elif kind==3:
        paths=[np.stack((60+u*210,270-210*np.sin(u*math.pi)),1)]
        for k in range(4):paths.append(np.stack((100+k*32+u*15,140+u*125),1))
    else:
        paths=[np.stack((40+u*240,100+np.sin(u*3.8)*55),1),np.stack((50+u*240,205+np.sin(u*3.8)*55),1)]
    theta=rng.uniform(-.7,.7);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);paths=[np.clip((p-size/2)@rot.T+size/2,12,size-13) for p in paths]
    center=np.zeros((size,size),np.uint8);alpha=np.zeros((size,size),np.float32)
    for p in paths:
        xy=np.rint(p).astype(np.int32);cv2.polylines(center,[xy],False,1,1);layer=np.zeros_like(alpha);cv2.polylines(layer,[xy],False,float(rng.uniform(.3,.85)),int(rng.integers(1,6)));alpha=np.maximum(alpha,layer)
    dist=distance_transform_edt(center==0);prob=np.exp(-dist**2/(2*1.1**2)).astype(np.float32);prob=np.clip(prob+rng.normal(0,.018,prob.shape),0,1)
    # Some physical ink gaps, while the reference trajectory stays continuous.
    if seed%3==0:
        p=paths[0][100];cv2.circle(alpha,tuple(np.rint(p).astype(int)),2,0,-1)
    paper=np.ones((size,size,3),np.float32)*rng.uniform(215,254,(1,1,3));rgb=np.clip(paper*(1-alpha[:,:,None])+rng.normal(0,.8,(size,size,1)),0,255).astype(np.uint8)
    return rgb,paths,prob

def pair_data(paths,prob,rgb,seed):
    rng=np.random.default_rng(seed);ink=density(rgb);pieces=[];ids=[];features_list=[];targets=[]
    for identity,p in enumerate(paths):
        p=dense(p)
        if len(p)<25:continue
        # Multiple artificial cuts of a known reference path teach same-stroke
        # continuity; this does not claim the original artist's pen lifts.
        for _ in range(3):
            k=int(rng.integers(9,len(p)-9));gap=int(rng.integers(1,min(9,len(p)-k-1)));a=p[max(0,k-45):k+1];b=p[k+gap:min(len(p),k+gap+46)]
            if len(a)<3 or len(b)<3:continue
            features_list.append(features(a,1,b,0,prob,ink));targets.append(1.);pieces.extend((a,b));ids.extend((identity,identity))
    if pieces:
        endpoints=np.array([p[k] for p in pieces for k in (0,-1)]);tree=cKDTree(endpoints)
        candidates=list(tree.query_pairs(15));rng.shuffle(candidates)
        for a,b in candidates:
            ia,ea=divmod(a,2);ib,eb=divmod(b,2)
            if ids[ia]==ids[ib]:continue
            features_list.append(features(pieces[ia],ea,pieces[ib],eb,prob,ink));targets.append(0.)
            if sum(v==0 for v in targets)>=max(8,2*sum(v==1 for v in targets)):break
    return features_list,targets

def train(real_records):
    if (ROOT/'link_training_summary.json').exists():raise FileExistsError('Preserve link run')
    rows=[];ys=[]
    for i in range(160):
        rgb,paths,p=synthetic(510000+i);x,y=pair_data(paths,p,rgb,910000+i);rows+=x;ys+=y
    for i,r in enumerate(real_records):
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=json.loads(Path(r['labels']).read_text(encoding='utf-8'));paths=[s['points'] for s in doc['strokes']];p=np.load(r['prediction'])['after'][0];x,y=pair_data(paths,p,rgb,940000+i);rows+=x;ys+=y
    train_x=np.asarray(rows,np.float32);train_y=np.asarray(ys,np.float32);valid_x=[];valid_y=[]
    for i in range(35):
        rgb,paths,p=synthetic(520000+i);x,y=pair_data(paths,p,rgb,920000+i);valid_x+=x;valid_y+=y
    valid_x=np.asarray(valid_x,np.float32);valid_y=np.asarray(valid_y,np.float32);torch.manual_seed(1212);model=net();opt=torch.optim.AdamW(model.parameters(),lr=.002);rng=np.random.default_rng(1212);pos=np.flatnonzero(train_y>0);neg=np.flatnonzero(train_y==0);start=time.perf_counter()
    for step in range(500):
        index=np.r_[rng.choice(pos,128),rng.choice(neg,128)];x=torch.from_numpy(train_x[index]);y=torch.from_numpy(train_y[index]);opt.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(model(x)[:,0],y);loss.backward();opt.step()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite link loss')
    with torch.no_grad():scores=model(torch.from_numpy(valid_x))[:,0].sigmoid().numpy()
    admissible=(valid_x[:,1]>=.6)&(np.minimum(valid_x[:,2],valid_x[:,3])>=.45)&((valid_x[:,12]>=.22)|(valid_x[:,0]<=.17))
    vpos=(valid_y>0)&admissible;vneg=(valid_y==0)&admissible
    if not vpos.any() or not vneg.any():raise RuntimeError('No valid hard-pair calibration set')
    choices=[]
    for threshold in np.arange(.7,.999,.01):
        fp=float(np.mean(scores[vneg]>=threshold));tp=float(np.mean(scores[vpos]>=threshold))
        if fp<=.01:choices.append((tp,-threshold,threshold,fp))
    best=max(choices) if choices else (0,-1.001,1.001,0);threshold=float(best[2]);torch.save(dict(model=model.state_dict(),threshold=threshold,features=14),ROOT/'link_model.pt')
    report=dict(successful_updates=500,real_reference_images=len(real_records),procedural_training_scenes=160,procedural_validation_scenes=35,pair_samples=len(train_y),positive_pairs=int((train_y>0).sum()),negative_pairs=int((train_y==0).sum()),threshold=threshold,validation_admissible_positive_pairs=int(vpos.sum()),validation_admissible_negative_pairs=int(vneg.sum()),validation_same_path_recall=float(best[0]),validation_different_path_false_join_rate=float(best[3]),validation_is_procedural_not_real_accuracy=True,elapsed_seconds=time.perf_counter()-start)
    jsave(ROOT/'link_training_summary.json',report);print('LINK_TRAINED',json.dumps(report),flush=True);return model.eval(),threshold

def load():
    d=torch.load(ROOT/'link_model.pt',map_location='cpu',weights_only=False);model=net();model.load_state_dict(d['model']);return model.eval(),float(d['threshold'])

def connect(paths,prob,rgb,model,threshold,max_gap=8):
    paths=[dense(p) for p in paths if len(p)>=2];ink=density(rgb);events=[];h,w=prob.shape
    for iteration in range(6):
        if len(paths)<2:break
        endpoints=np.array([p[k] for p in paths for k in (0,-1)]);tree=cKDTree(endpoints);pairs=[];fs=[]
        for a,b in sorted(tree.query_pairs(max_gap)):
            ia,ea=divmod(a,2);ib,eb=divmod(b,2)
            if ia==ib:continue
            feat=features(paths[ia],ea,paths[ib],eb,prob,ink)
            if feat[1]<.6 or min(feat[2],feat[3])<.45:continue
            if feat[12]<.22 and feat[0]>.17:continue
            pairs.append((a,b,ia,ea,ib,eb));fs.append(feat)
        if not pairs:break
        with torch.no_grad():scores=model(torch.from_numpy(np.asarray(fs,np.float32)))[:,0].sigmoid().numpy()
        by_end={}
        for i,(a,b,*_) in enumerate(pairs):
            by_end.setdefault(a,[]).append((float(scores[i]),i));by_end.setdefault(b,[]).append((float(scores[i]),i))
        selected=[]
        for i,(a,b,*_) in enumerate(pairs):
            if scores[i]<threshold:continue
            aa=sorted(by_end[a],reverse=True);bb=sorted(by_end[b],reverse=True)
            if aa[0][1]!=i or bb[0][1]!=i:continue
            if (len(aa)>1 and aa[0][0]-aa[1][0]<.08) or (len(bb)>1 and bb[0][0]-bb[1][0]<.08):continue
            selected.append((float(scores[i]),i))
        used=set();merged=[]
        for score,i in sorted(selected,reverse=True):
            _,_,ia,ea,ib,eb=pairs[i]
            if ia in used or ib in used:continue
            a=paths[ia][::-1] if ea==0 else paths[ia];b=paths[ib][::-1] if eb==1 else paths[ib];gap=np.linalg.norm(a[-1]-b[0]);bridge=np.linspace(a[-1],b[0],max(2,math.ceil(gap*2)+1));merged.append(np.concatenate((a,bridge[1:-1],b)));used.update((ia,ib));events.append(dict(iteration=iteration,score=score,gap_pixels=float(gap),bridge=bridge.round(2).tolist(),status='learned_connection_hypothesis_not_verified_human_pen_lift'))
        if not used:break
        paths=merged+[p for k,p in enumerate(paths) if k not in used]
    return paths,events
