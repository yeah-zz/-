# -*- coding: utf-8 -*-
"""Hard endpoint pairs with explicit abstention; not real brush-lift truth."""
import sys,json,time,math
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
import numpy as np,torch,cv2
from torch import nn
from scipy.ndimage import distance_transform_edt
from stroke_topology_v13 import ROOT,jsave
from stroke_link_v12 import features,density,dense

def net():return nn.Sequential(nn.Linear(14,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,3))

def samples(seed,n=1500):
    rng=np.random.default_rng(seed);x=[];y=[];strata=[]
    for i in range(n):
        gap=rng.uniform(2.5,7.5);spacing=rng.uniform(2.5,5.8);theta=rng.uniform(-1.2,1.2);u=np.array([math.cos(theta),math.sin(theta)]);v=np.array([-u[1],u[0]]);origin=np.array([64.,64.]);cut=origin-u*gap/2;other=origin+u*gap/2
        a=np.array([cut-u*38,cut]);b=np.array([other,other+u*38]);offset=v*spacing
        lines=[np.array([origin-u*48,origin+u*48]),np.array([origin-u*48+offset,origin+u*48+offset])];rgb=np.full((128,128,3),rng.integers(220,255),np.uint8);center=np.zeros((128,128),np.uint8)
        for line in lines:cv2.polylines(rgb,[np.rint(line).astype(np.int32)],False,(45,45,45),int(rng.integers(1,3)));cv2.polylines(center,[np.rint(line).astype(np.int32)],False,1,1)
        prob=np.exp(-distance_transform_edt(center==0)**2/(2*1.2**2)).astype(np.float32);ink=density(rgb)
        # Positive follows one visible lane; hard negative jumps to a nearby
        # parallel lane while satisfying the older geometric candidate gates.
        x.append(features(dense(a),1,dense(b),0,prob,ink));y.append(0);strata.append('same_visible_lane')
        x.append(features(dense(a),1,dense(b+offset),0,prob,ink));y.append(1);strata.append('different_near_parallel_lane')
        # With no visible bridge, a short collinear gap can represent either
        # a dry stroke or a true lift. Do not invent a uniquely correct lift.
        ambiguous_rgb=rgb.copy();cv2.line(ambiguous_rgb,tuple(np.rint(cut).astype(int)),tuple(np.rint(other).astype(int)),tuple(int(z) for z in rgb[0,0]),4)
        ambiguous_ink=density(ambiguous_rgb);amb_prob=prob.copy();cv2.line(amb_prob,tuple(np.rint(cut).astype(int)),tuple(np.rint(other).astype(int)),float(rng.uniform(.15,.7)),2)
        x.append(features(dense(a),1,dense(b),0,amb_prob,ambiguous_ink));y.append(2);strata.append('observationally_ambiguous_gap')
    return np.asarray(x,np.float32),np.asarray(y,np.int64),strata

def admissible(x):return (x[:,1]>=.6)&(np.minimum(x[:,2],x[:,3])>=.45)&((x[:,12]>=.22)|(x[:,0]<=.17))

class SafeScorer(nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def forward(self,x):
        p=self.base(x).softmax(-1);score=(p[:,0]*(1-p[:,2])).clamp(1e-6,1-1e-6);return torch.logit(score)[:,None]

def train():
    folder=ROOT/'pairs';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_summary.json').exists():raise FileExistsError('Preserve pair training')
    x,y,_=samples(810000,1800);vx,vy,_=samples(820000,600);tx,ty,strata=samples(830000,600);assert int(((vy==1)&admissible(vx)).sum())>=300
    torch.manual_seed(1324);torch.set_num_threads(4);model=net();opt=torch.optim.AdamW(model.parameters(),lr=.0015);rng=np.random.default_rng(1324);start=time.perf_counter()
    for step in range(800):
        idx=np.concatenate([rng.choice(np.flatnonzero(y==c),96) for c in range(3)]);opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(torch.from_numpy(x[idx])),torch.from_numpy(y[idx]));loss.backward();opt.step()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite pair loss')
    model.eval();scorer=SafeScorer(model)
    with torch.no_grad():vs=scorer(torch.from_numpy(vx))[:,0].sigmoid().numpy();ts=scorer(torch.from_numpy(tx))[:,0].sigmoid().numpy();class_probs=model(torch.from_numpy(tx)).softmax(-1).numpy()
    allowed=admissible(vx);positive=(vy==0)&allowed;negative=(vy==1)&allowed;ambiguous=(vy==2)&allowed;choices=[]
    for threshold in np.arange(.7,.999,.005):
        fp=float(np.mean(vs[negative]>=threshold));unsafe=float(np.mean(vs[ambiguous]>=threshold));tp=float(np.mean(vs[positive]>=threshold))
        if fp<=.01 and unsafe<=.01:choices.append((tp,-threshold,threshold))
    threshold=max(choices)[2] if choices else 1.001;torch.save(dict(model=model.state_dict(),threshold=threshold,classes=['same','different','uncertain']),folder/'final.pt')
    gate=admissible(tx);result={}
    for c,name in enumerate(('same','different','uncertain')):
        mask=(ty==c)&gate;result[name]=dict(admissible_pairs=int(mask.sum()),automatic_join_fraction=float(np.mean(ts[mask]>=threshold)),classification_accuracy=float(np.mean(class_probs[mask].argmax(1)==c)))
    summary=dict(successful_updates=800,training_pairs=len(y),calibration_pairs=len(vy),unseen_test_pairs=len(ty),calibration_admissible_negative_pairs=int(negative.sum()),threshold=float(threshold),unseen_test=result,elapsed_seconds=time.perf_counter()-start,all_data_procedural=True,real_artwork_accuracy_not_established=True,ambiguous_pairs_not_treated_as_known_brush_lifts=True)
    jsave(folder/'training_summary.json',summary);np.savez_compressed(folder/'unseen_pair_scores.npz',features=tx,targets=ty,scores=ts,admissible=gate);print('HARD_PAIRS_TRAINED',json.dumps(summary),flush=True)

def load():
    z=torch.load(ROOT/'pairs'/'final.pt',map_location='cpu',weights_only=False);m=net();m.load_state_dict(z['model']);return SafeScorer(m.eval()),float(z['threshold'])
if __name__=='__main__':train()
