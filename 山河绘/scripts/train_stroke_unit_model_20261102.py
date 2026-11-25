# -*- coding: utf-8 -*-
"""Train a new relationship module from known vectors and partial user curves.

Labels: 0 continuous fragments; 1 same rendered ink-band duplicate; 2 separate.
Synthetic counts are not real paintings. Original image extractor stays frozen.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import hashlib,json,time
import cv2,numpy as np,torch
from torch import nn
from stroke_unit_core_20261102 import pair_features,resample,FEATURES

ROOT=LAB/'stroke_unit_model_20261102'


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


class RelationNet(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(len(FEATURES),96),nn.SiLU(),
            nn.Linear(96,96),nn.SiLU(),nn.Dropout(.05),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,3))
    def forward(self,x):return self.net(x)


def curve(start,angle,length,turn):
    t=np.linspace(0,1,32);u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-u[1],u[0]])
    return np.asarray(start)+t[:,None]*length*u+(t*t)[:,None]*turn*v


def synthetic(count,seed):
    rng=np.random.default_rng(seed);x=[];y=[]
    for i in range(count):
        label=i%3;angle=rng.uniform(-np.pi,np.pi);u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-u[1],u[0]])
        center=np.array([64.,64.]);width=int(rng.integers(1,8));lp,lq=rng.uniform(5,38,2)
        bg=rng.uniform(.72,.98);gray=np.full((128,128),bg,np.float32)
        ink=float(rng.uniform(.04,.38))
        if label==0:
            gap=rng.uniform(0,3.) if rng.random()<.6 else 0.
            p=curve(center-u*gap/2,angle+np.pi,lp,rng.normal(0,lp*.12))
            q=curve(center+u*gap/2,angle+rng.normal(0,.28),lq,rng.normal(0,lq*.12))
            strokes=[p,q,np.array([p[0],q[0]])]
        elif label==1:
            p=curve(center-u*lp/2,angle,lp,rng.normal(0,lp*.12))
            offset=rng.uniform(-min(1.2,width*.3),min(1.2,width*.3))
            q=p+v*offset+rng.normal(0,.06,p.shape)
            if rng.random()<.5:q=q[::-1]
            strokes=[p]
        else:
            if rng.random()<.6:
                p=curve(center-u*lp/2,angle,lp,rng.normal(0,lp*.08))
                q=p+v*rng.uniform(width+1.,width+5.)
                if rng.random()<.5:q=q[::-1]
            else:
                p=curve(center,angle,lp,rng.normal(0,lp*.1))
                q=curve(center+rng.normal(0,1.,2),angle+rng.choice([-1,1])*rng.uniform(.4,2.4),lq,rng.normal(0,lq*.1))
            strokes=[p,q]
        for path in strokes:cv2.polylines(gray,[np.rint(path).astype(np.int32)],False,ink,width,cv2.LINE_AA)
        gray=cv2.GaussianBlur(gray,(0,0),float(rng.uniform(.2,.65)))
        gray=np.clip(gray+rng.normal(0,.005,gray.shape),0,1).astype(np.float32)
        x.append(pair_features(p,q,gray));y.append(label)
        if (i+1)%2000==0:print('synthetic',seed,i+1,flush=True)
    return np.stack(x),np.array(y,np.int64)


def human_augmented(row,count,seed):
    """Only explicitly user-drawn s_* curves. Drafts and unmarked pixels excluded."""
    d=json.loads(Path(row['labels']).read_text('utf-8'));rgb=cv2.imread(row['image'])
    h,w=rgb.shape[:2];factor=496/max(h,w);gray=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),None,fx=factor,fy=factor).astype(np.float32)/255
    curves=[resample(np.array(s['points'],float)*factor,.8,240) for s in d['strokes'] if s['id'].startswith('s_') and len(s.get('points',[]))>=3]
    rng=np.random.default_rng(seed);x=[];y=[]
    for i in range(count):
        label=i%3;p=curves[int(rng.integers(len(curves)))];cut=int(rng.integers(2,max(3,len(p)-2)))
        p0=p[max(0,cut-16):cut+1][::-1];q0=p[cut:min(len(p),cut+17)]
        if min(len(p0),len(q0))<2:continue
        if label==0:a,b=p0,q0
        elif label==1:
            a=p0;b=a+rng.normal(0,.1,a.shape)
            if rng.random()<.5:b=b[::-1]
        else:
            other=curves[int(rng.integers(len(curves)))];ix=int(np.argmin(np.linalg.norm(other-p0[0],axis=1)))
            a=p0;b=other[max(0,ix-8):min(len(other),ix+9)]
            if len(b)<2 or np.mean(np.linalg.norm(a[:,None]-b[None],axis=2).min(axis=1))<1.5:continue
        x.append(pair_features(a,b,gray));y.append(label)
    return np.stack(x),np.array(y,np.int64),len(curves)


@torch.no_grad()
def probabilities(model,x):
    out=[];model.eval()
    for start in range(0,len(x),2048):out.append(model(torch.from_numpy(x[start:start+2048]).cuda()).softmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,3),np.float32)


def binary_stats(p,y,label,threshold):
    pred=p[:,label]>=threshold;gold=y==label;tp=int((pred&gold).sum());fp=int((pred&~gold).sum());fn=int((~pred&gold).sum())
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),threshold=float(threshold))


def train():
    ROOT.mkdir(exist_ok=True);cv2.setNumThreads(2);torch.set_num_threads(4)
    torch.manual_seed(1102);np.random.seed(1102)
    manifest=json.loads((LAB/'reviewed_rocks_v14_20261024'/'corrected'/'manifest.json').read_text('utf-8'))['records']
    records={r['id']:r for r in manifest}
    sets={}
    for split,n,seed in [('train',12000,110201),('calibration',2400,110202),('test',3000,110203)]:
        x,y=synthetic(n,seed);sets[split]=(x,y)
    user_counts=[]
    for ident,split,count in [('R01','train',900),('R02','train',900),('R03','calibration',600),('R04','test',600)]:
        row=records[ident];x,y,curves=human_augmented(row,count,110200+int(ident[1:]));key='partial_'+split
        if key in sets:sets[key]=(np.concatenate([sets[key][0],x]),np.r_[sets[key][1],y])
        else:sets[key]=(x,y)
        user_counts.append(dict(id=ident,split=split,user_drawn_curves=curves,augmented_pairs=len(y),
            label_file=row['labels'],sha256=hashlib.sha256(Path(row['labels']).read_bytes()).hexdigest(),
            source_group=row['source_group'],source_page=row['source_page']))
    np.savez_compressed(ROOT/'training_data.npz',**{k+'_'+suffix:value for k,(x,y) in sets.items() for suffix,value in [('x',x),('y',y)]})
    save(ROOT/'protocol.json',dict(seed=1102,features=FEATURES,classes=['continuous_fragments','same_rendered_ink_band','separate'],
        synthetic_pairs={k:len(v[1]) for k,v in sets.items() if not k.startswith('partial')},
        real_partial_curve_augmentations=user_counts,user_drafts_excluded=True,unmarked_not_negative=True,
        upstream_pixel_model_frozen=True,new_model_role='fragment continuity and ink-band relation, not a new semantic image encoder',
        same_source_page_shared_by_partial_train_and_test=True,partial_test_not_independent_real_painting_accuracy=True,
        real_batch_images_used_for_training=False,robot_connected=False))
    x=np.r_[sets['train'][0],sets['partial_train'][0]];y=np.r_[sets['train'][1],sets['partial_train'][1]]
    cx=np.r_[sets['calibration'][0],sets['partial_calibration'][0]];cy=np.r_[sets['calibration'][1],sets['partial_calibration'][1]]
    xt=torch.from_numpy(x).cuda();yt=torch.from_numpy(y).cuda();model=RelationNet().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.0005)
    best=1e9;history=[];start=time.monotonic()
    for step in range(1,1501):
        model.train();ix=torch.randint(len(x),(512,),device='cuda');out=model(xt[ix]);loss=nn.functional.cross_entropy(out,yt[ix])
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%150==0:
            pred=probabilities(model,cx);val=float(-np.log(np.maximum(pred[np.arange(len(cy)),cy],1e-8)).mean())
            row=dict(step=step,loss=float(loss),calibration_ce=val,elapsed_s=time.monotonic()-start);history.append(row)
            print('TRAIN',json.dumps(row),flush=True)
            if val<best:
                best=val;torch.save(dict(model=model.cpu().state_dict(),step=step,features=FEATURES,classes=3),ROOT/'model.pt');model.cuda()
    saved=torch.load(ROOT/'model.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);model.cuda()
    cp=probabilities(model,cx);thresholds={}
    for label,name,floor in [(0,'continuous',.99),(1,'duplicate',.995)]:
        choices=[]
        for threshold in [.5,.6,.7,.8,.85,.9,.93,.95,.97,.98,.99,.995,.999,.9999]:
            stat=binary_stats(cp,cy,label,threshold)
            if stat['precision']>=floor and stat['tp']>=20:choices.append(stat)
        thresholds[name]=max(choices,key=lambda v:(v['recall'],v['precision']))['threshold'] if choices else 1.01
    save(ROOT/'thresholds.json',thresholds)
    evaluation={}
    for split in ['calibration','test','partial_calibration','partial_test']:
        xx,yy=sets[split];p=probabilities(model,xx)
        evaluation[split]=dict(samples=len(yy),argmax_accuracy=float((p.argmax(1)==yy).mean()),
            continuous=binary_stats(p,yy,0,thresholds['continuous']),duplicate=binary_stats(p,yy,1,thresholds['duplicate']))
    save(ROOT/'training_history.json',history);save(ROOT/'evaluation.json',evaluation)
    result=dict(device=torch.cuda.get_device_name(0),steps_executed=1500,retained_step=saved['step'],
        parameters=sum(p.numel() for p in model.parameters()),elapsed_train_s=time.monotonic()-start,
        thresholds=thresholds,evaluation=evaluation,synthetic_and_partial_augmentation_only=True,
        new_real_painting_ground_truth_accuracy_established=False)
    save(ROOT/'training_summary.json',result);print('TRAIN_COMPLETE',json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':train()
