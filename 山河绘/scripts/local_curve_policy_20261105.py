# -*- coding: utf-8 -*-
"""Image-conditioned next-point policy; inspect source again after each step.

Direction/stop supervision from known rendered curves and explicit real paths.
Not a robot policy, traditional cun classifier, or verified pen-lift predictor.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import math,hashlib,json,time
import numpy as np,cv2,torch
from torch import nn
from torch.nn import functional as F
from module_stroke_core_20261104 import samples,length

ROOT=LAB/'local_curve_policy_20261105';SIZE=48;GRID=17;SPACING=.5;STOP=GRID*GRID
SEED=6110517
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def arc_points(p,n):
    p=np.asarray(p,float);s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    keep=np.r_[True,np.diff(s)>1e-8];s=s[keep];p=p[keep]
    q=np.linspace(0,s[-1],n)
    return np.column_stack([np.interp(q,s,p[:,k]) for k in (0,1)])

def fields(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    contrast=np.maximum(0,cv2.GaussianBlur(gray,(0,0),2.2)-gray)
    return rgb,gray,contrast

def patch(f,seed,direction,history=None,scale=1.):
    rgb,gray,contrast=f;seed=np.asarray(seed,float);u=np.asarray(direction,float);u=u/max(np.linalg.norm(u),1e-8)
    v=np.array([-u[1],u[0]]);yy,xx=np.mgrid[:SIZE,:SIZE]
    xy=seed+((xx-23.5)[...,None]*u+(yy-23.5)[...,None]*v)*scale*.7
    x=xy[:,:,0].astype(np.float32);y=xy[:,:,1].astype(np.float32)
    color=cv2.remap(rgb,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    ink=cv2.remap(contrast,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    ink=np.clip(ink/max(.035,float(np.quantile(ink,.96))),0,1)
    recent=np.zeros((SIZE,SIZE),np.uint8)
    if history is not None and len(history)>1:
        d=np.asarray(history,float)-seed;coords=np.column_stack([d@u,d@v])/(scale*.7)+23.5
        cv2.polylines(recent,[np.rint(coords).astype(np.int32)],False,255,1,cv2.LINE_AA)
    return np.concatenate([color,np.uint8(ink[:,:,None]*255),recent[:,:,None]],2).transpose(2,0,1).copy()

class LocalCurveNet(nn.Module):
    def __init__(self):
        super().__init__();layers=[];c=5
        for out in [24,48,64]:layers += [nn.Conv2d(c,out,5,2,2),nn.SiLU()];c=out
        self.encoder=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(64*6*6,192),nn.SiLU(),nn.Linear(192,STOP+1))
    def forward(self,x):return self.head(self.encoder(x.float()/255 if x.dtype==torch.uint8 else x))

def synthetic_scene(rng,index,return_paths=False):
    side=96;yy,xx=np.mgrid[:side,:side];paper=np.array([.97,.955,.91])+rng.uniform(-.035,.025,3)
    rgb=np.ones((side,side,3))*paper
    for _ in range(3):
        x,y=rng.uniform(5,90,2);r=rng.uniform(15,45);blob=np.exp(-((xx-x)**2+(yy-y)**2)/(r*r))
        rgb+=blob[:,:,None]*np.array(rng.choice([[-.15,-.05,-.02],[-.04,-.02,-.08],[-.05,-.06,-.04]],1)[0])
    rgb=np.clip((rgb+rng.normal(0,.003,(side,side,1)))*255,0,255).astype(np.uint8)
    t=np.linspace(0,1,70);kind=index%4
    if kind==0:
        controls=rng.uniform([15,15],[80,80],(4,2));controls[0]=[15,rng.uniform(25,70)];controls[-1]=[80,rng.uniform(25,70)]
        target=(1-t[:,None])**3*controls[0]+3*(1-t[:,None])**2*t[:,None]*controls[1]+3*(1-t[:,None])*t[:,None]**2*controls[2]+t[:,None]**3*controls[3]
    elif kind==1:
        target=np.array([[15,80],[25,40],[37,22],[49,45],[58,32],[78,75]],float)
    elif kind==2:
        target=np.array([[14,60],[35,53],[48,30],[65,48],[83,53]],float) if rng.random()<.5 else np.array([[25,25],[25,78]],float)
    else:
        target=np.column_stack([15+65*t,65-32*t+9*np.sin(t*np.pi)])
    target=samples(target,.5);others=[]
    if rng.random()<.6:others.append(target+rng.choice([-1,1])*np.array([0,rng.uniform(3.5,8)]))
    if rng.random()<.55:others.append(np.array([[48,12],[48,84]],float))
    if kind==3:
        for j in range(5):
            k=int((j+1)*len(target)/7);others.append(np.array([target[k],target[k]+[rng.uniform(-8,4),-rng.uniform(7,14)]]))
    for p in [target]+others:
        layer=np.zeros((side*3,side*3),np.uint8);w=int(rng.integers(2,9))
        cv2.polylines(layer,[np.rint(p*3).astype(np.int32)],False,255,w,cv2.LINE_AA)
        alpha=cv2.resize(layer,(side,side),interpolation=cv2.INTER_AREA)/255*rng.uniform(.08,.92)
        rgb=(rgb*(1-alpha[:,:,None])+rng.uniform(15,70)*alpha[:,:,None]).astype(np.uint8)
    return (rgb,target,[target]+others) if return_paths else (rgb,target)

def examples(rgb,p,rng,count,known_end=False,recovery_states=False):
    p=samples(p,.4);total=length(p)
    if total<4:return [],[],[]
    ss=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];f=fields(rgb)
    out=[];targets=[];kinds=[]
    for j in range(count):
        reverse=rng.random()<.5;q=p[::-1] if reverse else p;s=ss if not reverse else total-ss[::-1]
        endpoint_example=known_end and j%9==0
        at=total if endpoint_example else rng.uniform(min(1,total*.1),max(1.1,total-2.2))
        interpolate=lambda x:np.array([np.interp(x,s,q[:,k]) for k in (0,1)])
        exact=interpolate(at);past=interpolate(max(0,at-2.5));u=exact-past;u=u/max(np.linalg.norm(u),1e-8)
        v=np.array([-u[1],u[0]]);scale=float(rng.uniform(.8,1.25))
        jitter=np.clip(rng.normal(0,.85 if recovery_states else .28,2),-1.6,1.6)*scale
        seed=exact+u*jitter[0]+v*jitter[1]
        history=np.array([interpolate(max(0,at-d)) for d in [5,3.5,2,0]])
        if recovery_states:
            # Recovery states force image use: neither the seed nor heading is
            # the correct teacher state. Recent history sometimes drifts too.
            angle=float(rng.uniform(-.55,.55));u=np.array([math.cos(angle)*u[0]-math.sin(angle)*u[1],math.sin(angle)*u[0]+math.cos(angle)*u[1]])
            v=np.array([-u[1],u[0]])
            if rng.random()<.5:history=history+(seed-exact)
        im=patch(f,seed,u,history,scale)
        if endpoint_example:target=STOP;kind='known_rendered_end'
        else:
            nxt=interpolate(min(total,at+1.8*scale));delta=(nxt-seed)/scale
            local=np.array([delta@u,delta@v]);xy=np.rint(local/SPACING).astype(int)+(GRID//2)
            if (xy<0).any() or (xy>=GRID).any():continue
            target=int(xy[1]*GRID+xy[0]);kind='next_point'
        out.append(im);targets.append(target);kinds.append(kind)
    return out,targets,kinds

def prepare(recovery_states=False):
    from context_curve_data_20261105 import source_records,bg_paths
    ROOT.mkdir(exist_ok=True);allrecords=source_records();manifest=[]
    for split,nscenes,seed in [('train',900,SEED),('calibration',120,SEED+1),('test',120,SEED+2)]:
        rng=np.random.default_rng(seed);xx=[];yy=[];origins=[]
        for i in range(nscenes):
            rgb,p,known=synthetic_scene(rng,i,True);a,b,k=examples(rgb,p,rng,9,True,recovery_states);xx+=a;yy+=b;origins+=['rendered_'+v for v in k]
            # Explicit synthetic blank sample, separate from faint true strokes.
            blank=np.full_like(rgb,245);f=fields(blank);xx.append(patch(f,[48,48],[1,0]));yy.append(STOP);origins.append('rendered_blank')
            if recovery_states:
                from scipy.spatial import cKDTree
                tree=cKDTree(np.concatenate([samples(v,.5) for v in known]));source=fields(rgb)
                for _ in range(2):
                    for _ in range(30):
                        seedpoint=rng.uniform(12,84,2)
                        if tree.query(seedpoint)[0]>7:break
                    else:continue
                    angle=rng.uniform(-math.pi,math.pi)
                    xx.append(patch(source,seedpoint,[math.cos(angle),math.sin(angle)]));yy.append(STOP);origins.append('known_rendered_off_ink')
        for record in allrecords:
            if record['split']!=split:continue
            n=len(xx)
            for p in record['refs']:
                a,b,k=examples(record['rgb'],p,rng,24 if split=='train' else 10,False,recovery_states)
                xx+=a;yy+=b;origins+=['explicit_real_next_point']*len(b)
            f=fields(record['rgb'])
            for p in bg_paths(record['background'],rng,40):
                u=p[-1]-p[0];xx.append(patch(f,p[0],u));yy.append(STOP);origins.append('explicit_real_background')
            manifest.append(dict(id=record['id'],split=split,image=record['image'],labels=record['labels'],
                image_sha256=sha(record['image']),labels_sha256=sha(record['labels']),examples=len(xx)-n,
                source_family=record['source_family']))
        x=np.stack(xx);y=np.array(yy,np.int64)
        np.savez_compressed(ROOT/(split+'.npz'),x=x,y=y,origin=np.asarray(origins))
        print('POLICY_DATA',split,len(y),flush=True)
    save(ROOT/'protocol.json',dict(records=manifest,seed=SEED,synthetic_scenes=dict(train=900,calibration=120,test=120),
        synthetic_scenes_are_not_real_paintings=True,whole_real_strokes_supply_ordered_next_point_supervision=True,
        real_reference_endpoints_not_stop_labels=True,unmarked_not_background=True,
        source_families_overlap=True,real_images_reused_from_previous_experiments=True,
        synthetic_and_partial_reference_test_not_full_image_accuracy=True,
        model='RGB/local-contrast/recent-path -> 17x17 next-point distribution + stop',
        recurrent_step=1.8,whole_proposal_guard_no_pointwise_fragmentation=True,robot_connected=False))
    protocol=json.loads((ROOT/'protocol.json').read_text('utf8'));protocol.update(recovery_state_augmentation=recovery_states,
        max_seed_jitter_px=1.6 if recovery_states else None,max_heading_jitter_rad=.55 if recovery_states else None,
        history_drift_augmentation=recovery_states,round1_free_rollout_failed_and_not_adopted=True)
    save(ROOT/'protocol.json',protocol)

@torch.no_grad()
def evaluate(model,data):
    xx,yy,orig=data;pred=[];conf=[]
    for i in range(0,len(yy),256):
        logits=model(xx[i:i+256]);pred.append(logits.argmax(1).cpu().numpy());conf.append(logits.softmax(1).amax(1).cpu().numpy())
    p=np.concatenate(pred);y=yy.cpu().numpy();real=orig=='explicit_real_next_point';step=y!=STOP
    px=np.column_stack([p%GRID,p//GRID]);tx=np.column_stack([y%GRID,y//GRID]);error=np.linalg.norm(px-tx,axis=1)*SPACING
    def summary(mask):
        return dict(count=int(mask.sum()),mean_next_point_error_px=float(np.mean(np.where(p[mask]==STOP,5,error[mask]))) if mask.any() else None,
                    within_1px=float(((error<=1)&(p!=STOP))[mask].mean()) if mask.any() else None,
                    premature_stop=int(((p==STOP)&mask).sum()))
    return dict(real_reference_seeded=summary(real),all_known_next_points=summary(step),
        stop_accuracy=float((p[y==STOP]==STOP).mean()),not_full_image_extraction_accuracy=True)

def load_data(split):
    z=np.load(ROOT/(split+'.npz'));return torch.as_tensor(z['x'],device='cuda'),torch.as_tensor(z['y'],device='cuda'),z['origin']

def loss(logits,y):
    # Spatially neighboring displacement bins share supervision. A hard class
    # boundary must not dominate a subpixel endpoint annotation difference.
    raw=F.cross_entropy(logits,y,label_smoothing=.01)
    good=y!=STOP
    if good.any():
        coords=torch.arange(STOP,device=y.device);px=coords%GRID;py=coords//GRID
        dx=px[None]-((y[good]%GRID)[:,None]);dy=py[None]-((y[good]//GRID)[:,None])
        target=torch.exp(-(dx*dx+dy*dy)/1.1);target/=target.sum(1,keepdim=True)
        smooth=-(target*logits[good,:STOP].log_softmax(1)).sum(1).mean()
        raw=.3*raw+.7*smooth
    return raw

def train(steps=2200):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; keep CPU free')
    torch.set_num_threads(4);torch.manual_seed(SEED);start=time.monotonic()
    training=load_data('train');calibration=load_data('calibration');net=LocalCurveNet().cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4);best=float('inf');logs=[]
    real=torch.as_tensor(np.where(training[2]=='explicit_real_next_point')[0],device='cuda')
    for step in range(1,steps+1):
        net.train();ids=torch.randint(len(training[1]),(48,),device='cuda')
        if step>steps//3:ids=torch.cat([ids[:24],real[torch.randint(len(real),(24,),device='cuda')]])
        x,y=training[0][ids],training[1][ids]
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x);value=loss(logits.float(),y)
        opt.zero_grad(set_to_none=True);value.backward();nn.utils.clip_grad_norm_(net.parameters(),3.);opt.step()
        if step%200==0:
            net.eval();metrics=evaluate(net,calibration);row=dict(step=step,loss=float(value.detach()),calibration=metrics)
            logs.append(row);score=metrics['real_reference_seeded']['mean_next_point_error_px']
            if step>=steps//3 and score<best:
                best=score;torch.save(dict(model=net.state_dict(),step=step,calibration=metrics),ROOT/'policy.pt')
            print('POLICY_TRAIN',json.dumps(row),flush=True)
    net.load_state_dict(torch.load(ROOT/'policy.pt',map_location='cuda',weights_only=True)['model']);net.eval()
    test=evaluate(net,load_data('test'))
    save(ROOT/'training_summary.json',dict(updates=steps,gpu=torch.cuda.get_device_name(),parameter_count=sum(p.numel() for p in net.parameters()),
        retained_step=torch.load(ROOT/'policy.pt',map_location='cpu',weights_only=True)['step'],logs=logs,test=test,
        seconds=time.monotonic()-start,model_sha256=sha(ROOT/'policy.pt'),no_independent_real_painting_test=True,robot_connected=False))
    print('POLICY_TEST',json.dumps(test),flush=True)

def load():
    net=LocalCurveNet().cuda();net.load_state_dict(torch.load(ROOT/'policy.pt',map_location='cuda',weights_only=True)['model']);return net.eval()

@torch.no_grad()
def trace_batch(model,rgb,seeds,directions,max_steps=96,histories=None):
    """Raw image/seed predictions. No references or baseline paths used inside."""
    f=fields(rgb);paths=[[np.array(p,float)] for p in seeds];u=[np.array(t,float)/max(np.linalg.norm(t),1e-8) for t in directions]
    active=set(range(len(paths)));stops={};h,w=rgb.shape[:2]
    for step in range(max_steps):
        ids=sorted(active)
        if not ids:break
        history=lambda i:((histories[i] if histories is not None else [])+paths[i])[-5:]
        x=np.stack([patch(f,paths[i][-1],u[i],history(i)) for i in ids])
        probs=model(torch.as_tensor(x,device='cuda')).softmax(1).cpu().numpy()
        for j,i in enumerate(ids):
            label=int(probs[j].argmax())
            if label==STOP:active.remove(i);stops[i]='model_stop';continue
            delta=np.array([label%GRID-GRID//2,label//GRID-GRID//2])*SPACING
            normal=np.array([-u[i][1],u[i][0]]);shift=delta[0]*u[i]+delta[1]*normal
            if np.linalg.norm(shift)<.55:active.remove(i);stops[i]='stalled';continue
            point=paths[i][-1]+shift
            if not (1<=point[0]<w-1 and 1<=point[1]<h-1):active.remove(i);stops[i]='image_boundary';continue
            if len(paths[i])>7 and np.linalg.norm(np.array(paths[i][:-5])-point,axis=1).min()<1:
                active.remove(i);stops[i]='loop_guard';continue
            paths[i].append(point);u[i]=shift/np.linalg.norm(shift)
    return [dict(points=np.asarray(p),stop=stops.get(i,'step_limit')) for i,p in enumerate(paths)]

@torch.no_grad()
def constrained_trace_batch(model,rgb,seeds,directions,max_steps=96,histories=None,use_model=True):
    """Choose only source-supported next moves during generation, never split.

    The same image/geometry search without logits is an explicit control. This
    is a hybrid proposal method, not claimed as pure neural curve prediction.
    """
    from hierarchical_paint_core_20261105 import source_fields
    from source_stroke_width_20261104 import sample
    f=fields(rgb);gray,contrast,ink=source_fields(rgb)
    paths=[[np.array(p,float)] for p in seeds];u=[np.array(t,float)/max(np.linalg.norm(t),1e-8) for t in directions]
    active=set(range(len(paths)));stops={};h,w=rgb.shape[:2]
    axes=np.stack(np.meshgrid(np.arange(GRID),np.arange(GRID)),axis=-1).reshape(-1,2)-GRID//2
    local=axes*SPACING;norm=np.linalg.norm(local,axis=1)
    valid=(norm>=1.25)&(norm<=2.75)&(local[:,0]>.4)
    labels=np.where(valid)[0];local=local[valid]
    for step in range(max_steps):
        ids=sorted(active)
        if not ids:break
        if use_model:
            history=lambda i:((histories[i] if histories is not None else [])+paths[i])[-5:]
            x=np.stack([patch(f,paths[i][-1],u[i],history(i)) for i in ids])
            probs=model(torch.as_tensor(x,device='cuda')).softmax(1).cpu().numpy()
        for j,i in enumerate(ids):
            normal=np.array([-u[i][1],u[i][0]]);delta=local[:,0:1]*u[i]+local[:,1:2]*normal
            points=paths[i][-1]+delta;inside=(points[:,0]>1)&(points[:,0]<w-2)&(points[:,1]>1)&(points[:,1]<h-2)
            axis=delta/np.linalg.norm(delta,axis=1,keepdims=True);nn=np.column_stack([-axis[:,1],axis[:,0]])
            center=sample(gray,points);ridge=np.full(len(points),-1.)
            for radius in (1.,2.,3.,5.):ridge=np.maximum(ridge,np.minimum(sample(gray,points+nn*radius),sample(gray,points-nn*radius))-center)
            chord=np.stack([paths[i][-1]+delta*t for t in np.linspace(.2,1,6)],axis=1)
            support=(sample(ink.astype(np.float32),chord.reshape(-1,2)).reshape(-1,6)>.5).mean(1)
            eligible=inside&(ridge>.004)&(support>=.83)
            if len(paths[i])>6:
                old=np.asarray(paths[i][:-4]);eligible&=np.linalg.norm(points[:,None]-old[None],axis=2).min(1)>1.1
            if not eligible.any():active.remove(i);stops[i]='source_supported_end';continue
            heading=axis@u[i];source=np.clip(ridge/.04,0,2)
            score=.75*source+.35*heading-.16*abs(norm[valid]-1.8)
            if use_model:score+=.18*np.log(np.maximum(probs[j,labels],1e-8))
            score[~eligible]=-np.inf;k=int(score.argmax());paths[i].append(points[k])
            if len(paths[i])>2:
                direction=paths[i][-1]-paths[i][max(0,len(paths[i])-4)]
                u[i]=direction/max(np.linalg.norm(direction),1e-8)
            else:u[i]=axis[k]
    return [dict(points=np.asarray(p),stop=stops.get(i,'step_limit')) for i,p in enumerate(paths)]

if __name__=='__main__':
    if len(sys.argv)<2:raise SystemExit('prepare | train')
    if sys.argv[1] in ('prepare_recovery','train_recovery'):
        ROOT=ROOT/'recovery_round2';SEED=6110521
    {'prepare':prepare,'train':train,'prepare_recovery':lambda:prepare(True),'train_recovery':train}[sys.argv[1]]()
