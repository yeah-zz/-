# -*- coding: utf-8 -*-
"""Seed-conditioned ordered curve prediction. Offline research, no robot API."""
from rock_v15_common import LAB, read, jsave, digest, FONT, SMALL
from pathlib import Path
import json, math, time, shutil
import cv2, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
from v31_profiles import observed

ROOT = LAB/'stroke_instances_v32_20261028'
V31 = LAB/'inverse_brush_curves_v31_20261027'
SIZE, N, STEP = 96, 20, 3.
ANCHOR = np.array([24.,48.], np.float32)
HOLDOUT = ['U03','stage_035','replay_R02']
SEED = 322810

def arc(p, step=1.):
    p=np.asarray(p,np.float32)
    if len(p)<2:return p
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    good=np.r_[True,np.diff(s)>1e-4];p=p[good];s=s[good]
    if s[-1]<.01:return p[:1]
    t=np.arange(0,s[-1],step);t=np.r_[t,s[-1]]
    return np.stack([np.interp(t,s,p[:,i]) for i in range(2)],1).astype(np.float32)

def canonical(image,p,u,scale=1.):
    v=np.array([-u[1],u[0]],np.float32);yy,xx=np.mgrid[:SIZE,:SIZE]
    xy=p+scale*((xx-24)[...,None]*u+(yy-48)[...,None]*v)
    return cv2.remap(image,xy[:,:,0].astype(np.float32),xy[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def sample_real(rgb,alpha,valid,path,rng):
    scale=float(rng.uniform(.75,1.3));q=arc(path,STEP*scale)
    if rng.random()<.5:q=q[::-1].copy()
    if len(q)<8:return None
    k=int(rng.integers(2,len(q)-4));p=q[k];u=p-q[max(0,k-3)];u=u/max(1e-6,np.linalg.norm(u));v=np.array([-u[1],u[0]])
    jitter=rng.uniform(-1.2,1.2,2);seed=p+scale*(u*jitter[0]+v*jitter[1])
    patch=canonical(np.stack([alpha,cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)/255],-1).astype(np.float32),seed,u,scale)
    t=np.zeros((N+1,2),np.float32);mask=np.zeros(N+1,np.float32)
    for j in range(min(N+1,len(q)-k)):
        xy=q[k+j];x,y=np.rint(xy).astype(int)
        if not (0<=y<valid.shape[0] and 0<=x<valid.shape[1] and valid[y,x]>.5):break
        delta=(xy-seed)/scale;z=ANCHOR+np.array([delta@u,delta@v])
        if not (2<=z[0]<94 and 2<=z[1]<94):break
        t[j]=z;mask[j]=1
    if mask.sum()<6:return None
    # Unknown/unlabelled image pixels and reference chain ends are NOT stop negatives.
    return np.uint8(np.clip(patch*255,0,255)),t,mask,np.zeros(N,np.float32),np.zeros(N,np.float32)

def procedural(rng):
    """Known ordered curves, not independently collected Chinese artworks."""
    x=np.arange(-5,N+5,dtype=np.float32)*STEP
    curvature=rng.uniform(-.9,.9);wave=rng.uniform(-.45,.45);freq=rng.uniform(.03,.08)
    theta=curvature*np.clip(x,0,None)/70+wave*(np.sin(freq*x)-np.sin(0.))
    dq=STEP*np.c_[np.cos(theta),np.sin(theta)]
    q=np.cumsum(dq,axis=0);q-=q[5];q+=ANCHOR+rng.uniform(-1.3,1.3,2)
    length=int(rng.integers(4,N+5));positive=rng.random()>.12
    stops=np.array([j<=length and positive for j in range(1,N+1)],np.float32)
    target=np.zeros((N+1,2),np.float32);mask=np.zeros(N+1,np.float32)
    for j in range(N+1):
        z=q[5+j]
        if positive and j<=length and 2<=z[0]<94 and 2<=z[1]<94:target[j]=z;mask[j]=1
        else:stops[max(0,j-1):]=0;break
    canvas=np.zeros((SIZE*2,SIZE*2),np.float32)
    def inkline(curve,width,a):
        nonlocal canvas
        layer=np.zeros_like(canvas)
        for i in range(1,len(curve)):
            w=max(1,int(round(2*width*(.55+.65*i/max(1,len(curve)-1)))))
            cv2.line(layer,tuple(np.rint(curve[i-1]*2).astype(int)),tuple(np.rint(curve[i]*2).astype(int)),float(a),w,cv2.LINE_AA)
        canvas=1-(1-canvas)*(1-layer)
    faint=rng.random()<.40;opacity=rng.uniform(.035,.17) if faint else rng.uniform(.2,.95)
    width=rng.uniform(.7,2.6) if rng.random()<.7 else rng.uniform(3,8)
    if positive:inkline(q[:min(len(q),6+length)],width,opacity)
    # Close parallel neighbours and crossing strokes have separate identities.
    kind=int(rng.integers(4))
    if kind in [1,3]:inkline(q+np.array([rng.uniform(-4,4),rng.choice([-1,1])*rng.uniform(4,14)]),rng.uniform(.8,4),rng.uniform(.08,.8))
    if kind in [2,3]:
        c=np.array([rng.uniform(40,75),48]);a=rng.uniform(.7,2.2);v=np.array([math.cos(a),math.sin(a)])
        inkline(c+np.linspace(-50,50,35)[:,None]*v,rng.uniform(.7,5),rng.uniform(.08,.85))
    if rng.random()<.2 and positive:
        gap=int(rng.integers(7,14));z=np.rint(q[gap]*2).astype(int);cv2.circle(canvas,tuple(z),int(rng.integers(2,5)),0,-1)
    for _ in range(int(rng.integers(0,14))):
        z=tuple(rng.integers(2,190,2));cv2.circle(canvas,z,int(rng.integers(1,3)),float(rng.uniform(.01,.25)),-1)
    alpha=cv2.resize(canvas,(SIZE,SIZE),interpolation=cv2.INTER_AREA)
    alpha=np.clip(alpha+rng.normal(0,rng.uniform(.001,.009),alpha.shape),0,1)
    paper=rng.uniform(.72,1.)+rng.normal(0,.003,alpha.shape)
    gray=np.clip(paper*(1-alpha),0,1)
    return np.uint8(np.clip(np.stack([alpha,gray],-1)*255,0,255)),target,mask,stops,np.ones(N,np.float32)

def pack(rows,file):
    names=['image','target','mask','stop','stop_mask']
    np.savez_compressed(file,**{key:np.stack([r[i] for r in rows]) for i,key in enumerate(names)})

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'protocol.json').exists():raise FileExistsError('Keep frozen v32 protocol')
    source=read(LAB/'joint_center_affinity_v24_20261026'/'training_manifest.json');records=[]
    for r in source['records']:
        if r['split']!='train':continue
        records.append(dict(id=r['id'],data=r['data'],labels=r['labels'],kind=r['kind'],split='validation' if r['id'] in HOLDOUT else 'train',data_sha256=digest(r['data']),labels_sha256=digest(r['labels'])))
    jsave(ROOT/'protocol.json',dict(seed=SEED,records=records,steps=2400,procedural_updates=1800,mixed_real_updates=600,batch_size=64,procedural_train_count=10000,procedural_validation_count=1000,canonical_size=SIZE,predicted_ordered_future_points=N,arc_step_px=STEP,lookahead_px=N*STEP,recurrent_advance_points=6,maximum_recurrent_chunks=12,seed_source='multiscale_image_hessian_ridges_not_reference_strokes',proposal_rule=dict(min_length=15,minimum_ink_fraction=.85,max_unbacked_run=3),addition_options=[dict(ink=.018,novel=.65),dict(ink=.035,novel=.8),dict(ink=.06,novel=.9)],calibration_criteria=dict(min_single_gain=.005,max_point_drop=.005,no_mixed_increase=True,max_duplicate_increase=.01),real_reference_endpoints_not_stop_labels=True,unknown_not_background=True,new_independent_artworks=0,reused_development_not_blind=True,robot_connected=False,protected_review_sha256=digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json'),protected_raster_sha256=digest(LAB/'clean_strokes_v12_20261024'/'final.pt')))
    rng=np.random.default_rng(SEED);real={s:[] for s in ['train','validation']};counts=[]
    for r in records:
        z=np.load(r['data']);rgb=z['rgb'];a=observed(rgb);doc=read(r['labels']);paths=[s['points'] for s in doc['strokes'] if len(s['points'])>1];rows=[]
        # Coordinates are from source label dimensions, not presumed resized.
        factor=np.array([rgb.shape[1]/doc['width'],rgb.shape[0]/doc['height']])
        paths=[np.asarray(p,np.float32)*factor for p in paths]
        for _ in range(600):
            p=paths[int(rng.integers(len(paths)))];v=sample_real(rgb,a,z['valid'],p,rng)
            if v is not None:rows.append(v)
            if len(rows)>=160:break
        real[r['split']].extend(rows);counts.append(dict(id=r['id'],split=r['split'],windows=len(rows),visible_chains=len(paths)))
        print('V32_REAL_WINDOWS',r['id'],len(rows),flush=True)
    for s,rows in real.items():pack(rows,ROOT/('real_'+s+'.npz'))
    for s,n,seed in [('train',10000,SEED+1),('validation',1000,SEED+2)]:
        rng=np.random.default_rng(seed);rows=[]
        for i in range(n):
            rows.append(procedural(rng))
            if (i+1)%2000==0:print('V32_PROCEDURAL',s,i+1,flush=True)
        pack(rows,ROOT/('procedural_'+s+'.npz'))
    jsave(ROOT/'data_summary.json',dict(real=counts,real_train_windows=len(real['train']),real_validation_windows=len(real['validation']),procedural_windows_are_not_painting_counts=True))
    shutil.copy2(__file__,ROOT/'model_source.py')

class CurveNet(nn.Module):
    def __init__(self):
        super().__init__();blocks=[];c=3
        for d in [16,32,64,96]:blocks.extend([nn.Conv2d(c,d,5,2,2),nn.SiLU()]);c=d
        self.enc=nn.Sequential(*blocks);self.head=nn.Sequential(nn.Flatten(),nn.Linear(96*6*6,192),nn.SiLU(),nn.Linear(192,2+2*N))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():self.head[-1].bias[2+N:2+2*N]=3
    def forward(self,x):
        z=self.head(self.enc(x));origin=z[:,:2].tanh()*3+x.new_tensor([24,48]);angle=math.pi*z[:,2:2+N].tanh();delta=STEP*torch.stack([angle.cos(),angle.sin()],-1);points=torch.cat([origin[:,None],origin[:,None]+delta.cumsum(1)],1)
        return points,z[:,2+N:2+2*N],angle

def features(x):
    x=x.float()/255 if x.dtype==torch.uint8 else x.float()
    a=x[:,0:1];norm=a/a.flatten(1).quantile(.97,dim=1).clamp_min(.04)[:,None,None,None]
    return torch.cat([a,norm.clamp(0,1),x[:,1:2]],1)

def loss_fn(output,target,mask,stop,stop_mask):
    p,logit,angle=output
    geom=(F.smooth_l1_loss(p/6,target/6,reduction='none').sum(-1)*mask).sum()/mask.sum().clamp_min(1)
    d=F.normalize(target[:,1:]-target[:,:-1],dim=-1);v=torch.stack([angle.cos(),angle.sin()],-1);dm=mask[:,1:]*mask[:,:-1]
    direction=((1-(d*v).sum(-1))*dm).sum()/dm.sum().clamp_min(1)
    termination=(F.binary_cross_entropy_with_logits(logit,stop,reduction='none')*stop_mask).sum()/stop_mask.sum().clamp_min(1)
    return geom+.25*direction+.18*termination

def load_data(name):
    z=np.load(ROOT/(name+'.npz'));out=[torch.as_tensor(z[k],device='cuda') for k in ['image','target','mask','stop','stop_mask']];out[0]=out[0].permute(0,3,1,2).contiguous();return out

def smoke():
    d=load_data('real_validation');batch=[a[:4] for a in d];m=CurveNet().cuda();x=features(batch[0]);out=m(x);loss=loss_fn(out,*batch[1:]);loss.backward();finite=bool(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
    result=dict(input_shape=list(x.shape),points_shape=list(out[0].shape),stop_shape=list(out[1].shape),loss=float(loss.detach()),finite_backward=finite,actual_device=str(x.device),gpu=torch.cuda.get_device_name());jsave(ROOT/'cuda_smoke.json',result);print('V32_SMOKE',result,flush=True)
    if not finite or list(x.shape)!=[4,3,96,96]:raise RuntimeError('Invalid model interface')

@torch.no_grad()
def validation(model,data):
    errs=[];loss=[];st=[]
    for i in range(0,len(data[0]),128):
        b=[x[i:i+128] for x in data];o=model(features(b[0]));dist=torch.linalg.vector_norm(o[0]-b[1],dim=-1);errs.extend(dist[b[2]>.5].cpu().tolist());loss.append(float(loss_fn(o,*b[1:])));known=b[4]>.5
        if known.any():st.extend(((o[1].sigmoid()>.5)==(b[3]>.5))[known].cpu().tolist())
    return dict(mean_ordered_point_error_px=float(np.mean(errs)),fraction_within_3px=float((np.asarray(errs)<=3).mean()),loss=float(np.mean(loss)),stop_accuracy=float(np.mean(st)) if st else None)

def train():
    if (ROOT/'curve_model.pt').exists():raise FileExistsError('Preserve trained model')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(SEED);g=torch.Generator(device='cuda').manual_seed(SEED)
    syn=load_data('procedural_train');real=load_data('real_train');sv=load_data('procedural_validation');rv=load_data('real_validation')
    net=CurveNet().cuda();opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-4);logs=[];best=1e9;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(1,2401):
        net.train();idx=torch.randint(len(syn[0]),(64,),device='cuda',generator=g);batch=[a[idx] for a in syn]
        if step>1800:
            idx=torch.randint(len(real[0]),(32,),device='cuda',generator=g);batch=[torch.cat([a[:32],b[idx]]) for a,b in zip(batch,real)];opt.param_groups[0]['lr']=1e-4
        # Contrast augmentation changes image appearance, never curve geometry.
        xx=batch[0].float()/255;gain=.45+.9*torch.rand(len(xx),1,1,1,device='cuda',generator=g);xx[:,0:1]=(xx[:,0:1]*gain).clamp(0,1)
        o=net(features(xx));loss=loss_fn(o,*batch[1:]);opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('Nonfinite curve training')
        opt.step()
        if step%100==0:
            net.eval();s=validation(net,sv);r=validation(net,rv);row=dict(step=step,loss=float(loss.detach()),procedural=s,real=r,elapsed_seconds=time.perf_counter()-start);logs.append(row);jsave(ROOT/'training_progress.json',dict(logs=logs));print('V32_TRAIN',json.dumps(row),flush=True)
            if step==1800:torch.save(dict(model=net.state_dict(),step=step),ROOT/'procedural_only.pt')
            if step>=1800 and r['mean_ordered_point_error_px']<best:best=r['mean_ordered_point_error_px'];torch.save(dict(model=net.state_dict(),step=step,real_validation=r),ROOT/'curve_model.pt')
    torch.cuda.synchronize();jsave(ROOT/'training_summary.json',dict(actual_updates=2400,procedural_updates=1800,mixed_real_updates=600,actual_device='cuda:0',gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,elapsed_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=digest(ROOT/'curve_model.pt'),logs=logs,new_independent_artworks=0,real_labels_partial_not_historical_pen_lifts=True))

def load(which='adapted'):
    m=CurveNet().cuda();m.load_state_dict(torch.load(ROOT/('curve_model.pt' if which=='adapted' else 'procedural_only.pt'),map_location='cuda',weights_only=True)['model']);return m.eval()

if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train,'smoke':smoke}[sys.argv[1]]()
