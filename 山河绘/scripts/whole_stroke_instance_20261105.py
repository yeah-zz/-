# -*- coding: utf-8 -*-
"""Query-conditioned whole-instance ownership plus simultaneous curve output.

No recurrent point walk, endpoint stitching, robot imports or brush commands.
Real ownership targets are reference-derived tubes with unknown pixels masked;
they are explicitly NOT expert pixel masks or historical pen-lift records.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,math,time
import numpy as np,cv2,torch
from torch import nn
from torch.nn import functional as F
from module_stroke_core_20261104 import samples,length

ROOT=LAB/'whole_stroke_instance_20261105';DATA_ROOT=ROOT;SIZE=96;POINTS=33;SEED=6110531
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arc_points(p,n=POINTS):
    p=np.asarray(p,float);s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    keep=np.r_[True,np.diff(s)>1e-8];s=s[keep];p=p[keep]
    q=np.linspace(0,s[-1],n)
    return np.column_stack([np.interp(q,s,p[:,k]) for k in (0,1)])
def raster(p,width=1):
    mask=np.zeros((SIZE,SIZE),np.uint8)
    if len(p)>1:cv2.polylines(mask,[np.rint(p).astype(np.int32)],False,255,width,cv2.LINE_AA)
    return mask

def crop_query(rgb,center,side,back,front):
    center=np.asarray(center,float);origin=center-side/2;yy,xx=np.mgrid[:SIZE,:SIZE]
    xy=origin+(np.stack([xx,yy],-1)+.5)*side/SIZE
    color=cv2.remap(rgb,xy[:,:,0].astype(np.float32),xy[:,:,1].astype(np.float32),cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,borderValue=(250,248,240))
    hints=[]
    for point in [back,front]:
        z=(np.asarray(point)-origin)/side*SIZE-.5
        hint=np.exp(-((xx-z[0])**2+(yy-z[1])**2)/(2*1.25**2))*255
        hints.append(np.uint8(hint))
    image=np.concatenate([color,hints[0][:,:,None],hints[1][:,:,None]],2).transpose(2,0,1).copy()
    return image,origin

def make_example(rgb,path,others,background,rng,synthetic=False,target_width=2.):
    p=np.asarray(path,float)
    if rng.random()<.5:p=p[::-1]
    dense=samples(p,.5);arc=length(p)
    if arc<5:return None
    k=int(rng.uniform(.30,.70)*(len(dense)-1));step=max(1,int(min(3.,arc*.05)/.5))
    back=dense[max(0,k-step)];front=dense[min(len(dense)-1,k+step)]
    lo=p.min(0);hi=p.max(0);extent=max(hi-lo)
    side=max(28,extent*rng.uniform(1.22,1.65));center=(lo+hi)/2+rng.uniform(-.07,.07,2)*side
    image,origin=crop_query(rgb,center,side,back,front)
    transform=lambda q:(np.asarray(q)-origin)/side*SIZE-.5
    q=transform(arc_points(p));mask=raster(transform(p),max(1,round(target_width)))
    if synthetic:valid=np.full((SIZE,SIZE),255,np.uint8)
    else:
        valid=mask.copy();unknown_band=raster(transform(p),6)>0
        yy,xx=np.mgrid[:SIZE,:SIZE];xy=origin+(np.stack([xx,yy],-1)+.5)*side/SIZE
        explicit=cv2.remap(background,xy[:,:,0].astype(np.float32),xy[:,:,1].astype(np.float32),cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        valid[explicit>0]=255
        for other in others:
            known=raster(transform(other),2)>0;valid[known&~unknown_band]=255
        # Near a traced centreline the tube is positive. The uncertain band
        # around it remains masked, rather than being declared negative ink.
        valid[mask>40]=255
    return image,mask,valid,(q/(SIZE-1)).astype(np.float32),1.,float(synthetic)

def synthetic_scene(rng,index):
    t=np.linspace(0,1,80);kind=index%5
    if kind==0:
        target=np.array([[18,100],[29,50],[43,23],[57,43],[72,34],[95,94]],float)
        others=[np.array([[x,65],[x-6,93]],float) for x in (33,49,66,83)]
    elif kind==1:
        target=np.column_stack([25+64*t,30+55*t+12*np.sin(t*np.pi)])
        others=[target+[rng.choice([-1,1])*rng.uniform(4,8),0],np.array([[20,65],[105,65]],float)]
    elif kind==2:
        target=np.array([[15,65],[37,52],[60,25],[82,49],[110,62]],float)
        others=[np.array([[30,69],[30,103]],float),np.array([[90,69],[90,103]],float),np.array([[28,103],[92,103]],float)]
    elif kind==3:
        target=np.column_stack([20+83*t,96-55*t+8*np.sin(t*np.pi)])
        others=[]
        for f in np.linspace(.2,.9,8):
            p=target[int(f*79)];others.append(np.array([p,p+[rng.uniform(-11,8),-rng.uniform(8,18)]]))
    else:
        controls=rng.uniform([20,20],[105,105],(4,2));controls[0]=[18,80];controls[-1]=[105,36]
        target=(1-t[:,None])**3*controls[0]+3*(1-t[:,None])**2*t[:,None]*controls[1]+3*(1-t[:,None])*t[:,None]**2*controls[2]+t[:,None]**3*controls[3]
        others=[target+[0,rng.uniform(5,10)]]
    angle=rng.uniform(-math.pi,math.pi);rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])*.85
    transform=lambda p:(np.asarray(p)-64)@rot.T+64
    target=transform(target);others=[transform(p) for p in others]
    yy,xx=np.mgrid[:128,:128];paper=np.array([.97,.955,.91])+rng.uniform(-.04,.025,3)
    image=np.ones((128,128,3))*paper
    for _ in range(3):
        x,y=rng.uniform(5,123,2);r=rng.uniform(20,60);blob=np.exp(-((xx-x)**2+(yy-y)**2)/(r*r))
        image+=blob[:,:,None]*np.array(rng.choice([[-.16,-.055,-.02],[-.04,-.03,-.08],[-.08,-.075,-.03]],1)[0])
    image=np.clip((image+rng.normal(0,.003,(128,128,1)))*255,0,255).astype(np.uint8)
    widths=[]
    for i,p in enumerate([target]+others):
        width=float(rng.uniform(.7,3.8));widths.append(width)
        layer=np.zeros((384,384),np.uint8);cv2.polylines(layer,[np.rint(p*3).astype(np.int32)],False,255,max(1,round(width*3)),cv2.LINE_AA)
        alpha=cv2.resize(layer,(128,128),interpolation=cv2.INTER_AREA)/255*rng.uniform(.07,.92)
        image=np.uint8(image*(1-alpha[:,:,None])+rng.uniform(15,70)*alpha[:,:,None])
    return image,target,others,widths[0]

def prepare():
    from context_curve_data_20261105 import source_records,bg_paths
    ROOT.mkdir(exist_ok=True);records=source_records();manifest=[]
    for split,n,seed in [('train',2600,SEED),('calibration',320,SEED+1),('test',320,SEED+2)]:
        rng=np.random.default_rng(seed);rows=[];kinds=[]
        for i in range(n):
            rgb,p,others,width=synthetic_scene(rng,i)
            result=make_example(rgb,p,others,None,rng,True,width)
            if result is not None:rows.append(result);kinds.append('known_rendered_instance')
            if i%6==0:
                image,_=crop_query(rgb,[64,64],128,[6,6],[10,6]);rows.append((image,np.zeros((SIZE,SIZE),np.uint8),np.full((SIZE,SIZE),255,np.uint8),np.zeros((POINTS,2),np.float32),0.,1.));kinds.append('known_rendered_no_query_instance')
        for rec in records:
            if rec['split']!=split:continue
            start=len(rows)
            for j,p in enumerate(rec['refs']):
                for _ in range(12 if split=='train' else 4):
                    others=[q for k,q in enumerate(rec['refs']) if k!=j]
                    result=make_example(rec['rgb'],p,others,rec['background'],rng,False,2.)
                    if result is not None:rows.append(result);kinds.append('explicit_real_reference_tube')
            manifest.append(dict(id=rec['id'],split=split,image=rec['image'],labels=rec['labels'],image_sha256=sha(rec['image']),
                labels_sha256=sha(rec['labels']),examples=len(rows)-start,source_family=rec['source_family']))
        names=['image','mask','valid','curve','presence','endpoint_valid']
        arrays={name:np.stack([r[j] for r in rows]) for j,name in enumerate(names)};arrays['kind']=np.asarray(kinds)
        np.savez_compressed(ROOT/(split+'.npz'),**arrays)
        print('INSTANCE_DATA',split,len(rows),dict(zip(*np.unique(kinds,return_counts=True))),flush=True)
    save(ROOT/'protocol.json',dict(records=manifest,seed=SEED,synthetic_scenes=dict(train=2600,calibration=320,test=320),
        query='oriented short fragment only, no whole reference at inference',
        outputs=['whole-instance ownership mask','33 simultaneously predicted ordered curve points','synthetic-only endpoint maps','presence'],
        real_mask_supervision='reference-derived 2px tubes, NOT expert pixel ownership masks',
        real_unmarked_pixels_ignored=True,real_endpoint_maps_not_supervised=True,
        curve_endpoints_are_visible_reference_extents_not_historical_brush_lifts=True,
        same_source_families_cross_splits=True,old_real_images_reused=True,new_independent_painting_test=False,
        selection_uses_calibration_only=True,robot_connected=False))

class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
    def forward(self,x):return self.net(x)
class WholeInstanceNet(nn.Module):
    def __init__(self):
        super().__init__();self.e1=Block(5,24);self.e2=Block(24,48);self.e3=Block(48,96);self.e4=Block(96,192);self.bottleneck=Block(192,192)
        self.d4=Block(384,96);self.d3=Block(192,48);self.d2=Block(96,24);self.d1=Block(48,24);self.maps=nn.Conv2d(24,3,1)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(192*6*6,256),nn.SiLU(),nn.Linear(256,POINTS*2+1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,image):
        x=image.float()/255 if image.dtype==torch.uint8 else image
        a=self.e1(x);b=self.e2(F.avg_pool2d(a,2));c=self.e3(F.avg_pool2d(b,2));d=self.e4(F.avg_pool2d(c,2));z=self.bottleneck(F.avg_pool2d(d,2))
        up=lambda t,skip:F.interpolate(t,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        y=self.d4(torch.cat([up(z,d),d],1));y=self.d3(torch.cat([up(y,c),c],1));y=self.d2(torch.cat([up(y,b),b],1));y=self.d1(torch.cat([up(y,a),a],1))
        logits=self.maps(y);out=self.head(z)
        grid=torch.linspace(0,1,SIZE,device=x.device);yy,xx=torch.meshgrid(grid,grid,indexing='ij')
        hints=x[:,3:5].clamp_min(0);mass=hints.sum((-2,-1)).clamp_min(1e-6)
        positions=torch.stack([(hints*xx).sum((-2,-1))/mass,(hints*yy).sum((-2,-1))/mass],-1)
        center=positions.mean(1);direction=F.normalize(positions[:,1]-positions[:,0],dim=-1)
        t=torch.linspace(-.35,.35,POINTS,device=x.device)
        base=center[:,None]+t[None,:,None]*direction[:,None]
        curve=(base+out[:,:POINTS*2].reshape(-1,POINTS,2).tanh()*.5).clamp(0,1)
        return dict(mask=logits[:,0:1],endpoints=logits[:,1:3],curve=curve,presence=out[:,-1],spatial_features=y)

class OwnershipAttentionNet(nn.Module):
    """Simultaneous point queries attend to predicted ownership and image keys."""
    def __init__(self):
        super().__init__();self.base=WholeInstanceNet();self.keys=nn.Conv2d(24,32,1)
        self.queries=nn.Sequential(nn.Linear(3,48),nn.SiLU(),nn.Linear(48,32))
        self.log_sigma=nn.Parameter(torch.tensor(-2.3));self.mask_weight=nn.Parameter(torch.tensor(1.5))
    def forward(self,image):
        out=self.base(image);curve=out['curve'];feat=F.avg_pool2d(out['spatial_features'],4)
        key=self.keys(feat).flatten(2).transpose(1,2)
        t=torch.linspace(0,1,POINTS,device=curve.device).expand(len(curve),-1)[...,None]
        query=self.queries(torch.cat([curve,t],-1));logits=query@key.transpose(1,2)/math.sqrt(32)
        grid=torch.linspace(0,1,24,device=curve.device);yy,xx=torch.meshgrid(grid,grid,indexing='ij');coords=torch.stack([xx,yy],-1).reshape(-1,2)
        distance=((curve[:,:,None]-coords[None,None])**2).sum(-1)
        sigma=self.log_sigma.exp().clamp(.03,.18)
        owner=F.avg_pool2d(out['mask'],4).flatten(2)[:,0]
        logits=logits-distance/(2*sigma*sigma)+self.mask_weight.clamp(.5,4)*owner[:,None]
        attention=logits.softmax(-1);refined=attention@coords
        out['curve']=.25*curve+.75*refined
        return out

def losses(out,batch):
    _,mask,valid,target,presence,endpoint_valid=batch
    mask=mask.float()/255;valid=valid.float()/255;good=presence>.5
    bce=F.binary_cross_entropy_with_logits(out['mask'][:,0],mask,reduction='none',pos_weight=mask.new_tensor(4.))
    ownership=(bce*valid).sum()/valid.sum().clamp_min(1)
    prob=out['mask'][:,0].sigmoid()*valid;truth=mask*valid
    dice=(1-(2*(prob*truth).sum((-2,-1))+1)/(prob.sum((-2,-1))+truth.sum((-2,-1))+1)).mean()
    curve=out['curve'];geom=mask.new_tensor(0.);direction=geom;arc=geom;consistency=geom
    if good.any():
        pred=curve[good];tar=target[good]
        geom=F.smooth_l1_loss(pred,tar,beta=.04)
        direction=(1-(F.normalize(pred[:,1:]-pred[:,:-1],dim=-1)*F.normalize(tar[:,1:]-tar[:,:-1],dim=-1)).sum(-1)).mean()
        pl=torch.linalg.vector_norm(pred[:,1:]-pred[:,:-1],dim=-1).sum(1);tl=torch.linalg.vector_norm(tar[:,1:]-tar[:,:-1],dim=-1).sum(1)
        arc=F.smooth_l1_loss(pl,tl)
        consistency=-F.grid_sample(out['mask'][good].sigmoid(),(pred*2-1)[:,None],align_corners=True).clamp_min(.01).log().mean()
    confidence=F.binary_cross_entropy_with_logits(out['presence'],presence.float())
    endpoint=mask.new_tensor(0.);epgood=good&(endpoint_valid>.5)
    if epgood.any():
        grid=torch.linspace(0,1,SIZE,device=mask.device);yy,xx=torch.meshgrid(grid,grid,indexing='ij')
        ends=target[epgood][:,[0,-1]];heat=torch.exp(-((xx[None,None]-ends[:,:,0,None,None])**2+(yy[None,None]-ends[:,:,1,None,None])**2)/(2*(1.6/SIZE)**2))
        endpoint=F.binary_cross_entropy_with_logits(out['endpoints'][epgood],heat,pos_weight=mask.new_tensor(20.))
    return ownership+dice+12*geom+.15*direction+.5*arc+.08*consistency+.25*confidence+.3*endpoint

def data(split):
    z=np.load(DATA_ROOT/(split+'.npz'));names=['image','mask','valid','curve','presence','endpoint_valid']
    tensors=[torch.as_tensor(z[k],device='cuda') for k in names]
    for i in (3,4,5):tensors[i]=tensors[i].float()
    return tensors,z['kind']

@torch.no_grad()
def evaluate(net,dataset):
    tensors,kinds=dataset;distance=[];coverage=[];ious=[];names=[];confidence=[]
    straight_distance=[]
    for begin in range(0,len(kinds),48):
        batch=[a[begin:begin+48] for a in tensors];out=net(batch[0]);good=batch[4]>.5
        pred=out['curve'][good].cpu().numpy();target=batch[3][good].cpu().numpy()
        for p,q in zip(pred,target):
            distance.append(float(np.linalg.norm(p-q,axis=1).mean()*SIZE))
            d=np.linalg.norm(q[:,None]-p[None],axis=2).min(1)*SIZE;coverage.append(float((d<=2).mean()))
        names += kinds[begin:begin+48][good.cpu().numpy()].tolist()
        prediction=out['mask'][:,0].sigmoid()>.5;truth=batch[1]>127;valid=batch[2]>127
        intersection=(prediction&truth&valid).sum((-2,-1));union=((prediction|truth)&valid).sum((-2,-1))
        ious += ((intersection+1)/(union+1)).cpu().tolist();confidence+=out['presence'].sigmoid().cpu().tolist()
    distance=np.array(distance);coverage=np.array(coverage);names=np.asarray(names)
    def summary(mask):
        return dict(count=int(mask.sum()),mean_ordered_curve_error_normalized_px=float(distance[mask].mean()) if mask.any() else None,
            mean_reference_point_coverage_2px=float(coverage[mask].mean()) if mask.any() else None,
            whole_curve_80pct_coverage_fraction=float((coverage[mask]>=.8).mean()) if mask.any() else None)
    return dict(real_reference_seeded=summary(names=='explicit_real_reference_tube'),
        known_rendered_seeded=summary(names=='known_rendered_instance'),
        mean_valid_pixel_ownership_iou=float(np.mean(ious)),not_full_image_extraction_accuracy=True)

def train(steps=2400,attention=False):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4);torch.manual_seed(SEED);net=(OwnershipAttentionNet() if attention else WholeInstanceNet()).cuda()
    if attention:net.base.load_state_dict(torch.load(DATA_ROOT/'instance_curve.pt',map_location='cuda',weights_only=True)['model'])
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4 if attention else 2e-4,weight_decay=1e-4)
    training=data('train');calibration=data('calibration');real=torch.as_tensor(np.where(training[1]=='explicit_real_reference_tube')[0],device='cuda')
    best=float('inf');logs=[];start=time.monotonic()
    for step in range(1,steps+1):
        net.train();ids=torch.randint(len(training[1]),(24,),device='cuda')
        if step>800:ids=torch.cat([ids[:12],real[torch.randint(len(real),(12,),device='cuda')]])
        batch=[a[ids] for a in training[0]]
        with torch.autocast('cuda',dtype=torch.bfloat16):out=net(batch[0]);value=losses(out,batch)
        opt.zero_grad(set_to_none=True);value.backward();grad=nn.utils.clip_grad_norm_(net.parameters(),3.)
        if not torch.isfinite(value) or not torch.isfinite(grad):raise ValueError('Nonfinite training')
        opt.step()
        if step%200==0:
            net.eval();metrics=evaluate(net,calibration);row=dict(step=step,loss=float(value.detach()),calibration=metrics,seconds=time.monotonic()-start);logs.append(row)
            score=metrics['real_reference_seeded']['mean_ordered_curve_error_normalized_px']
            if step>=800 and score<best:best=score;torch.save(dict(model=net.state_dict(),step=step,calibration=metrics),ROOT/'instance_curve.pt')
            print('INSTANCE_TRAIN',json.dumps(row),flush=True)
    ckpt=torch.load(ROOT/'instance_curve.pt',map_location='cuda',weights_only=True);net.load_state_dict(ckpt['model']);net.eval();test=evaluate(net,data('test'))
    save(ROOT/'training_summary.json',dict(updates=steps,retained_step=ckpt['step'],gpu=torch.cuda.get_device_name(),
        parameters=sum(p.numel() for p in net.parameters()),seconds=time.monotonic()-start,model_sha256=sha(ROOT/'instance_curve.pt'),
        calibration=ckpt['calibration'],test=test,logs=logs,new_independent_real_test=False,robot_connected=False,
        ownership_attention_decoder=attention,prior_test_seen_before_this_variant=attention,
        warm_start_sha256=sha(DATA_ROOT/'instance_curve.pt') if attention else None))
    print('INSTANCE_TEST',json.dumps(test),flush=True)

def load(attention=False):
    net=(OwnershipAttentionNet() if attention else WholeInstanceNet()).cuda();net.load_state_dict(torch.load(ROOT/'instance_curve.pt',map_location='cuda',weights_only=True)['model']);return net.eval()

@torch.no_grad()
def infer_queries(net,rgb,queries):
    """Only RGB and short seed fragments; no label or reference-file access."""
    items=[]
    for query in queries:
        x,origin=crop_query(rgb,query['center'],query['side'],query['back'],query['front']);items.append((x,origin,query))
    output=[]
    for begin in range(0,len(items),48):
        group=items[begin:begin+48];x=torch.as_tensor(np.stack([r[0] for r in group]),device='cuda');pred=net(x)
        curves=pred['curve'].float().cpu().numpy();masks=pred['mask'].sigmoid().float().cpu().numpy()[:,0];confidence=pred['presence'].sigmoid().float().cpu().numpy()
        for k,(_,origin,q) in enumerate(group):
            points=origin+(curves[k]*(SIZE-1)+.5)*q['side']/SIZE
            output.append(dict(points=points,mask=masks[k],origin=origin,side=q['side'],presence=float(confidence[k]),query=q))
    return output

if __name__=='__main__':
    if sys.argv[1]=='attention_train':
        ROOT=ROOT/'ownership_attention';ROOT.mkdir(exist_ok=True)
        protocol=json.loads((DATA_ROOT/'protocol.json').read_text('utf8'));protocol.update(variant='ownership-conditioned spatial attention for simultaneous points',
            data_root=str(DATA_ROOT),warm_start_sha256=sha(DATA_ROOT/'instance_curve.pt'),prior_variant_test_already_seen=True)
        save(ROOT/'protocol.json',protocol)
    {'prepare':prepare,'train':train,'attention_train':lambda:train(attention=True)}[sys.argv[1]]()
