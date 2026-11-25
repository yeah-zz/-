"""Offline direction/centerline learning, analytic weak teacher and stroke planning.

No hardware SDK. Real targets are weak labels, never claimed as human strokes.
"""
import math,json,time,hashlib,sys
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import distance_transform_edt,gaussian_filter1d

def ridge_fields(density):
    d=np.asarray(density,np.float32);h,w=d.shape;yy,xx=np.mgrid[:h,:w].astype(np.float32)
    best=np.zeros_like(d);tx=np.ones_like(d);ty=np.zeros_like(d);width=np.ones_like(d)
    for sigma in (.8,1.25,1.9,2.8):
        g=cv2.GaussianBlur(d,(0,0),sigma)
        a=cv2.Sobel(g,cv2.CV_32F,2,0,ksize=3)/4;b=cv2.Sobel(g,cv2.CV_32F,0,2,ksize=3)/4;c=cv2.Sobel(g,cv2.CV_32F,1,1,ksize=3)/4
        gap=np.sqrt((a-b)**2+4*c*c);large=(a+b+gap)/2;small=(a+b-gap)/2
        theta=.5*np.arctan2(2*c,a-b)
        tangent_x=np.cos(theta);tangent_y=np.sin(theta);nx=-tangent_y;ny=tangent_x
        side1=cv2.remap(g,xx+nx*1.2,yy+ny*1.2,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        side2=cv2.remap(g,xx-nx*1.2,yy-ny*1.2,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        coherence=np.clip(1-np.abs(large)/(np.abs(small)+1e-6),0,1)
        response=np.maximum(-small,0)*sigma**1.5*coherence
        response*=((g>=side1)&(g>=side2)&(g>.035)&(coherence>.32))
        choose=response>best;best[choose]=response[choose];tx[choose]=tangent_x[choose];ty[choose]=tangent_y[choose];width[choose]=sigma*2.35
    binary=(best>.004).astype(np.uint8)
    # Local ridge centers only; do not skeletonize the filled ink region.
    try:
        from skimage.morphology import skeletonize
        binary=skeletonize(binary).astype(np.uint8)
    except ImportError:pass
    return best*binary,tx,ty,width

def weak_targets(rgb,density):
    score,tx,ty,width=ridge_fields(density)
    valid=score>.004
    # Ignore red labels; black typography still requires crop review.
    r,g,b=rgb.transpose(2,0,1).astype(float)
    ignored=cv2.dilate(((r>g+18)&(r>b+15)).astype(np.uint8),np.ones((5,5),np.uint8))>0
    valid[ignored]=False
    if valid.sum():
        distance,idx=distance_transform_edt(~valid,return_indices=True)
        center=np.exp(-distance**2/(2*.85**2)).astype(np.float32)
        cos=(tx*tx-ty*ty)[tuple(idx)];sin=(2*tx*ty)[tuple(idx)];wi=width[tuple(idx)]/12
    else:center=np.zeros_like(density);cos=center.copy();sin=center.copy();wi=center.copy()
    target=np.stack((center,cos,sin,np.clip(wi,0,1))).astype(np.float32)
    confidence=np.ones_like(center)*.45;confidence[ignored]=0
    # Broad flat ink and wash are not known line-negative ground truth.
    confidence[(density>.25)&(center<.1)]=.12
    return target,confidence

def procedural(rng,size=256,count=None):
    h=w=size;ink=np.zeros((h,w),np.float32);center=np.zeros_like(ink);cs=np.zeros_like(ink);sn=np.zeros_like(ink);widthmap=np.zeros_like(ink);confidence=np.ones_like(ink)
    points=[]
    for _ in range(count or int(rng.integers(8,25))):
        length=rng.uniform(25,size*.6);start=rng.uniform(12,size-12,2)
        theta=rng.uniform(0,2*np.pi);direction=np.array([np.cos(theta),np.sin(theta)])
        normal=np.array([-direction[1],direction[0]]);t=np.linspace(0,1,max(30,int(length*1.8)))
        p=start[None]+t[:,None]*length*direction+np.sin(t[:,None]*np.pi)*normal*rng.uniform(-18,18)
        p=np.clip(p,3,size-4);taper=.2+.8*np.sin(np.minimum(t/.85,1)*np.pi/2)**1.2
        taper*=np.clip((1-t)/.12,0,1)*.8+.2
        widths=rng.uniform(1.5,8)*taper;alpha=rng.uniform(.14,.85)*(1-.20*t)
        layer=np.zeros_like(ink)
        for (x,y),wi,opacity in zip(p,widths,alpha):cv2.circle(layer,(round(x),round(y)),max(0,round((wi-1)/2)),float(opacity),-1)
        # Some dry-brush holes in the rendered ink, while truth is an open stroke.
        if rng.random()<.35:layer*=rng.uniform(.45,1,layer.shape).astype(np.float32)
        ink=np.maximum(ink,layer)
        dp=np.gradient(p,axis=0);theta=np.arctan2(dp[:,1],dp[:,0])
        for (x,y),a,wi in zip(p,theta,widths):
            x=round(x);y=round(y);center[y,x]=1;cs[y,x]=np.cos(2*a);sn[y,x]=np.sin(2*a);widthmap[y,x]=wi/12
        points.append(p)
    dist,idx=distance_transform_edt(center==0,return_indices=True)
    target=np.stack((np.exp(-dist**2/(2*.85**2)),cs[tuple(idx)],sn[tuple(idx)],widthmap[tuple(idx)])).astype(np.float32)
    confidence[(cv2.GaussianBlur(center,(0,0),1)>0.14)&(ink>.7)]=.5
    paper=np.ones((size,size,3),np.float32)*rng.uniform(205,254,(1,1,3))
    paper+=cv2.resize(rng.normal(0,1,(6,6)).astype(np.float32),(size,size))[:,:,None]*rng.uniform(0,7)
    rgb=np.clip(paper*(1-ink[:,:,None])+rng.normal(0,1,(size,size,1)),0,255).astype(np.uint8)
    return rgb,target,confidence,points

def model_new():
    import torch
    from .model import GongbiLineNet
    model=GongbiLineNet();model.head=torch.nn.Conv2d(24,4,1)
    with torch.no_grad():model.head.bias.zero_();model.head.bias[0]=-2
    return model

def loss_fn(logits,target,weight):
    import torch
    import torch.nn.functional as F
    p=logits.float();target=target.float();weight=weight.float()
    b=F.binary_cross_entropy_with_logits(p[:,0],target[:,0],pos_weight=p.new_tensor(7.),reduction='none')
    b=(b*weight).sum()/weight.sum().clamp_min(1)
    prob=p[:,0].sigmoid();truth=target[:,0];dice=1-(2*(prob*truth*weight).sum()+1)/((prob+truth)*weight).sum().add(1)
    mask=(truth>.25)*weight
    angle=1-F.cosine_similarity(p[:,1:3],target[:,1:3],dim=1,eps=1e-5)
    angle=(angle*mask).sum()/mask.sum().clamp_min(1)
    width=(F.smooth_l1_loss(p[:,3].sigmoid(),target[:,3],reduction='none')*mask).sum()/mask.sum().clamp_min(1)
    return b+.5*dice+.3*angle+2*width

def train(root,parent,rounds=3,steps=800):
    import torch
    from collections import OrderedDict
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(20261023);torch.backends.cudnn.benchmark=True
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    training=[r for r in manifest['real'] if r['split']=='train'];validation=[r for r in manifest['real'] if r['split']=='validation']
    cache=OrderedDict();size=256
    print('PRECOMPUTING_PROCEDURAL_OPEN_STROKES',flush=True)
    synthetic_rng=np.random.default_rng(73001)
    synthetic_pool=[procedural(synthetic_rng,size)[:3] for _ in range(160)]
    synthetic_validation=[procedural(np.random.default_rng(83000+i),size)[:3] for i in range(10)]
    validation_call=[0]
    def sample(rng,real_ratio,validation_index=None):
        real=validation_index is not None or rng.random()<real_ratio
        if real:
            rows=validation if validation_index is not None else training;r=rows[validation_index%len(rows)] if validation_index is not None else rows[int(rng.integers(len(rows)))]
            if r['id'] not in cache:
                rgb=np.asarray(Image_open(r['image']));z=np.load(r['direction_labels']);cache[r['id']]=(rgb,z['target'].astype(np.float32),z['confidence'].astype(np.float32));z.close()
                if len(cache)>70:cache.popitem(last=False)
            rgb,t,conf=cache[r['id']];h,w=rgb.shape[:2];s=int(size/rng.choice([.8,1.,1.2]))
            if h<s or w<s:
                pad=(max(0,s-h),max(0,s-w));rgb=cv2.copyMakeBorder(rgb,0,pad[0],0,pad[1],cv2.BORDER_REFLECT_101);t=np.pad(t,((0,0),(0,pad[0]),(0,pad[1])));conf=np.pad(conf,((0,pad[0]),(0,pad[1])));h,w=rgb.shape[:2]
            y=int(rng.integers(h-s+1));x=int(rng.integers(w-s+1));rgb=cv2.resize(rgb[y:y+s,x:x+s],(size,size));t=np.stack([cv2.resize(v[y:y+s,x:x+s],(size,size)) for v in t]);conf=cv2.resize(conf[y:y+s,x:x+s],(size,size));t[3]*=size/s
        else:
            pool=synthetic_validation if validation_call[0] else synthetic_pool
            rgb,t,conf=pool[int(rng.integers(len(pool)))];rgb=rgb.copy();t=t.copy();conf=conf.copy()
        if validation_index is None:
            if rng.random()<.5:rgb=rgb[:,::-1].copy();t=t[:,:,::-1].copy();t[2]*=-1;conf=conf[:,::-1].copy()
            rgb=np.clip(rgb.astype(float)*rng.uniform(.83,1.1)+rng.uniform(-8,8),0,255).astype(np.uint8)
        return torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255,torch.from_numpy(t.copy()),torch.from_numpy(conf.copy()),real
    model=model_new().cuda();ckpt=torch.load(parent/'best.pt',map_location='cpu',weights_only=False)
    state={k:v for k,v in ckpt['model'].items() if not k.startswith('head.')};model.load_state_dict(state,strict=False)
    del ckpt;all_summary=[];used=[0,0];start=time.perf_counter();rng=np.random.default_rng(7023)
    def validate():
        model.eval();values=[];metrics=[];vrng=np.random.default_rng(9707);validation_call[0]=1
        with torch.no_grad():
            for i in range(20):
                a=sample(vrng,0,validation_index=i) if i>=10 else sample(vrng,0)
                x,t,c,_=a;x=x[None].cuda();t=t[None].cuda();c=c[None].cuda()
                out=model(x);values.append(float(loss_fn(out,t,c)))
                if i<10:
                    theta=out[:,1:3];truth=t[:,1:3];valid=t[:,0]>.4
                    cos=torch.nn.functional.cosine_similarity(theta,truth,dim=1)
                    metrics.append(float((.5*torch.acos(cos.clamp(-1,1))[valid]).mean()*180/np.pi))
        validation_call[0]=0;model.train();return dict(loss=float(np.mean(values)),synthetic_orientation_error_degrees=float(np.mean(metrics)),real_labels='automatic ridge weak teacher, not expert',synthetic_cases=10,real_cases=10)
    for ri in range(1,rounds+1):
        run=root/'runs'/f'round_{ri}';run.mkdir(exist_ok=True);best=float('inf');head_start=model.head.weight.detach().clone()
        opt=torch.optim.AdamW(model.parameters(),lr=[1e-4,6e-5,3e-5][ri-1],weight_decay=1e-4)
        for step in range(1,steps+1):
            # Known individual open strokes stay in every round, preventing weak-label-only training.
            batch=[sample(rng,[.35,.6,.65][ri-1]) for _ in range(6)]
            for a in batch:used[int(a[3])]+=1
            x,t,c=[torch.stack([a[k] for a in batch]).cuda() for k in range(3)]
            for g in opt.param_groups:g['lr']=[1e-4,6e-5,3e-5][ri-1]*(.25+.75*.5*(1+math.cos(math.pi*step/steps)))
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x);loss=loss_fn(out,t,c)
            assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient; no update applied')
            opt.step()
            if step%50==0:
                row=dict(round=ri,step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start)
                with (run/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                print('TRAIN',json.dumps(row),flush=True)
            if step%200==0 or step==steps:
                v=validate();payload=dict(model=model.state_dict(),step=step,round=ri,validation=v,channels=['centerline_logit','cos_2theta','sin_2theta','relative_width_logit'],ready_for_robot=False)
                torch.save(payload,run/'last.pt')
                if v['loss']<best:best=v['loss'];torch.save(payload,run/'best.pt')
                print('VALIDATION',ri,step,json.dumps(v),flush=True)
        delta=float((model.head.weight.detach()-head_start).abs().mean());assert delta>0
        summary=dict(round=ri,successful_optimizer_updates=steps,head_mean_absolute_update=delta,best_validation_loss=best,validation_last=v)
        (run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');all_summary.append(summary)
        # Continue each next round from validation-selected weights, never from test results.
        model.load_state_dict(torch.load(run/'best.pt',map_location='cuda',weights_only=False)['model'])
    summary=dict(rounds=all_summary,total_successful_updates=rounds*steps,elapsed_seconds=time.perf_counter()-start,consumed_samples=dict(procedural=used[0],real_weak=used[1]),initialization=str(parent/'best.pt'),initialization_sha256=hashlib.sha256((parent/'best.pt').read_bytes()).hexdigest(),target_provenance='known procedural open strokes + automatic directional ridges; no expert pen-down labels',actual_drawing=False,ready_for_robot=False)
    (root/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('TRAINING_COMPLETE',json.dumps(summary),flush=True)

def Image_open(path):
    from PIL import Image
    return Image.open(path).convert('RGB')

def infer(model,rgb,tile=256,stride=192):
    import torch
    h,w=rgb.shape[:2];pad=cv2.copyMakeBorder(rgb,20,max(20,tile-h-20),20,max(20,tile-w-20),cv2.BORDER_REFLECT_101);ph,pw=pad.shape[:2]
    starts=lambda n:sorted(set(list(range(0,n-tile+1,stride))+[n-tile]))
    locations=[(y,x) for y in starts(ph) for x in starts(pw)];window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32);total=np.zeros((4,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32)
    with torch.no_grad():
        for start in range(0,len(locations),8):
            loc=locations[start:start+8];arr=np.stack([pad[y:y+tile,x:x+tile].transpose(2,0,1) for y,x in loc]);x=torch.from_numpy(arr).cuda().float()/255;out=model(x).float();out[:,0]=out[:,0].sigmoid();out[:,3]=out[:,3].sigmoid();out[:,1:3]=torch.nn.functional.normalize(out[:,1:3],dim=1);out=out.cpu().numpy()
            for (y,x),p in zip(loc,out):total[:,y:y+tile,x:x+tile]+=p*window;den[y:y+tile,x:x+tile]+=window
    return (total/den)[:,20:20+h,20:20+w]

def plan(ink,prediction=None):
    """Ridge graph with tangent-paired junctions, never hole-boundary tracing."""
    coverage,density=ink;h,w=density.shape
    response,tx,ty,teacher_width=ridge_fields(density)
    mask=(response>.004)&(coverage>.26)
    if prediction is not None:
        # Learned gate is an explicit second stage; analytic ridge localization
        # and orientation remain necessary with current weak real labels.
        mask&=(prediction[0]>.20)
        a=.5*np.arctan2(prediction[2],prediction[1]);px=np.cos(a);py=np.sin(a)
        sign=np.where(px*tx+py*ty>=0,1,-1)
        tx=.65*tx+.35*px*sign;ty=.65*ty+.35*py*sign
    mask[[0,-1],:]=False;mask[:,[0,-1]]=False
    ys,xs=np.nonzero(mask);points=np.stack((xs,ys),axis=1);index={tuple(p):i for i,p in enumerate(points)};adj=[set() for _ in points];edges={}
    for i,(x,y) in enumerate(points):
        for dx,dy in ((1,0),(0,1),(1,1),(-1,1)):
            j=index.get((x+dx,y+dy))
            if j is None:continue
            if dx and dy and ((x+dx,y) in index or (x,y+dy) in index):continue
            direction=np.array([dx,dy],float);direction/=np.linalg.norm(direction)
            a=abs(tx[y,x]*direction[0]+ty[y,x]*direction[1]);b=abs(tx[y+dy,x+dx]*direction[0]+ty[y+dy,x+dx]*direction[1])
            if max(a,b)<.40:continue
            adj[i].add(j);adj[j].add(i);edges[tuple(sorted((i,j)))]=True
    # Pair edges at each intersection by continuation angle. Odd leftover arms
    # become separate strokes, rather than circling the whole component.
    pairing={}
    for i,neighbors in enumerate(adj):
        candidates=[]
        for a in neighbors:
            for b in neighbors:
                if a>=b:continue
                va=points[a]-points[i];vb=points[b]-points[i];cos=float(va@vb/(np.linalg.norm(va)*np.linalg.norm(vb)))
                if cos<-.35:candidates.append((cos,a,b))
        used=set()
        for cos,a,b in sorted(candidates):
            if a in used or b in used:continue
            pairing[(i,a)]=b;pairing[(i,b)]=a;used.update((a,b))
    visited=set();raw=[]
    directed=[]
    for a,b in edges:
        if (a,b) not in pairing:directed.append((a,b))
        if (b,a) not in pairing:directed.append((b,a))
    directed+=list(edges)
    for a,b in directed:
        key=tuple(sorted((a,b)))
        if key in visited:continue
        path=[a,b];visited.add(key);previous,current=a,b
        while (current,previous) in pairing:
            nxt=pairing[(current,previous)];key=tuple(sorted((current,nxt)))
            if key in visited:break
            path.append(nxt);visited.add(key);previous,current=current,nxt
        raw.append(points[path].astype(float))
    paths=[];rejected=[]
    for p in raw:
        length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum());span=np.ptp(p,axis=0)
        if length<9:continue
        near_closed=np.linalg.norm(p[0]-p[-1])<max(3,.09*length)
        if near_closed and max(span)<38:
            rejected.append(dict(reason='small_near_closed_ridge_candidate',length=length,bounds=span.tolist()));continue
        # Smooth only within each already connected stroke. No joining over blank.
        q=np.stack([gaussian_filter1d(p[:,k],.8,mode='nearest') for k in range(2)],axis=1);q[0]=p[0];q[-1]=p[-1]
        paths.append(q)
    # Repair only tiny ink-supported gaps, with both tangents aligned. Never
    # connect two endpoints merely because nearest-neighbor travel is short.
    from scipy.spatial import cKDTree
    for _ in range(3):
        if len(paths)<2:break
        endpoints=np.array([p[k] for p in paths for k in (0,-1)]);tree=cKDTree(endpoints);candidates=[]
        for a,b in tree.query_pairs(4.5):
            ia,ea=divmod(a,2);ib,eb=divmod(b,2)
            if ia==ib:continue
            pa,pb=paths[ia],paths[ib];va=pa[0]-pa[min(5,len(pa)-1)] if ea==0 else pa[-1]-pa[max(0,len(pa)-6)];vb=pb[0]-pb[min(5,len(pb)-1)] if eb==0 else pb[-1]-pb[max(0,len(pb)-6)]
            va/=max(1e-5,np.linalg.norm(va));vb/=max(1e-5,np.linalg.norm(vb));gap=endpoints[b]-endpoints[a];distance=np.linalg.norm(gap)
            if va@vb>-.82:continue
            if distance>.5 and min(va@gap/distance,-vb@gap/distance)<.65:continue
            sample=np.rint(np.linspace(endpoints[a],endpoints[b],max(3,math.ceil(distance*2)+1))).astype(int)
            evidence=density[sample[:,1],sample[:,0]]
            if evidence.min()<.035 or evidence.mean()<.07:continue
            candidates.append((distance,ia,ea,ib,eb))
        used=set();merged=[]
        for _,ia,ea,ib,eb in sorted(candidates):
            if ia in used or ib in used:continue
            a=paths[ia][::-1] if ea==0 else paths[ia];b=paths[ib][::-1] if eb==1 else paths[ib]
            q=np.concatenate((a,b))
            if np.linalg.norm(q[0]-q[-1])<4:continue
            merged.append(q);used.update((ia,ib))
        paths=merged+[p for i,p in enumerate(paths) if i not in used]
    # Direction is ambiguous in a finished painting. The thin endpoint is a
    # heuristic start; otherwise prefer top-to-bottom, and disclose this.
    records=[];plan_img=np.full((h,w,3),255,np.uint8);replay=np.zeros((h,w),np.float32)
    for p in sorted(paths,key=lambda p:(p[:,1].min(),p[:,0].min())):
        xy=np.rint(p).astype(int);u=np.linspace(0,1,len(p));d=density[xy[:,1],xy[:,0]]
        tang=np.gradient(p,axis=0);tang/=np.maximum(np.linalg.norm(tang,axis=1)[:,None],1e-5);normal=np.stack((-tang[:,1],tang[:,0]),axis=1)
        widths=np.ones(len(p),np.float32)
        # Width from the local cross-stroke intensity profile, capped for this
        # pixel-scale preview; broad washes are deliberately not single strokes.
        for sign in (-1,1):
            alive=np.ones(len(p),bool)
            for radius in range(1,7):
                sample=np.rint(p+normal*radius*sign).astype(int);sample[:,0]=np.clip(sample[:,0],0,w-1);sample[:,1]=np.clip(sample[:,1],0,h-1)
                alive&=density[sample[:,1],sample[:,0]]>np.maximum(.045,d*.35)
                widths+=alive
        if prediction is not None:widths=.7*widths+.3*12*prediction[3,xy[:,1],xy[:,0]]
        widths=np.clip(gaussian_filter1d(widths,2),1,8);d=gaussian_filter1d(d,1.5)
        if widths[:min(5,len(p))].mean()>widths[-min(5,len(p)):].mean()+.6 or (abs(widths[0]-widths[-1])<.6 and p[0,1]>p[-1,1]):p=p[::-1];xy=xy[::-1];widths=widths[::-1];d=d[::-1]
        # Gentle virtual contact ramp, inferred/stylized, not recovered pressure.
        ramp=np.interp(u,[0,.08,.92,1],[.35,1,1,.4]);widths=np.maximum(1,widths*ramp)
        pressure=np.clip((widths-1)/7,0,1);color=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[int(len(records)*37%180),180,170]]]),cv2.COLOR_HSV2RGB)[0,0])
        cv2.polylines(plan_img,[np.rint(p).astype(np.int32)],False,color,1,cv2.LINE_AA)
        # Continuous supersampled trapezoid ribbon, not a pile of sparse disks.
        scale=3;layer=np.zeros((h*scale,w*scale),np.float32)
        for k in range(len(p)-1):
            delta=p[k+1]-p[k];norm=np.linalg.norm(delta)
            if norm<1e-4:continue
            n=np.array([-delta[1],delta[0]])/norm
            polygon=np.array([p[k]+n*widths[k]/2,p[k+1]+n*widths[k+1]/2,p[k+1]-n*widths[k+1]/2,p[k]-n*widths[k]/2])
            cv2.fillConvexPoly(layer,np.rint(polygon*scale).astype(np.int32),float((d[k]+d[k+1])/2))
        replay=np.maximum(replay,cv2.resize(layer,(w,h),interpolation=cv2.INTER_AREA))
        records.append(dict(order=len(records),xy_pixels=np.round(p,2).tolist(),xy_normalized=np.round(p/[max(1,w-1),max(1,h-1)],6).tolist(),width_pixels=np.round(widths,2).tolist(),relative_pressure=np.round(pressure,3).tolist(),ink_density=np.round(d,4).tolist(),role='directional_ink_line_candidate_not_verified_cunfa',direction_provenance='thin-endpoint or top-to-bottom heuristic; original motion unknown',virtual_pen='down'))
    info=dict(image_width=w,image_height=h,planned_strokes=len(records),rejected_small_closed_candidates=len(rejected),rejected=rejected,planner='directional ridge graph and tangent-paired junctions; learned gate '+str(prediction is not None),pixel_width_cap=8,physical_z=None,physical_speed=None,paper_transform=None,robot_connected=False,actually_drawn=False,ready_for_robot=False,pressure_units='relative 0..1 uncalibrated; not force or millimeters',stroke_identity='heuristic candidates; not recovered human strokes',strokes=records)
    return info,plan_img,np.rint(255*(1-replay)).astype(np.uint8)
