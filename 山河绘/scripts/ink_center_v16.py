# -*- coding: utf-8 -*-
"""Ink-ribbon specialist: independent weights, no robot SDK, no reference at inference."""
from rock_v15_common import *
V15=ROOT
ROOT=LAB/'ink_ribbon_v16_20261025'
import math,time,shutil
import numpy as np,cv2,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from human_mountain_v11 import fields,paste
from rock_pairs_v15 import reference_records,dense
from topology_trace_v13 import graph_paths

def photometry(rgb):
    a=rgb.astype(np.float32)/255;g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);paper=cv2.GaussianBlur(cv2.dilate(g,np.ones((41,41),np.uint8)),(0,0),9);contrast=np.clip((paper-g)/np.maximum(paper,.15),0,1);fine=np.clip((cv2.GaussianBlur(g,(0,0),3)-g)/np.maximum(paper,.15)*4,0,1);sat=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[:,:,1].astype(np.float32)/255
    x=np.concatenate((a,np.minimum(g/np.maximum(paper,.15),1)[:,:,None],fine[:,:,None],np.minimum(contrast*3,1)[:,:,None],sat[:,:,None]),2)
    return x.transpose(2,0,1).copy(),contrast,paper

class RibbonNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
        self.a=block(7,16);self.b=block(16,32);self.c=block(32,64);self.d=block(96,32);self.e=block(48,16);self.head=nn.Conv2d(16,5,1)
        with torch.no_grad():self.head.bias.zero_();self.head.bias[0]=-2;self.head.bias[1]=-1
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1));return self.head(self.e(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1)))

def geometry(paths,shape):
    doc=dict(width=shape[1],height=shape[0],strokes=[dict(role='contour',points=np.asarray(p).tolist()) for p in paths],regions=[]);center,angle,_,dist=fields(doc);return center,angle,dist

def generated(seed,size=256):
    rng=np.random.default_rng(seed);paths=[];widths=[];center=np.zeros((size,size),np.uint8);ink=np.zeros((size,size),np.float32);wi=np.zeros_like(ink);paper=np.ones((size,size,3),np.float32)*rng.uniform(190,255,(1,1,3));paper+=cv2.resize(rng.normal(0,1,(5,5)).astype(np.float32),(size,size))[:,:,None]*rng.uniform(0,5)
    # Palette washes are nuisance backgrounds, never target ink ribbons.
    if seed%4==0:
        color=np.asarray(rng.choice([[.65,.84,.75],[.53,.71,.83],[.84,.8,.66]]));wash=np.zeros((size,size),np.float32);cv2.ellipse(wash,(130,140),(100,90),0,0,360,.25,-1);wash=cv2.GaussianBlur(wash,(0,0),12);paper*=1-wash[:,:,None]*(1-color[None,None])
    lanes=int(rng.integers(3,7));u=np.linspace(0,1,200);kind=seed%3
    for k in range(lanes):
        if kind==0:p=np.stack((28+k*(size-56)/(lanes-1)+np.sin(u*2.5)*rng.uniform(-10,10),22+u*(size-44)),1)
        elif kind==1:p=np.stack((20+u*(size-40),30+k*(size-60)/(lanes-1)+np.sin(u*3)*rng.uniform(-10,10)),1)
        else:p=np.stack((25+u*(size-50),50+k*(size-100)/(lanes-1)+np.sin(u*math.pi)*rng.uniform(-25,25)),1)
        p=np.clip(p,12,size-13);paths.append(p);base=rng.choice([1.2,2.,3.5,7.,12.,18.,23.]);profile=base*np.interp(u,[0,.18,.82,1],[.35,.9,1,.5]);alpha=float(rng.choice([.035,.06,.10,.18,.35,.65,.88]));layer=np.zeros_like(ink)
        for (x,y),width in zip(p,profile):cv2.circle(layer,(round(x),round(y)),max(0,round(width/2)),alpha,-1)
        # Local dry-brush texture is not converted into multiple ground-truth lines.
        if seed%5==0:layer*=rng.uniform(.65,1,layer.shape).astype(np.float32)
        ink=np.maximum(ink,layer);xy=np.rint(p).astype(int);wi[xy[:,1],xy[:,0]]=profile/24
        widths.append(dict(width=float(base),alpha=alpha))
    rgb=np.clip(paper*(1-ink[:,:,None])+rng.normal(0,rng.uniform(.2,1.1),(size,size,1)),0,255).astype(np.uint8);c,angle,dist=geometry(paths,ink.shape);_,idx=distance_transform_edt(c==0,return_indices=True);width=wi[tuple(idx)];target=np.concatenate((np.exp(-dist[None]**2/(2*1.05**2)),(ink[None]>.008).astype(np.float32),angle,width[None])).astype(np.float32);mask=np.ones((4,size,size),np.float32);mask[2]=(dist<2.5);mask[3]=(dist<2.5)
    return rgb,target,mask,dict(paths=[p.tolist() for p in paths],stroke_properties=widths,seed=seed)

def textured_generation(seed,size=256,blank=False):
    rng=np.random.default_rng(seed+960000);rgb,t,m,meta=generated(seed,size)
    # New noise seeds and actual background negatives teach the faint-line
    # model that grain and page-edge contrast are not intended brush strokes.
    if blank:
        paper=np.ones((size,size,3),np.float32)*rng.uniform(185,250,(1,1,3));paper+=cv2.resize(rng.normal(0,1,(7,7)).astype(np.float32),(size,size))[:,:,None]*rng.uniform(1,8);rgb=np.clip(paper,0,255).astype(np.uint8);t[:]=0;m[:]=0;m[0:2]=1
    texture=rng.normal(0,rng.uniform(.8,3.5),(size,size,1));rgb=np.clip(rgb.astype(np.float32)+texture,0,255).astype(np.uint8)
    if seed%3==0:
        edge=np.ones((size,size),np.float32);cv2.rectangle(edge,(int(rng.integers(2,9)),int(rng.integers(2,9))),(size-4,size-4),float(rng.uniform(.86,.96)),1);rgb=np.clip(rgb*edge[:,:,None],0,255).astype(np.uint8)
    return rgb,t,m,meta

def weak_real(rgb):
    _,alpha,_=photometry(rgb);low=alpha>.06;high=alpha>.14
    for binary in (low,high):binary[:4]=False;binary[-4:]=False;binary[:,:4]=False;binary[:,-4:]=False
    a=skeletonize(low);b=skeletonize(high);near_b=distance_transform_edt(~b) if b.any() else np.full(b.shape,1e4);accepted=a&(near_b<2.5);degree=cv2.filter2D(a.astype(np.float32),-1,np.ones((3,3),np.float32))-a;accepted&=degree<=2
    empty=np.zeros((8,*a.shape),np.float32);paths,_=graph_paths(accepted,empty,False,min_length=16)
    if not paths:return None
    c,angle,dist=geometry(paths,a.shape);ink_width=distance_transform_edt(low)*2;_,idx=distance_transform_edt(c==0,return_indices=True);width=np.clip(ink_width[tuple(idx)]/24,0,1);target=np.concatenate((np.exp(-dist[None]**2/(2*1.05**2)),(alpha[None]>.09).astype(np.float32),angle,width[None])).astype(np.float32)
    # Soft analytic labels, not expert confirmation. Unknown gray ink is not negative.
    center_conf=((dist<14)&low)|((alpha<.01)&(dist>3));ink_conf=(alpha>.14)|(alpha<.01);mask=np.stack((center_conf,ink_conf,dist<2.5,dist<2.5)).astype(np.float32)*.35;mask[:,:5]=0;mask[:,-5:]=0;mask[:,:,:5]=0;mask[:,:,-5:]=0
    return target,mask

def prepare():
    (ROOT/'data').mkdir(parents=True,exist_ok=True);records=[];human=reference_records();new=read(V15/'new_labels'/'manifest.json')['records'];content={r['id']:r for r in read(V15/'curated_manifest.json')['records']}
    for r in human+[dict(r,split=r['split']) for r in new]:
        if r['split']!='train':continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=read(r['labels']);c,a,m,d=fields(doc);t=np.concatenate((np.exp(-d[None]**2/(2*1.05**2)),(d[None]<1.8).astype(np.float32),a,np.zeros_like(d[None]))).astype(np.float32);mc=m[0]+m[1];mask=np.stack((mc,mc*.5,m[2],np.zeros_like(d))).astype(np.float32)
        # R01's surviving old geometry is not an endpoint annotation.
        source='user_partial' if r in human else 'assistant_partial';path=ROOT/'data'/(r['id']+'.npz');np.savez_compressed(path,rgb=rgb,target=t,mask=mask);records.append(dict(id=r['id'],file=str(path),kind=source,image=r['image'],labels=r['labels'],image_sha256=digest(r['image']),label_sha256=digest(r['labels']),strokes=len(doc['strokes'])))
    for r in content.values():
        if r['split']!='train':continue
        rgb=np.asarray(Image.open(r['content_image']).convert('RGB'));z=weak_real(rgb)
        if z is None:continue
        t,m=z
        if r['id']=='sina_6586334590_14':m[:,:85,:110]=0
        path=ROOT/'data'/('weak_'+r['id']+'.npz');np.savez_compressed(path,rgb=rgb,target=t,mask=m);records.append(dict(id='weak_'+r['id'],file=str(path),kind='analytic_soft_not_expert',image=r['content_image'],image_sha256=digest(r['content_image']),source_family=r['source_group']))
    jsave(ROOT/'manifest.json',dict(records=records,synthetic_training_seeds=[161000,161159],synthetic_calibration_seeds=[162000,162019],synthetic_test_seeds=[163000,163039],heldout_current_training_ids=['R02','R03','R04','A05','N01','N10','N11','T01','T02','U_A06'],raster_parent_sha256=digest(PARENT),source_artwork_independence_not_established=True,unknown_not_background=True));print('V16_DATA_PREPARED',len(records),flush=True)

def loss_fn(o,t,m):
    o=o.float();b=F.binary_cross_entropy_with_logits(o[:,0],t[:,0],reduction='none',pos_weight=o.new_tensor(4));center=(b*m[:,0]).sum()/m[:,0].sum().clamp_min(1);prob=o[:,0].sigmoid();dice=1-(2*(prob*t[:,0]*m[:,0]).sum()+1)/(((prob+t[:,0])*m[:,0]).sum()+1);b=F.binary_cross_entropy_with_logits(o[:,1],t[:,1],reduction='none',pos_weight=o.new_tensor(2));ink=(b*m[:,1]).sum()/m[:,1].sum().clamp_min(1);angle=((1-F.cosine_similarity(o[:,2:4],t[:,2:4],dim=1))*m[:,2]).sum()/m[:,2].sum().clamp_min(1);width=(F.smooth_l1_loss(o[:,4].sigmoid(),t[:,4],reduction='none')*m[:,3]).sum()/m[:,3].sum().clamp_min(1);return center+.35*dice+.7*ink+.15*angle+.2*width

def crop(rgb,t,m,rng,fade=True,size=256):
    rgb=rgb.copy();t=t.copy();m=m.copy();h,w=rgb.shape[:2];pad_h=max(0,size-h);pad_w=max(0,size-w)
    if pad_h or pad_w:rgb=cv2.copyMakeBorder(rgb,0,pad_h,0,pad_w,cv2.BORDER_REFLECT_101);t=np.pad(t,((0,0),(0,pad_h),(0,pad_w)));m=np.pad(m,((0,0),(0,pad_h),(0,pad_w)));h,w=rgb.shape[:2]
    x=int(rng.integers(w-size+1));y=int(rng.integers(h-size+1));rgb=rgb[y:y+size,x:x+size];t=t[:,y:y+size,x:x+size];m=m[:,y:y+size,x:x+size]
    if rng.random()<.5:rgb=rgb[:,::-1].copy();t=t[:,:,::-1].copy();t[3]*=-1;m=m[:,:,::-1].copy()
    if fade and rng.random()<.55:
        _,_,paper=photometry(rgb);ratio=float(rng.choice([.2,.35,.6,.8]));paper_rgb=np.maximum(paper[:,:,None]*255,rgb.astype(np.float32));rgb=np.clip(paper_rgb+(rgb-paper_rgb)*ratio,0,255).astype(np.uint8)
    return photometry(rgb)[0],t,m

def train(stage):
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(160250+stage);rng=np.random.default_rng(160250+stage);folder=ROOT/f'stage_{stage:02d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_summary.json').exists():raise FileExistsError('preserve completed experiment')
    shutil.copy2(Path(__file__),folder/'source_at_start.py')
    records=read(ROOT/'manifest.json')['records'];native={r['id']:r.get('native') for r in reference_records()}
    def arrays(r):
        archive=np.load(r['file']);z={key:archive[key].copy() for key in ['rgb','target','mask']};rect=native.get(r['id'])
        if rect:
            left,top,w,h=rect;valid=np.zeros(z['rgb'].shape[:2],np.float32);valid[top:top+h,left:left+w]=1;z['mask']*=valid[None]
        return z
    partial=[arrays(r) for r in records if r['kind']!='analytic_soft_not_expert'];weak=[arrays(r) for r in records if r['kind']=='analytic_soft_not_expert'];pool=[generated(161000+i)[:3] for i in range(160)];model=RibbonNet().cuda()
    if stage>1:model.load_state_dict(torch.load(ROOT/f'stage_{stage-1:02d}'/'final.pt',map_location='cuda',weights_only=False)['model'])
    if stage==3:pool=[textured_generation(164000+i,blank=(i%4==0))[:3] for i in range(160)]
    opt=torch.optim.AdamW(model.parameters(),lr=.00065 if stage==1 else .00025,weight_decay=.0001);steps=1200 if stage<3 else 800;start=time.perf_counter();counts=dict(synthetic=0,partial=0,analytic_soft=0)
    for step in range(1,steps+1):
        batch=[]
        for _ in range(6):
            chance=rng.random()
            if chance<(.6 if stage==1 else .45):rgb,t,m=pool[int(rng.integers(len(pool)))];counts['synthetic']+=1
            else:
                group=partial if chance<.85 else weak;z=group[int(rng.integers(len(group)))];rgb,t,m=z['rgb'],z['target'],z['mask'];counts['partial' if group is partial else 'analytic_soft']+=1
            batch.append(crop(rgb,t,m,rng,fade=stage>1))
        x,t,m=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(3)];opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:group['lr']=(.00065 if stage==1 else .00025)*(.25+.75*.5*(1+math.cos(math.pi*step/steps)))
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=loss_fn(model(x),t,m)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(grad):raise RuntimeError('nonfinite gradient')
        opt.step()
        if step%100==0:
            row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start)
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('V16_TRAIN',stage,json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),stage=stage,steps=steps,input_channels=7,heads=['center','ink_ribbon','axis_cos2','axis_sin2','apparent_width_pixels_div24'],parent_sha256=digest(PARENT)),folder/'final.pt');jsave(folder/'training_summary.json',dict(successful_updates=steps,elapsed_seconds=time.perf_counter()-start,consumed_samples=counts,original_parent_unchanged=digest(PARENT)==read(ROOT/'manifest.json')['raster_parent_sha256'],independent_artwork_test=False,robot_connected=False));shutil.copy2(Path(__file__),folder/'source_snapshot.py')

@torch.no_grad()
def predict(model,rgb,tile=256,stride=192):
    h,w=rgb.shape[:2];pad=cv2.copyMakeBorder(rgb,20,max(20,tile-h-20),20,max(20,tile-w-20),cv2.BORDER_REFLECT_101);ph,pw=pad.shape[:2];starts=lambda n:sorted(set(list(range(0,n-tile+1,stride))+[n-tile]));loc=[(y,x) for y in starts(ph) for x in starts(pw)];window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32);total=np.zeros((5,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32)
    # Features are calculated once on the whole padded image; tile boundaries
    # do not produce different paper estimates in overlapping windows.
    features=photometry(pad)[0]
    for start in range(0,len(loc),8):
        group=loc[start:start+8];arr=np.stack([features[:,y:y+tile,x:x+tile] for y,x in group]);o=model(torch.from_numpy(arr).cuda()).float();o[:,[0,1,4]]=o[:,[0,1,4]].sigmoid();o[:,2:4]=F.normalize(o[:,2:4],dim=1)
        for (y,x),p in zip(group,o.cpu().numpy()):total[:,y:y+tile,x:x+tile]+=p*window;den[y:y+tile,x:x+tile]+=window
    return (total/den)[:,20:20+h,20:20+w]

def decode(pred,high=.55,low=.28,kind='ribbon'):
    p,ink=pred[:2];h,w=p.shape
    if kind=='center' or kind.startswith('joint_'):
        theta=.5*np.arctan2(pred[3],pred[2]);yy,xx=np.mgrid[:h,:w].astype(np.float32);nx=-np.sin(theta);ny=np.cos(theta);g=cv2.GaussianBlur(p,(0,0),.65);a=cv2.remap(g,xx+nx*1.5,yy+ny*1.5,cv2.INTER_LINEAR);b=cv2.remap(g,xx-nx*1.5,yy-ny*1.5,cv2.INTER_LINEAR);candidate=(p>low)&(g>=a)&(g>=b)
        if kind.startswith('joint_'):candidate&=ink>float(kind.split('_')[1])
    else:candidate=ink>low
    n,labels,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8);good=np.zeros(n,bool);strong=(ink>high) if kind=='ribbon' else (p>high);good[np.unique(labels[strong])]=True;good[0]=False;good[stats[:,cv2.CC_STAT_AREA]<10]=False;binary=good[labels];sk=skeletonize(binary);sk[:3]=False;sk[-3:]=False;sk[:,:3]=False;sk[:,-3:]=False
    # A whole dark wash cannot be declared one wide brush stroke.
    broad=distance_transform_edt(binary)>18;sk[broad]=False;field=np.zeros((8,h,w),np.float32);field[0]=np.maximum(p,ink*.8);field[1:3]=pred[2:4];field[3:5]=pred[2:4];paths,info=graph_paths(sk,field,True,min_length=8);return paths,sk,dict(**info,broad_ink_pixels_quarantined=int(broad.sum()))

def monochrome_route(rgb):
    sat=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[:,:,1]/255;return bool(np.median(sat)<.25 and np.quantile(sat,.95)<.4)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','1','2','3']);args=p.parse_args();prepare() if args.mode=='prepare' else train(int(args.mode))
