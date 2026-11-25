# -*- coding: utf-8 -*-
"""V18 local-paper cross-section supervision; derived annotations, offline only."""
from aligned_targets_v17 import *
import aligned_targets_v17 as legacy
import ink_center_v16 as v16
import eval_aligned_v17 as prior_eval
import json,time,base64,io
from scipy.ndimage import gaussian_filter1d
import torch
from torch import nn
from torch.nn import functional as F
from evaluate_clean_strokes_v12 import path_metrics,line_view
from ink_eval_v16 import normal_diagnostics
ROOT=LAB/'cross_sections_v18_20261025'
OLD=legacy.ROOT
OFFSETS=np.arange(-24,24.01,.5,dtype=np.float32)


def sample(gray,p,u):
    n=np.array([-u[1],u[0]],np.float32)
    coords=p[None,None]+OFFSETS[None,:,None]*n+np.arange(-4,5,dtype=np.float32)[:,None,None]*u
    return cv2.remap(gray,coords[:,:,0].astype(np.float32),coords[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)


def cross_sections(rgb,doc,bg,valid):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    residual=gray-cv2.GaussianBlur(gray,(0,0),1.5)
    neg=residual[bg>0]
    noise=float(np.median(np.abs(neg-np.median(neg)))*1.4826) if len(neg)>20 else .004
    noise=max(.001,noise)
    groups=[];reasons={};total=0
    def reject(reason):reasons[reason]=reasons.get(reason,0)+1
    for stroke in doc['strokes']:
        q=dense(stroke['points'],step=2);sections=[]
        for k,p in enumerate(q):
            total+=1;u=q[min(len(q)-1,k+2)]-q[max(0,k-2)];length=np.linalg.norm(u)
            if length<1:sections.append(None);reject('degenerate');continue
            u=u/length;raw=sample(gray,p,u)
            # Local sloping paper model from robust upper flank quantiles.
            av=np.sort(raw,axis=0)[1:-1].mean(0)
            l=float(np.quantile(av[OFFSETS<-15],.75));r=float(np.quantile(av[OFFSETS>15],.75))
            paper=np.interp(OFFSETS,[-20,20],[l,r]).clip(.15,1)
            contrast=np.maximum((paper-av)/paper,0)
            profile=gaussian_filter1d(contrast,.7)
            ns=max(.001,noise/max(float(paper.mean()),.15)/2.3)
            # Peak finding scans the annotation neighborhood, not an entire image.
            peaks=np.flatnonzero((profile>=np.r_[profile[0],profile[:-1]])&(profile>=np.r_[profile[1:],profile[-1]])&(np.abs(OFFSETS)<=10))
            candidates=[]
            for peak in peaks:
                pv=float(profile[peak])
                if pv<max(.009,3.2*ns):continue
                threshold=max(2*ns,min(.018,pv*.22));on=(profile>threshold).astype(np.uint8)[None]
                # Subpixel grain gaps within one ribbon may close; larger gaps remain separate.
                on=cv2.morphologyEx(on,cv2.MORPH_CLOSE,np.ones((1,3),np.uint8))[0].astype(bool)
                a=int(peak);b=int(peak)+1
                while a>0 and on[a-1]:a-=1
                while b<len(on) and on[b]:b+=1
                width=(b-a)*.5
                if a==0 or b==len(on) or not 1<=width<=32:continue
                weight=np.maximum(profile[a:b]-threshold*.5,0);centroid=float(np.sum(OFFSETS[a:b]*weight)/max(1e-6,weight.sum()))
                midpoint=float((OFFSETS[a]+OFFSETS[b-1])*.5);shift=.65*midpoint+.35*centroid
                if abs(shift)>9:continue
                pos=p+shift*np.array([-u[1],u[0]]);xy=np.rint(pos).astype(int)
                if not (2<=xy[0]<gray.shape[1]-2 and 2<=xy[1]<gray.shape[0]-2) or not valid[xy[1],xy[0]]:continue
                # Confidence follows SNR / longitudinal support, not absolute darkness.
                center_raw=np.maximum((paper[peak]-raw[:,peak])/paper[peak],0)
                support=float((center_raw>max(.005,ns)).mean())
                snr=pv/max(ns,1e-4);quality=float(np.clip((snr-2)/5,.25,1)*(.5+.5*support))
                if support<.55:continue
                entry=dict(p=p,u=u,n=np.array([-u[1],u[0]]),center=pos,shift=shift,left=float(OFFSETS[a]),right=float(OFFSETS[b-1]),width=width,peak=pv,quality=quality,snr=snr,threshold=threshold,profile=profile,paper=paper,raw=raw,index=k,stroke_id=stroke['id'],faint=pv<.18)
                if not any(abs(e['shift']-shift)<1 for e in candidates):candidates.append(entry)
            if not candidates:sections.append(None);reject('no_reliable_local_band');continue
            sections.append(sorted(candidates,key=lambda e:abs(e['shift']))[:3])
        # Smooth track choice disfavours jumping between nearby painted strokes.
        chosen=[None]*len(sections);start=0
        while start<len(sections):
            if sections[start] is None:start+=1;continue
            end=start+1
            while end<len(sections) and sections[end] is not None:end+=1
            costs=[];parents=[]
            for i in range(start,end):
                candidates=sections[i];base=np.array([.035*e['shift']**2-math.log(max(e['quality'],.05)) for e in candidates])
                if i==start:costs.append(base);parents.append([-1]*len(candidates))
                else:
                    previous=sections[i-1];trans=np.array([[costs[-1][j]+.25*(e['shift']-v['shift'])**2+.015*(e['width']-v['width'])**2 for j,v in enumerate(previous)] for e in candidates]);par=trans.argmin(1);costs.append(base+trans[np.arange(len(candidates)),par]);parents.append(par.tolist())
            pick=int(costs[-1].argmin())
            for i in range(end-1,start-1,-1):chosen[i]=sections[i][pick];pick=parents[i-start][pick]
            start=end
        for i,e in enumerate(chosen):
            if e is None:continue
            neighbors=[chosen[k] for k in range(max(0,i-2),min(len(chosen),i+3)) if chosen[k] is not None]
            if len(neighbors)<3:chosen[i]=None;reject('insufficient_continuity');continue
            if abs(e['shift']-float(np.median([n['shift'] for n in neighbors])))>3.5:chosen[i]=None;reject('unstable_normal_shift');continue
            # A second similar nearby band is ambiguous even if DP picked one.
            alternatives=[a for a in sections[i] if abs(a['shift']-e['shift'])>2 and abs(a['shift'])<6 and a['quality']>.8*e['quality']]
            if alternatives:e['quality']*=.4;e['ambiguous_adjacent']=True
        groups.append(chosen)
    return groups,dict(total=total,accepted=sum(e is not None for g in groups for e in g),rejections=reasons,pixel_noise_sigma=noise,derived_not_expert=True)


def targets(rgb,doc,bg,valid):
    h,w=rgb.shape[:2];groups,audit=cross_sections(rgb,doc,bg,valid)
    ink=np.zeros((h,w),np.uint8);center=ink.copy();edge=ink.copy();tube=ink.copy();conf=np.zeros((h,w),np.float32);sx=conf.copy();sy=conf.copy();count=conf.copy();width=conf.copy();alpha=conf.copy();faint=conf.copy();profile_records=[]
    for group in groups:
        for a,b in zip(group[:-1],group[1:]):
            if a is None or b is None:continue
            if float(a['u']@b['u'])<.65 or np.linalg.norm(a['center']-b['center'])>7:continue
            lefta=a['p']+a['left']*a['n'];righta=a['p']+a['right']*a['n'];leftb=b['p']+b['left']*b['n'];rightb=b['p']+b['right']*b['n']
            layer=np.zeros((h,w),np.uint8);cv2.fillPoly(layer,[np.rint([lefta,leftb,rightb,righta]).astype(np.int32)],1)
            halo=np.zeros_like(layer);cv2.fillPoly(halo,[np.rint([lefta-a['n']*2,leftb-b['n']*2,rightb+b['n']*2,righta+a['n']*2]).astype(np.int32)],1)
            quality=min(a['quality'],b['quality']);ink|=layer;tube|=halo;conf=np.maximum(conf,layer*quality+(halo-layer)*quality*.25)
            aa=tuple(np.rint(a['center']).astype(int));bb=tuple(np.rint(b['center']).astype(int));line=np.zeros_like(layer);cv2.line(line,aa,bb,1,1);center|=line
            for c,d in [(lefta,leftb),(righta,rightb)]:cv2.line(edge,tuple(np.rint(c).astype(int)),tuple(np.rint(d).astype(int)),1,1)
            delta=b['center']-a['center'];theta=math.atan2(delta[1],delta[0]);sx+=line*math.cos(2*theta);sy+=line*math.sin(2*theta);count+=line;width+=line*(a['width']+b['width'])/96
            alpha=np.maximum(alpha,layer*(a['peak']+b['peak'])*.5)
            if a['faint'] and b['faint']:faint=np.maximum(faint,layer*quality)
        profile_records.extend([e for e in group if e is not None])
    # Conflicts with expanded visible ribbons are ignored rather than forced to background.
    conflict=cv2.dilate(ink,np.ones((5,5),np.uint8))>0;safe_bg=bg.copy();safe_bg[conflict]=0
    active=valid>0;ink&=active.astype(np.uint8);center&=active.astype(np.uint8);edge&=active.astype(np.uint8);tube&=active.astype(np.uint8);conf*=active;safe_bg*=active
    d,idx=distance_transform_edt(center==0,return_indices=True);norm=np.sqrt(sx*sx+sy*sy);coherent=norm/np.maximum(count,1);angle=(np.stack((sx,sy))/np.maximum(norm,1e-6)[None])[:,idx[0],idx[1]];width=(width/np.maximum(count,1))[tuple(idx)]
    dt_edge=distance_transform_edt(edge==0);edge_t=np.exp(-dt_edge**2/(2*.65**2))*tube
    t=np.concatenate((np.exp(-d[None]**2/(2*1.05**2))*ink[None],ink[None],angle,width[None],edge_t[None],alpha[None],safe_bg[None])).astype(np.float32)
    mc=np.maximum(conf*tube,safe_bg);mi=mc.copy();ma=((d<2)&(ink>0)&(coherent[tuple(idx)]>.75))*conf;me=mc.copy();m=np.stack((mc,mi,ma,ma,me,np.maximum(conf*ink,safe_bg),np.maximum(conf*ink,safe_bg))).astype(np.float32)
    m*=active[None];assert np.isfinite(t).all() and np.isfinite(m).all();assert not np.any((t[0]>.5)&(t[1]<.5))
    audit.update(faint_samples=sum(e['faint'] for e in profile_records),bold_samples=sum(e['width']>=7 and e['peak']>=.18 for e in profile_records),ink_pixels=int(ink.sum()),explicit_safe_background=int(safe_bg.sum()),background_conflicts_ignored=int(((bg>0)&conflict).sum()),center_outside_ribbon=0)
    return t,m,faint,profile_records,audit


def prepare():
    (ROOT/'data').mkdir(parents=True,exist_ok=True);out=[];display=[];review=[]
    for r in legacy.records():
        rgb,doc,bg,valid=legacy.source(r);t,m,faint,profiles,audit=targets(rgb,doc,bg,valid);z=np.load(OLD/'data'/(r['id']+'.npz'));ct=np.zeros_like(t);ct[:5]=z['target'];cm=np.zeros_like(m);cm[:4]=z['mask'];file=ROOT/'data'/(r['id']+'.npz');np.savez_compressed(file,rgb=rgb,target=t,mask=m,control_target=ct,control_mask=cm,faint=faint,background=bg,valid=valid)
        jsave(ROOT/'data'/(r['id']+'_audit.json'),audit);out.append(dict(r,data=str(file),data_sha256=digest(file),image_sha256=digest(r['image']),label_sha256=digest(r['labels']),audit=audit))
        overlay=rgb.copy();band=t[1]>.5;overlay[band]=(overlay[band]*.45+np.array([30,140,235])*.55).astype(np.uint8);overlay[t[5]>.6]=[240,170,20];overlay[t[0]>.65]=[230,30,40]
        if r['id'] in ['R01','R02','N06','N09','N12']:
            tiles=[rgb,overlay,np.uint8(255*(1-t[0])),np.uint8(255*(1-np.maximum(m[0],0)))];display.append((r['id'],tiles))
        if r['split']=='train':
            subsets=[('faint',sorted([e for e in profiles if e['faint']],key=lambda e:e['peak'])),('bold',sorted([e for e in profiles if e['width']>=7],key=lambda e:-e['width']))]
            for kind,items in subsets:
                picked=[]
                for e in items:
                    if e.get('ambiguous_adjacent'):continue
                    if any(np.linalg.norm(e['center']-p)<20 for p in picked):continue
                    picked.append(e['center']);review.append(dict(id=r['id']+'_'+str(len(review)),sample_id=r['id'],kind=kind,p=e['p'].tolist(),normal=e['n'].tolist(),center=e['shift'],left=e['left'],right=e['right'],contrast=e['peak'],snr=e['snr'],derived_not_expert=True,clip=base64.b64encode(cv2.imencode('.png',cv2.cvtColor(cv2.remap(rgb,(e['p'][0]+(np.arange(192,dtype=np.float32)[None]-95.5)*e['n'][0]/3+(np.arange(96,dtype=np.float32)[:,None]-47.5)*e['u'][0]/3).astype(np.float32),(e['p'][1]+(np.arange(192,dtype=np.float32)[None]-95.5)*e['n'][1]/3+(np.arange(96,dtype=np.float32)[:,None]-47.5)*e['u'][1]/3).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101),cv2.COLOR_RGB2BGR))[1]).decode()))
                    if len(picked)>=3:break
        print('V18_LABEL',r['id'],audit['accepted'],audit['total'],'faint',audit['faint_samples'],flush=True)
    board=Image.new('RGB',(1600,370*len(display)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((6,5),'v18 训练标签审计（自动派生，非预测）：蓝墨带、橙边缘、红中心；白色未监督',font=FONT,fill='black')
    for i,(k,tiles) in enumerate(display):
        for j,a in enumerate(tiles):d.text((400*j+5,35+370*i),k+' '+['原图','横截面派生标签','中心标签','监督置信度'][j],font=SMALL,fill='black');paste(board,a,400*j+5,65+370*i,390,330)
    board.save(ROOT/'LABEL_AUDIT.jpg',quality=95);jsave(ROOT/'review_candidates.json',review)
    jsave(ROOT/'manifest.json',dict(records=out,init_weights=str(OLD/'aligned'/'final.pt'),init_sha256=digest(OLD/'aligned'/'final.pt'),parent_sha256=digest(PARENT),training_ids=[r['id'] for r in out if r['split']=='train'],calibration_ids=['N07','N08'],development_ids=['R03','R04','N01','N10','N11','T01','T02'],labels_are_automatic_derived_not_expert=True,expert_originals_unchanged=True,physical_ink_concentration_not_measured=True,robot_connected=False))
    print('V18_PREPARED',flush=True)


class SectionNet(v16.RibbonNet):
    def __init__(self):
        super().__init__();self.head=nn.Conv2d(16,8,1)
    def initialize(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)['model'];w=state.pop('head.weight');b=state.pop('head.bias');self.load_state_dict(state,strict=False)
        with torch.no_grad():self.head.weight[:5]=w;self.head.bias[:5]=b;self.head.bias[5:]=torch.tensor([-2.,-2.,1.])


def loss_fn(o,t,m):
    o=o.float();p=o.sigmoid()
    def balanced(logit,target,mask):
        l=F.binary_cross_entropy_with_logits(logit,target,reduction='none');pos=(target>.5)*mask;neg=(target<=.5)*mask
        return (l*pos).sum()/pos.sum().clamp_min(1)+(l*neg).sum()/neg.sum().clamp_min(1)
    center=balanced(o[:,0],t[:,0],m[:,0]);ink=balanced(o[:,1],t[:,1],m[:,1]);dice=1-(2*(p[:,0]*t[:,0]*m[:,0]).sum()+1)/(((p[:,0]+t[:,0])*m[:,0]).sum()+1)
    angle=((1-F.cosine_similarity(o[:,2:4],t[:,2:4],dim=1))*m[:,2]).sum()/m[:,2].sum().clamp_min(1)
    width=(F.smooth_l1_loss(p[:,4],t[:,4],reduction='none')*m[:,3]).sum()/m[:,3].sum().clamp_min(1)
    edge=balanced(o[:,5],t[:,5],m[:,4]);contrast=(F.smooth_l1_loss(p[:,6],t[:,6],reduction='none')*m[:,5]).sum()/m[:,5].sum().clamp_min(1);bg=balanced(o[:,7],t[:,7],m[:,6]);sides=(p[:,0]*t[:,5]*(t[:,4]>.13)*m[:,4]).sum()/m[:,4].sum().clamp_min(1)
    return center+.7*ink+.25*dice+.12*angle+.15*width+.2*edge+.2*contrast+.1*bg+.1*sides


def train(variant):
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(182510);rng=np.random.default_rng(182510);folder=ROOT/variant;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'final.pt').exists():raise FileExistsError('Completed checkpoint is immutable')
    manifest=read(ROOT/'manifest.json');rows=[r for r in manifest['records'] if r['split']=='train'];data=[]
    for r in rows:
        z=np.load(r['data']);prefix='control_' if variant=='control' else '';data.append(dict(id=r['id'],rgb=z['rgb'].copy(),t=z[prefix+'target'].copy(),m=z[prefix+'mask'].copy(),faint=z['faint'].copy(),bg=z['background'].copy()))
    model=SectionNet();model.initialize(manifest['init_weights']);model=model.cuda();opt=torch.optim.AdamW([dict(params=list(model.a.parameters())+list(model.b.parameters())+list(model.c.parameters()),lr=.00003),dict(params=list(model.d.parameters())+list(model.e.parameters())+list(model.head.parameters()),lr=.00018)],weight_decay=.0001);steps=1000;start=time.perf_counter();usage=dict(user=0,assistant=0,faint_focused=0,background_focused=0);config=dict(variant=variant,steps=steps,seed=182510,training_ids=manifest['training_ids'],calibration_ids=manifest['calibration_ids'],init_sha256=manifest['init_sha256'],heads=['center','ribbon','cos2','sin2','width_div48','both_edges','apparent_contrast','background'],real_cross_sections_automatic_not_expert=True,matched_schedule=True,robot_connected=False);jsave(folder/'config.json',config);shutil.copy2(Path(__file__),folder/'source_at_start.py')
    for step in range(1,steps+1):
        batch=[]
        for _ in range(6):
            group=[a for a in data if a['id'].startswith('R')] if rng.random()<.55 else [a for a in data if not a['id'].startswith('R')];a=group[int(rng.integers(len(group)))];usage['user' if a['id'].startswith('R') else 'assistant']+=1;rgb=a['rgb'];t=a['t'];m=a['m'];chance=rng.random();location=a['faint']>.3 if chance<.4 else a['bg']>0 if chance<.55 else None
            if location is not None and location.any():
                ys,xs=np.nonzero(location);k=int(rng.integers(len(xs)));h,w=rgb.shape[:2];x=int(np.clip(xs[k]-128,0,max(0,w-256)));y=int(np.clip(ys[k]-128,0,max(0,h-256)));rgb=rgb[y:y+256,x:x+256];t=t[:,y:y+256,x:x+256];m=m[:,y:y+256,x:x+256];usage['faint_focused' if chance<.4 else 'background_focused']+=1
            # Keep real faint examples visible. Avoid fading them again into imperceptible targets.
            batch.append(v16.crop(rgb,t,m,rng,fade=False))
        x,t,m=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(3)];factor=.35+.65*.5*(1+math.cos(math.pi*step/steps))
        for g,lr in zip(opt.param_groups,[.00003,.00018]):g['lr']=lr*factor
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=loss_fn(model(x),t,m)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward();g=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(g):raise RuntimeError('nonfinite gradient')
        opt.step()
        if step%100==0:
            row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start)
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('V18_TRAIN',variant,json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),config=config),folder/'final.pt');jsave(folder/'training_summary.json',dict(successful_updates=steps,usage=usage,elapsed_seconds=time.perf_counter()-start,parent_unchanged=digest(PARENT)==manifest['parent_sha256'],robot_connected=False));print('V18_TRAINED',variant,flush=True)


@torch.no_grad()
def predict(model,rgb):
    h,w=rgb.shape[:2];tile=256;stride=192;pad=cv2.copyMakeBorder(rgb,20,max(20,tile-h-20),20,max(20,tile-w-20),cv2.BORDER_REFLECT_101);ph,pw=pad.shape[:2];starts=lambda n:sorted(set(list(range(0,n-tile+1,stride))+[n-tile]));loc=[(y,x) for y in starts(ph) for x in starts(pw)];window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32);total=np.zeros((8,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32);features=v16.photometry(pad)[0]
    for start in range(0,len(loc),8):
        group=loc[start:start+8];arr=np.stack([features[:,y:y+tile,x:x+tile] for y,x in group]);o=model(torch.from_numpy(arr).cuda()).float();o[:,[0,1,4,5,6,7]]=o[:,[0,1,4,5,6,7]].sigmoid();o[:,2:4]=F.normalize(o[:,2:4],dim=1)
        for (y,x),p in zip(group,o.cpu().numpy()):total[:,y:y+tile,x:x+tile]+=p*window;den[y:y+tile,x:x+tile]+=window
    return (total/den)[:,20:20+h,20:20+w]


def load(variant):
    m=SectionNet().cuda();m.load_state_dict(torch.load(ROOT/variant/'final.pt',map_location='cuda',weights_only=False)['model']);return m.eval()


def evaluate(variant,mode):
    # Reuse a fixed v17 decoder grid and identical selection formula for fair comparison.
    oldroot=prior_eval.ROOT;oldload=prior_eval.load;oldpredict=prior_eval.v16.predict
    prior_eval.ROOT=ROOT;prior_eval.load=lambda v:load(v);prior_eval.v16.predict=predict
    try:
        prior_eval.calibrate(variant) if mode=='calibrate' else prior_eval.evaluate(variant)
    finally:prior_eval.ROOT=oldroot;prior_eval.load=oldload;prior_eval.v16.predict=oldpredict


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','calibrate','evaluate']);p.add_argument('variant',nargs='?',choices=['control','profiles']);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='train':train(a.variant)
    else:evaluate(a.variant,a.mode)
