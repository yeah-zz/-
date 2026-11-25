# -*- coding: utf-8 -*-
"""V21 offline, staged diagnosis and instance-preserving stroke supervision."""
from structure_train_v20 import *
import hashlib
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from v21_visible_references import ANNOT
from evaluate_clean_strokes_v12 import path_metrics,line_view
from ink_eval_v16 import normal_diagnostics
BASE20=ROOT
ROOT=LAB/'stroke_diagnosis_v21_20261025'

def candidates():
    ROOT.mkdir(parents=True,exist_ok=True)
    rows=read(V15/'curated_manifest.json')['records']
    selected=[]
    for r in rows:
        selected.append(dict(id=r['id'],image=r['content_image'],group=r.get('source_group',r['id'].rsplit('_',1)[0])))
    for n,group in [(1,'basic'),(18,'pima'),(23,'guatie'),(29,'heye'),(35,'jiesuo'),(41,'douban'),(43,'zhedai'),(49,'yuntou'),(55,'midian'),(61,'dingtou'),(65,'yudian'),(69,'mountains')]:
        p=BASE20/'paired_curriculum'/'images'/('stage_'+str(n).zfill(3)+'.png')
        selected.append(dict(id='stage_'+str(n).zfill(3),image=str(p),group=group))
    jsave(ROOT/'candidates.json',dict(records=selected))
    board=Image.new('RGB',(1200,330*math.ceil(len(selected)/4)),'white');d=ImageDraw.Draw(board)
    for i,r in enumerate(selected):
        im=Image.open(r['image']).convert('RGB');x=(i%4)*300;y=(i//4)*330
        d.text((x+3,y+3),r['id']+' '+str(im.size),font=SMALL,fill='black');paste(board,np.asarray(im),x+3,y+30,294,294)
    board.save(ROOT/'CANDIDATES.jpg',quality=95)
    print('V21_CANDIDATES',len(selected),flush=True)

def canvases():
    items=read(ROOT/'candidates.json')['records'];out=[];(ROOT/'images').mkdir(exist_ok=True)
    stages={1:[.08,.28,.91,.98],18:[.10,.36,.68,1.],23:[.05,.22,.91,1.],29:[.12,.25,.70,.78],35:[.10,.35,.80,1.],41:[.10,.20,.90,.90],43:[.10,.20,.90,.60],49:[.12,.17,.84,.56],55:[.20,.22,.90,.77],65:[.25,.26,.75,.74]}
    for r in items:
        sid=r['id']
        if sid.startswith('sina') and int(sid.rsplit('_',1)[1]) not in [27,28]:box=[0,0,1,1]
        elif sid=='kuhw_29483_00':box=[0,0,.975,1]
        elif sid.startswith('stage') and int(sid[-3:]) in stages:box=stages[int(sid[-3:])]
        else:continue
        im=Image.open(r['image']).convert('RGB');b=[round(v*(im.width if k%2==0 else im.height)) for k,v in enumerate(box)];im=im.crop(b);p=ROOT/'images'/(sid+'.png');im.save(p)
        out.append(dict(r,image=str(p),original_image=r['image'],crop_box=b,shape=[im.height,im.width]))
    jsave(ROOT/'selected_images.json',dict(records=out,full_stage_source_excluded_from_train_if_heldout=True))
    for start in range(0,len(out),6):
        board=Image.new('RGB',(1200,440*2),'white');d=ImageDraw.Draw(board)
        for i,r in enumerate(out[start:start+6]):
            a=Image.open(r['image']);a=a.resize((360,360));x=(i%3)*400+25;y=(i//3)*440+40;board.paste(a,(x,y));d.text((x,y-28),r['id'],font=FONT,fill='black')
            for q in range(0,101,10):
                xx=x+round(q*3.6);yy=y+round(q*3.6);d.line((xx,y,xx,y+360),fill='#cccccc',width=1);d.line((x,yy,x+360,yy),fill='#cccccc',width=1);d.text((xx-5,y+363),str(q),font=SMALL,fill='black');d.text((x-22,yy-6),str(q),font=SMALL,fill='black')
        board.save(ROOT/('CANVAS_'+str(start//6)+'.jpg'),quality=96)
    print('V21_SELECTED',len(out),flush=True)

def parsed(s,shape):
    h,w=shape;p=np.array([[float(v) for v in token.split(',')] for token in s.split()],np.float32);p*=np.array([w-1,h-1])/100
    q=dense(p).astype(np.float32)
    if len(q)>5:q=gaussian_filter1d(q,.8,axis=0)
    return q

def prepared_reference(r):
    rgb=np.asarray(Image.open(r['image']).convert('RGB'));h,w=rgb.shape[:2];key=r['id'];short=key.rsplit('_',1)[1] if key.startswith('sina') else key
    paths=[parsed(s,(h,w)) for s in ANNOT[short]]
    # Bounded image-only correction of drawn anchor chains. No model output is used.
    alpha=v16.photometry(rgb)[1];offsets=np.arange(-7,7.1,.5,dtype=np.float32);fixed=[];widths=[]
    for q in paths:
        tangent=np.gradient(q,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-6);normal=np.c_[-tangent[:,1],tangent[:,0]];xy=q[:,None]+normal[:,None]*offsets[None,:,None];sample=cv2.remap(alpha,xy[:,:,0],xy[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);shifts=[];ww=[]
        for a in sample:
            peak=float(a.max());binary=a>max(.008,.27*peak);n,cc,stats,_=cv2.connectedComponentsWithStats(binary[None].astype(np.uint8),8);choices=[]
            for k in range(1,n):
                ids=np.nonzero(cc[0]==k)[0];mid=float(offsets[ids].mean());width=float(len(ids)*.5)
                if abs(mid)<=4.5:choices.append((abs(mid),mid,width))
            best=min(choices) if choices else (0,0,1.5);shifts.append(np.clip(best[1],-3,3));ww.append(best[2])
        shift=gaussian_filter1d(np.array(shifts,np.float32),2);fixed.append(q+normal*shift[:,None]);widths.append(np.asarray(ww,np.float32))
    return rgb,fixed,widths

def make_targets(rgb,paths,widths):
    h,w=rgb.shape[:2];_,alpha,_=v16.photometry(rgb);c,angle,d=v16.geometry(paths,(h,w));_,idx=distance_transform_edt(c==0,return_indices=True);wm=np.zeros((h,w),np.float32)
    for q,ww in zip(paths,widths):
        xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);wm[xy[:,1],xy[:,0]]=ww
    wm=wm[tuple(idx)];ink=(alpha>.025).astype(np.float32);inkdist=distance_transform_edt(alpha<.012);nearest_photo=alpha[tuple(idx)];supported=(nearest_photo>.009);known_bg=(inkdist>5)&(d>7)&(alpha<.006);ribbon_known=((alpha>.045)&(d<20))|known_bg;center_known=((d<2.6)&supported)|((d>3)&(d<12)&(alpha>.045)&(wm<13))|known_bg
    # Untraced ink and ambiguous multi-stroke crossings are not background.
    valid=np.ones((h,w),np.float32);valid[:3]=0;valid[-3:]=0;valid[:,:3]=0;valid[:,-3:]=0
    crossing=np.zeros((h,w),bool)
    if len(paths)>1:
        ds=[]
        for q in paths:
            z=np.ones((h,w),np.uint8);xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);z[xy[:,1],xy[:,0]]=0;ds.append(distance_transform_edt(z))
        near=np.partition(np.stack(ds),1,axis=0)[:2];crossing=near[1]<2.5
    target=np.concatenate((np.exp(-d[None]**2/(2*1.05**2)),ink[None],angle,np.clip(wm[None]/48,0,1))).astype(np.float32);mask=np.stack((center_known*.7,ribbon_known*.5,(d<2)&supported&~crossing,(d<2)&supported&~crossing&(wm<13))).astype(np.float32)*valid
    return target,mask,dict(annotated_chains=len(paths),unknown_ink_pixels=int(((alpha>.045)&(d>12)).sum()),weak_positive_ink_pixels=int(((alpha>.045)&(d<20)).sum()),masked_crossing_pixels=int(crossing.sum()),visible_ink_near_reference_fraction=float(((alpha>.045)&(d<12)).sum()/max(1,(alpha>.045).sum())))

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);folder=ROOT/'labels';folder.mkdir(exist_ok=True);data=ROOT/'data';data.mkdir(exist_ok=True);records=[];pages=[]
    for r in read(ROOT/'selected_images.json')['records']:
        rgb,paths,widths=prepared_reference(r);h,w=rgb.shape[:2];key=r['id'];split='train'
        if key.startswith('sina'):
            n=int(key.rsplit('_',1)[1]);group='sina_rough_' + ('172' if n in [17,18] else '202' if n in [20,21,22] else '293' if n in [29,30,31] else '256' if n in [25,26] else str(n))
            if n in [20,21,22,24]:split='calibration'
            if n in [29,30,31]:split='development_holdout'
        else:group=r['group']
        if key in ['kuhw_29483_00','stage_018','stage_029','stage_049','stage_065']:split='development_holdout'
        if key=='stage_055':split='calibration'
        doc=dict(sample_id=key,width=w,height=h,strokes=[dict(id='visible_'+str(i),role='visible_chain',points=q.round(3).tolist()) for i,q in enumerate(paths)],regions=[],source='assistant_visible_ink_trace_before_model_predictions',complete_expert_ground_truth=False,historical_pen_lifts_unknown=True,cropped_boundary_ends_not_pen_lifts=True)
        label=folder/(key+'.json');jsave(label,doc);target,mask,info=make_targets(rgb,paths,widths);file=data/(key+'.npz');np.savez_compressed(file,rgb=rgb,target=target,mask=mask);records.append(dict(r,labels=str(label),label_sha256=digest(label),image_sha256=digest(r['image']),data=str(file),data_sha256=digest(file),split=split,work_group=group,reference_quality='assistant_near_complete_visible_chains_with_untraced_ink_unknown',**info))
        page=Image.new('RGB',(1000,310),'#f5f3ec');d=ImageDraw.Draw(page);d.text((5,3),key+' '+split+' 辅助参考（非模型输出）',font=SMALL,fill='black')
        overlay=rgb.copy();refim=line_view((h,w),paths,True);z=(target[0]>.4);overlay[z]=[220,40,50]
        for j,a in enumerate([rgb,overlay,refim]):paste(page,a,j*333+3,31,327,270)
        pages.append(page)
    groups={s:sorted({r['work_group'] for r in records if r['split']==s}) for s in ['train','calibration','development_holdout']}
    assert not set(groups['train'])&set(groups['calibration']);assert not set(groups['train'])&set(groups['development_holdout'])
    manifest=dict(records=records,groups=groups,init_weights=str(OLD/'aligned'/'final.pt'),init_sha256=digest(OLD/'aligned'/'final.pt'),v19_replay_manifest_sha256=digest(V19/'manifest_v2.json'),external_X01_X04_excluded=True,ancestor_source_exposure_possible=True,original_sources_only_three_tutorial_families=True,references_not_expert_gold=True,robot_connected=False)
    jsave(ROOT/'manifest.json',manifest)
    for start in range(0,len(pages),6):
        board=Image.new('RGB',(1000,310*len(pages[start:start+6])),'white')
        for i,page in enumerate(pages[start:start+6]):board.paste(page,(0,i*310))
        board.save(ROOT/('REFERENCE_AUDIT_'+str(start//6)+'.jpg'),quality=95)
    shutil.copy2(Path(__file__),ROOT/'source_before_training.py');shutil.copy2(Path(__file__).with_name('v21_visible_references.py'),ROOT/'visible_reference_source.py')
    print('V21_PREPARED',len(records),sum(r['annotated_chains'] for r in records),{s:sum(r['split']==s for r in records) for s in groups},flush=True)

def footprint_targets():
    from v21_background_regions import BACKGROUND
    manifest=read(ROOT/'manifest.json');folder=ROOT/'data_ribbon_v2';folder.mkdir(exist_ok=True);rows=[];pages=[]
    for r in manifest['records']:
        if r['split']!='train':continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));h,w=rgb.shape[:2];paths=[np.asarray(s['points'],np.float32) for s in read(r['labels'])['strokes']];alpha=v16.photometry(rgb)[1];ink=np.zeros((h,w),np.float32);known=np.zeros((h,w),np.float32);width=np.zeros((h,w),np.float32);wc=np.zeros((h,w),np.float32);off=np.arange(-24,24.01,.5,dtype=np.float32);accepted=0;rejected=0
        for p in paths:
            tangent=np.gradient(p,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-6);normal=np.c_[-tangent[:,1],tangent[:,0]]
            for i in range(2,len(p)-2):
                q=p[i]+off[:,None]*normal[i];v=cv2.remap(alpha,q[:,0].astype(np.float32)[None],q[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0];v=cv2.GaussianBlur(v[None],(5,1),.5)[0];near=np.abs(off)<4;peak=float(v[near].max());noise=float(np.median(np.r_[v[:8],v[-8:]]));cut=max(.009,noise+.28*max(0,peak-noise));binary=v>cut;n,cc,stats,_=cv2.connectedComponentsWithStats(binary[None].astype(np.uint8),8);candidates=[]
                for k in range(1,n):
                    ids=np.flatnonzero(cc[0]==k);mid=float(off[ids].mean());ww=len(ids)*.5
                    if abs(mid)<4 and 1<=ww<=38 and ids[0]>1 and ids[-1]<len(off)-2:candidates.append((abs(mid),ids,ww))
                if not candidates or peak<.015:rejected+=1;continue
                _,ids,ww=min(candidates,key=lambda v:v[0]);lo,hi=float(off[ids[0]]),float(off[ids[-1]]);extended=(off>=lo-4)&(off<=hi+4);xy=np.rint(q).astype(int);inside=(xy[:,0]>=3)&(xy[:,0]<w-3)&(xy[:,1]>=3)&(xy[:,1]<h-3);ep=xy[extended&inside];ip=xy[(off>=lo)&(off<=hi)&inside];known[ep[:,1],ep[:,0]]=1;ink[ip[:,1],ip[:,0]]=1;point=np.rint(p[i]).astype(int);width[point[1],point[0]]=ww/48;wc[point[1],point[0]]=1;accepted+=1
        # Closing within a two-pixel scan footprint only fills between adjacent scans.
        ink=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));known=np.maximum(known,ink);c,angle,dist=v16.geometry(paths,(h,w));_,idx=distance_transform_edt(c==0,return_indices=True);width=width[tuple(idx)];wc=wc[tuple(idx)];bg=np.zeros((h,w),np.uint8);key=r['id'].rsplit('_',1)[1] if r['id'].startswith('sina') else r['id']
        for b in BACKGROUND[key]:
            x0,y0,x1,y1=[round(v*(w-1 if k%2==0 else h-1)/100) for k,v in enumerate(b)];bg[y0:y1,x0:x1]=1
        # Reference safety margin prevents rectangles from crossing traced pale lines.
        bg[dist<6]=0;conflict=(bg>0)&(ink>0);bg[conflict]=0;known=np.maximum(known,bg);ink[bg>0]=0
        target=np.concatenate((np.exp(-dist[None]**2/(2*1.05**2)),ink[None],angle,width[None])).astype(np.float32);center_known=((known>0)&((dist<3)|(ink>0)|(bg>0)))|((dist>=3)&(dist<10)&(ink>0));mask=np.stack((center_known*.85,known*.85,((dist<2)&(wc>0))*.7,((dist<2)&(wc>0))*.5)).astype(np.float32);mask[:,:3]=0;mask[:,-3:]=0;mask[:,:,:3]=0;mask[:,:,-3:]=0;file=folder/(r['id']+'.npz');np.savez_compressed(file,rgb=rgb,target=target,mask=mask);rows.append(dict(id=r['id'],data=str(file),sha256=digest(file),explicit_background_pixels=int(bg.sum()),measured_sections=accepted,rejected_sections=rejected,labels_are_assistant_and_image_derived_not_expert=True));ov=rgb.copy();ov[bg>0]=((ov[bg>0].astype(float)+np.array([30,210,60]))*.5).astype(np.uint8);ov[ink>0]=[80,130,240];ov[target[0]>.6]=[220,35,45];page=Image.new('RGB',(1100,330),'white');d=ImageDraw.Draw(page);d.text((5,3),r['id']+' 墨带监督审计：蓝墨带／红中心／绿明确空白（非预测）',font=SMALL,fill='black');paste(page,rgb,5,32,535,285);paste(page,ov,555,32,535,285);pages.append(page)
    jsave(ROOT/'footprint_v2_manifest.json',dict(records=rows,source_label_manifest_sha256=digest(ROOT/'manifest.json'),training_ids=[r['id'] for r in rows],original_first_trial_preserved=True))
    for start in range(0,len(pages),4):
        board=Image.new('RGB',(1100,330*len(pages[start:start+4])),'white')
        for i,p in enumerate(pages[start:start+4]):board.paste(p,(0,i*330))
        board.save(ROOT/('FOOTPRINT_AUDIT_'+str(start//4)+'.jpg'),quality=95)
    print('V21_FOOTPRINT_V2',len(rows),sum(r['explicit_background_pixels'] for r in rows),flush=True)

def training_pool(variant):
    manifest=read(ROOT/'manifest.json');new=[];old=[]
    for r in manifest['records']:
        if r['split']=='train':
            z=np.load(ROOT/'data_ribbon_v2'/(r['id']+'.npz') if variant=='ribbon_improved' else r['data']);new.append((r['id'],v16.photometry(z['rgb'])[0],z['target'],z['mask']))
    for r in read(V19/'manifest_v2.json')['records']:
        z=np.load(r['data']);old.append((r['id'],v16.photometry(z['rgb'])[0],z['target'][:5],z['mask']))
    return manifest,new,old

def training_crop(row,rng):
    _,x,t,m=row;x=x.copy();t=t.copy();m=m.copy();h,w=x.shape[1:];ph=max(0,256-h);pw=max(0,256-w)
    if ph or pw:
        x=np.pad(x,((0,0),(0,ph),(0,pw)),mode='reflect');t=np.pad(t,((0,0),(0,ph),(0,pw)));m=np.pad(m,((0,0),(0,ph),(0,pw)));h,w=x.shape[1:]
    y=int(rng.integers(h-256+1));xx=int(rng.integers(w-256+1));x=x[:,y:y+256,xx:xx+256];t=t[:,y:y+256,xx:xx+256];m=m[:,y:y+256,xx:xx+256]
    if rng.random()<.5:x=x[:,:,::-1].copy();t=t[:,:,::-1].copy();m=m[:,:,::-1].copy();t[3]*=-1
    return x,t,m

def train(variant):
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(211025);rng=np.random.default_rng(211025);manifest,new,old=training_pool(variant);folder=ROOT/variant;folder.mkdir(exist_ok=True)
    if (folder/'final.pt').exists():raise FileExistsError('Preserve completed checkpoint')
    model=v16.RibbonNet().cuda();model.load_state_dict(torch.load(manifest['init_weights'],map_location='cuda',weights_only=False)['model']);opt=torch.optim.AdamW([dict(params=list(model.a.parameters())+list(model.b.parameters())+list(model.c.parameters()),lr=2e-5),dict(params=list(model.d.parameters())+list(model.e.parameters())+list(model.head.parameters()),lr=8e-5)],weight_decay=1e-4)
    config=dict(seed=211025,steps=1000,variant=variant,init_sha256=manifest['init_sha256'],new_training_ids=[r[0] for r in new] if variant!='control' else [],replay_ids=[r[0] for r in old],new_data_probability=.35 if variant!='control' else 0,objective='v17_balanced_center_ink_axis_width_unchanged',complete_expert_annotations=False,robot_connected=False);jsave(folder/'config.json',config);shutil.copy2(Path(__file__),folder/'source_at_start.py')
    if variant=='ribbon_improved':shutil.copy2(Path(__file__).with_name('v21_background_regions.py'),folder/'background_source_at_start.py')
    usage={};start=time.perf_counter()
    for step in range(1,1001):
        batch=[]
        for _ in range(6):
            choose_new=rng.random()<.35;oldpick=int(rng.integers(len(old)));newpick=int(rng.integers(len(new)));row=new[newpick] if choose_new and variant!='control' else old[oldpick];usage[row[0]]=usage.get(row[0],0)+1;batch.append(training_crop(row,rng))
        x,t,m=[torch.from_numpy(np.stack([b[k] for b in batch])).cuda() for k in range(3)];factor=.3+.7*.5*(1+math.cos(math.pi*step/1000))
        for g,lr in zip(opt.param_groups,[2e-5,8e-5]):g['lr']=lr*factor
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=train17.loss_fn(model(x),t,m)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
        opt.step()
        if step%100==0:
            log=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start)
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(log)+'\n')
            print('V21_TRAIN',variant,json.dumps(log),flush=True)
    torch.save(dict(model=model.state_dict(),config=config),folder/'final.pt');jsave(folder/'training_summary.json',dict(successful_updates=1000,usage=usage,elapsed_seconds=time.perf_counter()-start,robot_connected=False))

def load_model(variant):
    file=read(ROOT/'manifest.json')['init_weights'] if variant=='init' else ROOT/variant/'final.pt';m=v16.RibbonNet().cuda();m.load_state_dict(torch.load(file,map_location='cuda',weights_only=False)['model']);return m.eval()

def raster(paths,shape):
    a=np.zeros(shape,np.uint8)
    for q in paths:cv2.polylines(a,[np.rint(q).astype(np.int32)],False,1,1)
    return a

def diagnostic(ref,pred,paths,valid,rgb,bg):
    metric=path_metrics(ref,paths,valid);sk=raster(paths,rgb.shape[:2]);metric['background_line_density']=float(sk[bg].mean()) if bg.any() else None;metric['normal_diagnostics']=normal_diagnostics(ref,sk.astype(bool),rgb,valid=valid);points=np.concatenate([dense(q) for q in ref]);xy=np.rint(points).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<rgb.shape[1])&(xy[:,1]>=0)&(xy[:,1]<rgb.shape[0]);xy=xy[inside];xy=xy[valid[xy[:,1],xy[:,0]]>0];counts={}
    for channel,name,threshold in [(1,'ink_ref_recall_3px',.5),(0,'center_ref_recall_3px',.3)]:
        p=cv2.dilate(pred[channel],np.ones((7,7),np.uint8));counts[name]=float((p[xy[:,1],xy[:,0]]>=threshold).mean()) if len(xy) else 0.
    return dict(**metric,**counts,reference_point_count=int(len(xy)))

def calibration(variant):
    model=load_model(variant);cases=[]
    for r in read(ROOT/'manifest.json')['records']:
        if r['split']!='calibration':continue
        z=np.load(r['data']);rgb=z['rgb'];doc=read(r['labels']);ref=[s['points'] for s in doc['strokes']];cases.append((r,rgb,ref,v16.predict(model,rgb),z['mask']))
    choices=[]
    for kind,high,low in [('center',.65,.3),('center',.8,.45),('ribbon',.65,.35),('ribbon',.8,.5)]:
        rows=[]
        for r,rgb,ref,p,mask in cases:
            paths,sk,info=prior_eval.decode(p,kind,high,low);valid=np.ones(rgb.shape[:2],np.float32);bg=(mask[0]>0)&(mask[1]>0)&(np.max(p[:2],axis=0)<.1);met=path_metrics(ref,paths,valid);rows.append(dict(id=r['id'],**met))
        score=float(np.mean([.5*x['mean_best_single_path_coverage']+.5*x['point_coverage_3px']-.025*x['mixed_reference_paths_flagged'] for x in rows]));choices.append(dict(kind=kind,high=high,low=low,score=score,records=rows))
    folder=ROOT/variant;folder.mkdir(exist_ok=True);selected=max(choices,key=lambda x:x['score']);jsave(folder/'calibration.json',dict(selected=selected,candidates=choices,external_X01_X04_not_used=True,source_correlated_development_not_blind=True));print('V21_CALIBRATION',variant,selected['kind'],selected['score'],flush=True)

def evaluation_cases():
    out=[]
    for r in read(ROOT/'manifest.json')['records']:
        if r['split']!='development_holdout':continue
        z=np.load(r['data']);rgb=z['rgb'];doc=read(r['labels']);valid=np.ones(rgb.shape[:2],np.float32);valid[:3]=0;valid[-3:]=0;valid[:,:3]=0;valid[:,-3:]=0;out.append((r,rgb,[s['points'] for s in doc['strokes']],valid,(z['mask'][0]>0)&(z['target'][0]<.01)&(z['target'][1]<.1),z['target']))
    for r in read(BASE20/'external_manifest.json')['records']:
        if r['split']!='external_test':continue
        rgb=np.asarray(Image.open(r['image']).convert('L').convert('RGB'));doc=read(r['labels']);_,_,mask,_=fields(doc);out.append((dict(r,split='fixed_external_development'),rgb,[s['points'] for s in doc['strokes']],mask[3],mask[1]>0,None))
    for r in prior_eval.evaluation_records():
        if r['id'] not in ['R03','R04']:continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=read(r['labels']);_,_,mask,_=fields(doc);left,top,cw,ch=r['native'];native=np.zeros(rgb.shape[:2],np.float32);native[top:top+ch,left:left+cw]=1;out.append((dict(r,split='user_reviewed_old_development'),rgb,[s['points'] for s in doc['strokes']],mask[3]*native,(mask[1]*native)>0,None))
    return out

def evaluate(variant):
    torch.set_num_threads(4);cv2.setNumThreads(2);model=load_model(variant);cfg=read(ROOT/variant/'calibration.json')['selected'];folder=ROOT/variant/'evaluation';folder.mkdir(exist_ok=True);rows=[];pages=[]
    for r,rgb,ref,valid,bg,oracle in evaluation_cases():
        out=folder/r['id'];out.mkdir(exist_ok=True);pred=v16.predict(model,rgb);paths,sk,info=prior_eval.decode(pred,cfg['kind'],cfg['high'],cfg['low']);met=diagnostic(ref,pred,paths,valid,rgb,bg);row=dict(id=r['id'],split=r['split'],metrics=met,inference_reference_loaded=False,partial_assistant_reference=not r['id'].startswith('R'));np.savez_compressed(out/'prediction.npz',prediction=pred);jsave(out/'trajectory.json',dict(paths=[q.round(3).tolist() for q in paths],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,trace_info=info));jsave(out/'metrics.json',row)
        if oracle is not None:
            c,axis,dist=v16.geometry(ref,rgb.shape[:2]);ideal=np.concatenate((np.exp(-dist[None]**2/(2*1.05**2)),(dist[None]<3).astype(np.float32),axis,np.ones_like(dist[None])*.05)).astype(np.float32);oracle_paths,_,_=prior_eval.decode(ideal,cfg['kind'],cfg['high'],cfg['low']);row['oracle_decoder_diagnostic']=path_metrics(ref,oracle_paths,valid);row['oracle_diagnostic_is_reference_derived_not_prediction']=True
        rows.append(row);page=Image.new('RGB',(1800,350),'#f5f3ec');d=ImageDraw.Draw(page)
        for j,(title,a) in enumerate([('原图',rgb),('预测墨迹概率',np.uint8(255*(1-pred[1]))),('预测中心概率',np.uint8(255*(1-pred[0]))),('实际候选轨迹',line_view(rgb.shape[:2],paths,True)),('辅助参考线（非预测）',line_view(rgb.shape[:2],ref,True))]):d.text((360*j+5,4),r['id']+' '+title,font=SMALL,fill='black');paste(page,a,360*j+5,32,350,310)
        page.save(out/'diagnostic.jpg',quality=95);pages.append((r['id'],page));print('V21_EVAL',variant,r['id'],round(met['ink_ref_recall_3px'],3),round(met['center_ref_recall_3px'],3),round(met['mean_best_single_path_coverage'],3),flush=True)
    jsave(folder/'summary.json',dict(records=rows,variant=variant,not_blind=True,official_model_not_promoted=True))
    keys=['stage_029','stage_065','sina_6586334590_31','X01','X04','R04'];pages=dict(pages);board=Image.new('RGB',(1800,350*len(keys)+30),'#f5f3ec');d=ImageDraw.Draw(board);d.text((4,3),'v21 分阶段诊断：墨迹 → 中心 → 轨迹；末栏为辅助参考，不是模型效果',font=FONT,fill='black')
    for i,k in enumerate(keys):board.paste(pages[k],(0,30+i*350))
    board.save(ROOT/variant/'STAGE_DIAGNOSTIC.jpg',quality=95)

def oracle_diagnosis():
    cases=evaluation_cases()
    for variant in ['init','control','improved','ribbon_improved']:
        file=ROOT/variant/'evaluation'/'summary.json'
        if not file.exists():continue
        data=read(file);lookup={r['id']:r for r in data['records']};cfg=read(ROOT/variant/'calibration.json')['selected']
        for r,rgb,ref,valid,bg,targets in cases:
            if targets is None:continue
            c,axis,dist=v16.geometry(ref,rgb.shape[:2]);ideal=np.concatenate((np.exp(-dist[None]**2/(2*1.05**2)),(dist[None]<3).astype(np.float32),axis,np.ones_like(dist[None])*.05)).astype(np.float32);paths,_,_=prior_eval.decode(ideal,cfg['kind'],cfg['high'],cfg['low']);row=lookup[r['id']]
            if 'oracle_decoder_diagnostic' in row:row['earlier_target_ink_constrained_diagnostic']=row['oracle_decoder_diagnostic']
            row['oracle_decoder_diagnostic']=path_metrics(ref,paths,valid);row['oracle_diagnostic_is_reference_derived_not_prediction']=True
        jsave(file,data)
    print('V21_ORACLE_REFERENCE_ONLY_DIAGNOSTICS_UPDATED',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('variant',nargs='?',default='init');a=p.parse_args()
    if a.mode=='candidates':candidates()
    elif a.mode=='canvases':canvases()
    elif a.mode=='prepare':prepare()
    elif a.mode=='train':train(a.variant)
    elif a.mode=='calibrate':calibration(a.variant)
    elif a.mode=='evaluate':evaluate(a.variant)
    elif a.mode=='oracle':oracle_diagnosis()
    elif a.mode=='footprint':footprint_targets()
