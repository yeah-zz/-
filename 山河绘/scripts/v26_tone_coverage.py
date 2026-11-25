# -*- coding: utf-8 -*-
"""Matched tone-consistency experiment and spatially covered field tracing.

Research only. No robot imports. Configure the isolated CUDA runtime BEFORE torch.
"""
from rock_v15_common import LAB,read,jsave,digest,FONT,SMALL
from pathlib import Path
import os,json,math,time,hashlib,shutil
import cv2,numpy as np,torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw
import v24_train as prior
import v24_eval as old_eval
import v21_stroke_learning as v21

ROOT=LAB/'tone_coverage_v26_20261027'
V25=LAB/'joint_center_curves_v25_20261026'
INIT=V25/'fine_tuned.pt'
RATIOS=[.30,.55,.80]
METRICS=['point_coverage_3px','mean_best_single_path_coverage','mean_reference_fragments','path_count','mixed_reference_paths_flagged','low_contrast_ref_coverage_3px','dark_ref_coverage_3px','reference_center_distance_p90','paper_path_density','duplicate_point_fraction']

def fade_image(rgb,ratio):
    # Compress ink contrast relative to estimated local paper; do not uniformly
    # brighten the entire image. No geometric warp or target relabeling occurs.
    paper=v21.v16.photometry(rgb)[2]
    p=np.maximum(paper[:,:,None]*255,rgb.astype(np.float32))
    return np.clip(p+(rgb.astype(np.float32)-p)*ratio,0,255).astype(np.uint8)

def load_model(name):
    file=INIT if name=='base' else ROOT/name/'final.pt'
    m=prior.JointNet().cuda();m.load_state_dict(torch.load(file,map_location='cuda',weights_only=False)['model']);return m.eval()

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'manifest.json').exists():raise FileExistsError('Preserve frozen v26 manifest')
    if not torch.cuda.is_available():raise RuntimeError('GPU required for this experiment')
    x=torch.randn(2,3,32,32,device='cuda',requires_grad=True);conv=torch.nn.Conv2d(3,4,3,padding=1).cuda();y=conv(x).square().mean();y.backward();torch.cuda.synchronize()
    manifest=read(prior.ROOT/'training_manifest.json');records=[]
    for r in manifest['records']:
        if r['split']!='train':continue
        records.append(dict(id=r['id'],kind=r['kind'],data=r['data'],sha256=digest(r['data']),label=r.get('labels'),label_sha256=digest(r['labels']) if r.get('labels') else None))
    cases=[]
    for split in ['calibration','development']:
        for r,rgb,refs,valid,bg in old_eval.cases(split):
            cases.append(dict(id=r['id'],partition=split,group=r['split'],image_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),references_sha256=hashlib.sha256(json.dumps(refs,sort_keys=True,default=lambda a:np.asarray(a).tolist()).encode()).hexdigest(),partial_references=True))
    jsave(ROOT/'manifest.json',dict(init_checkpoint=str(INIT),init_sha256=digest(INIT),records=records,evaluation=cases,steps_per_model=600,seed=262710,training_tone_ratios=RATIOS,synthetic_test_ratio=.40,heldout_is_reused_development_not_blind=True,unknown_not_background=True,new_human_annotation_requested=False,robot_connected=False))
    jsave(ROOT/'device.json',dict(torch_version=torch.__version__,torch_file=torch.__file__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),compiled_arches=torch.cuda.get_arch_list(),forward_backward_verified=True,actual_tensor_device=str(y.device),new_dependencies_installed=False))
    shutil.copy2(Path(__file__),ROOT/'source_at_prepare.py')
    # Preview actual paired-image augmentation, not model output.
    selected=[r for r in manifest['records'] if r['split']=='train' and r['id'] in ['U02','U05','replay_R01']]
    board=Image.new('RGB',(1200,310*len(selected)+35),'#f4f2eb');d=ImageDraw.Draw(board);d.text((5,4),'v26 浓淡配对训练输入：同一几何／原图、0.55墨色、0.30墨色（非模型输出）',font=FONT,fill='black')
    for i,r in enumerate(selected):
        rgb=np.load(r['data'])['rgb']
        for j,(title,a) in enumerate([('原图',rgb),('墨色比例0.55',fade_image(rgb,.55)),('墨色比例0.30',fade_image(rgb,.30))]):
            d.text((400*j+5,35+310*i),r['id']+' '+title,font=SMALL,fill='black');v21.paste(board,a,400*j+5,60+310*i,390,275)
    board.save(ROOT/'TONE_INPUT_AUDIT.jpg',quality=95)
    print('V26_PREPARED',len(records),'training images',len(cases),'evaluation cases',torch.__version__,torch.cuda.get_device_name(),flush=True)

def pool():
    manifest,groups=prior.load_pool();out={};excluded=0
    for kind,rows in groups.items():
        out[kind]=[]
        for r in rows:
            rec=next(a for a in manifest['records'] if a['id']==r['id']);z=np.load(rec['data']);rgb=z['rgb'];t=r['t'].copy();m=r['m'].copy();valid=r['valid'][0].copy()
            if kind=='new_visible_reference':
                strict=np.load(V25/'strict_targets'/(r['id']+'.npz'));t[0]=strict['target'];m[0]=strict['mask']
            m*=valid[None]
            xs=[r['x'].astype(np.float16)]+[v21.v16.photometry(fade_image(rgb,v))[0].astype(np.float16) for v in RATIOS]
            h,w=t.shape[1:];ph=max(0,256-h);pw=max(0,256-w)
            if ph or pw:
                xs=[np.pad(x,((0,0),(0,ph),(0,pw)),mode='reflect') for x in xs];t=np.pad(t,((0,0),(0,ph),(0,pw)));m=np.pad(m,((0,0),(0,ph),(0,pw)))
            out[kind].append(dict(id=r['id'],x=xs,t=t,m=m,pairs=r['pairs']))
    return out

def crop(row,rng,level):
    h,w=row['t'].shape[1:]
    if row['pairs'] and rng.random()<.6:
        p=row['pairs'][int(rng.integers(len(row['pairs'])))];q=np.asarray(p['center'])+rng.uniform(-35,35,2);left=int(np.clip(q[0]-128,0,w-256));top=int(np.clip(q[1]-128,0,h-256))
    else:left=int(rng.integers(w-255));top=int(rng.integers(h-255))
    arr=[row['x'][0][:,top:top+256,left:left+256].astype(np.float32),row['x'][level+1][:,top:top+256,left:left+256].astype(np.float32),row['t'][:,top:top+256,left:left+256].copy(),row['m'][:,top:top+256,left:left+256].copy()]
    flip=rng.random()<.5;rot=int(rng.integers(4))
    if flip:
        arr=[a[:,:,::-1].copy() for a in arr];arr[2][3]*=-1
    if rot:
        arr=[np.rot90(a,rot,axes=(1,2)).copy() for a in arr]
        if rot%2:arr[2][2:4]*=-1
    return arr,(row['id'],level,left,top,flip,rot)

def train(name):
    if name not in ['control','tone']:raise ValueError(name)
    folder=ROOT/name;folder.mkdir(exist_ok=True)
    if (folder/'final.pt').exists():raise FileExistsError('Preserve completed checkpoint')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(262710);rng=np.random.default_rng(262710);groups=pool();model=load_model('base').train()
    for layer in [model.node_head,model.embedding_head]:
        for p in layer.parameters():p.requires_grad_(False)
    enc=list(model.a.parameters())+list(model.b.parameters())+list(model.c.parameters());dec=list(model.d.parameters())+list(model.e.parameters())+list(model.head.parameters());rates=[6e-6,2e-5];opt=torch.optim.AdamW([dict(params=enc,lr=rates[0]),dict(params=dec,lr=rates[1])],weight_decay=1e-4)
    cfg=dict(variant=name,steps=600,seed=262710,device='cuda',gpu=torch.cuda.get_device_name(),init_sha256=digest(INIT),same_original_sample_schedule=True,original_loss_weight=1 if name=='control' else .5,faded_supervision_weight=0 if name=='control' else .5,consistency_weight=0 if name=='control' else .25,all_consistency_terms_masked_by_known_supervision=True,width_is_apparent_pixels_not_robot_pressure=True,robot_connected=False)
    jsave(folder/'config.json',cfg);shutil.copy2(Path(__file__),folder/'source_at_training.py');schedule=hashlib.sha256();usage={};start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(1,601):
        batch=[]
        for _ in range(4):
            u=rng.random();kind='new_visible_reference' if u<.4 else 'v21_clear_replay' if u<.7 else 'reviewed_replay';row=groups[kind][int(rng.integers(len(groups[kind])))];level=int(rng.integers(3));a,meta=crop(row,rng,level);batch.append(a);schedule.update(json.dumps(meta).encode());usage[row['id']]=usage.get(row['id'],0)+1
        x,xf,t,m=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(4)];factor=.25+.75*.5*(1+math.cos(math.pi*step/600))
        for g,lr in zip(opt.param_groups,rates):g['lr']=lr*factor
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            o=model(x)[:,:5];orig=v21.train17.loss_fn(o,t,m);faded=orig.new_zeros(());cons=orig.new_zeros(())
            if name=='tone':
                of=model(xf)[:,:5];faded=v21.train17.loss_fn(of,t,m);a=o.float();b=of.float();cons=(((b[:,:2].sigmoid()-a[:,:2].sigmoid().detach())**2)*m[:,:2]).sum()/m[:,:2].sum().clamp_min(1);cons+=.1*((1-F.cosine_similarity(b[:,2:4],a[:,2:4].detach(),dim=1))*m[:,2]).sum()/m[:,2].sum().clamp_min(1);loss=.5*(orig+faded)+.25*cons
            else:loss=orig
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradients')
        opt.step()
        if step%100==0:
            log=dict(step=step,loss=float(loss.detach()),original_loss=float(orig.detach()),faded_loss=float(faded.detach()),consistency=float(cons.detach()),elapsed_seconds=time.perf_counter()-start)
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(log)+'\n')
            print('V26_TRAIN',name,json.dumps(log),flush=True)
    torch.cuda.synchronize();torch.save(dict(model=model.state_dict(),config=cfg),folder/'final.pt');jsave(folder/'summary.json',dict(updates=600,sample_usage=usage,sampler_sha256=schedule.hexdigest(),elapsed_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=digest(folder/'final.pt'),actual_device=str(next(model.parameters()).device),robot_connected=False));print('V26_TRAINED',name,flush=True)

def adaptive_curves(pred,cfg):
    """Spatially distributed seeds, vectorized bidirectional field integration.

    No reference inputs and no global 40-seed cap. A geometrical loop safety
    budget is recorded as a termination cause instead of hidden truncation.
    """
    center,ink=pred[:2];h,w=center.shape;theta=.5*np.arctan2(pred[3],pred[2]);yy,xx=np.mgrid[:h,:w].astype(np.float32);g=cv2.GaussianBlur(center,(0,0),.7);nx=-np.sin(theta);ny=np.cos(theta)
    a=cv2.remap(g,xx+nx,yy+ny,cv2.INTER_LINEAR);b=cv2.remap(g,xx-nx,yy-ny,cv2.INTER_LINEAR);ys,xs=np.where((g>=a)&(g>=b)&(center>=cfg['seed'])&(ink>=cfg['ink']))
    occupied=np.zeros((h,w),np.uint8);counts={};seeds=[]
    for i in np.argsort(-center[ys,xs]):
        x,y=int(xs[i]),int(ys[i]);cell=(x//64,y//64)
        if occupied[y,x] or counts.get(cell,0)>=16:continue
        seeds.append([x,y]);counts[cell]=counts.get(cell,0)+1;cv2.circle(occupied,(x,y),7,1,-1)
    if not seeds:return [],dict(seeds=0,safety_budget_hits=0,reference_used=False)
    seed=np.asarray(seeds,np.float32);pos=np.repeat(seed,2,axis=0);n=len(pos);direction=np.stack([np.cos(theta[pos[:,1].astype(int),pos[:,0].astype(int)]),np.sin(theta[pos[:,1].astype(int),pos[:,0].astype(int)])],1);direction[1::2]*=-1
    active=np.ones(n,bool);low=np.zeros(n,np.int16);length=np.ones(n,np.int32);budget=int(math.ceil(2*math.hypot(h,w)/1.25))+32;hist=np.full((budget+1,n,2),np.nan,np.float32);hist[0]=pos;reason=np.zeros(n,np.int16);field=pred[:4].transpose(1,2,0).copy()
    def query(points):return cv2.remap(field,points[:,0][None].astype(np.float32),points[:,1][None].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]
    for step in range(budget):
        ids=np.flatnonzero(active)
        if not len(ids):break
        p=pos[ids];v=query(p);ang=.5*np.arctan2(v[:,3],v[:,2]);d=np.c_[np.cos(ang),np.sin(ang)];dot=(d*direction[ids]).sum(1);d[dot<0]*=-1;sharp=np.abs(dot)<.17;normal=np.c_[-d[:,1],d[:,0]];q0=p+1.25*d;offset=np.array([-1.5,0,1.5],np.float32);qq=q0[:,None]+normal[:,None]*offset[None,:,None];values=cv2.remap(center,qq[:,:,0],qq[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);k=values.argmax(1);shift=offset[k].copy();mid=k==1;den=values[:,0]-2*values[:,1]+values[:,2];shift[mid]=np.clip(.75*(values[mid,0]-values[mid,2])/np.where(np.abs(den[mid])>1e-6,den[mid],-1e-6),-.75,.75);q=q0+normal*shift[:,None];fv=query(q);bad=(fv[:,0]<cfg['follow'])|(fv[:,1]<cfg['ink']);low[ids]=np.where(bad,low[ids]+1,0);outside=(q[:,0]<1)|(q[:,0]>=w-1)|(q[:,1]<1)|(q[:,1]>=h-1);loop=np.zeros(len(ids),bool)
        if step>16:
            loop=np.linalg.norm(q-hist[0,ids],axis=1)<1.3
            recent=hist[max(0,step-14):max(1,step-7),ids];loop|=np.nanmin(np.linalg.norm(recent-q[None],axis=2),axis=0)<.85
        stop=(low[ids]>=2)|outside|sharp|loop;reason[ids[stop]]=np.where(outside[stop],1,np.where(sharp[stop],2,np.where(loop[stop],3,4)));active[ids[stop]]=False;keep=ids[~stop];pos[keep]=q[~stop];direction[keep]=d[~stop];hist[step+1,keep]=pos[keep];length[keep]+=1
    reason[active]=5
    raw=[]
    for i in range(len(seed)):
        a=hist[:length[2*i],2*i];b=hist[:length[2*i+1],2*i+1];p=np.concatenate([b[::-1][:-1],a]);p=p[np.isfinite(p).all(1)]
        if len(p)>=12:raw.append(p)
    raw.sort(key=len,reverse=True);covered=np.zeros((h,w),np.uint8);paths=[];duplicates=0
    for p in raw:
        ij=np.rint(p).astype(np.int32);ij[:,0]=np.clip(ij[:,0],0,w-1);ij[:,1]=np.clip(ij[:,1],0,h-1);seen=covered[ij[:,1],ij[:,0]]>0
        if seen.mean()>.72 or np.count_nonzero(~seen)<8:duplicates+=1;continue
        paths.append(p);cv2.polylines(covered,[ij],False,1,5,cv2.LINE_8)
    return paths,dict(seeds=len(seed),seed_cells=len(counts),max_steps=budget,safety_budget_hits=int((reason==5).sum()),termination_counts={str(i):int((reason==i).sum()) for i in range(1,6)},raw_curves=len(raw),deduplicated_curves=duplicates,reference_used=False)

def infer():
    torch.set_num_threads(4);cv2.setNumThreads(2)
    for name in ['base','control','tone']:
        model=load_model(name)
        for partition in ['calibration','development']:
            for r,rgb,refs,valid,bg in old_eval.cases(partition):
                for condition,im in [('original',rgb),('synthetic_fade_040',fade_image(rgb,.40))]:
                    file=ROOT/'predictions'/name/condition/(r['id']+'.npz');file.parent.mkdir(parents=True,exist_ok=True)
                    if not file.exists():np.savez_compressed(file,prediction=prior.predict(model,im)[:5])
                print('V26_INFER',name,partition,r['id'],flush=True)
        del model;torch.cuda.empty_cache()

def raster(paths,shape):
    a=np.zeros(shape,np.uint8)
    for p in paths:
        if len(p)>1:cv2.polylines(a,[np.rint(p).astype(np.int32)],False,1,1,cv2.LINE_8)
    return a

def evaluate_paths(refs,paths,valid,bg,rgb):
    out=v21.path_metrics(refs,paths,valid);sk=raster(paths,valid.shape);dist=distance_transform_edt(sk==0);points=[]
    for p in refs:
        q=v21.dense(np.asarray(p,np.float32));xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,sk.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,sk.shape[0]-1);xy=xy[valid[xy[:,1],xy[:,0]]>.5];points.extend(xy.tolist())
    if points:
        xy=np.asarray(points,int);alpha=v21.v16.photometry(rgb)[1];strength=cv2.dilate(alpha,np.ones((5,5),np.uint8))[xy[:,1],xy[:,0]];faint=(strength>=.015)&(strength<.10);dark=strength>=.20;d=dist[xy[:,1],xy[:,0]];out['low_contrast_ref_coverage_3px']=float((d[faint]<=3).mean()) if faint.any() else None;out['dark_ref_coverage_3px']=float((d[dark]<=3).mean()) if dark.any() else None;out['reference_center_distance_p90']=float(np.percentile(d,90));out['low_contrast_reference_points']=int(faint.sum())
    else:out.update(low_contrast_ref_coverage_3px=None,dark_ref_coverage_3px=None,reference_center_distance_p90=None,low_contrast_reference_points=0)
    out['paper_path_density']=float(sk[bg>0].mean()) if np.any(bg) else None
    seen=np.zeros_like(sk);dup=0;count=0
    for p in paths:
        xy=np.rint(p).astype(int);xy[:,0]=np.clip(xy[:,0],0,sk.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,sk.shape[0]-1);dup+=int((seen[xy[:,1],xy[:,0]]>0).sum());count+=len(xy);cv2.polylines(seen,[xy.astype(np.int32)],False,1,3)
    out['duplicate_point_fraction']=float(dup/max(1,count));return out

def calibrate():
    # Threshold selection uses only original calibration images and base weights.
    options=[]
    configs=[dict(seed=.55,follow=.22,ink=.35),dict(seed=.65,follow=.30,ink=.45),dict(seed=.65,follow=.35,ink=.50)]
    for cfg in configs:
        rows=[]
        for r,rgb,refs,valid,bg in old_eval.cases('calibration'):
            p=np.load(ROOT/'predictions'/'base'/'original'/(r['id']+'.npz'))['prediction'];paths,info=adaptive_curves(p,cfg);rows.append(dict(id=r['id'],metrics=evaluate_paths(refs,paths,valid,bg,rgb),decoder=info))
        score=float(np.mean([.5*r['metrics']['point_coverage_3px']+.5*r['metrics']['mean_best_single_path_coverage']-.025*r['metrics']['mixed_reference_paths_flagged']-min(.1,10*(r['metrics']['paper_path_density'] or 0)) for r in rows]));options.append(dict(config=cfg,score=score,records=rows));print('V26_CALIBRATION',cfg,round(score,4),flush=True)
    jsave(ROOT/'calibration.json',dict(selected=max(options,key=lambda r:r['score']),options=options,weights='base',development_not_used=True,same_config_for_all_models=True))

def evaluate():
    cv2.setNumThreads(2);cfg=read(ROOT/'calibration.json')['selected']['config'];records=[];views={};keys=['H01','H02','stage_065','sina_6586334590_31','X04','R04']
    variants=[('baseline','base','graph'),('decode_only','base','adaptive'),('control_graph','control','graph'),('control_adaptive','control','adaptive'),('tone_only','tone','graph'),('tone_and_decode','tone','adaptive')]
    for r,rgb,refs,valid,bg in old_eval.cases('development'):
        for condition in ['original','synthetic_fade_040']:
            row=dict(id=r['id'],group=r['split'],condition=condition,metrics={},decoder={})
            for variant,model_name,decoder in variants:
                pred=np.load(ROOT/'predictions'/model_name/condition/(r['id']+'.npz'))['prediction']
                if decoder=='graph':paths,_,_=old_eval.decode(pred,dict(node=None,affinity=None));info=dict(reference_used=False,decoder='previous_graph')
                else:paths,info=adaptive_curves(pred,cfg)
                row['metrics'][variant]=evaluate_paths(refs,paths,valid,bg,rgb);row['decoder'][variant]=info;dest=ROOT/'trajectories'/condition/variant/(r['id']+'.json');jsave(dest,dict(paths=[p.round(3).tolist() for p in paths],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,reference_labels_used_at_inference=False,decoder_info=info))
                if r['id'] in keys:views[r['id'],condition,variant]=paths
            records.append(row);print('V26_EVAL',condition,r['id'],{k:round(v['mean_best_single_path_coverage'],3) for k,v in row['metrics'].items()},flush=True)
    groups={}
    for condition in ['original','synthetic_fade_040']:
        groups[condition]={}
        for g in sorted(set(r['group'] for r in records)):
            rr=[r for r in records if r['condition']==condition and r['group']==g];groups[condition][g]={}
            for variant,_,_ in variants:
                groups[condition][g][variant]={}
                for metric in METRICS:
                    values=[r['metrics'][variant][metric] for r in rr if r['metrics'][variant][metric] is not None];groups[condition][g][variant][metric]=float(np.mean(values)) if values else None
    jsave(ROOT/'evaluation.json',dict(records=records,groups=groups,partial_reference_metrics_not_full_image_accuracy=True,natural_faint_metric_is_local_contrast_proxy=True,synthetic_fade_test_not_real_faint_gold=True,known_development_not_blind=True))
    lookup={r['id']:rgb for r,rgb,refs,valid,bg in old_eval.cases('development')}
    for condition in ['original','synthetic_fade_040']:
        board=Image.new('RGB',(2000,len(keys)*305+40),'#f4f2eb');d=ImageDraw.Draw(board);d.text((5,4),'v26 '+condition+'：输入／基线／只改解码／只改浓淡训练／两项结合（离线像素轨迹）',font=FONT,fill='black')
        for i,key in enumerate(keys):
            rgb=lookup[key];im=rgb if condition=='original' else fade_image(rgb,.40)
            for j,(title,a) in enumerate([('输入',im),('基线',v21.line_view(rgb.shape[:2],views[key,condition,'baseline'],True)),('只改解码',v21.line_view(rgb.shape[:2],views[key,condition,'decode_only'],True)),('只改浓淡训练',v21.line_view(rgb.shape[:2],views[key,condition,'tone_only'],True)),('浓淡训练+解码',v21.line_view(rgb.shape[:2],views[key,condition,'tone_and_decode'],True))]):
                d.text((400*j+5,40+305*i),key+' '+title,font=SMALL,fill='black');v21.paste(board,a,400*j+5,66+305*i,390,232)
        board.save(ROOT/('COMPARISON_'+condition+'.jpg'),quality=95)
    print('V26_EVALUATION_FINISHED',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','infer','calibrate','evaluate','finalize']);p.add_argument('--variant',choices=['control','tone']);a=p.parse_args()
    {'prepare':prepare,'train':lambda:train(a.variant),'infer':infer,'calibrate':calibrate,'evaluate':evaluate,'finalize':lambda:__import__('v26_finalize').main()}[a.mode]()
