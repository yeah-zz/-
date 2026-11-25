# -*- coding: utf-8 -*-
"""Real faint-line emphasis and conservative junction/overlap refinement.

Offline research. Import the isolated E-drive runtime before torch.
"""
from rock_v15_common import LAB,read,jsave,digest,FONT,SMALL,PARENT
from pathlib import Path
import json,math,time,hashlib,shutil
import cv2,numpy as np,torch
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
import v26_tone_coverage as prev
import v23_junction_relations as rel
from stroke_link_v12 import dense,tangent

ROOT=LAB/'separation_real_faint_v27_20261027'
REVIEW=LAB/'cross_sections_v18_20261025'
GROUPS=['development_holdout','fixed_external_development','new_work_development','user_reviewed_old_development','legacy_real_faint_development']

def cases(partition):
    result=list(prev.old_eval.cases(partition))
    if partition=='development':
        for r in read(REVIEW/'manifest.json')['records']:
            if r['id'] not in ['N10','N11']:continue
            z=np.load(r['data']);doc=read(r['labels']);refs=[s['points'] for s in doc['strokes']];result.append((dict(r,split='legacy_real_faint_development'),z['rgb'],refs,z['valid'],z['background']>0))
    return result

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'manifest.json').exists():raise FileExistsError('Preserve frozen experiment')
    (ROOT/'focus').mkdir(exist_ok=True);source=read(prev.prior.ROOT/'training_manifest.json');review=read(REVIEW/'expert_cross_sections'/'reviewed.json')['annotations'];candidate={r['id']:r for r in read(REVIEW/'review_candidates.json')};records=[];audit=[]
    for r in source['records']:
        if r['split']!='train':continue
        z=np.load(r['data']);rgb=z['rgb'];t=z['target'].copy();m=z['mask'].copy();valid=z['valid'].copy()
        if r['kind']=='new_visible_reference':
            strict=np.load(prev.V25/'strict_targets'/(r['id']+'.npz'));t[0]=strict['target'];m[0]=strict['mask']
        known=(m[0]>0)&(valid>0);center=(t[0]>.5)&known;alpha=prev.v21.v16.photometry(rgb)[1];local=cv2.dilate(alpha,np.ones((3,3),np.uint8));weak=center&(local>=.008)&(local<.15)
        # Reweight existing valid center supervision near user-confirmed faint
        # cross-sections. Their initial center guesses do not become new gold.
        manual=np.zeros(valid.shape,np.uint8);anchors=[];sid=r['id'].replace('replay_','')
        for a in review:
            if a['sample_id']!=sid or a['label']!='faint' or not a.get('reviewed'):continue
            c=candidate[a['id']];p=np.asarray(c['p'])+np.asarray(c['normal'])*float(a['center']);x,y=np.rint(p).astype(int);cv2.circle(manual,(int(x),int(y)),14,1,-1);anchors.append(dict(id=a['id'],p=p.tolist(),presence_confirmed=True,exact_geometry_not_assumed=True))
        selected=(weak|(center&(manual>0)))&known;n,labels,stats,_=cv2.connectedComponentsWithStats(selected.astype(np.uint8),8);good=np.zeros(n,bool);good[stats[:,cv2.CC_STAT_AREA]>=4]=True;good[0]=False;selected=good[labels]
        weight=cv2.GaussianBlur(cv2.dilate(selected.astype(np.float32),np.ones((5,5),np.uint8)),(0,0),1);weight=np.clip(weight,0,1)*valid
        # The weighting mask does not create any new positive/negative labels.
        file=ROOT/'focus'/(r['id']+'.npz');np.savez_compressed(file,weight=weight.astype(np.float32),selected=selected.astype(np.uint8));ys,xs=np.where(selected);indices=np.arange(0,len(xs),max(1,len(xs)//100));points=np.c_[xs[indices],ys[indices]].tolist() if len(xs) else []
        records.append(dict(id=r['id'],kind=r['kind'],data=r['data'],data_sha256=digest(r['data']),labels=r.get('labels'),labels_sha256=digest(r['labels']) if r.get('labels') else None,focus=str(file),focus_sha256=digest(file),selected_real_faint_center_pixels=int(selected.sum()),manual_faint_anchors=anchors,focus_points=points,source='existing_image_and_valid_reference_masks_not_model_prediction'))
        ov=rgb.copy();ov[selected]=[230,35,35]
        for a in anchors:cv2.circle(ov,tuple(np.rint(a['p']).astype(int)),14,(45,130,230),1)
        if r['id'] in ['replay_R01','replay_R02','replay_N04','replay_N09','U04','U05']:audit.append((r['id'],rgb,ov))
    evals=[]
    for partition in ['calibration','development']:
        for r,rgb,refs,valid,bg in cases(partition):evals.append(dict(id=r['id'],partition=partition,group=r['split'],image_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),labels_sha256=hashlib.sha256(json.dumps(refs,default=lambda a:np.asarray(a).tolist()).encode()).hexdigest()))
    jsave(ROOT/'manifest.json',dict(records=records,evaluation=evals,init_checkpoint=str(prev.INIT),init_sha256=digest(prev.INIT),user_review_sha256=digest(REVIEW/'expert_cross_sections'/'reviewed.json'),confirmed_faint_cross_sections=sum(len(r['manual_faint_anchors']) for r in records),steps=600,seed=272710,training_focus_probability=.60,loss_focus_multiplier=3,all_unknown_pixels_remain_unknown=True,new_independent_artworks=0,new_independent_expert_gold=False,robot_connected=False))
    board=Image.new('RGB',(1200,len(audit)*290+40),'#f4f2eb');d=ImageDraw.Draw(board);d.text((5,5),'v27 真实淡线重点监督审计：红=原有可信参考中的低对比度中心；蓝圈=已确认淡墨横截面位置',font=FONT,fill='black')
    for i,(key,a,b) in enumerate(audit):
        d.text((5,40+i*290),key+' 原图 / 重点监督（非模型预测）',font=SMALL,fill='black');prev.v21.paste(board,a,5,65+i*290,590,225);prev.v21.paste(board,b,605,65+i*290,590,225)
    board.save(ROOT/'REAL_FAINT_SUPERVISION_AUDIT.jpg',quality=95);shutil.copy2(Path(__file__),ROOT/'source_at_prepare.py');print('V27_PREPARED',[(r['id'],r['selected_real_faint_center_pixels'],len(r['manual_faint_anchors'])) for r in records],flush=True)

def load_model(name):
    if name=='base':return prev.load_model('base')
    model=prev.prior.JointNet().cuda();model.load_state_dict(torch.load(ROOT/name/'final.pt',map_location='cuda',weights_only=False)['model']);return model.eval()

def training_pool():
    manifest=read(ROOT/'manifest.json');_,groups=prev.prior.load_pool();lookup={r['id']:r for r in manifest['records']};rows=[]
    for group in groups.values():
        for row in group:
            r=lookup[row['id']];t=row['t'].copy();m=row['m'].copy();valid=row['valid'][0];weight=np.load(r['focus'])['weight']
            if row['kind']=='new_visible_reference':
                strict=np.load(prev.V25/'strict_targets'/(row['id']+'.npz'));t[0]=strict['target'];m[0]=strict['mask']
            m*=valid[None];m=np.concatenate([m,weight[None]],0);x=row['x'].astype(np.float16);h,w=t.shape[1:];ph=max(0,256-h);pw=max(0,256-w)
            if ph or pw:x=np.pad(x,((0,0),(0,ph),(0,pw)),mode='reflect');t=np.pad(t,((0,0),(0,ph),(0,pw)));m=np.pad(m,((0,0),(0,ph),(0,pw)))
            rows.append(dict(id=row['id'],x=[x,x,x,x],t=t,m=m,pairs=[dict(center=p) for p in r['focus_points']] or row['pairs'],has_focus=bool(r['selected_real_faint_center_pixels'])))
    return rows

def train(variant):
    folder=ROOT/variant;folder.mkdir(exist_ok=True)
    if (folder/'final.pt').exists():raise FileExistsError('Preserve completed checkpoint')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(272710);rng=np.random.default_rng(272710);rows=training_pool();focused=[r for r in rows if r['has_focus']];model=load_model('base').train()
    for layer in [model.node_head,model.embedding_head]:
        for p in layer.parameters():p.requires_grad_(False)
    enc=list(model.a.parameters())+list(model.b.parameters())+list(model.c.parameters());dec=list(model.d.parameters())+list(model.e.parameters())+list(model.head.parameters());rates=[6e-6,2e-5];opt=torch.optim.AdamW([dict(params=enc,lr=rates[0]),dict(params=dec,lr=rates[1])],weight_decay=1e-4);schedule=hashlib.sha256();usage={};start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    cfg=dict(variant=variant,steps=600,seed=272710,init_sha256=digest(prev.INIT),manifest_sha256=digest(ROOT/'manifest.json'),actual_device=str(next(model.parameters()).device),gpu=torch.cuda.get_device_name(),real_images_only=True,faint_center_geometry_not_relabelled=True,focus_loss_multiplier=3 if variant=='faint' else 1,matched_sampling=True,robot_connected=False);jsave(folder/'config.json',cfg);shutil.copy2(Path(__file__),folder/'source_at_training.py')
    for step in range(1,601):
        batch=[]
        for _ in range(4):
            pool=focused if rng.random()<.60 else rows;row=pool[int(rng.integers(len(pool)))];arr,meta=prev.crop(row,rng,0);batch.append(arr);schedule.update(json.dumps(meta).encode());usage[row['id']]=usage.get(row['id'],0)+1
        x,unused,t,m=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(4)];supervision=m[:,:4]*(1+2*m[:,4:5]) if variant=='faint' else m[:,:4]
        for g,lr in zip(opt.param_groups,rates):g['lr']=lr*(.25+.75*.5*(1+math.cos(math.pi*step/600)))
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=prev.v21.train17.loss_fn(model(x)[:,:5],t,supervision)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradients')
        opt.step()
        if step%100==0:
            log=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start)
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(log)+'\n')
            print('V27_TRAIN',variant,json.dumps(log),flush=True)
    torch.cuda.synchronize();torch.save(dict(model=model.state_dict(),config=cfg),folder/'final.pt');jsave(folder/'summary.json',dict(updates=600,actual_device=str(next(model.parameters()).device),gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,elapsed_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),schedule_sha256=schedule.hexdigest(),usage=usage,checkpoint_sha256=digest(folder/'final.pt')));print('V27_TRAINED',variant,flush=True)

def junctions(pred,rgb):
    graph=rel.graph(pred,rgb);accepted=set(rel.original_pairings(graph));by_node={}
    for i,c in enumerate(graph['candidates']):
        key=tuple(c['node']);node=by_node.setdefault(key,dict(ends=set(),allowed=set()));node['ends'].update([c['a'],c['b']])
        if i in accepted:node['allowed'].add(frozenset([c['a'],c['b']]))
    out=[]
    for xy,node in by_node.items():
        if len(node['ends'])<3:continue
        ends=sorted(node['ends']);vectors=np.stack([tangent(graph['chains'][e[0]],e[1]==0,span=6) for e in ends]);out.append(dict(p=np.asarray(xy),ends=ends,vectors=vectors,allowed=node['allowed']))
    return out

def runs(mask):
    a=np.r_[False,mask,False].astype(np.int8);starts=np.flatnonzero(np.diff(a)==1);ends=np.flatnonzero(np.diff(a)==-1);return list(zip(starts,ends))

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-6)

def refine(paths,pred,rgb,cfg,nodes=None):
    nodes=junctions(pred,rgb) if nodes is None and cfg['junction'] else nodes or [];pieces=[];cuts=0;discarded=0
    for original in paths:
        p=dense(np.asarray(original,np.float32)).astype(np.float32)
        if len(p)<7:continue
        cut=[];tree=cKDTree(p)
        if cfg['junction']:
            for node in nodes:
                dist,i=tree.query(node['p']);i=int(i)
                if dist>2.5 or i<6 or i>=len(p)-6:continue
                u=unit(p[i-6]-p[i]);v=unit(p[i+6]-p[i]);vectors=node['vectors'];best=(-1,None)
                for a in range(len(vectors)):
                    for b in range(len(vectors)):
                        if a==b:continue
                        qa=float(vectors[a]@u);qb=float(vectors[b]@v);score=min(qa,qb)
                        if score>best[0]:best=(score,frozenset([node['ends'][a],node['ends'][b]]))
                if best[0]<.80 or best[1] not in node['allowed']:cut.append(i)
        cut=sorted(set(cut));cuts+=len(cut);bounds=[0]+cut+[len(p)-1]
        for a,b in zip(bounds[:-1],bounds[1:]):
            if b-a>=6:pieces.append(p[a:b+1])
            else:discarded+=b-a
    pieces.sort(key=lambda p:float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),reverse=True);accepted=[];old_points=[];old_tangents=[];tree=None;removed=0;events=0
    for p in pieces:
        tang=unit(np.gradient(p,axis=0));duplicate=np.zeros(len(p),bool)
        if tree is not None:
            dd,ii=tree.query(p);aligned=np.abs((tang*np.asarray(old_tangents)[ii]).sum(1))>.94;duplicate=(dd<=cfg['radius'])&aligned
        keep=np.ones(len(p),bool)
        for a,b in runs(duplicate):
            if b-a>=4:keep[a:b]=False;removed+=b-a;events+=1
        added=[]
        for a,b in runs(keep):
            if b-a>=7:added.append(p[a:b])
            else:discarded+=b-a
        if not added:continue
        accepted.extend(added)
        for a in added:old_points.extend(a.tolist());old_tangents.extend(unit(np.gradient(a,axis=0)).tolist())
        tree=cKDTree(np.asarray(old_points))
    return accepted,dict(junction_cut_events=int(cuts),overlap_intervals_removed=int(events),overlap_points_removed=int(removed),short_fragment_points_discarded=int(discarded),input_curves=len(paths),output_curves=len(accepted),radius=cfg['radius'],junction_enabled=cfg['junction'],reference_used=False)

def infer():
    torch.set_num_threads(4);cv2.setNumThreads(2)
    for name in ['base','control','faint']:
        model=load_model(name)
        for partition in ['calibration','development']:
            for r,rgb,refs,valid,bg in cases(partition):
                dest=ROOT/'predictions'/name/(r['id']+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():np.savez_compressed(dest,prediction=prev.prior.predict(model,rgb)[:5])
                print('V27_INFER',name,partition,r['id'],flush=True)
        del model;torch.cuda.empty_cache()

def calibrate():
    cfg26=read(prev.ROOT/'calibration.json')['selected']['config'];configs=[dict(radius=r,junction=j) for r in [1.,1.5] for j in [False,True]];prepared=[]
    for r,rgb,refs,valid,bg in cases('calibration'):
        pred=np.load(ROOT/'predictions'/'base'/(r['id']+'.npz'))['prediction'];paths,info=prev.adaptive_curves(pred,cfg26);prepared.append((r,rgb,refs,valid,bg,pred,paths,junctions(pred,rgb)))
    candidates=[]
    def aggregate(rows):
        return {k:float(np.mean([r[k] for r in rows])) for k in ['point_coverage_3px','mean_best_single_path_coverage','mixed_reference_paths_flagged','duplicate_point_fraction']}
    base=aggregate([prev.evaluate_paths(refs,paths,valid,bg,rgb) for r,rgb,refs,valid,bg,pred,paths,nodes in prepared])
    for cfg in configs:
        rows=[]
        for r,rgb,refs,valid,bg,pred,paths,nodes in prepared:
            output,info=refine(paths,pred,rgb,cfg,nodes);rows.append(prev.evaluate_paths(refs,output,valid,bg,rgb))
        agg=aggregate(rows);score=.5*agg['point_coverage_3px']+.5*agg['mean_best_single_path_coverage']-.025*agg['mixed_reference_paths_flagged']-.1*agg['duplicate_point_fraction'];eligible=agg['point_coverage_3px']>=base['point_coverage_3px']-.01 and agg['mean_best_single_path_coverage']>=base['mean_best_single_path_coverage']-.03 and agg['mixed_reference_paths_flagged']<=base['mixed_reference_paths_flagged'] and agg['duplicate_point_fraction']<=base['duplicate_point_fraction']-.02;candidates.append(dict(config=cfg,metrics=agg,score=score,eligible=bool(eligible)));print('V27_CALIBRATE',cfg,agg,'eligible',eligible,flush=True)
    joint=[r for r in candidates if r['config']['junction']];eligible=[r for r in joint if r['eligible']];selected=max(eligible or joint,key=lambda r:r['score']);dedup_trial=max([r for r in candidates if not r['config']['junction']],key=lambda r:r['score']);jsave(ROOT/'calibration.json',dict(baseline=base,candidates=candidates,selected=selected,dedup_trial=dedup_trial,gate_passed=bool(eligible),calibration_only=True,same_config_for_all_models=True,prior_decoder_config=cfg26))

def evaluate():
    cv2.setNumThreads(2);choice=read(ROOT/'calibration.json');cfg=choice['selected']['config'];cfg26=choice['prior_decoder_config'];records=[];variants=['baseline_graph','previous_adaptive','dedup_only','separation_dedup','control_graph','control_refined','faint_graph','faint_refined']
    for r,rgb,refs,valid,bg in cases('development'):
        row=dict(id=r['id'],group=r['split'],metrics={},diagnostics={})
        for name in ['base','control','faint']:
            pred=np.load(ROOT/'predictions'/name/(r['id']+'.npz'))['prediction'];graph,_,_=prev.old_eval.decode(pred,dict(node=None,affinity=None));adaptive,adaptive_info=prev.adaptive_curves(pred,cfg26);nodes=junctions(pred,rgb);processed,processed_info=refine(adaptive,pred,rgb,cfg,nodes)
            results=[('baseline_graph',graph,{}),('previous_adaptive',adaptive,adaptive_info),('separation_dedup',processed,processed_info)] if name=='base' else [(name+'_graph',graph,{}),(name+'_refined',processed,processed_info)]
            if name=='base':
                only,info=refine(adaptive,pred,rgb,choice['dedup_trial']['config'],nodes);results.append(('dedup_only',only,info))
            for variant,paths,info in results:
                row['metrics'][variant]=prev.evaluate_paths(refs,paths,valid,bg,rgb);row['diagnostics'][variant]=info;jsave(ROOT/'trajectories'/variant/(r['id']+'.json'),dict(paths=[p.round(3).tolist() for p in paths],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,reference_labels_used_at_inference=False,diagnostics=info))
        records.append(row);print('V27_EVAL',r['id'],{k:round(v['mean_best_single_path_coverage'],3) for k,v in row['metrics'].items()},flush=True)
    metrics=prev.METRICS;groups={}
    for g in sorted(set(r['group'] for r in records)):
        rr=[r for r in records if r['group']==g];groups[g]={v:{} for v in variants}
        for v in variants:
            for m in metrics:
                values=[r['metrics'][v][m] for r in rr if r['metrics'][v][m] is not None];groups[g][v][m]=float(np.mean(values)) if values else None
    jsave(ROOT/'evaluation.json',dict(records=records,groups=groups,partial_references_not_image_accuracy=True,no_new_blind_test=True,calibration_config=cfg,calibration_gate_passed=choice['gate_passed']))
    keys=['H02','R04','X04','N10','N11'];lookup={r['id']:rgb for r,rgb,refs,valid,bg in cases('development')};board=Image.new('RGB',(2000,len(keys)*310+40),'#f4f2eb');d=ImageDraw.Draw(board);d.text((5,5),'v27：原图／v26连续轨迹／局部去重／分笔与去重／真实淡线训练+分笔去重（全为离线结果）',font=FONT,fill='black')
    for i,key in enumerate(keys):
        rgb=lookup[key];images=[('输入',rgb)]
        for v,title in [('previous_adaptive','v26连续轨迹'),('dedup_only','只做局部去重'),('separation_dedup','分笔与去重'),('faint_refined','真实淡线训练+分笔去重')]:images.append((title,prev.v21.line_view(rgb.shape[:2],read(ROOT/'trajectories'/v/(key+'.json'))['paths'],True)))
        for j,(title,im) in enumerate(images):d.text((j*400+5,40+i*310),key+' '+title,font=SMALL,fill='black');prev.v21.paste(board,im,j*400+5,65+i*310,390,240)
    board.save(ROOT/'RESULT_COMPARISON.jpg',quality=95);print('V27_EVALUATED',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','infer','calibrate','evaluate','union_calibrate','union_evaluate','finalize']);p.add_argument('--variant',choices=['control','faint']);a=p.parse_args();{'prepare':prepare,'train':lambda:train(a.variant),'infer':infer,'calibrate':calibrate,'evaluate':evaluate,'union_calibrate':lambda:__import__('v27_union_trial').calibrate(),'union_evaluate':lambda:__import__('v27_union_trial').evaluate(),'finalize':lambda:__import__('v27_finalize').main()}[a.mode]()
