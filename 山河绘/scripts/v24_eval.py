# -*- coding: utf-8 -*-
"""Frozen calibration and work-group development checks for v24 joint training."""
from v24_train import ROOT,V21,JointNet,load_model,predict
import v21_stroke_learning as v21
from v23_junction_relations import assemble,original_pairings,endpoint_segment,arc
from topology_trace_v13 import graph_paths
from stroke_link_v12 import tangent
from rock_v15_common import read,jsave,FONT,SMALL
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from collections import defaultdict
from PIL import Image,ImageDraw
import numpy as np,cv2,torch,math,itertools

def cases(split):
    result=[]
    if split=='calibration':
        for r in read(V21/'manifest.json')['records']:
            if r['split']!='calibration':continue
            z=np.load(r['data']);rgb=z['rgb'];refs=[p['points'] for p in read(r['labels'])['strokes']];valid=np.ones(rgb.shape[:2],np.float32);bg=(z['mask'][0]>0)&(z['target'][0]<.01)&(z['target'][1]<.1);result.append((r,rgb,refs,valid,bg))
    else:
        for r,rgb,refs,valid,bg,_ in v21.evaluation_cases():
            if r['id'] in ['stage_018','stage_029']:r=dict(r,split='legacy_related_probe')
            result.append((r,rgb,refs,valid,bg))
    for r in read(ROOT/'training_manifest.json')['records']:
        if r['kind']!='new_visible_reference':continue
        if (split=='calibration' and r['split']!='calibration') or (split!='calibration' and r['split']!='new_work_development'):continue
        z=np.load(r['data']);rgb=z['rgb'];refs=[p['points'] for p in read(r['labels'])['strokes']];valid=z['valid']*(z['mask'][0]>0);result.append((r,rgb,refs,valid,z['background']>0))
    return result

def infer():
    torch.set_num_threads(4);cv2.setNumThreads(2);allcases=cases('calibration')+cases('development')
    for variant in ['parent','control','joint']:
        folder=ROOT/'predictions'/variant;folder.mkdir(parents=True,exist_ok=True);model=v21.load_model('ribbon_improved') if variant=='parent' else load_model(variant)
        for r,rgb,refs,valid,bg in allcases:
            dest=folder/(r['id']+'.npz')
            if dest.exists():continue
            prior=V21/'ribbon_improved'/'evaluation'/r['id']/'prediction.npz'
            p=np.load(prior)['prediction'] if variant=='parent' and prior.exists() else v21.v16.predict(model,rgb) if variant=='parent' else predict(model,rgb)
            np.savez_compressed(dest,prediction=p);print('V24_INFER',variant,r['id'],p.shape,flush=True)
        del model;torch.cuda.empty_cache()

def skeleton_with_nodes(pred,cut):
    if cut is None:return v21.prior_eval.decode(pred[:5],'center',.65,.30)[1]
    center,ink=pred[:2];h,w=center.shape;theta=.5*np.arctan2(pred[3],pred[2]);yy,xx=np.mgrid[:h,:w].astype(np.float32);nx=-np.sin(theta);ny=np.cos(theta);g=cv2.GaussianBlur(center,(0,0),.65);a=cv2.remap(g,xx+nx*1.5,yy+ny*1.5,cv2.INTER_LINEAR);b=cv2.remap(g,xx-nx*1.5,yy-ny*1.5,cv2.INTER_LINEAR)
    node=cv2.dilate((pred[5]>cut).astype(np.uint8),np.ones((5,5),np.uint8))>0;candidate=((center>.30)&(g>=a)&(g>=b)&(ink>.5))|(node&(center>.65)&(ink>.5))
    n,labels,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8);good=np.zeros(n,bool);good[np.unique(labels[center>.65])]=True;good[stats[:,cv2.CC_STAT_AREA]<12]=False;good[0]=False;binary=good[labels];sk=skeletonize(binary);sk[:3]=False;sk[-3:]=False;sk[:,:3]=False;sk[:,-3:]=False;sk[distance_transform_edt(binary)>24]=False
    pad=np.pad(sk,1);degree=cv2.filter2D(sk.astype(np.float32),-1,np.ones((3,3),np.float32))-sk
    for dy in [-1,1]:
        for dx in [-1,1]:degree-=(pad[1+dy:1+dy+h,1+dx:1+dx+w]&(pad[1:1+h,1+dx:1+dx+w]|pad[1+dy:1+dy+h,1:1+w]))
    expanded=cv2.dilate(((degree>2)&sk).astype(np.uint8),np.ones((5,5),np.uint8))*sk.astype(np.uint8);n,cc,stats,_=cv2.connectedComponentsWithStats(expanded,8);large=np.zeros(n,bool);large[stats[:,cv2.CC_STAT_AREA]>48]=True;large[0]=False;sk[large[cc]]=False;return sk

def decode(pred,cfg):
    if cfg['node'] is None and cfg['affinity'] is None and cfg.get('veto') is None:return v21.prior_eval.decode(pred[:5],'center',.65,.30)
    sk=skeleton_with_nodes(pred,cfg['node']);h,w=sk.shape;field=np.zeros((8,h,w),np.float32);field[0]=np.maximum(pred[0],pred[1]*.8);field[1:3]=pred[2:4];field[3:5]=pred[2:4]
    if cfg['affinity'] is None and cfg.get('veto') is None:return graph_paths(sk,field,True,min_length=8)[0],sk,dict(node_threshold=cfg['node'],affinity_joins=0)
    chains,_=graph_paths(sk,field,False,min_length=0);nodes=defaultdict(list)
    for i,p in enumerate(chains):
        nodes[tuple(np.round(p[0],4))].append((i,0));nodes[tuple(np.round(p[-1],4))].append((i,1))
    candidates=[];descriptors={}
    def descriptor(e):
        if e in descriptors:return descriptors[e]
        _,q=endpoint_segment(chains[e[0]],e[1])
        if len(q)<4:descriptors[e]=None;return None
        q=q[:12];samples=[cv2.remap(im,q[:,0][None],q[:,1][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0] for im in pred[6:14]];emb=np.asarray(samples).mean(1);emb/=max(float(np.linalg.norm(emb)),1e-6);descriptors[e]=emb;return emb
    for xy,ends in nodes.items():
        for a,b in itertools.combinations(ends,2):
            if a[0]==b[0]:continue
            ea=descriptor(a);eb=descriptor(b);sim=float(ea@eb) if ea is not None and eb is not None else -2
            candidates.append(dict(a=a,b=b,node=list(xy),similarity=sim))
    g=dict(chains=chains,candidates=candidates,pred=pred,shape=sk.shape);base=original_pairings(g);accepted=[i for i in base if cfg.get('veto') is None or candidates[i]['similarity']==-2 or candidates[i]['similarity']>=cfg['veto']];used={e for i in accepted for e in [candidates[i]['a'],candidates[i]['b']]};eligible=[]
    for i,c in enumerate(candidates):
        a,b=c['a'],c['b']
        if cfg['affinity'] is None or a in used or b in used or c['similarity']<cfg['affinity']:continue
        if min(arc(chains[a[0]])[-1],arc(chains[b[0]])[-1])<8:continue
        if float(tangent(chains[a[0]],a[1]==0)@(-tangent(chains[b[0]],b[1]==0)))<-.2:continue
        eligible.append(i)
    for i in sorted(eligible,key=lambda i:-candidates[i]['similarity']):
        c=candidates[i];a,b=c['a'],c['b']
        if a in used or b in used:continue
        alternatives=[j for j in eligible if j!=i and ({candidates[j]['a'],candidates[j]['b']}&{a,b})]
        if alternatives and c['similarity']-max(candidates[j]['similarity'] for j in alternatives)<.1:continue
        edges=defaultdict(set)
        for j in accepted:
            u,v=candidates[j]['a'][0],candidates[j]['b'][0];edges[u].add(v);edges[v].add(u)
        todo=[a[0]];seen=set(todo)
        while todo:
            for k in edges[todo.pop()]:
                if k not in seen:seen.add(k);todo.append(k)
        if b[0] in seen:continue
        accepted.append(i);used.update([a,b])
    return assemble(g,accepted),sk,dict(node_threshold=cfg['node'],affinity_threshold=cfg['affinity'],negative_veto=cfg.get('veto'),affinity_joins=len(set(accepted)-set(base)),affinity_removed=len(set(base)-set(accepted)),reference_used_at_inference=False)

def calibrate():
    OUT=ROOT/'joint';options=[];cc=cases('calibration')
    configs=[dict(node=node,affinity=affinity,veto=None) for node in [None,.55,.75] for affinity in [None,.75,.90]]+[dict(node=None,affinity=affinity,veto=0.) for affinity in [None,.75,.90]]
    for cfg in configs:
            rows=[]
            for r,rgb,refs,valid,bg in cc:
                pred=np.load(ROOT/'predictions'/'joint'/(r['id']+'.npz'))['prediction'];paths,sk,info=decode(pred,cfg);m=v21.path_metrics(refs,paths,valid);m['background_density']=float(sk[bg].mean()) if np.any(bg) else 0.;rows.append(dict(id=r['id'],**m))
            score=float(np.mean([.5*r['point_coverage_3px']+.5*r['mean_best_single_path_coverage']-.025*r['mixed_reference_paths_flagged']-min(.1,10*r['background_density']) for r in rows]));options.append(dict(config=cfg,score=score,single=float(np.mean([r['mean_best_single_path_coverage'] for r in rows])),mixed=sum(r['mixed_reference_paths_flagged'] for r in rows),background=float(np.mean([r['background_density'] for r in rows])),records=rows))
    baseline=options[0];allowed=[o for o in options if o['score']>=baseline['score']+.005 and o['single']>=baseline['single']+.005 and o['mixed']<=baseline['mixed'] and o['background']<=baseline['background']*1.1+.0002];selected=max(allowed,key=lambda o:o['score']) if allowed else baseline;trial=max(options,key=lambda o:o['score']);jsave(OUT/'calibration.json',dict(selected=selected,trial=trial,auxiliary_decoding_enabled=bool(allowed),candidates=options,heldout_works_not_used_for_selection=True));print('V24_CALIBRATION',bool(allowed),selected['config'],baseline['score'],selected['score'],flush=True)

def evaluate():
    cv2.setNumThreads(2);decision=read(ROOT/'joint'/'calibration.json');rows=[];images={};folder=ROOT/'evaluation';folder.mkdir(exist_ok=True)
    for r,rgb,refs,valid,bg in cases('development'):
        key=r['id'];row=dict(id=key,split=r['split'],metrics={},auxiliary_decoding_enabled=decision['auxiliary_decoding_enabled'],partial_assistant_reference=not key.startswith('R'))
        views={}
        for variant in ['parent','control','joint']:
            pred=np.load(ROOT/'predictions'/variant/(key+'.npz'))['prediction'];cfg=decision['selected']['config'] if variant=='joint' else dict(node=None,affinity=None);paths,sk,info=decode(pred,cfg);met=v21.diagnostic(refs,pred[:5],paths,valid,rgb,bg);row['metrics'][variant]=met;views[variant]=paths;out=folder/variant/key;out.mkdir(parents=True,exist_ok=True);jsave(out/'trajectory.json',dict(paths=[p.round(3).tolist() for p in paths],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,variant=variant,decoder_config=cfg,trace_info=info,reference_labels_used_at_inference=False))
        rows.append(row);images[key]=(rgb,views);print('V24_EVAL',key,{v:round(m['mean_best_single_path_coverage'],3) for v,m in row['metrics'].items()},flush=True)
    groups={}
    for group in sorted(set(r['split'] for r in rows)):
        rr=[r for r in rows if r['split']==group];groups[group]={v:{k:float(np.mean([r['metrics'][v][k] for r in rr])) for k in ['point_coverage_3px','mean_best_single_path_coverage','mean_reference_fragments','path_count','mixed_reference_paths_flagged','ink_ref_recall_3px','center_ref_recall_3px']} for v in ['parent','control','joint']}
    jsave(ROOT/'evaluation.json',dict(records=rows,groups=groups,calibration=decision['selected']['config'],fixed_old_sets_are_development_not_blind=True,new_work_references_are_partial_assistant_labels=True))
    keys=['H01','H02','stage_065','sina_6586334590_31','X04','R04'];board=Image.new('RGB',(1600,335*len(keys)+45),'#f5f3ec');d=ImageDraw.Draw(board);d.text((5,5),'v24：原图 / v21 研究基线 / 匹配新数据对照组 / 联合中心线与关系训练（全为离线预测）',font=FONT,fill='black')
    for i,key in enumerate(keys):
        rgb,views=images[key]
        for j,(title,a) in enumerate([('原图',rgb),('原研究基线',v21.line_view(rgb.shape[:2],views['parent'],True)),('新数据对照组',v21.line_view(rgb.shape[:2],views['control'],True)),('联合训练候选',v21.line_view(rgb.shape[:2],views['joint'],True))]):
            d.text((j*400+5,45+i*335),key+' '+title,font=SMALL,fill='black');v21.paste(board,a,j*400+5,72+i*335,390,295)
    board.save(ROOT/'RESULT_COMPARISON.jpg',quality=95);print('V24_GROUPS',groups,flush=True)

def probe_auxiliary():
    # This option was frozen by calibration before development evaluation. It
    # failed the declared gain gate, so these outputs remain explicitly trials.
    decision=read(ROOT/'joint'/'calibration.json');cfg=decision['trial']['config'];rows=[];images={};out=ROOT/'auxiliary_trial';out.mkdir(exist_ok=True)
    for r,rgb,refs,valid,bg in cases('development'):
        key=r['id'];pred=np.load(ROOT/'predictions'/'joint'/(key+'.npz'))['prediction'];paths,sk,info=decode(pred,cfg);met=v21.diagnostic(refs,pred[:5],paths,valid,rgb,bg);rows.append(dict(id=key,split=r['split'],metrics=met,info=info));folder=out/key;folder.mkdir(exist_ok=True);jsave(folder/'trial_trajectory.json',dict(paths=[p.round(3).tolist() for p in paths],decoder_config=cfg,ready_for_robot=False,physical_z=None,actually_drawn=False,calibration_gate_passed=False));images[key]=(rgb,paths);print('V24_AUX_TRIAL',key,round(met['mean_best_single_path_coverage'],3),flush=True)
    groups={g:{k:float(np.mean([r['metrics'][k] for r in rows if r['split']==g])) for k in ['point_coverage_3px','mean_best_single_path_coverage','mean_reference_fragments','path_count','mixed_reference_paths_flagged']} for g in sorted(set(r['split'] for r in rows))}
    jsave(out/'evaluation.json',dict(config=cfg,records=rows,groups=groups,not_enabled=True,calibration_choice_was_frozen_before_development=True))
    keys=['H01','H02','stage_065','sina_6586334590_31','X04','R04'];board=Image.new('RGB',(1500,325*len(keys)+42),'#f5f3ec');d=ImageDraw.Draw(board);d.text((5,5),'v24 辅助解码试验：原图 / 联合网络当前解码 / 未通过校准门槛的节点与关系候选',font=FONT,fill='black')
    for i,key in enumerate(keys):
        rgb,paths=images[key];actual=read(ROOT/'evaluation'/'joint'/key/'trajectory.json')['paths']
        for j,(title,a) in enumerate([('原图',rgb),('联合网络当前解码',v21.line_view(rgb.shape[:2],actual,True)),('未启用的辅助候选',v21.line_view(rgb.shape[:2],paths,True))]):
            d.text((j*500+5,42+325*i),key+' '+title,font=SMALL,fill='black');v21.paste(board,a,j*500+5,68+325*i,490,285)
    board.save(out/'RESULT_COMPARISON.jpg',quality=95);print('V24_AUX_GROUPS',groups,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['infer','calibrate','evaluate','probe']);a=p.parse_args();{'infer':infer,'calibrate':calibrate,'evaluate':evaluate,'probe':probe_auxiliary}[a.mode]()
