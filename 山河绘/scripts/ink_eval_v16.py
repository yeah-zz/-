# -*- coding: utf-8 -*-
"""Calibration and separate broad-ribbon/faint-line evaluation, including failures."""
from ink_center_v16 import *
from rock_pairs_v15 import load as load_links,connect,probability,frozen_paths
from rock_eval_v15 import quality_audit
from evaluate_clean_strokes_v12 import path_metrics,line_view
from gongbi_training.stroke_train_v08 import model_new as old_model,infer as old_infer

def load(stage):
    model=RibbonNet().cuda();model.load_state_dict(torch.load(ROOT/f'stage_{stage:02d}'/'final.pt',map_location='cuda',weights_only=False)['model']);return model.eval()

def raster(paths,shape):
    out=np.zeros(shape,np.uint8)
    for p in paths:cv2.polylines(out,[np.rint(p).astype(np.int32)],False,1,1)
    return out

def normal_diagnostics(paths,sk,rgb=None,properties=None,valid=None):
    dilated=cv2.dilate(sk.astype(np.uint8),np.ones((3,3),np.uint8));alpha=photometry(rgb)[1] if rgb is not None else None;double=[];bold_hit=[];faint_hit=[];point_dist=distance_transform_edt(~sk.astype(bool)) if sk.any() else np.full(sk.shape,1e5)
    for index,path in enumerate(paths):
        q=dense(path);step=12
        for k in range(12,len(q)-12,step):
            if valid is not None:
                cx,cy=np.rint(q[k]).astype(int)
                if not (0<=cx<sk.shape[1] and 0<=cy<sk.shape[0]) or not valid[cy,cx]:continue
            delta=q[min(len(q)-1,k+4)]-q[max(0,k-4)];u=delta/max(1e-6,np.linalg.norm(delta));n=np.array([-u[1],u[0]]);center=q[k];radius=18;offset=np.arange(-radius,radius+1);sample=center[None]+offset[:,None]*n;xs=sample[:,0].astype(np.float32)[None];ys=sample[:,1].astype(np.float32)[None];ink=cv2.remap(alpha,xs,ys,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0] if alpha is not None else None
            if properties is not None:
                width=properties[index]['width'];contrast=properties[index]['alpha'];span=max(3,int(min(radius,width*.65+2)));start=radius-span;end=radius+span+1
            else:
                near=ink[radius-4:radius+5];peak=int(np.argmax(near))+radius-4;contrast=float(ink[peak]);start=peak;end=peak+1
                if contrast<.02:continue
                while start>0 and ink[start-1]>.3*contrast:start-=1
                while end<len(ink) and ink[end]>.3*contrast:end+=1
                width=end-start;start=max(0,start-2);end=min(len(offset),end+2)
            xy=np.rint(center).astype(int);xy[0]=np.clip(xy[0],0,sk.shape[1]-1);xy[1]=np.clip(xy[1],0,sk.shape[0]-1);hit=bool(point_dist[xy[1],xy[0]]<=3)
            if width>=7 and contrast>=.10:
                vals=cv2.remap(dilated,xs,ys,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)[0][start:end]>0;runs=int(np.sum(vals&~np.r_[False,vals[:-1]]));double.append(runs>=2);bold_hit.append(hit)
            if .02<=contrast<=.18:faint_hit.append(hit)
    return dict(bold_samples=len(double),bold_multiple_center_fraction=float(np.mean(double)) if double else None,bold_reference_coverage=float(np.mean(bold_hit)) if bold_hit else None,faint_samples=len(faint_hit),faint_reference_coverage=float(np.mean(faint_hit)) if faint_hit else None,real_width_and_contrast_are_automatic_proxies=rgb is not None)

def complete_metrics(paths,sk,target,meta):
    truth=target[0]>.5;d=distance_transform_edt(~sk) if sk.any() else np.full(sk.shape,1e5);dt=distance_transform_edt(~truth);rec=float((d[truth]<=3).mean());prec=float((dt[sk]<=3).mean()) if sk.any() else 0.;return dict(recall=rec,precision=prec,f1=2*rec*prec/max(1e-6,rec+prec),**normal_diagnostics(meta['paths'],sk,properties=meta['stroke_properties']))

def calibrate(stage):
    torch.set_num_threads(4);cv2.setNumThreads(2);model=load(stage);options=[('ribbon',.45,.22),('ribbon',.6,.30),('ribbon',.7,.4),('center',.55,.25),('center',.7,.35)];pool=[]
    if stage>=3:options += [('joint_0.55',.7,.35),('joint_0.7',.7,.35),('joint_0.85',.7,.35)]
    for i in range(20):
        rgb,target,mask,meta=generated(162000+i);pool.append((predict(model,rgb),target,meta))
    r=next(r for r in reference_records() if r['id']=='R02');rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=read(r['labels']);_,_,mask,_=fields(doc);cal_real=predict(model,rgb);rows=[]
    for kind,high,low in options:
        values=[]
        for pred,target,meta in pool:
            paths,sk,_=decode(pred,high,low,kind);values.append(complete_metrics(paths,sk,target,meta))
        paths,sk,_=decode(cal_real,high,low,kind);partial=path_metrics([s['points'] for s in doc['strokes']],paths,mask[3]);background_density=float(sk[mask[1]>0].mean());mean=lambda key:float(np.mean([v[key] for v in values if v[key] is not None]));score=.6*mean('f1')+.2*mean('faint_reference_coverage')+.2*partial['mean_best_single_path_coverage']-.15*mean('bold_multiple_center_fraction')-(min(1.,background_density*50) if stage>=3 else 0);rows.append(dict(kind=kind,high=high,low=low,selection_score=score,complete_synthetic_f1=mean('f1'),synthetic_faint_coverage=mean('faint_reference_coverage'),synthetic_bold_multiple_center_fraction=mean('bold_multiple_center_fraction'),R02_partial_single_path_coverage=partial['mean_best_single_path_coverage'],R02_explicit_background_line_density=background_density))
    selected=max(rows,key=lambda r:r['selection_score']);jsave(ROOT/f'stage_{stage:02d}'/'calibration.json',dict(selected=selected,candidates=rows,selection_data=['synthetic_162000_162019','R02_partial'],heldout_test_not_used_for_selection=True));print('V16_CALIBRATED',stage,json.dumps(selected),flush=True)

def evaluate(stage,fusion=False):
    torch.set_num_threads(4);cv2.setNumThreads(2);model=load(stage);run_root=ROOT/f'stage_{stage:02d}'/('observed_fusion' if fusion else '');settings=read(run_root/'calibration.json')['selected'];link,threshold=load_links(4);folder=run_root/'evaluation';folder.mkdir(exist_ok=True);old=old_model().cuda();old.load_state_dict(torch.load(PARENT,map_location='cuda',weights_only=False)['model']);old.eval();controlled=[];example=[]
    def selected_decode(pred,rgb):
        if fusion:
            from ribbon_fusion_v16 import fusion_decode
            return fusion_decode(pred,rgb,**settings['config'])
        return decode(pred,settings['high'],settings['low'],settings['kind'])
    for i in range(40):
        rgb,target,mask,meta=generated(163000+i);base=old_infer(old,rgb);before=frozen_paths(base);before,_,_=connect(before,base[0],rgb,link,threshold);before=[p for p in before if np.linalg.norm(np.diff(p,axis=0),axis=1).sum()>=12];pred=predict(model,rgb);paths,sk,_=selected_decode(pred,rgb);row=dict(seed=meta['seed'],before=complete_metrics(before,raster(before,rgb.shape[:2]).astype(bool),target,meta),after=complete_metrics(paths,sk,target,meta),programmed_truth_not_real_artwork=True)
        if fusion:
            from ribbon_fusion_v16 import fusion_decode
            config=dict(settings['config']);config['neural']=False;aa,ass,_=fusion_decode(pred,rgb,**config);row['analytic_no_neural']=complete_metrics(aa,ass,target,meta)
        controlled.append(row)
        if i in [0,1,2,3]:example.append((rgb,before,paths,meta))
    jsave(folder/'controlled_test.json',dict(records=controlled,independent_seeds=True,not_independent_artwork_validation=True))
    examples=Image.new('RGB',(1500,350*len(example)+40),'#f5f3ec');d=ImageDraw.Draw(examples);d.text((8,4),'可控诊断：程序生成图的真实模型输出；不是山水画准确率',font=FONT,fill='black')
    for i,(rgb,before,after,meta) in enumerate(example):
        for j,(title,a) in enumerate([('程序墨线输入',rgb),('v15 模型输出',line_view(rgb.shape[:2],before,True)),('本阶段模型输出',line_view(rgb.shape[:2],after,True))]):d.text((j*500+5,40+i*350),str(meta['seed'])+' '+title,font=SMALL,fill='black');paste(examples,a,j*500+8,70+i*350,484,308)
    examples.save(folder/'controlled_comparison.jpg',quality=95)
    refs=reference_records();oldrecords={r['id']:r for r in read(LAB/'clean_strokes_v12_20261024'/'manifest.json')['records']}
    for k in ['T01','T02','U_A06']:
        r=oldrecords[k];box=r.get('source_box',[0,0,10000,10000]);left,top=r.get('padding',[0,0]);refs.append(dict(id=k,image=r['image'],labels=r['labels'],split='reused_development',native=[left,top,box[2]-box[0],box[3]-box[1]]))
    new=read(V15/'new_labels'/'manifest.json')['records'];content={r['id']:r for r in read(V15/'curated_manifest.json')['records']};refs+=new;rows=[]
    for r in refs:
        full=np.asarray(Image.open(r['image']).convert('RGB'));doc=read(r['labels']);h,w=full.shape[:2];left,top,cw,ch=r.get('native',[0,0,w,h]);cw=min(cw,w-left);ch=min(ch,h-top)
        if r['id'].startswith('N'):
            cr=content.get(r['source_id']);rgb=np.asarray(Image.open(cr['content_image']).convert('RGB')) if cr else full;ch,cw=rgb.shape[:2];before=read(V15/'round_04'/'new_source_evaluation'/r['id']/'trajectory.json')['paths']
        else:rgb=full[top:top+ch,left:left+cw];before=read(V15/'round_04'/'evaluation'/r['id']/'trajectory.json')['paths']
        route=monochrome_route(rgb);pred=None
        if route:
            pred=predict(model,rgb);paths,sk,info=selected_decode(pred,rgb);fake=np.zeros((6,ch,cw),np.float32);fake[0]=np.maximum(pred[0],pred[1]*.8);fake[1:3]=pred[2:4];paths,events,uncertain=connect(paths,fake[0],rgb,link,threshold,max_gap=8);paths=[p+np.array([left,top]) for p in paths]
        else:paths=[np.asarray(p) for p in before];events=[];info=dict(preserved_original_color_branch=True)
        _,_,m,_=fields(doc);valid=np.zeros((h,w),np.float32);valid[top:top+ch,left:left+cw]=1;m*=valid[None]
        if r['id']=='U_A06':m[:,:86,:96]=0
        reference=[s['points'] for s in doc['strokes']];before_sk=raster(before,(h,w)).astype(bool);after_sk=raster(paths,(h,w)).astype(bool);row=dict(id=r['id'],split=r['split'],monochrome_specialist_used=route,before=path_metrics(reference,before,m[3]),after=path_metrics(reference,paths,m[3]),before_diagnostics=normal_diagnostics(reference,before_sk,full,valid=m[3]),after_diagnostics=normal_diagnostics(reference,after_sk,full,valid=m[3]),expert_gold=not r['id'].startswith(('N','T')),reference_is_partial=True,independent_artwork_test=False);rows.append(row);out=folder/r['id'];out.mkdir(exist_ok=True);jsave(out/'metrics.json',row);jsave(out/'trajectory.json',dict(paths=[p.round(2).tolist() for p in paths],ready_for_robot=False,actually_drawn=False,monochrome_specialist_used=route,trace_info=info));
        if pred is not None:np.savez_compressed(out/'prediction.npz',prediction=pred)
        board=Image.new('RGB',(1600,430),'#f5f3ec');d=ImageDraw.Draw(board)
        for j,(title,a) in enumerate([('原图',full),('局部参考（非预测）',line_view((h,w),reference)),('v15 候选轨迹',line_view((h,w),before,True)),('本阶段候选轨迹',line_view((h,w),paths,True))]):d.text((j*400+6,5),r['id']+' '+title,font=SMALL,fill='black');paste(board,a,j*400+8,38,384,383)
        board.save(out/'comparison.jpg',quality=95);print('V16_REAL',stage,r['id'],route,round(row['before']['mean_best_single_path_coverage'],3),round(row['after']['mean_best_single_path_coverage'],3),flush=True)
    jsave(folder/'real_summary.json',dict(records=rows,automatic_bold_and_faint_partition_not_expert_tags=True,source_holdout_families_not_completely_independent=True))
    for name,ids in [('reviewed_rocks',['R01','R02','R03','R04']),('new_rocks',['N01','N10','N11']),('transfer',['T01','T02','U_A06'])]:
        board=Image.new('RGB',(1600,430*len(ids)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,4),'v16 stage '+str(stage)+' '+name+'：参考非预测，颜色分笔',font=FONT,fill='black')
        for i,k in enumerate(ids):board.paste(Image.open(folder/k/'comparison.jpg'),(0,35+430*i))
        board.save(folder/(name+'.jpg'),quality=95)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibrate','evaluate','evaluate_fusion']);p.add_argument('stage',type=int);args=p.parse_args();calibrate(args.stage) if args.mode=='calibrate' else evaluate(args.stage,fusion=args.mode=='evaluate_fusion')
