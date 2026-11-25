# -*- coding: utf-8 -*-
"""Same decoder candidates for control/aligned, natural-image and source-role reporting."""
from aligned_targets_v17 import *
import torch
from torch.nn import functional as F
from skimage.morphology import skeletonize
import ink_center_v16 as v16
from topology_trace_v13 import graph_paths
from evaluate_clean_strokes_v12 import path_metrics,line_view
from ink_eval_v16 import normal_diagnostics

def load(variant):
    m=v16.RibbonNet().cuda();m.load_state_dict(torch.load(ROOT/variant/'final.pt',map_location='cuda',weights_only=False)['model']);return m.eval()

def decode(pred,kind='ribbon',high=.7,low=.4):
    center,ink=pred[:2];h,w=ink.shape
    if kind=='ribbon':candidate=ink>low;strong=(ink>high)&(center>.25)
    else:
        theta=.5*np.arctan2(pred[3],pred[2]);yy,xx=np.mgrid[:h,:w].astype(np.float32);nx=-np.sin(theta);ny=np.cos(theta);g=cv2.GaussianBlur(center,(0,0),.65);a=cv2.remap(g,xx+nx*1.5,yy+ny*1.5,cv2.INTER_LINEAR);b=cv2.remap(g,xx-nx*1.5,yy-ny*1.5,cv2.INTER_LINEAR);candidate=(center>low)&(g>=a)&(g>=b)&(ink>.5);strong=center>high
    n,labels,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8);good=np.zeros(n,bool);good[np.unique(labels[strong])]=True;good[stats[:,cv2.CC_STAT_AREA]<12]=False;good[0]=False;binary=good[labels];sk=skeletonize(binary);sk[:3]=False;sk[-3:]=False;sk[:,:3]=False;sk[:,-3:]=False;broad=distance_transform_edt(binary)>24;sk[broad]=False
    # Ambiguous large junction clusters are not exhaustively paired.
    pad=np.pad(sk,1);degree=cv2.filter2D(sk.astype(np.float32),-1,np.ones((3,3),np.float32))-sk
    for dy in (-1,1):
        for dx in (-1,1):degree-=(pad[1+dy:1+dy+h,1+dx:1+dx+w]&(pad[1:1+h,1+dx:1+dx+w]|pad[1+dy:1+dy+h,1:1+w]))
    expanded=cv2.dilate(((degree>2)&sk).astype(np.uint8),np.ones((5,5),np.uint8))*sk.astype(np.uint8);nc,cc,ss,_=cv2.connectedComponentsWithStats(expanded,8);large=np.zeros(nc,bool);large[ss[:,cv2.CC_STAT_AREA]>48]=True;large[0]=False;crowded=large[cc];sk[crowded]=False;field=np.zeros((8,h,w),np.float32);field[0]=np.maximum(center,ink*.8);field[1:3]=pred[2:4];field[3:5]=pred[2:4];paths,info=graph_paths(sk,field,True,min_length=8);return paths,sk,dict(**info,broad_area_quarantined_pixels=int(broad.sum()),crowded_junction_quarantined_pixels=int(crowded.sum()))

def calibrate(variant):
    torch.set_num_threads(4);cv2.setNumThreads(2);model=load(variant);cases=[]
    for r in records():
        if r['split']!='calibration':continue
        rgb,doc,bg,valid=source(r);pred=v16.predict(model,rgb);cases.append((rgb,doc,bg,valid,pred))
    choices=[]
    for kind,high,low in [('ribbon',.65,.35),('ribbon',.8,.5),('center',.65,.3),('center',.8,.45)]:
        result=[]
        for rgb,doc,bg,valid,pred in cases:
            paths,sk,_=decode(pred,kind,high,low);metric=path_metrics([s['points'] for s in doc['strokes']],paths,valid);diag=normal_diagnostics([s['points'] for s in doc['strokes']],sk,rgb,valid=valid);density=float(sk[bg>0].mean()) if bg.any() else 0.;result.append(dict(id=doc['sample_id'],metric=metric,background_line_density=density,double_center_proxy=diag['bold_multiple_center_fraction']))
        point=float(np.mean([r['metric']['point_coverage_3px'] for r in result]));single=float(np.mean([r['metric']['mean_best_single_path_coverage'] for r in result]));bg_density=float(np.mean([r['background_line_density'] for r in result]));double=float(np.mean([r['double_center_proxy'] for r in result if r['double_center_proxy'] is not None]));score=.45*point+.55*single-.12*double-min(.5,bg_density*30);choices.append(dict(kind=kind,high=high,low=low,score=score,mean_point_coverage=point,mean_single_path_coverage=single,mean_background_line_density=bg_density,mean_double_center_proxy=double,records=result))
    selected=max(choices,key=lambda r:r['score']);jsave(ROOT/variant/'calibration.json',dict(selected=selected,candidates=choices,calibration_ids=['N07','N08'],calibration_reused_source_family_not_blind=True));print('V17_CALIBRATED',variant,selected['kind'],selected['mean_single_path_coverage'],flush=True)

def evaluation_records():
    cases=records();old={r['id']:r for r in read(LAB/'clean_strokes_v12_20261024'/'manifest.json')['records']};human={r['id']:r for r in reference_records()}
    for k in ['A01','A02','A03','A04','A05']:cases.append(dict(human[k],split='unchanged_color_control'))
    for k in ['T01','T02','U_A06']:
        r=old[k];im=Image.open(r['image']);box=r.get('source_box',[0,0,im.width,im.height]);cases.append(dict(id=k,image=r['image'],labels=r['labels'],split='unchanged_color_control' if k=='U_A06' else 'reused_development',native=[*r.get('padding',[0,0]),box[2]-box[0],box[3]-box[1]],provenance=r.get('provenance','previous_partial')))
    return cases

def evaluate(variant):
    torch.set_num_threads(4);cv2.setNumThreads(2);model=load(variant);policy=read(ROOT/variant/'calibration.json')['selected'];folder=ROOT/variant/'evaluation';folder.mkdir(exist_ok=True);rows=[];curated={r['id']:r for r in read(V15/'curated_manifest.json')['records']}
    for r in evaluation_records():
        full=np.asarray(Image.open(r['image']).convert('RGB'));doc=read(r['labels']);h,w=full.shape[:2];left,top,cw,ch=r.get('native',[0,0,w,h]);cw=min(cw,w-left);ch=min(ch,h-top);rgb=full[top:top+ch,left:left+cw]
        if r['id'].startswith('N'):
            cr=curated.get(r['source_id']);rgb=np.asarray(Image.open(cr['content_image']).convert('RGB')) if cr else full;ch,cw=rgb.shape[:2]
        route=v16.monochrome_route(rgb);baseline=read(V16/'protected_refinement'/r['id']/'trajectory.json')['paths'];native=np.zeros((h,w),np.float32);native[top:top+ch,left:left+cw]=1;_,_,mask,_=fields(doc);mask*=native[None]
        if r['id']=='U_A06':mask[:,:86,:96]=0
        reference=[s['points'] for s in doc['strokes']];out=folder/r['id'];out.mkdir(exist_ok=True)
        if route:
            pred=v16.predict(model,rgb);paths,sk,info=decode(pred,policy['kind'],policy['high'],policy['low']);paths=[p+np.array([left,top]) for p in paths];np.savez_compressed(out/'prediction.npz',prediction=pred);Image.fromarray(np.uint8(255*(1-pred[1]))).save(out/'ink_probability.png')
        else:paths=[np.asarray(p) for p in baseline];info=dict(original_color_branch_preserved=True)
        def raster(paths):
            sk=np.zeros((h,w),np.uint8)
            for p in paths:cv2.polylines(sk,[np.rint(p).astype(np.int32)],False,1,1)
            return sk.astype(bool)
        oldsk=raster(baseline);newsk=raster(paths);bg=np.zeros((h,w),np.float32)
        if r['id'] in {a['id'] for a in records()}:_,_,bg,_=source(r);bg*=native
        else:bg=mask[1]
        row=dict(id=r['id'],split=r['split'],monochrome_specialist_used=route,before=path_metrics(reference,baseline,mask[3]),after=path_metrics(reference,paths,mask[3]),before_diagnostics=normal_diagnostics(reference,oldsk,full,valid=mask[3]),after_diagnostics=normal_diagnostics(reference,newsk,full,valid=mask[3]),before_background_density=float(oldsk[bg>0].mean()) if bg.any() else None,after_background_density=float(newsk[bg>0].mean()) if bg.any() else None,reference_is_partial=True,new_source_labels_not_expert=r['id'].startswith(('N','T')),independent_artwork_test=False);rows.append(row);jsave(out/'metrics.json',row);jsave(out/'trajectory.json',dict(paths=[p.round(2).tolist() for p in paths],ready_for_robot=False,actually_drawn=False,variant=variant,trace_info=info));Image.fromarray(line_view((h,w),paths,True)).save(out/'prediction.png');board=Image.new('RGB',(1600,430),'#f5f3ec');d=ImageDraw.Draw(board)
        for j,(title,a) in enumerate([('原图',full),('局部参考（非预测）',line_view((h,w),reference)),('v16 受限校正',line_view((h,w),baseline,True)),('v17 '+variant+' 模型输出',line_view((h,w),paths,True))]):d.text((j*400+7,5),r['id']+' '+title,font=SMALL,fill='black');paste(board,a,j*400+7,38,386,383)
        board.save(out/'comparison.jpg',quality=95);print('V17_REAL',variant,r['id'],r['split'],round(row['before']['mean_best_single_path_coverage'],3),round(row['after']['mean_best_single_path_coverage'],3),flush=True)
    jsave(folder/'summary.json',dict(records=rows,variant=variant,partial_metrics_not_accuracy=True,original_references_not_used_in_inference=True))
    for name,ids in [('reviewed_rocks',['R01','R02','R03','R04']),('new_rocks',['N01','N06','N10','N11']),('transfer',['T01','T02','U_A06'])]:
        board=Image.new('RGB',(1600,430*len(ids)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,4),'v17 '+variant+' '+name+'；R01/R02 已参与本轮训练',font=FONT,fill='black')
        for i,k in enumerate(ids):board.paste(Image.open(folder/k/'comparison.jpg'),(0,35+430*i))
        board.save(folder/(name+'.jpg'),quality=95)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibrate','evaluate']);p.add_argument('variant',choices=['control','aligned']);a=p.parse_args();calibrate(a.variant) if a.mode=='calibrate' else evaluate(a.variant)
