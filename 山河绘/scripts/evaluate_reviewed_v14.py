# -*- coding: utf-8 -*-
"""User-reviewed small fit: direct inference, tiling and oracle separated."""
import sys,json
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from skimage.morphology import skeletonize
from reviewed_rocks_v14 import ROOT,PARENT,model_new,direct,infer_tiled,FONT,SMALL,paste,jsave,fields
from topology_trace_v13 import graph_paths
from evaluate_clean_strokes_v12 import path_metrics,line_view

def extract(pred,pair=True):
    p=pred[0];smooth=cv2.GaussianBlur(p.astype(np.float32),(0,0),.65);h,w=p.shape;yy,xx=np.mgrid[:h,:w].astype(np.float32);ridge=np.zeros((h,w),bool)
    for start in (1,3):
        theta=.5*np.arctan2(pred[start+1],pred[start]);nx=-np.sin(theta);ny=np.cos(theta);a=cv2.remap(smooth,xx+nx*1.5,yy+ny*1.5,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);b=cv2.remap(smooth,xx-nx*1.5,yy-ny*1.5,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);selected=(smooth>=a)&(smooth>=b)&(smooth-(a+b)*.5>.001)
        if start==3:selected&=pred[7]>.5
        ridge|=selected
    selected=((p>=.3)&ridge)|((pred[6]>.55)&(p>.3));n,labels,stats,_=cv2.connectedComponentsWithStats(selected.astype(np.uint8),8);good=np.zeros(n,bool);good[np.unique(labels[p>=.5])]=True;good[0]=False;good[stats[:,cv2.CC_STAT_AREA]<12]=False;sk=skeletonize(good[labels]);sk[[0,-1],:]=False;sk[:,[0,-1]]=False;paths,info=graph_paths(sk,pred,pair);return paths,info,sk

def evaluate(mode):
    torch.set_num_threads(4);cv2.setNumThreads(2);folder=ROOT/mode/'evaluation';folder.mkdir(parents=True,exist_ok=True);records=json.loads((ROOT/'manifest.json').read_text())['records'];model=model_new().cuda();model.load_state_dict(torch.load(ROOT/mode/'final.pt',map_location='cuda',weights_only=False)['model']);model.eval();rows=[];board=Image.new('RGB',(2000,470*4+45),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),('前三张训练拟合，R04 未加入此阶段优化' if mode=='small' else '四张复核图上的训练拟合，非独立画作准确率')+'；参考图与预测分栏',font=FONT,fill='black')
    for i,r in enumerate(records):
        out=folder/r['id'];out.mkdir(exist_ok=True);rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=json.loads(Path(r['labels']).read_text(encoding='utf-8'));_,_,mask,_=fields(doc);left,top,cw,ch=r['native_content'];valid=np.zeros(rgb.shape[:2],np.uint8);valid[top:top+ch,left:left+cw]=1;mask*=valid[None];reference=[s['points'] for s in doc['strokes']];p=direct(model,rgb);tiled=infer_tiled(model,rgb);truth=np.load(r['target'])['target']
        def get(pred,pair=True):
            paths,info,_=extract(pred[:,top:top+ch,left:left+cw],pair);return [p+np.array([left,top]) for p in paths],info
        before=[np.asarray(p,float) for p in json.loads((LAB/'clean_strokes_v12_20261024'/'evaluation'/r['source_id']/'trajectory.json').read_text())['paths']];raw,raw_info=get(p,False);after,info=get(p,True);tile_paths,tile_info=get(tiled,True);oracle,oracle_info=get(truth,True);score=lambda paths:path_metrics(reference,paths,mask[3]*valid)
        result=dict(id=r['id'],stage_role='training_fit' if mode!='small' or r['id']!='R04' else 'same_motif_reused_development',before=score(before),direct_unpaired=score(raw),direct_paired=score(after),tiled_paired=score(tile_paths),known_reference_oracle=score(oracle),independent_test=False,background_high_response_before=None,background_high_response_direct=float(np.mean(p[0][mask[1]>0]>=.5)) if mask[1].any() else None)
        rows.append(result);jsave(out/'metrics.json',result);np.savez_compressed(out/'probabilities.npz',direct=p,tiled=tiled);jsave(out/'trajectory.json',dict(paths=[p.round(2).tolist() for p in after],ready_for_robot=False,physical_z=None,actually_drawn=False,junction_pairings=info['junction_pairings']))
        ref_image=line_view(rgb.shape[:2],reference);before_image=line_view(rgb.shape[:2],before);after_image=line_view(rgb.shape[:2],after,True);oracle_image=line_view(rgb.shape[:2],oracle,True)
        for name,a in [('original',rgb),('user_reference',ref_image),('before',before_image),('after',after_image),('reference_input_diagnostic',oracle_image)]:Image.fromarray(a).save(out/(name+'.png'))
        one=Image.new('RGB',(2000,470),'#f5f3ec');od=ImageDraw.Draw(one)
        for j,(label,a) in enumerate([('原图',rgb),('你的复核参考（非预测）',ref_image),('旧版候选轨迹',before_image),('本阶段原尺寸预测',after_image),('正确参考输入追踪诊断',oracle_image)]):od.text((j*400+8,5),r['id']+' '+label,font=SMALL,fill='black');paste(one,a,j*400+8,45,384,410)
        one.save(out/'comparison.jpg',quality=95);board.paste(one,(0,45+i*470));print('REVIEWED_CHECK',mode,r['id'],round(result['before']['mean_best_single_path_coverage'],3),round(result['direct_paired']['mean_best_single_path_coverage'],3),'tiled',round(result['tiled_paired']['mean_best_single_path_coverage'],3),'oracle',round(result['known_reference_oracle']['mean_best_single_path_coverage'],3),flush=True)
    board.save(folder/'comparison.jpg',quality=95);jsave(folder/'metrics.json',dict(records=rows,all_references_user_reviewed=True,unmarked_unknown=True,precision_not_available_outside_explicit_background=True,same_motif_training_and_development=True))
    fit=[r for r in rows if mode!='small' or r['id']!='R04'];loc=float(np.mean([r['direct_paired']['point_coverage_3px'] for r in fit]));single=float(np.mean([r['direct_paired']['mean_best_single_path_coverage'] for r in fit]));jsave(folder/'summary.json',dict(mean_reference_point_coverage=loc,mean_best_single_path_coverage=single,localization_gate_passed=loc>=.9,stroke_completeness_gate_passed=single>=.9,independent_test=False,do_not_treat_fit_as_generalization=True))
    return rows
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['small','main','replay']);a=p.parse_args();evaluate(a.mode)
