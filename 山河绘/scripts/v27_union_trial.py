# -*- coding: utf-8 -*-
"""Unique graph-edge ownership for overlapping continuous curves.

A separately frozen experiment following the clipping failure. Calibrate before
development evaluation; no pixel/reference geometry is read by this decoder.
"""
from rock_v15_common import read,jsave,FONT,SMALL
import v27_separation_faint as exp
import numpy as np,cv2
from PIL import Image,ImageDraw
from skimage.morphology import skeletonize
from topology_trace_v13 import graph_paths

ROOT=exp.ROOT/'union_trial'

def decode(paths,pred,config):
    union=exp.prev.raster(paths,pred.shape[1:]);sk=skeletonize(union>0);field=np.zeros((8,*sk.shape),np.float32);field[0]=np.maximum(pred[0],pred[1]*.8);field[1:3]=pred[2:4];field[3:5]=pred[2:4];out,info=graph_paths(sk,field,pair=config['pair'],min_length=config['min_length']);return out,dict(unique_graph_edge_ownership=True,union_pixels=int(union.sum()),skeleton_pixels=int(sk.sum()),pairing=config['pair'],minimum_path_length=config['min_length'],reference_used=False)

def calibrate():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'calibration.json').exists():raise FileExistsError('Preserve frozen union calibration')
    cfg26=read(exp.prev.ROOT/'calibration.json')['selected']['config'];prepared=[]
    for r,rgb,refs,valid,bg in exp.cases('calibration'):
        pred=np.load(exp.ROOT/'predictions'/'base'/(r['id']+'.npz'))['prediction'];paths,_=exp.prev.adaptive_curves(pred,cfg26);prepared.append((r,rgb,refs,valid,bg,pred,paths))
    def aggregate(rows):return {k:float(np.mean([r[k] for r in rows])) for k in ['point_coverage_3px','mean_best_single_path_coverage','mixed_reference_paths_flagged','duplicate_point_fraction']}
    base=aggregate([exp.prev.evaluate_paths(refs,p,valid,bg,rgb) for r,rgb,refs,valid,bg,pred,p in prepared]);options=[]
    for cfg in [dict(pair=True,min_length=6),dict(pair=True,min_length=8),dict(pair=False,min_length=6)]:
        rows=[]
        for r,rgb,refs,valid,bg,pred,paths in prepared:
            p,info=decode(paths,pred,cfg);rows.append(exp.prev.evaluate_paths(refs,p,valid,bg,rgb))
        a=aggregate(rows);score=.5*a['point_coverage_3px']+.5*a['mean_best_single_path_coverage']-.025*a['mixed_reference_paths_flagged']-.1*a['duplicate_point_fraction'];eligible=a['point_coverage_3px']>=base['point_coverage_3px']-.01 and a['mean_best_single_path_coverage']>=base['mean_best_single_path_coverage']-.03 and a['mixed_reference_paths_flagged']<=base['mixed_reference_paths_flagged'] and a['duplicate_point_fraction']<base['duplicate_point_fraction']-.02;options.append(dict(config=cfg,metrics=a,score=score,eligible=bool(eligible)));print('V27_UNION_CALIBRATION',cfg,a,'eligible',eligible,flush=True)
    eligible=[o for o in options if o['eligible']];selected=max(eligible or options,key=lambda o:o['score']);jsave(ROOT/'calibration.json',dict(baseline=base,options=options,selected=selected,gate_passed=bool(eligible),prior_adaptive_config=cfg26,calibration_only=True,frozen_before_development=True))

def evaluate():
    choice=read(ROOT/'calibration.json');cfg=choice['selected']['config'];cfg26=choice['prior_adaptive_config'];records=[];views={};old={r['id']:r for r in read(exp.ROOT/'evaluation.json')['records']}
    for r,rgb,refs,valid,bg in exp.cases('development'):
        row=dict(id=r['id'],group=r['split'],metrics={},diagnostics={})
        for name in ['base','control','faint']:
            pred=np.load(exp.ROOT/'predictions'/name/(r['id']+'.npz'))['prediction'];paths,_=exp.prev.adaptive_curves(pred,cfg26);out,info=decode(paths,pred,cfg);row['metrics'][name]=exp.prev.evaluate_paths(refs,out,valid,bg,rgb);row['diagnostics'][name]=info;jsave(ROOT/'trajectories'/name/(r['id']+'.json'),dict(paths=[p.round(3).tolist() for p in out],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,reference_labels_used_at_inference=False,diagnostics=info));views[r['id'],name]=out
        records.append(row);print('V27_UNION_EVAL',r['id'],{n:round(v['mean_best_single_path_coverage'],3) for n,v in row['metrics'].items()},flush=True)
    groups={}
    for g in sorted(set(r['group'] for r in records)):
        rr=[r for r in records if r['group']==g];groups[g]={}
        for name in ['base','control','faint']:
            groups[g][name]={}
            for k in exp.prev.METRICS:
                vals=[r['metrics'][name][k] for r in rr if r['metrics'][name][k] is not None];groups[g][name][k]=float(np.mean(vals)) if vals else None
    jsave(ROOT/'evaluation.json',dict(records=records,groups=groups,choice=choice,no_reference_at_decoding=True,development_not_blind=True))
    keys=['H02','R04','X04','N10','N11'];lookup={r['id']:rgb for r,rgb,refs,valid,bg in exp.cases('development')};board=Image.new('RGB',(1600,310*len(keys)+40),'#f4f2eb');d=ImageDraw.Draw(board);d.text((5,5),'v27 图结构去重：原图／v26连续轨迹／共享线段分配／真实淡线训练+共享线段分配（离线）',font=FONT,fill='black')
    for i,key in enumerate(keys):
        rgb=lookup[key];previous=read(exp.ROOT/'trajectories'/'previous_adaptive'/(key+'.json'))['paths'];images=[('输入',rgb),('v26连续轨迹',exp.prev.v21.line_view(rgb.shape[:2],previous,True)),('图结构分笔与去重',exp.prev.v21.line_view(rgb.shape[:2],views[key,'base'],True)),('淡线训练+图结构去重',exp.prev.v21.line_view(rgb.shape[:2],views[key,'faint'],True))]
        for j,(title,im) in enumerate(images):d.text((j*400+5,40+i*310),key+' '+title,font=SMALL,fill='black');exp.prev.v21.paste(board,im,j*400+5,65+i*310,390,240)
    board.save(ROOT/'RESULT_COMPARISON.jpg',quality=95);print('V27_UNION_FINISHED',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibrate','evaluate']);a=p.parse_args();{'calibrate':calibrate,'evaluate':evaluate}[a.mode]()
