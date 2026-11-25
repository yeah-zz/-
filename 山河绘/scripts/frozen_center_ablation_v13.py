# -*- coding: utf-8 -*-
"""Keep v12 center probabilities fixed; evaluate new topology separately."""
import sys,json
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
import numpy as np
from PIL import Image,ImageDraw
from stroke_topology_v13 import ROOT,OLD,jsave
from topology_trace_v13 import adapt_old,center_skeleton,graph_paths
from stroke_pairs_v13 import load
from stroke_link_v12 import connect,dense
from evaluate_clean_strokes_v12 import path_metrics,line_view
from human_mountain_v11 import fields,FONT,SMALL,paste

def run():
    records=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))['records'];folder=ROOT/'frozen_center_ablation';folder.mkdir(exist_ok=True);scorer,threshold=load();rows=[]
    for r in records:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=json.loads(Path(r['labels']).read_text(encoding='utf-8'));_,_,mask,_=fields(doc)
        if r['id']=='U_A06':mask[:,0:86,0:96]=0
        with np.load(OLD/'evaluation'/r['id']/'probabilities.npz') as z:old=z['after'].copy()
        with np.load(ROOT/'evaluation'/r['id']/'probabilities.npz') as z:new=z['prediction'].copy()
        left,top=r.get('padding',[0,0]);box=r.get('source_box',[0,0,rgb.shape[1],rgb.shape[0]]);w,h=box[2]-box[0],box[3]-box[1];old_content=adapt_old(old[:,top:top+h,left:left+w]);sk=center_skeleton(old_content);hybrid=new[:,top:top+h,left:left+w].copy();hybrid[0]=old_content[0]
        unpaired,info0=graph_paths(sk,hybrid,False);paired,info1=graph_paths(sk,hybrid,True);shift=np.array([left,top]);unpaired=[p+shift for p in unpaired];paired=[p+shift for p in paired];linked,events=connect(paired,old[0],rgb,scorer,threshold);baseline=[np.asarray(p,float) for p in json.loads((OLD/'evaluation'/r['id']/'trajectory.json').read_text())['paths']];reference=[s['points'] for s in doc['strokes']]
        score=lambda p:path_metrics(reference,p,mask[3]);row=dict(id=r['id'],before_v12_final=score(baseline),fixed_pixels_unpaired=score([dense(p) for p in unpaired]),fixed_pixels_junction_pairing=score([dense(p) for p in paired]),fixed_pixels_final=score(linked),junction_pairings=len(info1['junction_pairings']),endpoint_joins=len(events),old_center_probabilities_unchanged=True,real_independent_accuracy=False);rows.append(row)
        out=folder/r['id'];out.mkdir(exist_ok=True);after=line_view(rgb.shape[:2],linked,True);Image.fromarray(after).save(out/'paths.png');jsave(out/'trajectory.json',dict(paths=[p.round(2).tolist() for p in linked],ready_for_robot=False,physical_z=None,actually_drawn=False))
        board=Image.new('RGB',(1600,460),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),r['id']+' | 固定 v12 主线概率，连接方法对照；仍是部分标注开发图',font=FONT,fill='black')
        for j,(label,a) in enumerate([('原图',rgb),('v12 最终轨迹',line_view(rgb.shape[:2],baseline)),('同一像素图的分支',line_view(rgb.shape[:2],unpaired,True)),('新配对与连接',after)]):d.text((j*400+8,42),label,font=SMALL,fill='black');paste(board,a,j*400+8,78,384,370)
        board.save(out/'comparison.jpg',quality=95);print('FROZEN_CONTROL',r['id'],round(row['before_v12_final']['mean_best_single_path_coverage'],3),round(row['fixed_pixels_final']['mean_best_single_path_coverage'],3),flush=True)
    jsave(folder/'metrics.json',dict(records=rows,old_center_probabilities_and_ridge_skeleton_fixed=True,real_images_not_independent_tests=True,partial_reference_mixed_paths_not_actual_false_join_rate=True))
    ids=['T01','T02','U_A06'];board=Image.new('RGB',(1600,460*3+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),'固定旧主线概率的模块对照；没有重新训练或修改权重',font=FONT,fill='black')
    for i,key in enumerate(ids):board.paste(Image.open(folder/key/'comparison.jpg'),(0,35+460*i))
    board.save(folder/'comparison.jpg',quality=95)
if __name__=='__main__':run()
