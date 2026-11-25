# -*- coding: utf-8 -*-
"""New-source partial draft comparison and unlabelled batch QC, no gold laundering."""
from rock_v15_common import *
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from rock_pairs_v15 import load,connect,frozen_paths,probability,dense
from rock_eval_v15 import quality_audit
from human_mountain_v11 import fields,paste
from human_mountain_eval_v11 import extract
from evaluate_clean_strokes_v12 import path_metrics,line_view

def evaluate(round_index):
    torch.set_num_threads(4);cv2.setNumThreads(2);model,threshold=load(round_index);folder=ROOT/f'round_{round_index:02d}'/'new_source_evaluation';folder.mkdir(exist_ok=True);rows=[]
    refs=read(ROOT/'new_labels'/'manifest.json')['records'];heldout=[r for r in refs if r['split']=='development'];content={r['id']:r for r in read(ROOT/'curated_manifest.json')['records']}
    board=Image.new('RGB',(1600,410*len(heldout)+40),'#f5f3ec');d=ImageDraw.Draw(board);d.text((7,5),'新增来源开发对照；参考为助手局部草标，非用户复核/完整准确率',font=FONT,fill='black')
    for r in refs:
        cr=content.get(r['source_id']);rgb=np.asarray(Image.open(cr['content_image'] if cr else r['image']).convert('RGB'));p=probability(rgb,(ROOT/'auto_predictions' if cr else ROOT/'probabilities')/(r['source_id']+'.npz'));_,old=extract(p);raw=frozen_paths(p);paths,events,uncertain=connect(raw,p[0],rgb,model,threshold);paths=[q for q in paths if np.linalg.norm(np.diff(q,axis=0),axis=1).sum()>=12];doc=read(r['labels']);_,_,mask,_=fields(doc);native=np.zeros(mask.shape[1:],np.float32);native[:rgb.shape[0],:rgb.shape[1]]=1;mask*=native[None];reference=[s['points'] for s in doc['strokes']];row=dict(id=r['id'],source_id=r['source_id'],split=r['split'],before=path_metrics(reference,old,mask[3]),after=path_metrics(reference,paths,mask[3]),native_content=[0,0,rgb.shape[1],rgb.shape[0]],auto_joins=len(events),partial_assistant_reference=True,expert_accuracy=False,automatic_quality=quality_audit(paths,p[0],rgb,uncertain));rows.append(row);out=folder/r['id'];out.mkdir(exist_ok=True);jsave(out/'metrics.json',row);jsave(out/'trajectory.json',dict(paths=[q.round(2).tolist() for q in paths],ready_for_robot=False,actually_drawn=False));Image.fromarray(line_view(rgb.shape[:2],paths,True)).save(out/'prediction.png')
        if r['split']=='development':
            i=heldout.index(r)
            for j,(title,a) in enumerate([('原图',rgb),('助手局部参考（非预测）',line_view(rgb.shape[:2],reference)),('冻结 v12 原追踪',line_view(rgb.shape[:2],old)),('本轮候选轨迹',line_view(rgb.shape[:2],paths,True))]):d.text((j*400+6,40+i*410),r['id']+' '+title,font=SMALL,fill='black');paste(board,a,j*400+7,73+i*410,385,370)
        print('NEW_PARTIAL_EVAL',round_index,r['id'],round(row['before']['mean_best_single_path_coverage'],3),round(row['after']['mean_best_single_path_coverage'],3),flush=True)
    board.save(folder/'new_source_comparison.jpg',quality=95);jsave(folder/'summary.json',dict(records=rows,new_drafts_not_expert_gold=True))
    batch=[]
    for r in read(ROOT/'curated_manifest.json')['records']:
        rgb=np.asarray(Image.open(r['content_image']).convert('RGB'));p=probability(rgb,ROOT/'auto_predictions'/(r['id']+'.npz'));raw=frozen_paths(p);paths,events,uncertain=connect(raw,p[0],rgb,model,threshold);paths=[q for q in paths if np.linalg.norm(np.diff(q,axis=0),axis=1).sum()>=12];qc=quality_audit(paths,p[0],rgb,uncertain);dest=ROOT/'batch_results'/r['id'];dest.mkdir(parents=True,exist_ok=True);jsave(dest/'trajectory.json',dict(paths=[q.round(2).tolist() for q in paths],events=events,uncertain=uncertain,ready_for_robot=False,actually_drawn=False));jsave(dest/'automatic_review.json',qc);Image.fromarray(line_view(rgb.shape[:2],paths,True)).save(dest/'prediction.png');batch.append(dict(id=r['id'],image=r['content_image'],split=r['split'],quality=qc,automatic_join_count=len(events),prediction=str(dest/'prediction.png')))
    jsave(ROOT/'batch_results'/'summary.json',dict(records=batch,review_is_algorithmic_not_expert=True,all_images_processed=True,requires_per_image_user_review=False,does_not_claim_all_results_correct=True))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('round',type=int);a=p.parse_args();evaluate(a.round)
