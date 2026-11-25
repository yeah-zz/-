# -*- coding: utf-8 -*-
"""Common untouched raster, endpoint CNN and automatic quality flags."""
from rock_v15_common import *
import numpy as np,torch,cv2,math
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from rock_pairs_v15 import load,connect,frozen_paths,probability,reference_records,dense,density
from human_mountain_v11 import fields,paste
from evaluate_clean_strokes_v12 import path_metrics,line_view

def quality_audit(paths,p,rgb,uncertain):
    flags=[];ink=density(rgb);h,w=p.shape
    for i,q in enumerate(paths):
        q=dense(q);xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);pr=p[xy[:,1],xy[:,0]];length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum())
        if len(q)>3 and np.linalg.norm(q[0]-q[-1])<2 and length<45:flags.append(dict(path=i,reason='small_closed_curve_may_be_genuine',severity='uncertain'))
        if float((pr>.35).mean())<.8:flags.append(dict(path=i,reason='weak_raster_support',severity='uncertain'))
        if float((np.minimum.reduce([q[:,0],w-1-q[:,0],q[:,1],h-1-q[:,1]])<3).mean())>.1:flags.append(dict(path=i,reason='possible_image_frame_or_crop_boundary',severity='uncertain'))
    trees=[cKDTree(dense(q)) for q in paths]
    for i,q in enumerate(paths):
        q=dense(q)
        if len(q)<28:continue
        if any(j!=i and float((tree.query(q)[0]<3).mean())>.6 for j,tree in enumerate(trees)):flags.append(dict(path=i,reason='possible_double_edge_or_neighbor_stroke',severity='uncertain'))
    return dict(status='needs_algorithm_attention' if flags or uncertain else 'passed_geometric_checks_only',flags=flags,ambiguous_endpoint_pairs=len(uncertain),not_expert_certified=True,not_proof_of_semantic_correctness=True,ready_for_robot=False)

def evaluate(round_index):
    model,threshold=load(round_index);torch.set_num_threads(4);cv2.setNumThreads(2);folder=ROOT/f'round_{round_index:02d}'/'evaluation';folder.mkdir(parents=True,exist_ok=True);records=reference_records();old=LAB/'clean_strokes_v12_20261024';oldrecords={r['id']:r for r in read(old/'manifest.json')['records']}
    for identity in ['T01','T02','U_A06']:
        r=oldrecords[identity];left,top=r.get('padding',[0,0]);box=r.get('source_box',[0,0,10000,10000]);records.append(dict(id=identity,image=r['image'],labels=r['labels'],split='reused_development',family='previous_development',native=[left,top,box[2]-box[0],box[3]-box[1]]))
    summary=[]
    for r in records:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=read(r['labels']);p=probability(rgb,ROOT/'probabilities'/(r['id']+'.npz'));_,_,mask,_=fields(doc);h,w=p.shape[1:];left,top,cw,ch=r.get('native',[0,0,w,h]);cw=min(cw,w-left);ch=min(ch,h-top);sub=p[:,top:top+ch,left:left+cw];source=rgb[top:top+ch,left:left+cw];raw=frozen_paths(sub);paths,events,uncertain=connect(raw,sub[0],source,model,threshold);raw=[q+np.array([left,top]) for q in raw];paths=[q+np.array([left,top]) for q in paths if np.linalg.norm(np.diff(q,axis=0),axis=1).sum()>=12];reference=[s['points'] for s in doc['strokes']]
        if r['id']=='U_A06':mask[:,:86,:96]=0
        baseline_id=next((q['source_id'] for q in read(PREVIOUS/'manifest.json')['records'] if q['id']==r['id']),r['id'] if r['id'].startswith(('T','U_')) else 'U_'+r['id']);base=read(old/'evaluation'/baseline_id/'trajectory.json')['paths'];before=path_metrics(reference,base,mask[3]);after=path_metrics(reference,paths,mask[3]);unpaired=path_metrics(reference,raw,mask[3]);audit=quality_audit(paths,p[0],rgb,uncertain);out=folder/r['id'];out.mkdir(exist_ok=True);row=dict(id=r['id'],role=r['split'],before=before,after=after,unpaired=unpaired,automatic_joins=len(events),automatic_quality=audit,independent_test=False,raster_frozen=True);summary.append(row);jsave(out/'metrics.json',row);jsave(out/'trajectory.json',dict(paths=[q.round(2).tolist() for q in paths],events=events,uncertain=uncertain,ready_for_robot=False,actually_drawn=False));plan=line_view(rgb.shape[:2],paths,True)
        one=Image.new('RGB',(1600,430),'#f5f3ec');d=ImageDraw.Draw(one)
        for j,(title,a) in enumerate([('原图',rgb),('用户参考/局部草标（非预测）',line_view(rgb.shape[:2],reference)),('v12 原有轨迹',line_view(rgb.shape[:2],base)),('冻结定位＋本轮分笔网络',plan)]):d.text((j*400+7,5),r['id']+' '+title,font=SMALL,fill='black');paste(one,a,j*400+7,38,386,382)
        one.save(out/'comparison.jpg',quality=95);Image.fromarray(plan).save(out/'prediction.png');print('V15_REAL_EVAL',round_index,r['id'],round(before['mean_best_single_path_coverage'],3),round(after['mean_best_single_path_coverage'],3),'joins',len(events),flush=True)
    jsave(folder/'summary.json',dict(records=summary,threshold=threshold,unmarked_not_negative=True,calibration_is_shared_artwork=True,independent_accuracy_unavailable=True))
    for title,ids in [('reviewed_rocks',['R01','R02','R03','R04']),('transfer',['T01','T02','U_A06'])]:
        board=Image.new('RGB',(1600,430*len(ids)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,4),'round '+str(round_index)+' '+title+'；颜色分笔，非压力/浓淡',font=FONT,fill='black')
        for i,identity in enumerate(ids):board.paste(Image.open(folder/identity/'comparison.jpg'),(0,35+i*430))
        board.save(folder/(title+'.jpg'),quality=95)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('round',type=int);a=parser.parse_args();evaluate(a.round)
