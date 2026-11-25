# -*- coding: utf-8 -*-
"""Controlled pre-graph ridge retention experiment with frozen v21 outputs."""
import math
import argparse
from pathlib import Path
import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
import v21_stroke_learning as v21
from topology_trace_v13 import graph_paths
from v23_junction_relations import ROOT, BASE

OUT=ROOT/'center_band'
POLICIES=[dict(name='baseline',band=None),dict(name='hybrid_080',band=.80),dict(name='hybrid_065',band=.65),dict(name='center_band',band=.30)]

def degree_map(sk):
    h,w=sk.shape;pad=np.pad(sk,1);degree=cv2.filter2D(sk.astype(np.float32),-1,np.ones((3,3),np.float32))-sk
    for dy in [-1,1]:
        for dx in [-1,1]:degree-=(pad[1+dy:1+dy+h,1+dx:1+dx+w]&(pad[1:1+h,1+dx:1+dx+w]|pad[1+dy:1+dy+h,1:1+w]))
    return degree

def short_bridges(sk,pred,max_gap):
    h,w=sk.shape;deg=degree_map(sk);end_yx=np.argwhere(sk&(deg==1));n,cc=cv2.connectedComponents(sk.astype(np.uint8),8)
    def neighbors(y,x):
        result=[]
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                yy=y+dy;xx=x+dx
                if (not dx and not dy) or yy<0 or xx<0 or yy>=h or xx>=w or not sk[yy,xx]:continue
                if dx and dy and (sk[y,xx] or sk[yy,x]):continue
                result.append((yy,xx))
        return result
    ends=[]
    for y,x in end_yx:
        p=(int(y),int(x));visited={p};q=p;walk=[p]
        for _ in range(8):
            nxt=[a for a in neighbors(*q) if a not in visited]
            if len(nxt)!=1:break
            q=nxt[0];walk.append(q);visited.add(q)
        if len(walk)<5:continue
        tangent=np.array([x-walk[-1][1],y-walk[-1][0]],np.float32);tangent/=max(float(np.linalg.norm(tangent)),1e-6)
        ends.append(dict(xy=np.array([x,y],np.float32),t=tangent,component=int(cc[y,x])))
    if len(ends)<2:return sk.copy(),[]
    tree=cKDTree(np.stack([a['xy'] for a in ends]));candidates=[]
    for i,j in sorted(tree.query_pairs(max_gap)):
        a=ends[i];b=ends[j]
        if a['component']==b['component']:continue
        delta=b['xy']-a['xy'];gap=float(np.linalg.norm(delta));direction=delta/max(gap,1e-6);alignment=float(a['t']@(-b['t']))
        if gap<1.4 or alignment<-.20 or a['t']@direction<-.10 or b['t']@(-direction)<-.10:continue
        q=a['xy']+np.linspace(0,1,max(9,round(gap*3)))[:,None]*delta
        cx=q[:,0].astype(np.float32)[None];cy=q[:,1].astype(np.float32)[None]
        center=cv2.remap(pred[0],cx,cy,cv2.INTER_LINEAR)[0];ink=cv2.remap(pred[1],cx,cy,cv2.INTER_LINEAR)[0]
        if center.mean()<.65 or np.quantile(center,.2)<.45 or ink.min()<.5:continue
        xy=np.rint(q).astype(int);third=False
        for xx,yy in xy[2:-2]:
            labels=np.unique(cc[max(0,yy-1):min(h,yy+2),max(0,xx-1):min(w,xx+2)])
            if any(k not in [0,a['component'],b['component']] for k in labels):third=True;break
        if third:continue
        candidates.append(dict(i=i,j=j,gap=gap,alignment=alignment,score=gap+2*(1-alignment),center_mean=float(center.mean())))
    accepted=[];used=set();parents=list(range(n));out=sk.copy()
    def find(i):
        while parents[i]!=i:i=parents[i]
        return i
    for c in sorted(candidates,key=lambda c:c['score']):
        i,j=c['i'],c['j']
        if i in used or j in used:continue
        competitors=[z['score'] for z in candidates if z is not c and ({z['i'],z['j']}&{i,j})]
        if competitors and min(competitors)-c['score']<1.:continue
        a=ends[i];b=ends[j];u=find(a['component']);v=find(b['component'])
        if u==v:continue
        parents[u]=v;used.update([i,j]);canvas=out.astype(np.uint8);cv2.line(canvas,tuple(a['xy'].astype(int)),tuple(b['xy'].astype(int)),1,1);out=canvas>0
        accepted.append(dict(start=a['xy'].tolist(),end=b['xy'].tolist(),gap=c['gap'],alignment=c['alignment'],center_mean=c['center_mean']))
    return out,accepted

def decode(pred,cfg):
    if cfg['name']=='baseline':return v21.prior_eval.decode(pred,'center',.65,.30)
    if cfg['name'].startswith('bridge'):
        _,sk,_=v21.prior_eval.decode(pred,'center',.65,.30);sk,events=short_bridges(sk,pred,cfg['gap']);h,w=sk.shape
        field=np.zeros((8,h,w),np.float32);field[0]=np.maximum(pred[0],pred[1]*.8);field[1:3]=pred[2:4];field[3:5]=pred[2:4]
        paths,info=graph_paths(sk,field,True,min_length=8)
        return paths,sk,dict(**info,short_bridges=events)
    center,ink=pred[:2];h,w=center.shape;theta=.5*np.arctan2(pred[3],pred[2]);yy,xx=np.mgrid[:h,:w].astype(np.float32)
    nx=-np.sin(theta);ny=np.cos(theta);smooth=cv2.GaussianBlur(center,(0,0),.65)
    a=cv2.remap(smooth,xx+nx*1.5,yy+ny*1.5,cv2.INTER_LINEAR);b=cv2.remap(smooth,xx-nx*1.5,yy-ny*1.5,cv2.INTER_LINEAR)
    ridge=(center>.30)&(smooth>=a)&(smooth>=b)&(ink>.5)
    # Retain an already confident center band across unstable local orientations.
    # This introduces no line pixels unsupported by the predicted center/ink heads.
    candidate=((center>cfg['band'])&(ink>.5)) if cfg['name']=='center_band' else (ridge|((center>cfg['band'])&(ink>.5)))
    n,labels,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8);good=np.zeros(n,bool);good[np.unique(labels[center>.65])]=True;good[stats[:,cv2.CC_STAT_AREA]<12]=False;good[0]=False;binary=good[labels];sk=skeletonize(binary);sk[:3]=False;sk[-3:]=False;sk[:,:3]=False;sk[:,-3:]=False
    broad=distance_transform_edt(binary)>24;sk[broad]=False
    pad=np.pad(sk,1);degree=cv2.filter2D(sk.astype(np.float32),-1,np.ones((3,3),np.float32))-sk
    for dy in [-1,1]:
        for dx in [-1,1]:degree-=(pad[1+dy:1+dy+h,1+dx:1+dx+w]&(pad[1:1+h,1+dx:1+dx+w]|pad[1+dy:1+dy+h,1:1+w]))
    expanded=cv2.dilate(((degree>2)&sk).astype(np.uint8),np.ones((5,5),np.uint8))*sk.astype(np.uint8);nc,cc,ss,_=cv2.connectedComponentsWithStats(expanded,8);large=np.zeros(nc,bool);large[ss[:,cv2.CC_STAT_AREA]>48]=True;large[0]=False;crowded=large[cc];sk[crowded]=False
    field=np.zeros((8,h,w),np.float32);field[0]=np.maximum(center,ink*.8);field[1:3]=pred[2:4];field[3:5]=pred[2:4]
    paths,info=graph_paths(sk,field,True,min_length=8)
    return paths,sk,dict(**info,broad_area_quarantined_pixels=int(broad.sum()),crowded_junction_quarantined_pixels=int(crowded.sum()))

def run():
    cv2.setNumThreads(2);OUT.mkdir(exist_ok=True);cal=[]
    for r in v21.read(BASE/'manifest.json')['records']:
        if r['split']!='calibration':continue
        pred=np.load(ROOT/'predictions'/(r['id']+'.npz'))['prediction'];rgb=np.asarray(Image.open(r['image']).convert('RGB'));refs=[s['points'] for s in v21.read(r['labels'])['strokes']];cal.append((r['id'],rgb,refs,pred,np.ones(rgb.shape[:2],np.float32)))
    options=[]
    for cfg in POLICIES:
        rows=[]
        for key,rgb,refs,pred,valid in cal:
            paths,sk,info=decode(pred,cfg);met=v21.path_metrics(refs,paths,valid);rows.append(dict(id=key,**met))
        score=float(np.mean([.5*r['point_coverage_3px']+.5*r['mean_best_single_path_coverage']-.025*r['mixed_reference_paths_flagged'] for r in rows]))
        options.append(dict(policy=cfg,score=score,single=float(np.mean([r['mean_best_single_path_coverage'] for r in rows])),mixed=sum(r['mixed_reference_paths_flagged'] for r in rows),records=rows))
    baseline=options[0];selected=max(options,key=lambda r:r['score'])
    enabled=selected['score']>=baseline['score']+.005 and selected['single']>=baseline['single']+.005 and selected['mixed']<=baseline['mixed']
    v21.jsave(OUT/'calibration.json',dict(options=options,selected=selected['policy'],enabled=bool(enabled),baseline_score=baseline['score'],selected_score=selected['score'],fixed_development_not_used_for_selection=True))
    print('V23_CENTER_CALIBRATION',bool(enabled),[(r['policy']['name'],round(r['score'],4),round(r['single'],4),r['mixed']) for r in options],flush=True)
    rows=[];images={}
    for r,rgb,refs,valid,bg,_ in v21.evaluation_cases():
        key=r['id'];pred=np.load(BASE/'ribbon_improved'/'evaluation'/key/'prediction.npz')['prediction'];split=r['split']
        if key in ['stage_018','stage_029']:split='legacy_related_probe'
        before,sk0,_=decode(pred,POLICIES[0]);candidate,sk1,info=decode(pred,selected['policy'])
        m0=v21.path_metrics(refs,before,valid);m1=v21.path_metrics(refs,candidate,valid)
        row=dict(id=key,split=split,before=m0,trial=m1,background_density_before=float(sk0[bg].mean()) if np.any(bg) else None,background_density_trial=float(sk1[bg].mean()) if np.any(bg) else None)
        rows.append(row);images[key]=(rgb,before,candidate)
        folder=OUT/'evaluation'/key;folder.mkdir(parents=True,exist_ok=True)
        v21.jsave(folder/'trial_trajectory.json',dict(paths=[q.round(3).tolist() for q in candidate],policy=selected['policy'],references_used_at_inference=False,physical_z=None,ready_for_robot=False,actually_drawn=False,info=info))
        print('V23_CENTER_EVAL',key,round(m0['mean_best_single_path_coverage'],3),round(m1['mean_best_single_path_coverage'],3),m0['mixed_reference_paths_flagged'],m1['mixed_reference_paths_flagged'],flush=True)
    groups={}
    for split in sorted(set(r['split'] for r in rows)):
        rr=[r for r in rows if r['split']==split];groups[split]={tag:{m:float(np.mean([r[tag][m] for r in rr])) for m in ['point_coverage_3px','mean_best_single_path_coverage','mean_reference_fragments','mixed_reference_paths_flagged','path_count']} for tag in ['before','trial']}
    keys=['stage_065','sina_6586334590_31','X01','X04','R04','stage_049'];board=Image.new('RGB',(1500,320*len(keys)+45),'#f5f3ec');d=ImageDraw.Draw(board)
    d.text((8,6),'v23 骨架连续性试验：原图 / 原方向抑制解码 / 中心带保留候选（非实画）',font=v21.FONT,fill='black')
    for i,key in enumerate(keys):
        rgb,before,trial=images[key]
        for j,(title,a) in enumerate([('原图',rgb),('原路径',v21.line_view(rgb.shape[:2],before,True)),('中心带候选',v21.line_view(rgb.shape[:2],trial,True))]):
            d.text((j*500+6,45+320*i),key+' '+title,font=v21.SMALL,fill='black');v21.paste(board,a,j*500+6,70+320*i,485,285)
    board.save(OUT/'RESULT_COMPARISON.jpg',quality=95)
    v21.jsave(OUT/'evaluation.json',dict(groups=groups,records=rows,calibration_enabled=bool(enabled),selected=selected['policy'],not_blind=True,extractor_weights_updated=False))
    print('V23_CENTER_GROUPS',groups,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bridges',action='store_true');a=p.parse_args()
    if a.bridges:
        OUT=ROOT/'center_bridges';POLICIES=[dict(name='baseline',band=None)]+[dict(name='bridge_'+str(gap),gap=gap) for gap in [3,5,7]]
    run()
