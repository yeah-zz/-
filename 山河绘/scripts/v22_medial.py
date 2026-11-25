# -*- coding: utf-8 -*-
"""Frozen-pixel ink-band medial tracing versus directional center NMS."""
from v22_junctions import *
import v22_junctions as junction
OUT=ROOT/'medial_experiment'
CAL_BG={'20':[[37,31,73,80],[1,1,22,45]],'21':[[30,2,52,19],[38,58,58,86]],'22':[[38,25,62,41],[1,1,24,28]],'24':[[36,25,69,78],[1,1,25,25]],'stage_055':[[2,2,95,22],[2,32,24,99]]}

def clear_bg(r,rgb,refs):
    key=r['id'].rsplit('_',1)[1] if r['id'].startswith('sina') else r['id'];h,w=rgb.shape[:2];bg=np.zeros((h,w),np.uint8)
    for box in CAL_BG[key]:
        x0,y0,x1,y1=[round(v*(w-1 if k%2==0 else h-1)/100) for k,v in enumerate(box)];bg[y0:y1,x0:x1]=1
    _,_,d=v16.geometry(refs,(h,w));bg[d<6]=0;return bg>0

def skeleton_for(pred,rgb,cfg,policy):
    oldpaths,oldsk,oldinfo=prior_eval.decode(pred,policy['kind'],policy['high'],policy['low'])
    if cfg['mode']=='old':return oldsk,oldinfo
    alpha=v16.photometry(rgb)[1];mask=pred[1]>cfg['low']
    if cfg['mode'] in ['photo','hybrid']:mask&=alpha>cfg['photo']
    n,cc,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8);keep=np.zeros(n,bool);keep[np.unique(cc[(pred[0]>.45)&(pred[1]>.65)])]=True;keep[stats[:,cv2.CC_STAT_AREA]<10]=False;keep[0]=False;mask=keep[cc]
    # Fill small enclosed dry-ink holes, never close a white channel between strokes.
    n,cc,stats,_=cv2.connectedComponentsWithStats((~mask).astype(np.uint8),8);fill=np.zeros(n,bool);h,w=mask.shape
    for k in range(1,n):
        x,y,bw,bh,area=stats[k]
        if area<=12 and x>0 and y>0 and x+bw<w and y+bh<h:fill[k]=True
    mask|=fill[cc];sk=skeletonize(mask)
    if cfg['mode']=='hybrid':
        extra=oldsk&(distance_transform_edt(~mask)>4)&(alpha>.004)&(pred[0]>.6);n,cc,stats,_=cv2.connectedComponentsWithStats(extra.astype(np.uint8),8);keep=np.zeros(n,bool);keep[stats[:,cv2.CC_STAT_AREA]>=8]=True;keep[0]=False;sk|=keep[cc]
    broad=distance_transform_edt(mask)>24;sk[broad]=False;sk[:3]=False;sk[-3:]=False;sk[:,:3]=False;sk[:,-3:]=False
    # Bound the same very large ambiguous clusters as the original decoder.
    degree=cv2.filter2D(sk.astype(np.float32),-1,np.ones((3,3),np.float32))-sk;pad=np.pad(sk,1)
    for dy in [-1,1]:
        for dx in [-1,1]:degree-=(pad[1+dy:1+dy+h,1+dx:1+dx+w]&(pad[1:1+h,1+dx:1+dx+w]|pad[1+dy:1+dy+h,1:1+w]))
    expanded=cv2.dilate(((degree>2)&sk).astype(np.uint8),np.ones((5,5),np.uint8))*sk;n,cc,ss,_=cv2.connectedComponentsWithStats(expanded.astype(np.uint8),8);large=np.zeros(n,bool);large[ss[:,cv2.CC_STAT_AREA]>48]=True;large[0]=False;sk[large[cc]]=False
    return sk,dict(small_enclosed_holes_filled=int(fill.sum()),broad_pixels_quarantined=int(broad.sum()),large_junction_pixels_quarantined=int(large[cc].sum()),mode=cfg['mode'])

def configs():
    geometry=[dict(mode='old'),dict(mode='ink',low=.5),dict(mode='photo',low=.35,photo=.008),dict(mode='photo',low=.35,photo=.02),dict(mode='photo',low=.5,photo=.008),dict(mode='hybrid',low=.35,photo=.008)]
    nodes=[dict(mode='legacy'),read(ROOT/'calibration.json')['selected']['config']]
    return [dict(geometry=g,node=n) for g in geometry for n in nodes]

def calibrate_medial():
    OUT.mkdir(exist_ok=True);model,params=load_junction();policy=read(ROOT/'manifest.json')['old_skeleton_decode_policy'];cases=[]
    for r in read(ROOT/'manifest.json')['records']:
        if r['split']!='calibration':continue
        z=np.load(ROOT/'cache'/(r['id']+'.npz'));refs=[s['points'] for s in read(r['labels'])['strokes']];cases.append((r,z['rgb'],z['prediction'],refs))
    choices=[]
    for cfg in configs():
        records=[]
        for r,rgb,pred,refs in cases:
            sk,_=skeleton_for(pred,rgb,cfg['geometry'],policy);g=graph_parts(sk);paths,info=decode_graph(g,pred,rgb,cfg['node'],model,params);met=path_metrics(refs,paths,np.ones(rgb.shape[:2]));bg=clear_bg(r,rgb,refs);rast=v21.raster(paths,rgb.shape[:2]);met['background_line_density']=float(rast[bg].mean()) if bg.any() else 0.;met['small_closed_paths']=int(loop_count(paths));records.append(dict(id=r['id'],metrics=met))
        score=float(np.mean([.55*x['metrics']['mean_best_single_path_coverage']+.45*x['metrics']['point_coverage_3px']-.04*x['metrics']['mixed_reference_paths_flagged']-min(.2,20*x['metrics']['background_line_density']) for x in records]));choices.append(dict(config=cfg,score=score,records=records));print('V22_MEDIAL_CAL',cfg,round(score,4),flush=True)
    base=choices[0];basepoint=np.mean([r['metrics']['point_coverage_3px'] for r in base['records']]);basemix=sum(r['metrics']['mixed_reference_paths_flagged'] for r in base['records']);basebg=np.mean([r['metrics']['background_line_density'] for r in base['records']]);baseloop=sum(r['metrics']['small_closed_paths'] for r in base['records']);allowed=[r for r in choices if np.mean([x['metrics']['point_coverage_3px'] for x in r['records']])>=basepoint-.005 and sum(x['metrics']['mixed_reference_paths_flagged'] for x in r['records'])<=basemix and np.mean([x['metrics']['background_line_density'] for x in r['records']])<=basebg+.0005 and sum(x['metrics']['small_closed_paths'] for x in r['records'])<=baseloop];selected=max(allowed,key=lambda r:r['score']);jsave(OUT/'calibration.json',dict(selected=selected,candidates=choices,calibration_work_ids=[r['id'] for r,_,_,_ in cases],explicit_paper_rectangles=CAL_BG,external_evaluation_not_used_for_selection=True,not_blind_development=True));print('V22_MEDIAL_SELECTED',selected['config'],selected['score'],flush=True)

def evaluate_medial():
    cfg=read(OUT/'calibration.json')['selected']['config'];model,params=load_junction();policy=read(ROOT/'manifest.json')['old_skeleton_decode_policy'];rows=[];pages=[]
    for r,rgb,refs,valid,bg,_ in v21.evaluation_cases():
        key=r['id'];cached=BASE21/'ribbon_improved'/'evaluation'/key;pred=np.load(cached/'prediction.npz')['prediction'];old=read(cached/'trajectory.json')['paths'];sk,mask_info=skeleton_for(pred,rgb,cfg['geometry'],policy);g=graph_parts(sk);paths,info=decode_graph(g,pred,rgb,cfg['node'],model,params);oldmet=v21.diagnostic(refs,pred,old,valid,rgb,bg);newmet=v21.diagnostic(refs,pred,paths,valid,rgb,bg)
        for m,p in [(oldmet,old),(newmet,paths)]:m['small_closed_path_count']=int(loop_count([np.asarray(q) for q in p]))
        folder=OUT/'evaluation'/key;folder.mkdir(parents=True,exist_ok=True);jsave(folder/'trajectory.json',dict(paths=[p.round(3).tolist() for p in paths],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,reference_used_at_inference=False,config=cfg,mask_info=mask_info,graph_info=info));Image.fromarray(np.uint8(255*(~sk))).save(folder/'medial_skeleton.png');row=dict(id=key,split='legacy_related_probe' if key in ['stage_018','stage_029'] else r['split'],before=oldmet,after=newmet);rows.append(row);jsave(folder/'metrics.json',row);page=Image.new('RGB',(1800,365),'#f5f3ec');d=ImageDraw.Draw(page)
        for j,(title,a) in enumerate([('原图',rgb),('v21 原轨迹',line_view(rgb.shape[:2],old,True)),('v22 校准选定候选',line_view(rgb.shape[:2],paths,True)),('候选中心骨架',np.uint8(255*(~sk)))]):d.text((450*j+5,5),key+' '+title,font=SMALL,fill='black');paste(page,a,450*j+5,35,440,320)
        page.save(folder/'comparison.jpg',quality=95);pages.append((key,page));print('V22_MEDIAL_EVAL',key,round(oldmet['mean_best_single_path_coverage'],3),round(newmet['mean_best_single_path_coverage'],3),flush=True)
    aggregate={}
    for split in ['development_holdout','fixed_external_development','user_reviewed_old_development','legacy_related_probe']:
        group=[r for r in rows if r['split']==split];aggregate[split]={}
        for mode in ['before','after']:
            x=[r[mode] for r in group];aggregate[split][mode]={k:float(np.mean([a[k] for a in x])) for k in ['point_coverage_3px','mean_best_single_path_coverage','mean_reference_fragments','path_count']};aggregate[split][mode].update(mixed_flags=sum(a['mixed_reference_paths_flagged'] for a in x),small_closed_path_count=sum(a['small_closed_path_count'] for a in x))
    jsave(OUT/'evaluation'/'summary.json',dict(records=rows,aggregate=aggregate,config=cfg,frozen_pixel_weights=True,not_blind=True));selected_ids=['stage_065','stage_049','sina_6586334590_31','X01','X04','R04'];board=Image.new('RGB',(1800,365*len(selected_ids)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((5,4),'v22 墨带中心细化试验：均为真实离线预测，颜色区分路径，未在机械臂执行',font=FONT,fill='black')
    for i,key in enumerate(selected_ids):board.paste(dict(pages)[key],(0,35+365*i))
    board.save(OUT/'RESULT_COMPARISON.jpg',quality=95);print('V22_MEDIAL_AGGREGATE',json.dumps(aggregate),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode');a=p.parse_args()
    if a.mode=='calibrate':calibrate_medial()
    elif a.mode=='evaluate':evaluate_medial()
