# -*- coding: utf-8 -*-
"""Bounded image-supported edits to stable paths; no reference input at inference."""
from ink_eval_v16 import *
from ribbon_fusion_v16 import observed_ink
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

def center_ribbons(paths,rgb,pred,strength=1.):
    alpha=observed_ink(rgb);h,w=alpha.shape;result=[];marked=[];count=0
    for path in paths:
        q=dense(path);out=q.copy();broad=np.zeros(len(q),bool);shift=np.zeros_like(q)
        for k in range(5,len(q)-5):
            delta=q[k+4]-q[k-4];norm=np.linalg.norm(delta)
            if norm<3:continue
            u=delta/norm;n=np.array([-u[1],u[0]]);offset=np.arange(-14,15,dtype=np.float32)*.75;sample=q[k][None]+offset[:,None]*n;xs=sample[:,0].astype(np.float32)[None];ys=sample[:,1].astype(np.float32)[None];values=cv2.remap(alpha,xs,ys,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0];near=values[10:19];peak=int(np.argmax(near))+10;maximum=float(values[peak])
            if maximum<.16:continue
            threshold=max(.055,maximum*.27);mask=values>threshold;starts=np.flatnonzero(mask&~np.r_[False,mask[:-1]]);ends=np.flatnonzero(mask&~np.r_[mask[1:],False])+1;candidates=[(a,b) for a,b in zip(starts,ends) if a<=peak<b]
            if not candidates:continue
            a,b=candidates[0];width=(b-a)*.75
            if width<6 or width>24 or a==0 or b==len(values):continue
            # The existing line must touch this very band. A white gap or a
            # separate nearby thin stroke is not allowed to jump into it.
            if min(abs(offset[a]),abs(offset[b-1]))>3 and not (a<=14<b):continue
            weights=np.maximum(values[a:b]-threshold*.5,0);move=float(np.sum(offset[a:b]*weights)/max(1e-6,weights.sum()));location=q[k]+n*move;ix,iy=np.rint(location).astype(int)
            if not (0<=ix<w and 0<=iy<h) or pred[1,iy,ix]<.7:continue
            shift[k]=n*np.clip(move,-6,6)*strength;broad[k]=True
        shift=gaussian_filter1d(shift,1.2,axis=0);shift[~broad]=0;out+=shift;out[:,0]=np.clip(out[:,0],0,w-1);out[:,1]=np.clip(out[:,1],0,h-1);result.append(out);marked.append(broad);count+=int((np.linalg.norm(shift,axis=1)>.2).sum())
    # Drop only an almost fully redundant path that has been centered onto
    # the same broad band. Keep unmatched branches and thin neighboring lines.
    drop=set();events=[]
    for a in range(len(result)):
        if a in drop or len(result[a])<20 or marked[a].mean()<.35:continue
        for b in range(a):
            if b in drop or marked[b].mean()<.35:continue
            short,long=(a,b) if len(result[a])<=len(result[b]) else (b,a);dist,_=cKDTree(result[long]).query(result[short]);near=dist<2
            if near.mean()<.88:continue
            missing=~near;start=np.flatnonzero(missing&~np.r_[False,missing[:-1]]);end=np.flatnonzero(missing&~np.r_[missing[1:],False])+1
            if len(start) and max(end-start)>10:continue
            drop.add(short);events.append(dict(removed_path=short,kept_path=long,overlap=float(near.mean()),reason='nearly_redundant_paths_centered_inside_same_observed_ink_band'))
            if short==a:break
    return [p for i,p in enumerate(result) if i not in drop],dict(centered_points=count,redundant_paths_removed=len(drop),deduplication=events)

def recover_faint(paths,rgb,legacy,specialist,old,link,link_threshold):
    alpha=observed_ink(rgb);paper=cv2.medianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),61).astype(np.float32);paper_rgb=np.maximum(rgb.astype(np.float32),paper[:,:,None]);predictions=[]
    for gain in [1.6,2.3]:
        enhanced=np.clip(paper_rgb+(rgb.astype(np.float32)-paper_rgb)*gain,0,255).astype(np.uint8);predictions.append(old_infer(old,enhanced))
    candidates=frozen_paths(predictions[0]);h,w=alpha.shape;base=raster(paths,(h,w)).astype(bool);distance=distance_transform_edt(~base) if base.any() else np.full(base.shape,1e4);dark_distance=distance_transform_edt(alpha<=.22);added=[];rejected=0;quarantined=[]
    for p in candidates:
        p=dense(p)
        if len(p)<32:continue
        xy=np.rint(p).astype(int);xs=np.clip(xy[:,0],0,w-1);ys=np.clip(xy[:,1],0,h-1);inside=(xs>5)&(xs<w-6)&(ys>5)&(ys<h-6);new=distance[ys,xs]>3
        stable=(predictions[0][0,ys,xs]>.5)&(predictions[1][0,ys,xs]>.5);original=legacy[0,ys,xs]>.16;visible=alpha[ys,xs]>.025;neural=specialist[0,ys,xs]>.5
        if inside.mean()>.95 and new.mean()>.8 and stable.mean()>.9 and original.mean()>.75 and visible.mean()>.8 and neural.mean()>.65:
            clear_of_dark=float((dark_distance[ys,xs]>6).mean())
            if clear_of_dark<.8:quarantined.append(dict(points=p.round(2).tolist(),fraction_clear_of_dark_band=clear_of_dark,reason='possible_second_edge_of_existing_dark_ribbon'));rejected+=1
            else:added.append(p)
        else:rejected+=1
    return paths+added,dict(faint_paths_added=len(added),candidate_paths_rejected=rejected,quarantined_dark_edge_candidates=quarantined,require_original_neural_support=True,require_two_exposure_agreement=True)

def calibrate():
    torch.set_num_threads(4);cv2.setNumThreads(2);model=load(3);r=next(r for r in reference_records() if r['id']=='R02');rgb=np.asarray(Image.open(r['image']).convert('RGB'));pred=predict(model,rgb);doc=read(r['labels']);_,_,mask,_=fields(doc);reference=[s['points'] for s in doc['strokes']];baseline=read(V15/'round_04'/'evaluation'/'R02'/'trajectory.json')['paths'];before=path_metrics(reference,baseline,mask[3]);rows=[]
    for strength in [0.,.35,.65,1.]:
        paths,info=center_ribbons(baseline,rgb,pred,strength);metric=path_metrics(reference,paths,mask[3]);sk=raster(paths,rgb.shape[:2]);density=float(sk[mask[1]>0].mean());rows.append(dict(strength=strength,reference_metrics=metric,background_density=density,info=info))
    allowed=[r for r in rows if r['reference_metrics']['point_coverage_3px']>=before['point_coverage_3px']-.01 and r['reference_metrics']['mean_best_single_path_coverage']>=before['mean_best_single_path_coverage']-.01]
    selected=max(allowed,key=lambda r:(r['reference_metrics']['mean_best_single_path_coverage'],r['reference_metrics']['point_coverage_3px']))
    folder=ROOT/'protected_refinement';folder.mkdir(exist_ok=True);jsave(folder/'calibration.json',dict(selected=selected,candidates=rows,only_R02_used_for_calibration=True));print('PROTECTED_CALIBRATED',json.dumps(selected),flush=True)

def evaluate():
    torch.set_num_threads(4);cv2.setNumThreads(2);folder=ROOT/'protected_refinement';strength=read(folder/'calibration.json')['selected']['strength'];model=load(3);old=old_model().cuda();old.load_state_dict(torch.load(PARENT,map_location='cuda',weights_only=False)['model']);old.eval();link,lt=load_links(4);records=reference_records()+read(V15/'new_labels'/'manifest.json')['records'];lookup={r['id']:r for r in read(V15/'curated_manifest.json')['records']};oldrows={r['id']:r for r in read(LAB/'clean_strokes_v12_20261024'/'manifest.json')['records']}
    if (folder/'summary.json').exists() and not (folder/'before_dark_edge_veto').exists():
        backup=folder/'before_dark_edge_veto';backup.mkdir();shutil.copy2(folder/'summary.json',backup/'summary.json')
        for name in ['faint_candidates.jpg','comparison.jpg']:
            if (folder/name).exists():shutil.copy2(folder/name,backup/name)
    for k in ['T01','T02','U_A06']:
        r=oldrows[k];im=Image.open(r['image']);box=r.get('source_box',[0,0,im.width,im.height]);records.append(dict(id=k,image=r['image'],labels=r['labels'],split='reused_development',native=[*r.get('padding',[0,0]),box[2]-box[0],box[3]-box[1]]))
    rows=[]
    for r in records:
        full=np.asarray(Image.open(r['image']).convert('RGB'));h,w=full.shape[:2];left,top,cw,ch=r.get('native',[0,0,w,h]);cw=min(cw,w-left);ch=min(ch,h-top);rgb=full[top:top+ch,left:left+cw]
        if r['id'].startswith('N'):
            cr=lookup.get(r['source_id']);rgb=np.asarray(Image.open(cr['content_image']).convert('RGB')) if cr else full;ch,cw=rgb.shape[:2];baseline=read(V15/'round_04'/'new_source_evaluation'/r['id']/'trajectory.json')['paths'];cache=V15/'auto_predictions'/(r['source_id']+'.npz')
        else:baseline=read(V15/'round_04'/'evaluation'/r['id']/'trajectory.json')['paths'];cache=V15/'probabilities'/(r['id']+'.npz')
        paths=[np.asarray(p)-np.array([left,top]) for p in baseline];route=monochrome_route(rgb);info={};weak={}
        if route:
            pred=predict(model,rgb);paths,info=center_ribbons(paths,rgb,pred,strength);legacy=np.load(cache)['prediction'][:,top:top+ch,left:left+cw];paths,weak=recover_faint(paths,rgb,legacy,pred,old,link,lt)
        after=[p+np.array([left,top]) for p in paths];doc=read(r['labels']);_,_,mask,_=fields(doc);native=np.zeros((h,w),np.float32);native[top:top+ch,left:left+cw]=1;mask*=native[None]
        if r['id']=='U_A06':mask[:,:86,:96]=0
        reference=[s['points'] for s in doc['strokes']];row=dict(id=r['id'],split=r['split'],before=path_metrics(reference,baseline,mask[3]),after=path_metrics(reference,after,mask[3]),correction=info,faint_recovery=weak,monochrome_route=route);rows.append(row);out=folder/r['id'];out.mkdir(exist_ok=True);jsave(out/'metrics.json',row);jsave(out/'trajectory.json',dict(paths=[p.round(2).tolist() for p in after],ready_for_robot=False,actually_drawn=False,protected_refinement=True));board=Image.new('RGB',(1600,430),'#f5f3ec');d=ImageDraw.Draw(board)
        for j,(title,a) in enumerate([('原图',full),('局部参考（非预测）',line_view((h,w),reference)),('v15 轨迹',line_view((h,w),baseline,True)),('旧轨迹上的受限校正',line_view((h,w),after,True))]):d.text((j*400+6,5),r['id']+' '+title,font=SMALL,fill='black');paste(board,a,j*400+8,38,384,383)
        board.save(out/'comparison.jpg',quality=95);print('PROTECTED_REAL',r['id'],round(row['before']['mean_best_single_path_coverage'],3),round(row['after']['mean_best_single_path_coverage'],3),info.get('centered_points',0),weak.get('faint_paths_added',0),flush=True)
    jsave(folder/'summary.json',dict(records=rows,original_parent_unchanged=True,uses_no_reference_at_inference=True,actual_faint_addition_not_guaranteed=True));board=Image.new('RGB',(1600,430*4+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,4),'受限校正对照：仍有错误；颜色分笔，未接实机',font=FONT,fill='black')
    for i,k in enumerate(['R01','R02','N06','N10']):board.paste(Image.open(folder/k/'comparison.jpg'),(0,35+i*430))
    board.save(folder/'comparison.jpg',quality=95)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['calibrate','evaluate']);a=p.parse_args();calibrate() if a.mode=='calibrate' else evaluate()
