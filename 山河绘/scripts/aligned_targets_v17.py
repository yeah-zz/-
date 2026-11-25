# -*- coding: utf-8 -*-
"""Image-anchored real ribbon/center supervision; preserve original user documents."""
import sys,math,shutil
from pathlib import Path
sys.dont_write_bytecode=True
from rock_v15_common import LAB,PARENT,read,jsave,digest,FONT,SMALL
import numpy as np,cv2
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from human_mountain_v11 import fields,paste
from rock_pairs_v15 import reference_records,dense
from ribbon_fusion_v16 import observed_ink
V15=LAB/'autonomous_strokes_v15_20261025'
V16=LAB/'ink_ribbon_v16_20261025'
ROOT=LAB/'aligned_real_supervision_v17_20261025'

# Small blank-paper rectangles inspected by the assistant, not user/expert gold.
# No untraced painting region is automatically relabelled as background.
ASSISTANT_BG={
'N02':[235,5,282,40], 'N03':[5,5,65,45], 'N04':[3,3,35,28],
'N05':[3,3,65,45], 'N06':[5,5,65,35], 'N07':[3,3,70,42],
'N08':[3,3,95,50], 'N09':[5,5,80,60], 'N12':[5,5,120,60]}

def records():
    user=reference_records();new=read(V15/'new_labels'/'manifest.json')['records'];out=[]
    for r in user:
        if r['id'] in ['R01','R02']:out.append(dict(r,split='train',provenance='user_reviewed_partial'))
        elif r['id'] in ['R03','R04']:out.append(dict(r,split='development',provenance='user_reviewed_partial'))
    for r in new:
        split='calibration' if r['id'] in ['N07','N08'] else 'development' if r['id'] in ['N01','N10','N11'] else 'train'
        out.append(dict(r,split=split,provenance='assistant_partial_not_expert'))
    return out

def source(r):
    rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=read(r['labels']);h,w=rgb.shape[:2];native=np.zeros((h,w),np.float32);left,top,cw,ch=r.get('native',[0,0,w,h]);native[top:top+ch,left:left+cw]=1
    if r['id']=='N01':native[int(h*.77):]=0
    if r['id']=='N12':native[180:]=0
    _,_,m,_=fields(doc);bg=m[1].copy()*native;valid=m[3]*native
    if r['id'] in ASSISTANT_BG:
        x0,y0,x1,y1=ASSISTANT_BG[r['id']];bg[y0:y1,x0:x1]=valid[y0:y1,x0:x1]
    return rgb,doc,bg,valid

def anchored_targets(rgb,doc,bg,valid):
    h,w=rgb.shape[:2];alpha=observed_ink(rgb);smoothed=cv2.bilateralFilter((alpha*255).astype(np.uint8),5,15,2).astype(np.float32)/255
    # Estimate noise only from explicit user/assistant paper negatives.
    bg_values=alpha[bg>0];noise_mean=float(np.median(bg_values)) if len(bg_values) else .005;noise_sigma=float(np.median(np.abs(bg_values-noise_mean))*1.4826) if len(bg_values) else .004
    noise_floor=max(.012,noise_mean+1.5*noise_sigma/math.sqrt(9));ink=np.zeros((h,w),np.uint8);tube=ink.copy();center=ink.copy();sx=np.zeros((h,w),np.float32);sy=sx.copy();angle_count=sx.copy();width_map=sx.copy();confidence=sx.copy();profile_log=[];derived=[];accepted=0;total=0;rejected={};deltas=[]
    def reject(reason):rejected[reason]=rejected.get(reason,0)+1
    for si,stroke in enumerate(doc['strokes']):
        q=dense(stroke['points'],step=2);points=[];entries=[];span=[]
        for k,p in enumerate(q):
            total+=1;ia=max(0,k-2);ib=min(len(q)-1,k+2);tangent=q[ib]-q[ia];norm=np.linalg.norm(tangent)
            if norm<1:reject('degenerate');entries.append(None);continue
            u=tangent/norm;n=np.array([-u[1],u[0]]);offset=np.arange(-24,24.01,.5,dtype=np.float32);samples=p[None]+offset[:,None]*n;profile=np.zeros(len(offset),np.float32)
            for along in [-4,-3,-2,-1,0,1,2,3,4]:
                at=samples+u*along;profile+=cv2.remap(smoothed,at[:,0].astype(np.float32)[None],at[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]/9
            middle=(np.abs(offset)<=4);peak_candidates=np.flatnonzero(middle);peak=int(peak_candidates[np.argmax(profile[middle])]);peak_value=float(profile[peak]);threshold=max(noise_floor,.27*peak_value)
            if peak_value<max(.022,noise_mean+3*noise_sigma/math.sqrt(9)):reject('insufficient_visible_ink');entries.append(None);continue
            selected=profile>threshold;starts=np.flatnonzero(selected&~np.r_[False,selected[:-1]]);ends=np.flatnonzero(selected&~np.r_[selected[1:],False])+1;bands=[(a,b) for a,b in zip(starts,ends) if a<=peak<b]
            if not bands:reject('no_band');entries.append(None);continue
            a,b=bands[0]
            if a==0 or b==len(offset) or not .5<=(b-a)*.5<=46:reject('band_open_or_too_wide');entries.append(None);continue
            others=[(aa,bb) for aa,bb in zip(starts,ends) if (aa,bb)!=(a,b) and np.min(np.abs(offset[aa:bb]))<5]
            if others:reject('adjacent_band_ambiguous');entries.append(None);continue
            weights=np.maximum(profile[a:b]-threshold*.5,0);move=float(np.sum(offset[a:b]*weights)/max(weights.sum(),1e-6));width=float((b-a)*.5)
            if abs(move)>8 or (abs(move)>4 and width<10):reject('reference_ink_disagreement');entries.append(None);continue
            new=p+move*n;xy=np.rint(new).astype(int)
            if not (1<=xy[0]<w-1 and 1<=xy[1]<h-1) or valid[xy[1],xy[0]]==0:reject('outside_review_scope');entries.append(None);continue
            left=p+n*offset[a];right=p+n*offset[b-1];left=new-n*max(1,np.linalg.norm(new-left));right=new+n*max(1,np.linalg.norm(right-new));quality=float(np.clip((peak_value-noise_mean)/max(.06,3*noise_sigma),.25,1))*(.5 if abs(move)>4 else 1.);entry=dict(center=new,left=left,right=right,width=max(2,width),normal=n,tangent=u,quality=quality,index=k);entries.append(entry);accepted+=1;deltas.append(abs(move));points.append(new.tolist())
            if len(profile_log)<350:profile_log.append(dict(stroke=stroke['id'],original=p.round(2).tolist(),derived=new.round(2).tolist(),width=width,peak=peak_value,noise_floor=noise_floor,center_shift=abs(move)))
        for a,b in zip(entries[:-1],entries[1:]):
            if a is None or b is None:continue
            if float(a['tangent']@b['tangent'])<.65 or np.linalg.norm(a['center']-b['center'])>6:continue
            polygon=np.rint([a['left'],b['left'],b['right'],a['right']]).astype(np.int32);layer=np.zeros((h,w),np.uint8);cv2.fillPoly(layer,[polygon],1);ink|=layer
            halo=np.zeros_like(layer);n1=a['normal'];n2=b['normal'];hp=np.rint([a['left']-n1*1.5,b['left']-n2*1.5,b['right']+n2*1.5,a['right']+n1*1.5]).astype(np.int32);cv2.fillPoly(halo,[hp],1)
            # Outside the observed band, only visually empty local flanks are
            # soft negatives; unrelated/untraced ink remains unknown.
            halo[(layer==0)&(alpha>max(.02,noise_mean+2*noise_sigma))]=0;tube|=halo;quality=min(a['quality'],b['quality']);confidence=np.maximum(confidence,layer*quality);confidence=np.maximum(confidence,np.maximum(halo.astype(np.float32)-layer,0)*quality*.3)
            aa=np.rint(a['center']).astype(int);bb=np.rint(b['center']).astype(int);cv2.line(center,tuple(aa),tuple(bb),1,1);delta=b['center']-a['center'];theta=math.atan2(delta[1],delta[0]);direction=np.zeros_like(layer);cv2.line(direction,tuple(aa),tuple(bb),1,1);sx[direction>0]+=math.cos(2*theta);sy[direction>0]+=math.sin(2*theta);angle_count[direction>0]+=1;width_map[direction>0]+=(a['width']+b['width'])/96
        derived.append(dict(id=stroke['id'],visible_derived_points=points,accepted_points=len(points),reference_points=len(q)))
    ink=np.maximum(ink,center);tube=np.maximum(tube,ink);conflict=(ink>0)&(bg>0);conflict=cv2.dilate(conflict.astype(np.uint8),np.ones((3,3),np.uint8))>0;active=(valid>0)&~conflict;ink&=active.astype(np.uint8);center&=active.astype(np.uint8);tube&=active.astype(np.uint8);bg=bg*active
    d,idx=distance_transform_edt(center==0,return_indices=True);norm=np.sqrt(sx*sx+sy*sy);coherence=norm/np.maximum(angle_count,1);angle=(np.stack((sx,sy))/np.maximum(norm,1e-6)[None])[:,idx[0],idx[1]];width=(width_map/np.maximum(angle_count,1))[tuple(idx)];tc=np.exp(-d*d/(2*1.05**2))*ink;target=np.concatenate((tc[None],ink[None],angle,width[None])).astype(np.float32)
    mcenter=np.maximum(confidence*tube,bg);mink=np.maximum(confidence*ink,bg);mangle=((d<2)&(ink>0)&(coherence[tuple(idx)]>.75))*confidence;mwidth=mangle.copy();mask=np.stack((mcenter,mink,mangle,mwidth)).astype(np.float32);mask*=active[None]
    assert mask.min()>=0 and mask.max()<=1
    assert not np.any((target[0]>.5)&(target[1]<.5)&(mask[0]>0))
    audit=dict(accepted_reference_samples=accepted,total_reference_samples=total,acceptance_fraction=accepted/max(1,total),rejections=rejected,mean_center_shift=float(np.mean(deltas)) if deltas else None,max_center_shift=float(max(deltas)) if deltas else None,noise_mean=noise_mean,noise_sigma=noise_sigma,background_positive_conflict_pixels=int(conflict.sum()),ink_positive_pixels=int((ink>0).sum()),center_known_pixels=int((mask[0]>0).sum()),ink_known_pixels=int((mask[1]>0).sum()),center_outside_ink_pixels=0,profiles=profile_log,derived_paths=derived,derived_labels_not_expert_gold=True)
    return target,mask,audit

def control_targets(doc,bg,valid):
    c,a,m,d=fields(doc);target=np.concatenate((np.exp(-d[None]**2/(2*1.05**2)),(d[None]<1.8).astype(np.float32),a,np.zeros_like(d[None]))).astype(np.float32);conf=np.maximum(m[0],bg)*valid;mask=np.stack((conf,conf*.5,m[2]*valid,np.zeros_like(d))).astype(np.float32);return target,mask

def prepare():
    (ROOT/'data').mkdir(parents=True,exist_ok=True);out=[];cases=records();board=Image.new('RGB',(2000,390*len(cases)+40),'#f5f3ec');draw=ImageDraw.Draw(board);draw.text((8,4),'训练标签审计：自动派生监督，以下各栏都不是模型提取结果',font=FONT,fill='black')
    for i,r in enumerate(cases):
        rgb,doc,bg,valid=source(r);target,mask,audit=anchored_targets(rgb,doc,bg,valid);old,om=control_targets(doc,bg,valid);file=ROOT/'data'/(r['id']+'.npz');np.savez_compressed(file,rgb=rgb,target=target,mask=mask,control_target=old,control_mask=om,background=bg,valid=valid);jsave(ROOT/'data'/(r['id']+'_audit.json'),audit)
        overlay=rgb.copy();ink=(target[1]>.5)&(mask[1]>0);overlay[ink]=(overlay[ink].astype(float)*.4+np.array([20,150,235])*.6).astype(np.uint8);center=target[0]>.6;overlay[center]=[235,40,20];newcenter=np.full_like(rgb,255);newcenter[center]=[230,40,20];neg=np.full_like(rgb,255);neg[bg>0]=[70,175,70];neg[(mask[0]==0)&(bg==0)]=[225,225,225]
        for j,(title,a) in enumerate([('原图',rgb),('旧标签：固定窄墨带',np.uint8(255*(1-old[1]))),('新标签：蓝墨带／红中心',overlay),('派生中心（非预测）',newcenter),('绿色已知背景／灰色未知',neg)]):draw.text((j*400+6,40+i*390),r['id']+' '+title,font=SMALL,fill='black');paste(board,a,j*400+7,75+i*390,386,347)
        row=dict(r);row.update(data=str(file),image_sha256=digest(r['image']),label_sha256=digest(r['labels']),data_sha256=digest(file),accepted_samples=audit['accepted_reference_samples'],total_samples=audit['total_reference_samples'],acceptance_fraction=audit['acceptance_fraction'],ink_positive_pixels=audit['ink_positive_pixels'],center_known_pixels=audit['center_known_pixels'],background_pixels=int(bg.sum()),native_ignored_pixels=int((valid==0).sum()),label_provenance='derived_from_visible_ink_at_reviewed_or_assistant_anchors');out.append(row);print('ALIGNED_LABELS',r['id'],r['split'],audit['accepted_reference_samples'],audit['total_reference_samples'],audit['ink_positive_pixels'],flush=True)
    board.save(ROOT/'LABEL_AUDIT.jpg',quality=94)
    selected=Image.new('RGB',(2000,390*5+40),'#f5f3ec');sd=ImageDraw.Draw(selected);sd.text((8,4),'监督标签审计（非模型输出）：R01、R02、N06、N09、N12',font=FONT,fill='black')
    for j,identity in enumerate(['R01','R02','N06','N09','N12']):
        i=next(k for k,r in enumerate(cases) if r['id']==identity);selected.paste(board.crop((0,40+i*390,2000,40+(i+1)*390)),(0,40+j*390))
    selected.save(ROOT/'LABEL_AUDIT_SELECTED.jpg',quality=95);jsave(ROOT/'manifest.json',dict(records=out,training_ids=[r['id'] for r in out if r['split']=='train'],calibration_ids=['N07','N08'],development_ids=['R03','R04','N01','N10','N11','T01','T02','U_A06'],R02_now_training_not_holdout=True,original_user_documents_unchanged=True,unanchored_global_weak_positive_labels_removed=True,assistant_background_rectangles_not_user_gold=ASSISTANT_BG,init_weights=str(V16/'stage_03'/'final.pt'),init_sha256=digest(V16/'stage_03'/'final.pt'),stable_parent_sha256=digest(PARENT),target_semantics='stroke_ribbon_footprint_not_pixel_opacity',apparent_width_divisor=48,max_derived_shift=8,shifts_over4_downweighted=True,no_reference_at_inference=True,independent_artwork_accuracy_unavailable=True));print('V17_LABEL_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':prepare()
