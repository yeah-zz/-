# -*- coding: utf-8 -*-
"""Image-only constrained reference correction and withheld raw-section checks."""
from rock_v15_common import LAB,read,jsave,digest,FONT,SMALL
import v21_stroke_learning as v21
import v24_eval as previous
from pathlib import Path
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
import numpy as np,cv2,math,shutil

ROOT=LAB/'continuous_curves_v25_20261026'
V24=LAB/'joint_center_affinity_v24_20261026'
OFF=np.arange(-12,12.01,.5,dtype=np.float32)

def dense(p):return v21.dense(np.asarray(p,np.float32)).astype(np.float32)

def sample(im,q):
    q=np.asarray(q,np.float32);return cv2.remap(im.astype(np.float32),q[:,0][None],q[:,1][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]

def geometry(p):
    a=np.gradient(p,axis=0);a=gaussian_filter1d(a,1,axis=0);a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-6);return a,np.c_[-a[:,1],a[:,0]]

def profiles(alpha,p,t,n):
    q=p[:,None]+n[:,None]*OFF[None,:,None];result=0
    for d in [-1.5,0,1.5]:
        xy=q+t[:,None]*d;result=result+cv2.remap(alpha,xy[:,:,0],xy[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)/3
    return gaussian_filter1d(result,.6,axis=1)

def collect():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'inputs').mkdir(exist_ok=True);rows=[];seen=set()
    def add(r,rgb,refs,valid,bg):
        if r['id'] in seen:return
        seen.add(r['id']);key=r['id'];file=ROOT/'inputs'/(key+'.npz');np.savez_compressed(file,rgb=rgb,valid=valid.astype(np.float32),background=bg.astype(np.uint8));rows.append(dict(id=key,split=r['split'],work_group=r.get('work_group',key),input=str(file),input_sha256=digest(file),source_image=r['image'],source_image_sha256=digest(r['image']),source_labels=r['labels'],source_label_sha256=digest(r['labels']),references=[dense(p).round(4).tolist() for p in refs]))
    for r in read(V24/'training_manifest.json')['records']:
        if r['kind']=='reviewed_replay':continue
        z=np.load(r['data']);rgb=z['rgb'];refs=[s['points'] for s in read(r['labels'])['strokes']];valid=z['valid'];bg=z['background']>0
        if not bg.any():bg=(z['mask'][0]>0)&(z['target'][0]<.01)&(z['target'][1]<.1)
        add(r,rgb,refs,valid,bg)
    for split in ['calibration','development']:
        for r,rgb,refs,valid,bg in previous.cases(split):
            if r['id'] in ['C01','H01','H02']:continue
            add(r,rgb,refs,valid,bg)
    jsave(ROOT/'initial_manifest.json',dict(records=rows,training_ids=[r['id'] for r in rows if r['split']=='train'],all_other_splits_excluded_from_training=True,parent_weights=str(v21.ROOT/'ribbon_improved'/'final.pt'),parent_sha256=digest(v21.ROOT/'ribbon_improved'/'final.pt'),robot_connected=False))
    print('V25_INPUTS',len(rows),'train',sum(r['split']=='train' for r in rows),flush=True)

def raw_checks():
    folder=ROOT/'raw_checks';folder.mkdir(exist_ok=True);rows=[];cards=[]
    for r in read(ROOT/'initial_manifest.json')['records']:
        if r['id'] not in ['R03','R04','H01','H02']:continue
        z=np.load(r['input']);rgb=z['rgb'];alpha=v21.v16.photometry(rgb)[1];refs=[dense(p) for p in r['references']];trees=[cKDTree(p) for p in refs];possible=[]
        for k,p in enumerate(refs):
            t,n=geometry(p)
            for i in range(12,len(p)-12,17):
                if sample(z['valid'],p[i:i+1])[0]<.9:continue
                if any(tree.query(p[i])[0]<7 for j,tree in enumerate(trees) if j!=k):continue
                pr=profiles(alpha,p[i:i+1],t[i:i+1],n[i:i+1])[0]
                if pr.max()<.025:continue
                possible.append((k,i,p[i],t[i],n[i],pr))
        if not possible:continue
        picks=np.unique(np.linspace(0,len(possible)-1,min(8,len(possible))).astype(int))
        for j in picks:
            k,i,p,t,n,pr=possible[j];key='K'+str(len(rows)).zfill(2);meta=dict(id=key,image_id=r['id'],stroke_index=k,point_index=i,p=p.tolist(),t=t.tolist(),n=n.tolist(),offsets=OFF.tolist(),profile=pr.tolist());rows.append(meta);card=Image.new('RGB',(630,235),'white');d=ImageDraw.Draw(card);d.text((5,3),key+' '+r['id']+' 原图横截面（未显示修正或模型结果）',font=SMALL,fill='black')
            # Canonical section view: horizontal is normal offset in original px.
            xx,yy=np.meshgrid(np.linspace(-12,12,160),np.linspace(-7,7,110));xy=p+n*xx[:,:,None]+t*yy[:,:,None];patch=cv2.remap(rgb,xy[:,:,0].astype(np.float32),xy[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);card.paste(Image.fromarray(patch),(5,35));d.line((85,35,85,145),fill='#ee3333');d.text((10,151),'红线＝原参考位置 0',font=SMALL,fill='black')
            x0,y0,ww,hh=205,35,400,145;scale=max(float(pr.max())*1.1,.06)
            for tick in range(-12,13,2):
                x=x0+(tick+12)/24*ww;d.line((x,y0,x,y0+hh),fill='#dddddd');d.text((x-7,y0+hh+3),str(tick),font=SMALL,fill='black')
            d.line((x0,y0+hh,x0+ww,y0+hh),fill='black');pts=[(x0+(float(o)+12)/24*ww,y0+hh-float(v)/scale*hh) for o,v in zip(OFF,pr)];d.line(pts,fill='#222222',width=2);d.text((x0,y0-20),'原图相对墨色（越高越黑）；横轴为原图像素',font=SMALL,fill='black');cards.append(card)
    for start in range(0,len(cards),6):
        board=Image.new('RGB',(1260,235*math.ceil(min(6,len(cards)-start)/2)),'#eeeeee')
        for i,c in enumerate(cards[start:start+6]):board.paste(c,(i%2*630,i//2*235))
        board.save(folder/('SECTIONS_'+str(start//6)+'.jpg'),quality=97)
    jsave(folder/'locations.json',dict(records=rows,selected_without_corrected_references_or_new_model_predictions=True,all_images_excluded_from_v25_training=True));print('V25_RAW_CHECKS',len(rows),flush=True)

def corrected_reference(p,alpha,valid):
    p=dense(p);t,n=geometry(p);pr=profiles(alpha,p,t,n);cost=np.full((len(p),len(OFF)),30,np.float32);descriptors=[];reasons=[]
    for i,v in enumerate(pr):
        noise=float(np.median(np.r_[v[:5],v[-5:]]));peak=float(v.max());cut=max(.012,noise+.32*max(0,peak-noise));on=v>cut;nn,ll,_,_=cv2.connectedComponentsWithStats(on[None].astype(np.uint8),8);bands=[]
        for k in range(1,nn):
            ids=np.flatnonzero(ll[0]==k);width=len(ids)*.5;mid=float((OFF[ids[0]]+OFF[ids[-1]])*.5);strength=float(v[ids].max()-noise)
            if ids[0]>0 and ids[-1]<len(OFF)-1 and 1<=width<=16 and abs(mid)<=8 and strength>=.018:bands.append(dict(mid=mid,width=width,strength=strength))
        if sample(valid,p[i:i+1])[0]<.9:bands=[];reason='excluded_region'
        elif not bands:reason='no_unambiguous_bounded_band'
        else:reason='candidate'
        if bands:
            cost[i]=np.min(np.stack([((OFF-b['mid'])/.7)**2-.1*np.log(max(b['strength'],.001)) for b in bands]),axis=0)+.015*OFF**2
        else:cost[i]=.3*OFF**2
        descriptors.append(bands);reasons.append(reason)
    # Dynamic programming controls offset changes along the original chain.
    transition=.30*(OFF[:,None]-OFF[None,:])**2;transition[np.abs(OFF[:,None]-OFF[None,:])>2]=1e5;dp=cost[0].copy();back=[]
    for i in range(1,len(p)):
        z=dp[:,None]+transition;idx=z.argmin(0);back.append(idx);dp=cost[i]+z[idx,np.arange(len(OFF))]
    index=int(dp.argmin());shift=np.zeros(len(p),np.float32);shift[-1]=OFF[index]
    for i in range(len(p)-2,-1,-1):index=int(back[i][index]);shift[i]=OFF[index]
    shift=gaussian_filter1d(shift,.7);q=p+n*shift[:,None];h,w=alpha.shape;q[:,0]=np.clip(q[:,0],0,w-1);q[:,1]=np.clip(q[:,1],0,h-1);confidence=np.zeros(len(p),np.float32);width=np.zeros(len(p),np.float32);old_residual=[];new_residual=[]
    for i,bands in enumerate(descriptors):
        if not bands:continue
        ordered=sorted(bands,key=lambda b:abs(b['mid']-float(shift[i])));b=ordered[0]
        unique=len(ordered)==1 or abs(ordered[1]['mid']-b['mid'])>=5
        if not unique or abs(b['mid']-float(shift[i]))>1.25 or b['width']>12:continue
        if sample(valid,q[i:i+1])[0]<.9:continue
        confidence[i]=1.;width[i]=b['width'];old_residual.append(abs(b['mid']));new_residual.append(abs(b['mid']-float(shift[i])))
    # Require a small neighborhood of accepted sections for local curve targets.
    if len(confidence)>4:confidence=np.minimum(confidence,np.minimum(np.r_[0,confidence[:-1]],np.r_[confidence[1:],0]))
    return dict(points=q.round(4).tolist(),coarse_points=p.round(4).tolist(),confidence=confidence.tolist(),apparent_width=width.tolist(),offset=shift.tolist()),dict(points=len(p),accepted=int(confidence.sum()),mean_shift=float(np.abs(shift).mean()),max_shift=float(np.abs(shift).max()),image_band_residual_before=float(np.mean(old_residual)) if old_residual else None,image_band_residual_after=float(np.mean(new_residual)) if new_residual else None)

def refine():
    (ROOT/'refined').mkdir(exist_ok=True);rows=[];cards=[]
    for r in read(ROOT/'initial_manifest.json')['records']:
        z=np.load(r['input']);rgb=z['rgb'];alpha=v21.v16.photometry(rgb)[1];strokes=[];metrics=[]
        for p in r['references']:
            s,m=corrected_reference(p,alpha,z['valid']);strokes.append(s);metrics.append(m)
        file=ROOT/'refined'/(r['id']+'.json');jsave(file,dict(image_id=r['id'],strokes=strokes,image_only_geometric_correction=True,not_expert_ground_truth=True,unknown_sections_have_zero_confidence=True));rows.append(dict(id=r['id'],split=r['split'],file=str(file),sha256=digest(file),metrics=metrics,accepted_fraction=sum(m['accepted'] for m in metrics)/max(1,sum(m['points'] for m in metrics))))
        if r['id'] in ['U01','U02','U05','H01','R04']:
            card=Image.new('RGB',(1200,360),'#f5f3ec');d=ImageDraw.Draw(card);d.text((5,3),r['id']+' 中心参考修正审计：橙旧参考／蓝修正／灰为未知（非模型效果）',font=FONT,fill='black');ov=rgb.copy()
            for s in strokes:
                p=np.rint(s['coarse_points']).astype(np.int32);q=np.rint(s['points']).astype(np.int32);cv2.polylines(ov,[p],False,(230,130,35),1,cv2.LINE_AA)
                for a,b,c0,c1 in zip(q[:-1],q[1:],s['confidence'][:-1],s['confidence'][1:]):cv2.line(ov,tuple(a),tuple(b),(35,110,230) if min(c0,c1)>.5 else (155,155,155),1,cv2.LINE_AA)
            v21.paste(card,rgb,5,35,585,315);v21.paste(card,ov,605,35,585,315);cards.append(card)
        print('V25_REFINE',r['id'],round(rows[-1]['accepted_fraction'],3),flush=True)
    jsave(ROOT/'reference_manifest.json',dict(records=rows,initial_manifest_sha256=digest(ROOT/'initial_manifest.json'),method='bounded_image_band_DP_normal_offsets',labelled_checks_used_for_parameters=False,expert_accuracy_not_established=True))
    board=Image.new('RGB',(1200,360*len(cards)),'white')
    for i,c in enumerate(cards):board.paste(c,(0,i*360))
    board.save(ROOT/'REFERENCE_REFINEMENT.jpg',quality=95)
    print('V25_REFERENCE_DONE',len(rows),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','raw_checks','refine']);a=p.parse_args();{'collect':collect,'raw_checks':raw_checks,'refine':refine}[a.mode]()
