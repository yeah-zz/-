# -*- coding: utf-8 -*-
"""Image-anchored partial raster targets and frozen local curve relationships."""
from v24_sources import ROOT
import v21_stroke_learning as v21
from v24_visible_references import ANNOT,IGNORE,BACKGROUND
from v23_junction_relations import graph,endpoint_segment
from rock_v15_common import read,jsave,digest,FONT,SMALL
from scipy.ndimage import gaussian_filter1d,distance_transform_edt,maximum_filter1d
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from pathlib import Path
import numpy as np,cv2,torch,math,shutil

V21=v21.ROOT
V23=v21.LAB/'junction_relations_v23_20261026'

def densify(p):return v21.dense(np.asarray(p,np.float32)).astype(np.float32)

def ref_paths(key,shape,rgb):
    h,w=shape;alpha=v21.v16.photometry(rgb)[1];result=[]
    for s in ANNOT[key]:
        p=np.array([[float(v) for v in a.split(',')] for a in s.split()],np.float32);p*=np.array([w-1,h-1])/100;q=densify(p)
        if len(q)>4:q=gaussian_filter1d(q,.35,axis=0)
        tang=np.gradient(q,axis=0);tang/=np.maximum(np.linalg.norm(tang,axis=1,keepdims=True),1e-6);normal=np.c_[-tang[:,1],tang[:,0]];off=np.arange(-3,3.1,.5,dtype=np.float32);xy=q[:,None]+normal[:,None]*off[None,:,None];values=cv2.remap(alpha,xy[:,:,0],xy[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);shifts=[]
        for a in values:
            peak=float(a.max());on=a>max(.008,.3*peak);n,labels,_,_=cv2.connectedComponentsWithStats(on[None].astype(np.uint8),8);bands=[]
            for k in range(1,n):
                ids=np.flatnonzero(labels[0]==k)
                if ids[0]>0 and ids[-1]<len(off)-1 and abs(off[ids].mean())<=2.25:bands.append((abs(float(off[ids].mean())),float(off[ids].mean())))
            shifts.append(min(bands)[1] if bands and peak>.015 else 0.)
        shifts=np.clip(gaussian_filter1d(np.asarray(shifts,np.float32),1),-2.25,2.25);q=q+normal*shifts[:,None];q[:,0]=np.clip(q[:,0],0,w-1);q[:,1]=np.clip(q[:,1],0,h-1);result.append(q)
    return result

def valid_map(key,shape):
    h,w=shape;valid=np.ones(shape,np.float32);valid[:3]=0;valid[-3:]=0;valid[:,:3]=0;valid[:,-3:]=0
    for b in IGNORE.get(key,[]):
        x0,y0,x1,y1=[round(v*(w-1 if i%2==0 else h-1)/100) for i,v in enumerate(b)];valid[y0:y1+1,x0:x1+1]=0
    return valid

def sample(im,q):
    q=np.asarray(q,np.float32);return cv2.remap(im.astype(np.float32),q[:,0][None],q[:,1][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]

def curve_nodes_and_pairs(refs,shape,valid,alpha,key):
    h,w=shape;node=np.zeros(shape,np.float32);pairs=[];curves=[densify(p) for p in refs];trees=[cKDTree(p) for p in curves]
    def safe(q):return bool((sample(valid,q)>.9).all() and (sample(alpha,q)>.006).mean()>=.65)
    for k,q in enumerate(curves):
        if len(q)<21:continue
        idx=np.arange(8,len(q)-8);a=q[idx]-q[idx-6];b=q[idx+6]-q[idx];a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-6);b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-6);angle=np.arccos(np.clip((a*b).sum(1),-1,1));peaks=idx[(angle>.30)&(angle>=maximum_filter1d(angle,13))];taken=[]
        for i in peaks[np.argsort(-angle[peaks-8])]:
            if any(abs(int(i)-j)<12 for j in taken):continue
            support=q[i-8:i+9:2]
            if not safe(support):continue
            taken.append(int(i));x,y=np.rint(q[i]).astype(int);yy,xx=np.mgrid[max(0,y-10):min(h,y+11),max(0,x-10):min(w,x+11)];heat=np.exp(-((xx-x)**2+(yy-y)**2)/(2*3**2));sl=np.s_[max(0,y-10):min(h,y+11),max(0,x-10):min(w,x+11)];node[sl]=np.maximum(node[sl],heat)
            pairs.append(dict(image_id=key,p=q[i-8].round(3).tolist(),q=q[i+8].round(3).tolist(),center=q[i].round(3).tolist(),label=1,support=support.round(3).tolist(),provenance='assistant_visible_curve_turn'))
    # Nearby distinct traced chains supply separation examples. Unknown ink and
    # ambiguous overlapping center pixels do not become negative stroke labels.
    seen=[]
    for k,q in enumerate(curves):
        for l in range(k+1,len(curves)):
            other=curves[l]
            if len(q)<12 or len(other)<12:continue
            ii=np.arange(5,len(q)-5,12);dd,jj=trees[l].query(q[ii])
            for i,j,d in zip(ii,jj,dd):
                if not 3<=d<=24 or not 5<=j<len(other)-5:continue
                p=q[i];r=other[j];center=(p+r)*.5
                if not safe(np.stack([p,r])) or sample(valid,center[None])[0]<.9:continue
                if any(np.linalg.norm(center-c)<9 for c in seen):continue
                seen.append(center);pairs.append(dict(image_id=key,p=p.round(3).tolist(),q=r.round(3).tolist(),center=center.round(3).tolist(),label=0,support=[],provenance='nearby_distinct_assistant_chains'))
    node*=valid
    return node,pairs

def new_targets(rgb,refs,valid,key):
    h,w=valid.shape;c,axis,d=v21.v16.geometry(refs,(h,w));alpha=v21.v16.photometry(rgb)[1];raw=alpha>.035;trusted=np.zeros((h,w),np.uint8);center_trust=trusted.copy();center_band=trusted.copy();off=np.arange(-24,24.1,.5,dtype=np.float32)
    # Only the locally measured band containing the traced center may supply
    # center-side negatives. Other nearby, untraced ink remains unknown.
    for p in refs:
        p=densify(p);tangent=np.gradient(p,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-6);normal=np.c_[-tangent[:,1],tangent[:,0]]
        for i in range(1,len(p)-1,2):
            q=p[i]+off[:,None]*normal[i];v=sample(alpha,q);peak=float(v[np.abs(off)<=3].max());cut=max(.012,.30*peak);on=v>cut;n,lab,_,_=cv2.connectedComponentsWithStats(on[None].astype(np.uint8),8);choices=[]
            for j in range(1,n):
                ids=np.flatnonzero(lab[0]==j);mid=float(off[ids].mean())
                if ids[0]>0 and ids[-1]<len(off)-1 and abs(mid)<4:choices.append((abs(mid),ids))
            if not choices or peak<.02:continue
            ids=min(choices,key=lambda a:a[0])[1];mid=float(off[ids].mean());width=float(len(ids)*.5);xy=np.rint(q[ids]).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h);xy=xy[inside];trusted[xy[:,1],xy[:,0]]=1
            if abs(mid)<=2 and width<=20:
                center_band[xy[:,1],xy[:,0]]=1;pt=np.rint(p[i]).astype(int);cv2.circle(center_trust,tuple(pt),2,1,-1)
    ink=raw&(trusted>0);bg=np.zeros((h,w),bool)
    for b in BACKGROUND.get(key,[]):
        x0,y0,x1,y1=[round(v*(w-1 if i%2==0 else h-1)/100) for i,v in enumerate(b)];bg[y0:y1+1,x0:x1+1]=True
    bg&=(d>6)&(valid>0)&(alpha<.05);center=np.exp(-d**2/(2*1.05**2));known_c=(((center_trust>0)&(d<2.8))|((center_band>0)&(alpha>.045))|bg)&(valid>0);known_i=(((trusted>0)&(alpha>.08))|bg)&(valid>0)
    # Apparent image width is only an auxiliary target, never robot pressure.
    width=np.clip(2*distance_transform_edt(raw)/48,0,1);_,nearest=distance_transform_edt(c==0,return_indices=True);width=width[tuple(nearest)];width_valid=(d<2)&(width>.02)&(width<.80)&(valid>0)&(center_trust>0)
    dd=np.stack([distance_transform_edt(v21.raster([p],(h,w))==0).astype(np.float32) for p in refs]);two=np.partition(dd,1,axis=0)[:2] if len(refs)>1 else None;axis_valid=(d<2)&(valid>0)&(center_trust>0)
    if two is not None:axis_valid&=two[1]>3
    t=np.concatenate((center[None],ink[None].astype(np.float32),axis,width[None])).astype(np.float32);m=np.stack([known_c,known_i,axis_valid,width_valid]).astype(np.float32);return t,m,alpha,bg

def prepare():
    if any(ROOT.glob('*/final.pt')):raise FileExistsError('Preserve completed experiment data')
    (ROOT/'data').mkdir(exist_ok=True);(ROOT/'labels').mkdir(exist_ok=True);records=[];pair_rows=[];audits=[];arrays=[]
    for r in read(ROOT/'selected.json')['records']:
        rgb=np.asarray(Image.open(r['image']).convert('L').convert('RGB'));h,w=rgb.shape[:2];valid=valid_map(r['id'],(h,w));refs=ref_paths(r['id'],(h,w),rgb);t,m,alpha,bg=new_targets(rgb,refs,valid,r['id']);node,pairs=curve_nodes_and_pairs(refs,(h,w),valid*(m[0]>0),alpha,r['id']);node_mask=(m[0]>0).astype(np.float32);label=ROOT/'labels'/(r['id']+'.json');jsave(label,dict(id=r['id'],width=w,height=h,strokes=[dict(id='chain_'+str(i),points=p.round(3).tolist()) for i,p in enumerate(refs)],unknown_regions=IGNORE.get(r['id'],[]),source_image_sha256=r['image_sha256'],assistant_reference_not_expert=True,physical_pen_lifts_not_known=True));rec=dict(r,kind='new_visible_reference',labels=str(label),label_sha256=digest(label),data=str(ROOT/'data'/(r['id']+'.npz')),reference_chains=len(refs),reference_center_trust_fraction=float(m[0][t[0]>.5].mean()));records.append(rec);arrays.append((rec,rgb,t,m,np.stack([node,node_mask]),valid,bg));pair_rows.extend(dict(p,split=r['split'],work_group=r['work_group']) for p in pairs)
        ov=rgb.copy();ov[(t[0]>.5)&(m[0]>0)]=[225,45,45];ov[(node>.5)&(node_mask>0)]=[235,160,20];ov[valid==0]=((ov[valid==0].astype(float)+np.array([90,160,220]))*.5).astype(np.uint8);audits.append((r['id'],r['split'],rgb,ov));print('V24_LABELS',r['id'],len(refs),'local_pairs',len(pairs),'center_trust',round(rec['reference_center_trust_fraction'],3),flush=True)
    # Freeze the new visible references before any v24 weight is fitted.
    shutil.copy2(Path(__file__).with_name('v24_visible_references.py'),ROOT/'visible_references_before_training.py')
    for r in read(V21/'manifest.json')['records']:
        if r['split']!='train':continue
        z=np.load(V21/'data_ribbon_v2'/(r['id']+'.npz'));rgb=z['rgb'];refs=[s['points'] for s in read(r['labels'])['strokes']];valid=np.ones(rgb.shape[:2],np.float32);valid[:3]=0;valid[-3:]=0;valid[:,:3]=0;valid[:,-3:]=0;alpha=v21.v16.photometry(rgb)[1];node,pairs=curve_nodes_and_pairs(refs,rgb.shape[:2],valid,alpha,r['id']);node_mask=(z['mask'][0]>0).astype(np.float32);rec=dict(r,kind='v21_clear_replay',data=str(ROOT/'data'/('clear_'+r['id']+'.npz')));records.append(rec);arrays.append((rec,rgb,z['target'],z['mask'],np.stack([node,node_mask]),valid,np.zeros(valid.shape,bool)));pair_rows.extend(dict(p,split='train',work_group=r['work_group']) for p in pairs)
    for r in read(v21.LAB/'autonomous_review_v19_20261025'/'manifest_v2.json')['records']:
        z=np.load(r['data']);rgb=z['rgb'];valid=z['valid'].copy()
        if 'uncertain_exclusion' in z:valid*=z['uncertain_exclusion']==0
        rec=dict(r,id='replay_'+r['id'],kind='reviewed_replay',work_group='prior_'+r['id'],data=str(ROOT/'data'/('replay_'+r['id']+'.npz')));records.append(rec);arrays.append((rec,rgb,z['target'][:5],z['mask'],np.zeros((2,*rgb.shape[:2]),np.float32),valid,z['background']>0))
    # Reuse the actual-node relationships frozen and visually audited in v23,
    # but learn them through the image network rather than a separate MLP.
    lookup={r['id']:(r,rgb,valid) for r,rgb,t,m,node,valid,bg in arrays}
    for r in read(V23/'data_audit.json')['records']:
        if r['split']!='train' or r['policy']!=['center',.65,.30]:continue
        rec,rgb,valid=lookup[r['id']];pred=np.load(V23/'predictions'/(r['id']+'.npz'))['prediction'];g=graph(pred,rgb);alpha=v21.v16.photometry(rgb)[1]
        for c in r['known_pairs']:
            _,a=endpoint_segment(g['chains'][c['a'][0]],c['a'][1]);_,b=endpoint_segment(g['chains'][c['b'][0]],c['b'][1])
            if min(len(a),len(b))<4:continue
            p=np.mean(a[:12],axis=0);q=np.mean(b[:12],axis=0)
            if not (sample(valid,np.stack([p,q]))>.9).all() or not (sample(alpha,np.stack([p,q]))>.006).all():continue
            pair_rows.append(dict(image_id=r['id'],p=p.round(3).tolist(),q=q.round(3).tolist(),center=c['node'],label=c['label'],support=[],split='train',work_group=rec['work_group'],provenance='v23_real_node_audited_relationship'))
    # Keep a bounded balanced set of local references; one image cannot fill it.
    selected=[];rng=np.random.default_rng(242610);train=[p for p in pair_rows if p['split']=='train']
    for label in [0,1]:
        actual=[p for p in train if p['label']==label and p['provenance']=='v23_real_node_audited_relationship'];selected.extend(actual);remaining=[p for p in train if p['provenance']!='v23_real_node_audited_relationship'];grouped={key:[p for p in remaining if p['label']==label and p['image_id']==key] for key in sorted({p['image_id'] for p in remaining})};grouped={k:v for k,v in grouped.items() if v}
        for v in grouped.values():rng.shuffle(v)
        count=len(actual)
        while grouped and count<200:
            for key in list(grouped):
                selected.append(grouped[key].pop());count+=1
                if not grouped[key]:del grouped[key]
                if count>=200:break
    selected.extend(p for p in pair_rows if p['split']!='train');jsave(ROOT/'local_relations.json',dict(records=selected,generated_from_visible_reference_geometry_not_model_predictions=True,classes_are_geometric_chain_relations_not_verified_artist_strokes=True))
    # Parent probabilities constrain unknown pixels, rather than turning them
    # into background labels. They are weak regularization, not new gold.
    model=v21.load_model('ribbon_improved')
    for rec,rgb,t,m,node,valid,bg in arrays:
        teacher=v21.v16.predict(model,rgb)[:2] if rec['split']=='train' else np.zeros((2,*rgb.shape[:2]),np.float32)
        np.savez_compressed(rec['data'],rgb=rgb,target=t.astype(np.float32),mask=m.astype(np.float32),node=node.astype(np.float32),teacher=teacher.astype(np.float16),valid=valid.astype(np.float32),background=bg.astype(np.uint8));rec['data_sha256']=digest(rec['data'])
    manifest=dict(records=records,init_weights=str(V21/'ribbon_improved'/'final.pt'),init_sha256=digest(V21/'ribbon_improved'/'final.pt'),local_relations_sha256=digest(ROOT/'local_relations.json'),new_original_training_images=6,new_calibration_images=1,new_work_development_images=2,new_sources=read(ROOT/'source_manifest.json')['pages'],new_reference_chains=sum(r['reference_chains'] for r in records if r['kind']=='new_visible_reference'),training_local_pairs=sum(p['split']=='train' for p in selected),positive_train_pairs=sum(p['split']=='train' and p['label']==1 for p in selected),negative_train_pairs=sum(p['split']=='train' and p['label']==0 for p in selected),unknown_not_background=True,robot_connected=False,source_independence_beyond_work_groups_not_established=True)
    jsave(ROOT/'training_manifest.json',manifest)
    for start in range(0,len(audits),3):
        sub=audits[start:start+3];board=Image.new('RGB',(1200,350*len(sub)+32),'#f5f3ec');d=ImageDraw.Draw(board);d.text((5,3),'v24 新参考监督审计（非模型输出）：红中心／橙转折／蓝未知',font=FONT,fill='black')
        for i,(key,split,rgb,ov) in enumerate(sub):
            for j,a in enumerate([rgb,ov]):d.text((600*j+5,34+350*i),key+' '+split,font=SMALL,fill='black');v21.paste(board,a,600*j+5,60+350*i,590,310)
        board.save(ROOT/('REFERENCE_AUDIT_'+str(start//3)+'.jpg'),quality=95)
    print('V24_PREPARED',len(records),'local_pairs',manifest['training_local_pairs'],'new_chains',manifest['new_reference_chains'],flush=True)

if __name__=='__main__':prepare()
