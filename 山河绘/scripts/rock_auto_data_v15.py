# -*- coding: utf-8 -*-
"""Consistency-gated auto drafts, actual predicted-arc mining and trace audit."""
from rock_v15_common import *
import math,shutil
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from rock_pairs_v15 import probability,examples,pair_inputs,reference_records,frozen_paths,dense,density
from human_mountain_v11 import paste
from evaluate_clean_strokes_v12 import line_view
from gongbi_training.stroke_train_v08 import model_new,infer

def actual_arc_pairs(rgb,p,doc):
    paths=frozen_paths(p);refs=[dense(s['points']) for s in doc['strokes']];ink=density(rgb);ends=[];identities=[];ep=[]
    for i,q in enumerate(paths):
        for side in (0,1):
            near=q[:min(16,len(q))] if side==0 else q[-min(16,len(q)):]
            score=np.array([np.mean(cKDTree(ref).query(near)[0]<=3) for ref in refs]);order=np.argsort(-score)
            identity=int(order[0]) if score[order[0]]>=.8 and (len(order)<2 or score[order[0]]-score[order[1]]>=.2) else -1
            ends.append((i,side));identities.append(identity);ep.append(q[0 if side==0 else -1])
    rows=[]
    if not ep:return rows
    for a,b in sorted(cKDTree(ep).query_pairs(16)):
        ia,ea=ends[a];ib,eb=ends[b];ra,rb=identities[a],identities[b]
        if ia==ib or min(ra,rb)<0:continue
        patch,f=pair_inputs(paths[ia],ea,paths[ib],eb,p[0],ink)
        if f[1]<.25 or min(f[2],f[3])<-.05:continue
        rows.append((patch,f,int(ra==rb)))
    return rows

def save_pairs(path,rows):
    if not rows:raise RuntimeError('No eligible pair training data')
    patch,feat,y=zip(*rows);np.savez_compressed(path,patch=np.asarray(patch,np.float16),features=np.asarray(feat,np.float32),targets=np.asarray(y,np.float32))

def build():
    torch.set_num_threads(4);cv2.setNumThreads(2);rows=[];audit=[]
    if (ROOT/'curated_manifest.json').exists() and not (ROOT/'auto_gate_v1'/'curated_manifest.json').exists():
        dest=ROOT/'auto_gate_v1';dest.mkdir(exist_ok=True)
        for name in ['curated_manifest.json','pseudo_pairs.npz','auto_quality_contact.jpg']:shutil.copy2(ROOT/name,dest/name)
        shutil.copytree(ROOT/'auto_labels',dest/'auto_labels',dirs_exist_ok=True)
    for r in reference_records():
        if r['split']!='train':continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));p=probability(rgb,ROOT/'probabilities'/(r['id']+'.npz'));rr=actual_arc_pairs(rgb,p,read(r['labels']));rows+=rr;audit.append(dict(id=r['id'],source='user_reference_matched_actual_predicted_arcs',pairs=len(rr),positive=sum(x[2] for x in rr)))
    for i,r in enumerate(read(ROOT/'new_labels'/'manifest.json')['records']):
        if r['split']!='train':continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));p=probability(rgb,ROOT/'probabilities'/(r['source_id']+'.npz'));doc=read(r['labels']);rr=examples([s['points'] for s in doc['strokes']],p[0],rgb,155500+i,18);rr+=actual_arc_pairs(rgb,p,doc);rows+=rr;audit.append(dict(id=r['id'],source='assistant_visual_draft_not_expert',pairs=len(rr),positive=sum(x[2] for x in rr)))
    save_pairs(ROOT/'additional_real_pairs.npz',rows);jsave(ROOT/'additional_real_pairs_audit.json',audit)
    source=read(ROOT/'sources.json');selected=[]
    for r in source:
        identity=r['id']
        if identity.startswith('sina_6586334590_') and 14<=int(identity.rsplit('_',1)[1])<=31 and r['eligible_clear_candidate']:selected.append(r)
        elif identity.startswith('kuhw_29483_'):selected.append(r)
        elif identity=='kuhw_2256_01':selected.append(r)
    model=model_new().cuda();model.load_state_dict(torch.load(PARENT,map_location='cuda',weights_only=False)['model']);model.eval();pseudo=[];pseudo_sources=[];manifest=[];crops=[];panels=[]
    for r in selected:
        identity=r['id'];rgb=np.asarray(Image.open(r['image']).convert('RGB'));h,w=rgb.shape[:2]
        # Explicit content rectangles exclude printed labels/logos, not strokes.
        if identity.startswith('kuhw_29483_'):rgb=rgb[:int(h*.77)]
        if identity=='kuhw_2256_01':rgb=rgb[:180]
        h,w=rgb.shape[:2];p=probability(rgb,ROOT/'auto_predictions'/(identity+'.npz'))
        # Flip/brightness stability are correlated tests, not independent truth.
        alt=np.clip(rgb[:,::-1].astype(np.float32)*1.04+2,0,255).astype(np.uint8);p2=infer(model,alt)[:, :, ::-1].copy();ink=density(rgb);raw=frozen_paths(p);accepted=[];reasons=[]
        other_trees=[cKDTree(dense(q)) for q in raw]
        for qi,q in enumerate(raw):
            q=dense(q);xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);xs,ys=xy.T;length=np.linalg.norm(np.diff(q,axis=0),axis=1).sum();closed=np.linalg.norm(q[0]-q[-1])<3;support=float((p[0,ys,xs]>.45).mean());stable=float((p2[0,ys,xs]>.4).mean());visible=float((ink[ys,xs]>.035).mean())
            border_fraction=float((np.minimum.reduce([q[:,0],w-1-q[:,0],q[:,1],h-1-q[:,1]])<3).mean());near_parallel=False
            for j,tree in enumerate(other_trees):
                if j!=qi and float((tree.query(q)[0]<4).mean())>.55:near_parallel=True;break
            arrow_fraction=float(((q[:,0]<110)&(q[:,1]<85)).mean()) if identity=='sina_6586334590_14' else 0.
            if length>=28 and not closed and support>=.9 and stable>=.85 and visible>=.6 and border_fraction<.1 and not near_parallel and arrow_fraction<.15:accepted.append(q)
            else:reasons.append(dict(length=float(length),closed=bool(closed),support=support,transformed_support=stable,visible_ink_fraction=visible,border_fraction=border_fraction,possible_double_edge_or_close_neighbor=near_parallel,annotation_arrow_fraction=arrow_fraction))
        family_holdout=identity.startswith('kuhw_29483_') or identity.endswith(('_27','_28'));draft=ROOT/'auto_labels'/(identity+'.json');jsave(draft,dict(image=r['image'],content_shape=[h,w],strokes=[dict(points=q.round(2).tolist(),source='model_consistency_pseudo_not_expert') for q in accepted],rejected_candidates=reasons,expert_verified=False,robot_ready=False))
        if accepted and not family_holdout:
            generated=examples(accepted,p[0],rgb,156000+len(manifest),6);new_pairs=[x for x in generated if x[2]==1];pseudo+=new_pairs;pseudo_sources += [identity]*len(new_pairs)
        # Crops are counted as crops, never independent paintings.
        image_path=ROOT/'curated_images'/(identity+'.png');image_path.parent.mkdir(exist_ok=True);Image.fromarray(rgb).save(image_path)
        for k,box in enumerate([(0,0,w,h),(0,0,max(90,int(w*.7)),h),(max(0,int(w*.3)),0,w,h),(0,0,w,max(90,int(h*.7))),(0,max(0,int(h*.3)),w,h),(int(w*.15),int(h*.15),int(w*.85),int(h*.85))]):
            dest=ROOT/'crop_proposals'/(identity+'_'+str(k)+'.png');Image.fromarray(rgb).crop(box).save(dest);crops.append(dict(id=dest.stem,image=str(dest),source_id=identity,source_family=r['source_group'],box=list(box),split='development' if family_holdout else 'train',annotation='unlabeled_crop_not_additional_artwork'))
        manifest.append(dict(**r,content_image=str(image_path),pseudo_file=str(draft),candidate_paths=len(raw),accepted_paths=len(accepted),quarantined_paths=len(reasons),split='development' if family_holdout else 'train',independent_artwork_not_verified=True));panels.append((identity,rgb,line_view(rgb.shape[:2],accepted,True)));print('AUTO_QUALITY_GATE',identity,len(raw),len(accepted),'heldout',family_holdout,flush=True)
    if not pseudo:raise RuntimeError('No pseudo labels passed; do not silently pretend self-training occurred')
    pseudo=pseudo[:1600];save_pairs(ROOT/'pseudo_pairs.npz',pseudo);jsave(ROOT/'pseudo_pair_sources.json',pseudo_sources[:1600]);jsave(ROOT/'curated_manifest.json',dict(records=manifest,crops=crops,raw_image_count=len(manifest),crop_count=len(crops),source_pages=len(set(r['source_page'] for r in manifest)),accepted_auto_paths=sum(r['accepted_paths'] for r in manifest),pseudo_training_pairs=len(pseudo),pseudo_negatives=0,pseudo_are_not_expert_labels=True,heldout_sources_excluded=True,gate_version=2,border_double_edge_arrow_quarantined=True))
    board=Image.new('RGB',(1500,225*math.ceil(len(panels)/3)),'#f5f3ec');d=ImageDraw.Draw(board)
    for i,(identity,rgb,lines) in enumerate(panels):
        x=(i%3)*500;y=(i//3)*225;d.text((x+5,y+3),identity+' 自动入选候选（非金标准）',font=SMALL,fill='black');paste(board,rgb,x+4,y+30,240,186);paste(board,lines,x+252,y+30,240,186)
    board.save(ROOT/'auto_quality_contact.jpg',quality=93);print('AUTO_DATA_COMPLETE',len(rows),len(pseudo),len(manifest),len(crops),flush=True)

if __name__=='__main__':build()
