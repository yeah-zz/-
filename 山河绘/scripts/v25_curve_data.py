# -*- coding: utf-8 -*-
"""Matched coarse/corrected targets at the same image-generated anchors."""
from v25_curve_common import *
from v25_references import dense,sample
from scipy.spatial import cKDTree
import torch

V24=LAB/'joint_center_affinity_v24_20261026'
V23=LAB/'junction_relations_v23_20261026'

def excluded_sources():
    result=[]
    for r in read(LAB/'autonomous_review_v19_20261025'/'manifest_v2.json')['records']:
        z=np.load(r['data'])
        if 'uncertain_exclusion' in z and (z['uncertain_exclusion']>0).any():result.append(Path(r['image']).stem)
    return result

def parent_predictions():
    folder=ROOT/'parent_predictions';folder.mkdir(exist_ok=True);model=None
    for r in read(ROOT/'initial_manifest.json')['records']:
        dest=folder/(r['id']+'.npz')
        if dest.exists():continue
        options=[V24/'predictions'/'parent'/(r['id']+'.npz'),V23/'predictions'/(r['id']+'.npz')];cached=next((p for p in options if p.exists()),None)
        if cached:p=np.load(cached)['prediction']
        else:
            if model is None:model=v21.load_model('ribbon_improved')
            p=v21.v16.predict(model,np.load(r['input'])['rgb'])
        np.savez_compressed(dest,prediction=p);print('V25_PARENT',r['id'],flush=True)

def prepare():
    if any(ROOT.glob('*/final.pt')):raise FileExistsError('Preserve frozen curve targets after training')
    (ROOT/'curve_data').mkdir(exist_ok=True);rows=[];patches=[];coarse=[];corrected=[];weights=[];labels=[];seeds=[];angles=[];ids=[];splits=[];rng=np.random.default_rng(252610);excluded=excluded_sources();details=[]
    for r in read(ROOT/'initial_manifest.json')['records']:
        z=np.load(r['input']);rgb=z['rgb'];features,alpha,theta=axes_from_image(rgb);pred=np.load(ROOT/'parent_predictions'/(r['id']+'.npz'))['prediction'];anchors=automatic_seeds(pred);strokes=read(ROOT/'refined'/(r['id']+'.json'))['strokes'];paths=[np.asarray(s['points'],np.float32) for s in strokes];trees=[cKDTree(p) for p in paths];positive=negative=0;split='excluded_user_uncertainty_source' if r['id'] in excluded and r['split']=='train' else r['split']
        for anchor in anchors:
            distances=np.array([t.query(anchor)[0] for t in trees]);order=np.argsort(distances);k=int(order[0])
            if distances[k]>3.5 or (len(order)>1 and distances[order[1]]-distances[k]<2.5):continue
            i=int(trees[k].query(anchor)[1]);p=paths[k];conf=np.asarray(strokes[k]['confidence'],np.float32)
            if i<17 or i+17>=len(p) or conf[i]<.5:continue
            sample_indices=np.arange(i-16,i+17,2);ww=conf[sample_indices]
            if ww.mean()<.8 or min(ww[0],ww[-1])<.5:continue
            q=p[sample_indices];p0=np.asarray(strokes[k]['coarse_points'],np.float32)[sample_indices]
            if (sample(z['valid'],q)<.9).any():continue
            th=float(theta[round(anchor[1]),round(anchor[0])]);ql=to_local(q,anchor,th);cl=to_local(p0,anchor,th)
            if ql[-1,0]<ql[0,0]:ql=ql[::-1].copy();cl=cl[::-1].copy();ww=ww[::-1].copy()
            if np.max(np.abs(ql))>.85 or np.max(np.abs(cl))>.85:continue
            patches.append(extract_patch(features,anchor,th).astype(np.float16));coarse.append(cl);corrected.append(ql);weights.append(ww);labels.append(1.);seeds.append(anchor);angles.append(th);ids.append(r['id']);splits.append(split);rows.append(dict(image_id=r['id'],split=split,work_group=r['work_group'],seed=anchor.tolist(),theta=th,label=1,stroke_index=k,point_index=i));positive+=1
        # Explicitly reviewed paper only; untraced ink is never a blank target.
        bg=(z['background']>0)&(z['valid']>0)&(alpha<.025);bg=cv2.erode(bg.astype(np.uint8),np.ones((9,9),np.uint8))>0;ys,xs=np.nonzero(bg)
        for _ in range(min(12,max(3,positive//3)) if len(xs) else 0):
            j=int(rng.integers(len(xs)));anchor=np.array([xs[j],ys[j]],np.float32);th=float(theta[ys[j],xs[j]]);patches.append(extract_patch(features,anchor,th).astype(np.float16));coarse.append(np.zeros((POINTS,2),np.float32));corrected.append(np.zeros((POINTS,2),np.float32));weights.append(np.zeros(POINTS,np.float32));labels.append(0.);seeds.append(anchor);angles.append(th);ids.append(r['id']);splits.append(split);rows.append(dict(image_id=r['id'],split=split,work_group=r['work_group'],seed=anchor.tolist(),theta=th,label=0));negative+=1
        details.append(dict(id=r['id'],split=split,automatic_seed_count=len(anchors),matched_positive_curves=positive,known_paper_negatives=negative));print('V25_CURVE_DATA',r['id'],split,len(anchors),positive,negative,flush=True)
    file=ROOT/'curve_data'/'patches.npz';np.savez_compressed(file,x=np.stack(patches),coarse=np.stack(coarse),corrected=np.stack(corrected),weights=np.stack(weights),label=np.array(labels,np.float32),seeds=np.stack(seeds),theta=np.array(angles,np.float32),image_ids=np.array(ids),split=np.array(splits));jsave(ROOT/'curve_data'/'manifest.json',dict(records=rows,per_image=details,data=str(file),data_sha256=digest(file),initial_manifest_sha256=digest(ROOT/'initial_manifest.json'),reference_manifest_sha256=digest(ROOT/'reference_manifest.json'),check_intervals_sha256=digest(ROOT/'raw_checks'/'assistant_intervals.json'),excluded_user_uncertain_source_ids=excluded,same_inputs_for_coarse_and_corrected=True,all_anchors_are_image_generated_not_reference_seeds=True,curve_points=POINTS,target_arc_span_pixels=SPAN,local_prototype_not_global_instance_detector=True));print('V25_CURVE_DATA_DONE',len(rows),'train',sum(s=='train' for s in splits),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['parent','prepare']);a=p.parse_args();{'parent':parent_predictions,'prepare':prepare}[a.mode]()
