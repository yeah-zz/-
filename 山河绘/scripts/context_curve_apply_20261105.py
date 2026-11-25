# -*- coding: utf-8 -*-
"""Candidate inference is source-only. Reference audit is an explicit separate step."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json
import cv2,numpy as np,torch
from scipy.spatial import cKDTree
from context_curve_model_20261105 import ContextCurveNet,pack,predict,raster
from context_curve_data_20261105 import ROOT,source_records,candidate_label
from source_evidence_filter_20261104 import filter_curves
from module_stroke_core_20261104 import samples,length

def scores(model,rgb,paths):
    # Count curves, not repeated vertices. Remove the queried curve's own
    # contribution while retaining genuinely intersecting neighbors.
    counts=np.zeros(rgb.shape[:2],np.uint16)
    for p in paths:counts+=(raster(rgb.shape[:2],[p])>0).astype(np.uint16)
    x=[]
    for p in paths:
        own=raster(rgb.shape[:2],[p])>0
        others=np.where(counts-own.astype(np.uint16)>0,255,0).astype(np.uint8)
        x.append(pack(rgb,p,others))
    return predict(model,np.stack(x))

def load():
    d=json.loads((ROOT/'training_summary.json').read_text('utf8'))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ContextCurveNet().to(device);saved=torch.load(ROOT/'model.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model']);model.eval();return model,float(d['threshold'])

def select(model,threshold,rgb,rows,old_decisions=None):
    paths=[np.asarray(r['points'],float) for r in rows]
    p=scores(model,rgb,paths)
    source=old_decisions if old_decisions is not None else filter_curves(rgb,rows)['decisions']
    decisions=[]
    for i,(prob,d) in enumerate(zip(p,source)):
        m=d['metrics'];guard=bool(d['contour_guard'] or (d['texture_guard'] and m['ridge_support']>=.5 and m['directional_support']>=.55)
            or (m['ridge_support']>=.65 and m['directional_support']>=.55))
        requested=bool(prob[1]>=threshold)
        decisions.append(dict(curve=i,noise_probability=float(prob[1]),visible_probability=float(prob[0]),
            neural_reject_requested=requested,source_contour_or_clear_texture_guard=guard,
            disposition='reject' if requested and not guard else 'keep',
            reason='high context-background score' if requested and not guard else 'protected source contour/texture' if requested else 'retained',
            artist_stroke_identity_verified=False))
    selected=[d['curve'] for d in decisions if d['disposition']=='keep']
    rejected=[d['curve'] for d in decisions if d['disposition']=='reject']
    return dict(selected=selected,rejected=rejected,decisions=decisions,threshold=threshold,
        contour_guards_restored=sum(d['neural_reject_requested'] and d['source_contour_or_clear_texture_guard'] for d in decisions),
        annotations_read_at_inference=False,selected_geometry_unchanged=True,ready_for_robot=False)

def metrics(paths,refs,background,tolerance=2.):
    points=[samples(p,.5) for p in paths];tree=cKDTree(np.concatenate(points)) if points else None
    refdetails=[]
    for i,p in enumerate(refs):
        q=samples(p,.5);dist=tree.query(q)[0] if tree else np.full(len(q),np.inf)
        hit=dist<=tolerance;step=length(q)/max(1,len(q)-1);runs=[];n=0
        for v in hit:
            if not v:n+=1
            else:
                if n:runs.append(n*step)
                n=0
        if n:runs.append(n*step)
        refdetails.append(dict(reference=i,covered_fraction=float(hit.mean()),longest_missing_run_px=float(max(runs,default=0))))
    known_bg_count=0;known_bg_length=0.;known_bg_arc=0.;unknown=0;matched=0;multi_ref=0
    ref_trees=[cKDTree(samples(p,.5)) for p in refs]
    for p,q in zip(paths,points):
        xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,background.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,background.shape[0]-1)
        in_bg=background[xy[:,1],xy[:,0]]>0
        if ref_trees:
            near_ref=np.stack([t.query(q)[0] for t in ref_trees]).min(0)<=tolerance
            in_bg &= ~near_ref
        bg_weight=in_bg.astype(float)
        if len(q)>1:known_bg_arc+=float((np.linalg.norm(np.diff(q,axis=0),axis=1)*(bg_weight[:-1]+bg_weight[1:])/2).sum())
        label,_=candidate_label(p,refs,background,tolerance)
        if label==1:known_bg_count+=1;known_bg_length+=length(np.asarray(p))
        elif label==0:matched+=1
        else:unknown+=1
        contacts=sum((t.query(q)[0]<=tolerance).mean()>.15 for t in ref_trees)
        if contacts>=2:multi_ref+=1
    return dict(prediction_curves=len(paths),reference_curves=len(refs),
        mean_reference_coverage=float(np.mean([r['covered_fraction'] for r in refdetails])),
        reference_curves_80pct_covered=sum(r['covered_fraction']>=.8 for r in refdetails),
        known_background_prediction_curves=known_bg_count,known_background_prediction_length_px=known_bg_length,
        trace_length_inside_explicit_background_px=known_bg_arc,
        matched_reference_prediction_curves=matched,unmatched_unknown_curves=unknown,
        multi_reference_contacts_not_false_join_truth=multi_ref,reference_details=refdetails,
        tolerance_native_px=tolerance,unmarked_pixels_not_scored=True)

def audit(model,threshold):
    results=[]
    for record in source_records():
        if not record['prediction_file']:continue
        rows=[dict(points=p,source_curve_ids=[i],module=0) for i,p in enumerate(record['paths'])]
        # Labels are not passed into this inference call.
        candidate=select(model,threshold,record['rgb'],rows)
        before=metrics(record['paths'],record['refs'],record['background'])
        after=metrics([record['paths'][i] for i in candidate['selected']],record['refs'],record['background'])
        drop=max(a['covered_fraction']-b['covered_fraction'] for a,b in zip(before['reference_details'],after['reference_details']))
        results.append(dict(id=record['id'],split=record['split'],before=before,after=after,max_per_reference_coverage_drop=drop,
            passes_coverage_guard=drop<=.01 and after['known_background_prediction_length_px']<=before['known_background_prediction_length_px']+1e-5,
            prediction_file=record['prediction_file'],source_image=record['image'],labels=record['labels'],
            predictions_source_pixels_exactly_verified=True,selection=candidate))
    result=dict(records=results,passes_all_actual_output_coverage_guards=all(r['passes_coverage_guard'] for r in results),
        independent_real_painting_accuracy=False,source_families_overlap=True,
        evaluation_on_actual_archived_prediction_not_teacher_input=True,robot_connected=False)
    (ROOT/'actual_prediction_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('ACTUAL_AUDIT',[(r['id'],r['before']['mean_reference_coverage'],r['after']['mean_reference_coverage'],r['before']['known_background_prediction_curves'],r['after']['known_background_prediction_curves'],r['passes_coverage_guard']) for r in results],flush=True)
    return result

if __name__=='__main__':
    torch.set_num_threads(4);cv2.setNumThreads(2);m,t=load();audit(m,t)
