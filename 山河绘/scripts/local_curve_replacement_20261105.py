# -*- coding: utf-8 -*-
"""Whole-candidate replacement guard. Never clips a neural proposal into pieces."""
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from module_stroke_core_20261104 import samples,length
from source_evidence_filter_20261104 import fields,evidence

def tangents(q):
    q=gaussian_filter1d(q,.8,axis=0,mode='nearest');t=np.gradient(q,axis=0)
    return t/np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-8)

def whole_guard(rgb,rows,proposals):
    ff=fields(rgb);paths=[np.asarray(r['points'],float) for r in rows]
    sampled=[samples(p,.4) for p in paths];trees=[cKDTree(q) for q in sampled]
    tt=[tangents(q) for q in sampled];decisions=[]
    for index,r in enumerate(proposals):
        p=np.asarray(r['points'],float);arc=length(p) if len(p)>1 else 0.
        row=dict(proposal=index,accepted_candidate=False,covered_ids=[],reason='too_short',length_px=arc)
        if len(p)<2 or arc<12:decisions.append(row);continue
        if r.get('stop_forward') in ['step_limit','loop_guard'] or r.get('stop_backward') in ['step_limit','loop_guard']:
            row['reason']='incomplete_or_looping_neural_trace';decisions.append(row);continue
        ev=evidence(p,ff)
        row['source_evidence']={k:v for k,v in ev.items() if k not in ['points','tangents']}
        if ev['near_ink_support']<.95 or ev['ridge_support']<.75 or ev['directional_support']<.70:
            row['reason']='whole_curve_source_evidence_failed';decisions.append(row);continue
        q=samples(p,.4);tree=cKDTree(q);t=tangents(q);lo=q.min(0)-1.3;hi=q.max(0)+1.3
        covered=[]
        for i,old in enumerate(sampled):
            if np.linalg.norm(paths[i][0]-paths[i][-1])<.1:continue
            if (old.min(0)<lo).any() or (old.max(0)>hi).any():continue
            dist,ix=tree.query(old)
            if dist.max()>1.25 or (dist<=1.).mean()<.97:continue
            agreement=np.abs((tt[i]*t[ix]).sum(1))
            if (agreement>.8).mean()<.85:continue
            covered.append(i)
        row['covered_ids']=covered
        if len(covered)<2:row['reason']='does_not_replace_two_whole_fragments';decisions.append(row);continue
        if len({rows[i]['module'] for i in covered})!=1:
            row['reason']='cross_module_candidate';decisions.append(row);continue
        union=cKDTree(np.concatenate([sampled[i] for i in covered]));dd=union.query(q)[0]
        total=sum(length(paths[i]) for i in covered)
        if (dd<=1.5).mean()<.9 or dd.max()>3 or arc>total*1.15+3:
            row['reason']='large_unmatched_extension';decisions.append(row);continue
        row.update(accepted_candidate=True,reason='whole_source_supported_replacement',max_unmatched_distance_px=float(dd.max()))
        decisions.append(row)
    chosen=[];used=set();output=[]
    for d in sorted(decisions,key=lambda d:(-len(d['covered_ids']),-d['length_px'])):
        if not d['accepted_candidate'] or used.intersection(d['covered_ids']):continue
        ids=d['covered_ids'];used.update(ids);chosen.append(d['proposal'])
        phase=min(rows[i]['phase'] for i in ids);inherit=next(rows[i] for i in ids if rows[i]['phase']==phase)
        output.append(dict(points=np.asarray(proposals[d['proposal']]['points'],float),module=inherit['module'],phase=phase,
            phase_label=inherit.get('phase_label',''),source_curve_ids=[j for i in ids for j in rows[i]['source_curve_ids']],
            baseline_curve_ids=ids,neural_proposal_id=d['proposal'],role_verified=False,
            uncertain_module=any(rows[i].get('uncertain_module',False) for i in ids),
            length_px=d['length_px'],source_support=float(d['source_evidence']['near_ink_support']),
            artist_same_stroke_verified=False))
    for i,r in enumerate(rows):
        if i not in used:output.append({**r,'points':np.asarray(r['points'],float),'baseline_curve_ids':[i]})
    assert sorted(i for r in output for i in r['baseline_curve_ids'])==list(range(len(rows)))
    return output,dict(selected_proposals=chosen,replaced_input_ids=sorted(used),decisions=decisions,
        reasons=dict(Counter(d['reason'] for d in decisions)),no_pointwise_splitting=True,
        original_input_geometry_accounted_for=True,expert_accuracy_verified=False)
