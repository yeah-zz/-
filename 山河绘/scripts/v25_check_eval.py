# -*- coding: utf-8 -*-
from v25_references import ROOT,V24,dense
from rock_v15_common import read,jsave,digest
import numpy as np

def interval_error(value,limits):return float(max(limits[0]-value,value-limits[1],0))

def references():
    frozen=read(ROOT/'raw_checks'/'assistant_intervals.json');loc=read(ROOT/'raw_checks'/'locations.json')['records'];rows=[]
    for r in loc:
        if r['id'] not in frozen['intervals']:continue
        s=read(ROOT/'refined'/(r['image_id']+'.json'))['strokes'][r['stroke_index']];i=r['point_index'];q=np.asarray(s['points'][i]);offset=float((q-np.asarray(r['p']))@np.asarray(r['n']));accepted=bool(s['confidence'][i]>.5);used=offset if accepted else 0.;limits=frozen['intervals'][r['id']];rows.append(dict(id=r['id'],image_id=r['image_id'],interval=limits,raw_offset=0.,corrected_candidate_offset=offset,accepted=accepted,used_offset=used,before_error=interval_error(0,limits),after_error=interval_error(used,limits)))
    summary=dict(count=len(rows),uncertain_excluded=len(frozen['uncertain']),accepted=sum(r['accepted'] for r in rows),before_mean_interval_error=float(np.mean([r['before_error'] for r in rows])),after_mean_interval_error=float(np.mean([r['after_error'] for r in rows])),before_mean_error_same_accepted_subset=float(np.mean([r['before_error'] for r in rows if r['accepted']])),after_mean_error_same_accepted_subset=float(np.mean([r['after_error'] for r in rows if r['accepted']])),assistant_ranges_not_expert_true_centers=True,intervals_not_used_for_refinement_parameters=True)
    jsave(ROOT/'reference_check_results.json',dict(summary=summary,records=rows,interval_file_sha256=digest(ROOT/'raw_checks'/'assistant_intervals.json')));print('V25_REFERENCE_CHECK',summary,flush=True)

def trajectories():
    frozen=read(ROOT/'raw_checks'/'assistant_intervals.json');loc=read(ROOT/'raw_checks'/'locations.json')['records'];reports={}
    for variant in ['parent','coarse','corrected']:
        rows=[]
        for r in loc:
            if r['id'] not in frozen['intervals']:continue
            file=V24/'evaluation'/'parent'/r['image_id']/'trajectory.json' if variant=='parent' else ROOT/'evaluation'/variant/r['image_id']/'trajectory.json';paths=read(file)['paths'];points=np.concatenate([dense(p) for p in paths if len(p)>1]) if paths else np.zeros((0,2));delta=points-np.asarray(r['p']);normal=delta@np.asarray(r['n']);along=np.abs(delta@np.asarray(r['t']));ok=(along<=2)&(np.abs(normal)<=12);limits=frozen['intervals'][r['id']]
            if ok.any():
                ids=np.flatnonzero(ok);i=ids[np.argmin(np.linalg.norm(delta[ids],axis=1))];offset=float(normal[i]);error=interval_error(offset,limits)
            else:offset=None;error=12.
            rows.append(dict(id=r['id'],image_id=r['image_id'],detected=offset is not None,offset=offset,error_with_miss_penalty=error,interval=limits))
        reports[variant]=dict(count=len(rows),detected=sum(r['detected'] for r in rows),mean_error_with_12px_miss_penalty=float(np.mean([r['error_with_miss_penalty'] for r in rows])),conditional_interval_error=float(np.mean([r['error_with_miss_penalty'] for r in rows if r['detected']])) if any(r['detected'] for r in rows) else None,records=rows)
    jsave(ROOT/'trajectory_check_results.json',dict(reports=reports,small_assistant_checkset_not_expert_accuracy=True,missing_sections_penalized=True));print('V25_TRAJECTORY_CHECK',{k:{kk:vv for kk,vv in v.items() if kk!='records'} for k,v in reports.items()},flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['references','trajectories']);a=p.parse_args();{'references':references,'trajectories':trajectories}[a.mode]()
