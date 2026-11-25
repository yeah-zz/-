# -*- coding: utf-8 -*-
"""Calibrate the combined source guard + context threshold on calibration only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
import json,torch,cv2
from context_curve_data_20261105 import ROOT,source_records,write
from context_curve_apply_20261105 import load,select,metrics

def choose():
    model,raw_threshold=load();records=source_records();bank=[]
    for r in records:
        if r['split']!='calibration':continue
        rows=[dict(points=p,module=0,source_curve_ids=[i]) for i,p in enumerate(r['refs'])]
        result=select(model,0.,r['rgb'],rows)
        actual=None
        if r['paths']:
            actual=select(model,0.,r['rgb'],[dict(points=p,module=0,source_curve_ids=[i]) for i,p in enumerate(r['paths'])])
        bank.append((r,result,actual))
    choices=[]
    for t in [.5,.6,.7,.8,.9,.95,.97,raw_threshold]:
        teacher_removed=0;coverage_drop=0.
        for r,ref,actual in bank:
            teacher_removed+=sum(d['noise_probability']>=t and not d['source_contour_or_clear_texture_guard'] for d in ref['decisions'])
            if actual:
                keep=[i for i,d in enumerate(actual['decisions']) if d['noise_probability']<t or d['source_contour_or_clear_texture_guard']]
                a=metrics(r['paths'],r['refs'],r['background']);b=metrics([r['paths'][i] for i in keep],r['refs'],r['background'])
                coverage_drop=max(coverage_drop,max(x['covered_fraction']-y['covered_fraction'] for x,y in zip(a['reference_details'],b['reference_details'])))
        choices.append(dict(threshold=t,teacher_removed=teacher_removed,max_reference_coverage_drop=coverage_drop,
            passes=teacher_removed==0 and coverage_drop<=.01))
    valid=[r['threshold'] for r in choices if r['passes']];threshold=min(valid) if valid else raw_threshold
    # First use of test-image full teacher curves for this joint policy occurs
    # after the selected threshold is fixed; no threshold is fitted to them.
    tests=[]
    for r in records:
        if r['split']!='test':continue
        rows=[dict(points=p,module=0,source_curve_ids=[i]) for i,p in enumerate(r['refs'])]
        result=select(model,threshold,r['rgb'],rows)
        tests.append(dict(id=r['id'],positive_curves=len(rows),removed=len(result['rejected'])))
    report=dict(raw_network_threshold=raw_threshold,joint_threshold=threshold,calibration_choices=choices,
        calibration_ids=[r['id'] for r,_,_ in bank],test_teacher_curves=tests,
        test_teacher_preservation_passed=all(r['removed']==0 for r in tests),
        threshold_selected_without_test_labels=True,source_families_overlap=True,
        independent_real_painting_accuracy=False)
    write(ROOT/'joint_calibration.json',report);print('JOINT_CALIBRATION',json.dumps(report),flush=True)
    return report

if __name__=='__main__':torch.set_num_threads(4);cv2.setNumThreads(2);choose()
