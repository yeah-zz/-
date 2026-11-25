# -*- coding: utf-8 -*-
"""Reviewed positive curves and explicitly marked background only.

Source families may overlap across R01-R04. This is a regression check, not
independent full-painting extraction accuracy and not a fitting procedure.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib
from collections import Counter
import cv2,numpy as np
from source_evidence_filter_20261104 import filter_curves,CONFIG
from module_stroke_core_20261104 import samples

ROOT=LAB/'source_evidence_selection_20261104'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evaluate():
    ROOT.mkdir(exist_ok=True);records=json.loads((LAB/'reviewed_rocks_v14_20261024/corrected/manifest.json').read_text('utf8'))['records'];out=[]
    for r in records:
        if r['id'] not in ['R01','R02','R03','R04']:continue
        doc=json.loads(Path(r['labels']).read_text('utf8'));rgb=cv2.cvtColor(cv2.imread(r['image']),cv2.COLOR_BGR2RGB)
        positive=[s for s in doc['strokes'] if s['id'].startswith('s_') and len(s['points'])>=2]
        paths=[dict(points=s['points'],source_curve_ids=[i],module=0) for i,s in enumerate(positive)]
        result=filter_curves(rgb,paths)
        background=np.zeros(rgb.shape[:2],np.uint8)
        for region in doc['regions']:
            if region['kind']=='background':cv2.fillPoly(background,[np.rint(region['points']).astype(np.int32)],1)
        negative=[];rng=np.random.default_rng(11040500+int(r['id'][1:]));coords=np.argwhere(background>0)
        for _ in range(3000):
            if len(negative)>=60 or not len(coords):break
            y,x=coords[rng.integers(len(coords))];angle=rng.uniform(-np.pi,np.pi);arc=rng.uniform(3,12)
            p=np.array([[x,y],[x+arc*np.cos(angle),y+arc*np.sin(angle)]])
            q=np.rint(samples(p,.3)).astype(int)
            if (q.min(0)<0).any() or (q.max(0)>=[rgb.shape[1],rgb.shape[0]]).any():continue
            if not background[q[:,1],q[:,0]].all():continue
            negative.append(dict(points=p,source_curve_ids=[len(negative)],module=0))
        nr=filter_curves(rgb,negative) if negative else dict(removed=[],decisions=[])
        removed=[dict(stroke_id=positive[i]['id'],decision=result['decisions'][i]) for i in result['removed']]
        out.append(dict(id=r['id'],positive_user_curves=len(paths),positive_kept=len(result['selected']),positive_removed=removed,
            positive_dispositions=dict(Counter(x['disposition'] for x in result['decisions'])),
            explicit_background_probe_curves=len(negative),background_probe_removed=len(nr['removed']),
            background_probes_are_generated_tests_not_detector_predictions=True,
            image_sha256=sha(r['image']),labels_sha256=sha(r['labels']),same_source_family=r.get('source_group'),
            labels_not_read_at_inference=True,unmarked_pixels_not_scored=True))
    # Whole landscape traces: only files explicitly marked checked. A draft or
    # missing label is not silently treated as confirmed user supervision.
    fullroot=LAB/'user_review_v11_20261024'
    package=json.loads((fullroot/'package_manifest.json').read_text('utf8'));skipped=[]
    for row in package['samples']:
        label=fullroot/'labels'/(row['id']+'.json')
        if not label.exists():continue
        doc=json.loads(label.read_text('utf8'))
        if doc['review_state']!='checked_partial':skipped.append(dict(id=row['id'],reason=doc['review_state']));continue
        positive=[s for s in doc['strokes'] if s['id'].startswith('s_') and len(s['points'])>=2]
        if not positive:continue
        image=fullroot/row['image'];assert sha(image)==row['image_sha256']
        rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB);assert rgb.shape[:2]==(doc['height'],doc['width'])
        rows=[dict(points=s['points'],source_curve_ids=[i],module=0) for i,s in enumerate(positive)]
        result=filter_curves(rgb,rows)
        removed=[dict(stroke_id=positive[i]['id'],decision=result['decisions'][i]) for i in result['removed']]
        out.append(dict(id=row['id'],positive_user_curves=len(rows),positive_kept=len(result['selected']),positive_removed=removed,
            positive_dispositions=dict(Counter(x['disposition'] for x in result['decisions'])),
            explicit_background_probe_curves=0,background_probe_removed=0,
            image_sha256=sha(image),labels_sha256=sha(label),same_source_family=row['source_group'],
            labels_not_read_at_inference=True,unmarked_pixels_not_scored=True))
    total=sum(r['positive_user_curves'] for r in out);kept=sum(r['positive_kept'] for r in out)
    report=dict(records=out,configuration=CONFIG,user_positive_curves=total,user_positive_kept=kept,
        preservation_regression_passed=total==kept,source_families_may_overlap=True,
        independent_real_painting_accuracy=False,automatic_old_predictions_not_gold=True,
        test_only_no_parameter_fitting=True,unconfirmed_files_skipped=skipped,robot_connected=False)
    (ROOT/'reference_regression.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True);return report
if __name__=='__main__':evaluate()
