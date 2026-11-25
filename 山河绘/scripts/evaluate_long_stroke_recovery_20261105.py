# -*- coding: utf-8 -*-
"""Independent analytic geometry controls, not real-painting accuracy."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,math
from collections import Counter
import numpy as np,cv2
from long_stroke_recovery_20261105 import recover_long_strokes

ROOT=LAB/'long_stroke_recovery_20261105'

def evaluate(count=35,seed=6110507,filename='synthetic_controls.json'):
    rng=np.random.default_rng(seed);records=[]
    for kind in ['overlap','faint_gap','quarter_turn','white_gap','parallel','T_junction','ambiguous_Y']:
        for trial in range(count):
            if kind=='overlap':parts=[[[ -20,0],[-1,0]],[[-4,0],[20,0]]];source=[[[-20,0],[20,0]]];truth=[0,0]
            elif kind=='faint_gap':parts=[[[-20,0],[-2,0]],[[2,0],[20,0]]];source=[[[-20,0],[20,0]]];truth=[0,0]
            elif kind=='quarter_turn':
                parts=[[[-20,-6],[0,-6]],[[6,0],[6,20]]]
                arc=np.column_stack([6*np.cos(np.linspace(-math.pi/2,0,61)),6*np.sin(np.linspace(-math.pi/2,0,61))])
                source=[parts[0],arc,parts[1]];truth=[0,0]
            elif kind=='white_gap':parts=[[[-20,0],[-2,0]],[[2,0],[20,0]]];source=parts;truth=[0,1]
            elif kind=='parallel':parts=[[[-20,0],[18,0]],[[-18,2.5],[20,2.5]]];source=parts;truth=[0,1]
            elif kind=='T_junction':parts=[[[-20,0],[0,0]],[[0,0],[20,0]],[[0,0],[0,20]]];source=parts;truth=[0,0,1]
            else:parts=[[[-20,0],[0,0]],[[0,0],[20,-5]],[[0,0],[20,5]]];source=parts;truth=[0,1,2]
            angle=rng.uniform(-math.pi,math.pi);scale=rng.uniform(.85,1.15)
            rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])*scale
            offset=np.array([42.2,41.7])+rng.uniform(-1,1,2)
            transformed=lambda p:np.array(p,float)@rot.T+offset
            curves=[dict(points=transformed(p),module=0,uncertain_module=(trial%2==0)) for p in parts]
            # Draw analytic source at 4x, independently from connector code.
            high=np.full((336,336,3),250,np.uint8)
            shade=int(rng.integers(190,226)) if kind=='faint_gap' else int(rng.integers(15,85))
            for p in source:cv2.polylines(high,[np.rint(transformed(p)*4).astype(np.int32)],False,(shade,)*3,4,cv2.LINE_AA)
            rgb=cv2.resize(high,(84,84),interpolation=cv2.INTER_AREA)
            out,diag=recover_long_strokes(rgb,curves)
            wrong=sum(len({truth[i] for i in r['source_curve_ids']})>1 for r in out)
            expected_pairs=sum(truth[i]==truth[j] for i in range(len(truth)) for j in range(i+1,len(truth)))
            recovered=sum(truth[i]==truth[j] for r in out for k,i in enumerate(r['source_curve_ids']) for j in r['source_curve_ids'][k+1:])
            records.append(dict(kind=kind,trial=trial,expected_pairs=expected_pairs,recovered_pairs=recovered,
                wrong_merged_groups=wrong,accepted=len(diag['accepted'])))
    totals=dict(cases=len(records),wrong_merged_groups=sum(r['wrong_merged_groups'] for r in records),
        expected_pairs=sum(r['expected_pairs'] for r in records),recovered_pairs=sum(r['recovered_pairs'] for r in records))
    families={k:{name:sum(r[name] for r in records if r['kind']==k) for name in ['expected_pairs','recovered_pairs','wrong_merged_groups']}
              for k in sorted({r['kind'] for r in records})}
    report=dict(seed=seed,totals=totals,families=families,records=records,
        source='analytic synthetic controls only; no independent real-painting truth',
        geometric_randomization='rotation, subpixel translation, scale, ink shade',
        model_training=False,development_controls_not_blind_real_test=True,real_stroke_precision=None,real_stroke_recall=None)
    (ROOT/filename).write_text(json.dumps(report,indent=2),encoding='utf8')
    print('SYNTHETIC_ONLY',json.dumps(dict(totals=totals,families=families)),flush=True)
    return report
if __name__=='__main__':evaluate()
