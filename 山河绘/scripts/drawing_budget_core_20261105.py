# -*- coding: utf-8 -*-
"""Nested drawing-detail budgets. Never relabel omitted input as false ink.

No stroke is shortened, moved, fused or straightened here. Complete extraction
is kept separately. Structural roles are inherited heuristics, not expert truth.
"""
from collections import Counter
import math
import numpy as np
from scipy.spatial import cKDTree
from module_stroke_core_20261104 import samples,length

def features(curves):
    out=[]
    for r in curves:
        p=np.asarray(r['points'],float);q=samples(p,.7);arc=length(p)
        v=p[-1]-p[0];angle=math.atan2(v[1],v[0])%math.pi
        # Spatial bins are only a coverage proxy, not ground truth or semantics.
        cells={tuple(t) for t in np.floor(q/2).astype(int)}
        out.append(dict(arc=arc,cells=cells,angle_bin=int(angle/(math.pi/6))%6,
            module=r['module'],phase=r['phase'],support=float(r.get('source_support',.5)),
            information=(max(arc,.1)**.6)*(.7+.3*float(r.get('source_support',.5)))))
    return out

def select_budgets(curves,budgets=(300,500,700)):
    if not curves:raise ValueError('No curves')
    if any(isinstance(b,bool) or not isinstance(b,int) or b<1 for b in budgets):raise ValueError('Positive integer budgets required')
    f=features(curves);n=len(curves)
    protected={i for i,r in enumerate(curves) if r['phase']==0 or (r['phase']==1 and f[i]['arc']>=25)}
    modules=sorted({r['module'] for r in curves})
    # Preserve at least one strongest complete path per work region.
    for m in modules:
        ids=[i for i in range(n) if f[i]['module']==m]
        if not any(i in protected for i in ids):protected.add(max(ids,key=lambda i:f[i]['information']))
    info={m:sum(f[i]['information'] for i in range(n) if f[i]['module']==m) for m in modules}
    total=sum(info.values());counts=Counter(f[i]['module'] for i in protected)
    covered=set().union(*(f[i]['cells'] for i in protected)) if protected else set()
    orientations=Counter((f[i]['module'],f[i]['angle_bin']) for i in protected)
    # Endpoint relations influence retention only, never add a connecting path.
    ends=np.array([np.asarray(r['points'],float)[[0,-1]] for r in curves]).reshape(-1,2)
    neighbors=[set() for _ in curves]
    for a,b in cKDTree(ends).query_pairs(1.5):
        i,j=a//2,b//2
        if i!=j and f[i]['module']==f[j]['module']:neighbors[i].add(j);neighbors[j].add(i)
    selected=set(protected);ranking=sorted(protected);target=min(n,max(max(budgets),len(protected)))
    fine={i for i,a in enumerate(f) if a['phase']==2 and .8<=a['arc']<5 and a['support']>=.35}
    reserved_fine=0
    while len(selected)<target:
        def score(i):
            a=f[i];m=a['module'];novel=len(a['cells']-covered)/max(1,len(a['cells']))
            share=max(1,info[m]/total*target);fairness=np.clip((share+2)/(counts[m]+2),.45,2.5)**.65
            diversity=1+.35/(1+orientations[m,a['angle_bin']])
            connection=1+.18*min(2,len(neighbors[i]&selected))
            role=1.35 if a['phase']==1 else 1.
            return a['information']*(.25+.75*novel)*fairness*diversity*connection*role
        # A short needle or architectural cue can be necessary. Reserve a small
        # share of discretionary slots for source-supported fine details rather
        # than letting length ranking remove them all. Still no semantic claim.
        eligible=fine-selected if reserved_fine<math.floor((len(selected)-len(protected)+1)*.20) else set()
        i=max(eligible or (set(range(n))-selected),key=lambda i:(score(i),-i))
        if i in fine:reserved_fine+=1
        ranking.append(i);selected.add(i);covered.update(f[i]['cells']);counts[f[i]['module']]+=1
        orientations[f[i]['module'],f[i]['angle_bin']]+=1
    outputs={}
    for budget in sorted(set(budgets)):
        count=min(n,max(budget,len(protected)));chosen=set(ranking[:count])
        assert protected<=chosen
        outputs[budget]=dict(selected_ids=sorted(chosen),omitted_ids=sorted(set(range(n))-chosen),
            requested_budget=budget,actual_count=len(chosen),protected_ids=sorted(protected),
            protected_count=len(protected),budget_exceeded_for_structure=len(protected)>budget,
            budget_not_filled_due_to_input_size=n<budget,
            retained_short_detail_candidates=len(chosen&fine),
            reason='drawing detail budget; omitted paths are not classified as false extraction',
            selection_is_expert_verified=False)
    return outputs
