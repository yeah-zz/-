# -*- coding: utf-8 -*-
"""Offline decoder threshold sweep for v21; model weights and source labels stay frozen."""
import sys
from pathlib import Path
SRC = Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\stroke_diagnosis_v21_20261025\source_snapshot')
sys.path.insert(0, str(SRC))
import numpy as np
from PIL import Image, ImageDraw
import v21_stroke_learning as v21

ROOT = v21.ROOT
OUT = ROOT / 'decoder_sweep_v22_20261025'
VAR = 'ribbon_improved'

def get_calibration(model):
    cases=[]
    for r in v21.read(ROOT/'manifest.json')['records']:
        if r['split']!='calibration':
            continue
        z=np.load(r['data']);doc=v21.read(r['labels'])
        pred=v21.v16.predict(model,z['rgb'])
        refs=[s['points'] for s in doc['strokes']]
        cases.append((r['id'],z['rgb'],refs,pred,np.ones(z['rgb'].shape[:2],np.float32)))
    return cases

def score_case(pred,refs,valid,kind,high,low):
    paths,sk,info=v21.prior_eval.decode(pred,kind,high,low)
    m=v21.path_metrics(refs,paths,valid)
    m['paths']=paths
    return m

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    model=v21.load_model(VAR)
    cal=get_calibration(model)
    cases=[]
    for r,rgb,refs,valid,bg,_ in v21.evaluation_cases():
        f=ROOT/VAR/'evaluation'/r['id']/'prediction.npz'
        if not f.exists():
            continue
        pred=np.load(f)['prediction']
        cases.append((r['id'],r['split'],rgb,refs,valid,bg,pred))
    policies=[]
    # The existing .65/.30 policy is the fixed comparator. Others test a
    # somewhat stronger seed and a higher low threshold to suppress weak clutter.
    for high in [.55,.65,.72,.80,.88]:
        for low in [.25,.30,.35,.40,.45,.50]:
            if low>=high: continue
            rows=[]
            for key,rgb,refs,pred,valid in cal:
                m=score_case(pred,refs,valid,'center',high,low)
                rows.append(dict(id=key,point=m['point_coverage_3px'],single=m['mean_best_single_path_coverage'],mixed=m['mixed_reference_paths_flagged'],fragments=m['mean_reference_fragments'],paths=m['path_count']))
            score=float(np.mean([.5*x['single']+.5*x['point']-.025*x['mixed'] for x in rows]))
            policies.append(dict(kind='center',high=high,low=low,calibration_score=score,calibration=rows))
    selected=max(policies,key=lambda x:x['calibration_score'])
    base=next(x for x in policies if x['high']==.65 and x['low']==.30)
    # Evaluate fixed comparator and selected threshold only on already frozen
    # predictions; no references are read by the decoder itself.
    eval_rows=[]
    ids={x[0]:x for x in cases}
    for key,split,rgb,refs,valid,bg,pred in cases:
        row=dict(id=key,split=split)
        for tag,p in [('baseline',base),('selected',selected)]:
            m=score_case(pred,refs,valid,'center',p['high'],p['low'])
            raster=v21.raster(m.pop('paths'),rgb.shape[:2])
            row[tag]=dict(point=m['point_coverage_3px'],single=m['mean_best_single_path_coverage'],mixed=m['mixed_reference_paths_flagged'],fragments=m['mean_reference_fragments'],path_count=m['path_count'],explicit_bg_fpr=float(raster[bg].mean()) if np.asarray(bg).any() else None)
        eval_rows.append(row)
    def group_stats(group):
        rr=[x for x in eval_rows if x['split']==group]
        if not rr:return None
        return {tag:{k:float(np.mean([x[tag][k] for x in rr if x[tag][k] is not None])) for k in ['point','single','mixed','fragments','path_count','explicit_bg_fpr'] if any(x[tag][k] is not None for x in rr)} for tag in ['baseline','selected']}
    # Visualize a representative set. Only colored lines are predicted paths.
    board=Image.new('RGB',(1500,330*min(8,len(cases))),'#f5f3ec');d=ImageDraw.Draw(board)
    preferred=['stage_065','stage_049','sina_6586334590_31','X01','X04','R04','stage_029','kuhw_29483_00']
    chosen=[ids[k] for k in preferred if k in ids][:8]
    for i,c in enumerate(chosen):
        key,split,rgb,refs,valid,bg,pred=c
        d.text((5,330*i+3),f'{key} | {split}',fill='black')
        for j,(tag,p) in enumerate([('原图',None),('固定阈值',base),('校准选择',selected)]):
            x=j*500+5;y=330*i+25
            if p is None: a=rgb
            else:
                m=score_case(pred,refs,valid,'center',p['high'],p['low']);a=v21.line_view(rgb.shape[:2],m['paths'],True)
            d.text((x,y),tag,fill='black');v21.paste(board,a,x,y+20,490,280)
    board.save(OUT/'DECODER_COMPARISON.jpg',quality=94)
    result=dict(experiment='threshold-only decode sweep',model_variant=VAR,weights_sha256=v21.digest(ROOT/VAR/'final.pt'),calibration_ids=[x[0] for x in cal],candidate_count=len(policies),baseline={k:base[k] for k in ['high','low','calibration_score']},selected={k:selected[k] for k in ['high','low','calibration_score']},selected_is_new_neural_training=False,reference_used_at_inference=False,records=eval_rows,group_averages={g:group_stats(g) for g in sorted(set(x['split'] for x in eval_rows))},promotion_gate_passed=False,notes=['Small correlated calibration set; selected threshold may overfit.','Development and X/R sets have been viewed before and are not blind tests.','No robot execution; paths remain image pixel coordinates.'])
    v21.jsave(OUT/'summary.json',result)
    print('V22_DECODER_SWEEP',result['baseline'],result['selected'],result['group_averages'],flush=True)

if __name__=='__main__':
    run()
