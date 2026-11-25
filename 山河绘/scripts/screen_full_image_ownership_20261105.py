# -*- coding: utf-8 -*-
"""Frozen whole-curve background classifier applied to native RGB, separately.

No labels or accepted 500/700 paths enter screening. This is a disclosed
posthoc experiment prompted by observed full-image paper/wash false positives.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,copy,shutil,hashlib,time,gc,argparse
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from context_curve_model_20261105 import ContextCurveNet,pack,predict,raster
from source_evidence_filter_20261104 import fields,evidence
from hierarchical_paint_core_20261105 import hierarchy_schedule
from drawing_budget_core_20261105 import select_budgets
from full_image_ownership_core_20261105 import curve_metrics,compare_geometry
from build_full_image_ownership_20261105 import read,save,sha,draw,make_plan,INPUTS,SCENES,FONT,SMALL

RAW=LAB/'autonomous_full_image_20261105';ROOT=LAB/'autonomous_full_image_screened_20261105'
MODEL=LAB/'context_curve_selection_20261105';BASE=LAB/'drawing_budgets_20261105'

def score_rows(model,rgb,rows,factor):
    paths=[np.asarray(r['points'],float)*factor for r in rows]
    counts=np.zeros(rgb.shape[:2],np.uint16)
    for p in paths:counts+=(raster(rgb.shape[:2],[p])>0).astype(np.uint16)
    out=[]
    for start in range(0,len(paths),64):
        group=[]
        for p in paths[start:start+64]:
            own=raster(rgb.shape[:2],[p])>0
            others=np.where(counts-own.astype(np.uint16)>0,255,0).astype(np.uint8)
            group.append(pack(rgb,p,others))
        out.extend(predict(model,np.stack(group)).tolist())
    return np.asarray(out),paths

def reconstruct_modules(plan):
    out=[]
    for m in plan['modules']:
        mask=np.zeros((496,496),np.uint8)
        for p in m['outlines']:cv2.fillPoly(mask,[np.rint(p).astype(np.int32)],1)
        out.append({**m,'mask':mask>0})
    return out

def run(ident,model,threshold):
    dest=ROOT/ident;dest.mkdir(exist_ok=True);old=read(RAW/ident/'plan.json');rgb=np.array(Image.open(INPUTS[ident]).convert('RGB'))
    factor=max(rgb.shape[:2])/496.;rows=old['curves'];t=time.monotonic()
    cached=dest/'raw_context_scores.npz'
    if cached.exists():
        z=np.load(cached);prob=z['scores'];assert len(prob)==len(rows)
        paths=[np.asarray(r['points'],float)*factor for r in rows]
    else:
        prob,paths=score_rows(model,rgb,rows,factor);np.savez_compressed(cached,scores=prob)
    f=fields(rgb);decisions=[];keep=[];excluded=[]
    for i,(r,p,scores) in enumerate(zip(rows,paths,prob)):
        asked=float(scores[1])>=threshold
        guard=False;met=None
        if asked:
            e=evidence(p,f);met={k:v for k,v in e.items() if k not in ('points','tangents')}
            # Existing source guards and calibrated model threshold, unchanged.
            contour=e['length_px']>=14 and e['extent_px']>=7 and (e['edge_support']>=.25 or e['near_ink_support']>=.30)
            guard=contour or (e['ridge_support']>=.65 and e['directional_support']>=.55) or (e['coherence']>=.75 and e['length_px']>=3 and e['directional_support']>=.55 and e['ridge_support']>=.5)
        reject=asked and not guard
        decisions.append(dict(curve=i,background_probability=float(scores[1]),neural_reject_requested=asked,source_guard=bool(guard),
            disposition='excluded_candidate' if reject else 'keep_or_uncertain',source_metrics=met,
            no_expert_ground_truth=True))
        if reject:excluded.append(i)
        else:keep.append(i)
    selected=[copy.deepcopy(rows[i]) for i in keep]
    for r in selected:r['points']=np.asarray(r['points'],float)
    modules=reconstruct_modules(old)
    meta=dict(native_size=list(rgb.shape[1::-1]),native_to_viewer_scale=factor,
        algorithm='automatic source graph + frozen ownership masks + frozen full-curve context screen',
        raw_input_plan_sha256=sha(RAW/ident/'plan.json'),context_model_sha256=sha(MODEL/'model.pt'),
        context_threshold=threshold,posthoc_screen_after_failed_unfiltered_full_image=True,
        screening_excluded_candidates=len(excluded),automatic_prompt_count=old['raw_query_count'])
    plan=make_plan(ident,selected,modules,[dict(points=np.asarray(r['points'])) for r in rows],meta)
    save(dest/'plan.json',plan);save(dest/'screen_decisions.json',dict(threshold=threshold,selected_ids=keep,excluded_ids=excluded,decisions=decisions,
        thresholds_not_tuned_on_these_images=True,excluded_are_not_verified_false_ink=True))
    shutil.copy2(RAW/ident/'source.png',dest/'source.png');draw(selected).save(dest/'extraction.png');draw(selected,True).save(dest/'modules.png')
    controls={}
    for budget in (500,700):
        p=BASE/f'{ident}_{budget}'/'plan.json'
        if p.exists():controls[str(budget)]=compare_geometry(read(p)['curves'],selected)
    summary=dict(id=ident,unfiltered=curve_metrics([dict(points=np.asarray(r['points'])) for r in rows]),screened=curve_metrics(selected),
        excluded_candidates=len(excluded),neural_requests=sum(d['neural_reject_requested'] for d in decisions),source_guards=sum(d['neural_reject_requested'] and d['source_guard'] for d in decisions),
        baselines=controls,old_baselines_unchanged=True,whole_image_accuracy_not_established=True,new_training=False,
        module_reentries=plan['stats']['hierarchy_order']['module_reentries'],seconds=time.monotonic()-t)
    save(dest/'summary.json',summary)
    changes=np.array(Image.open(dest/'source.png').convert('RGB'));changes=np.uint8(changes*.4+255*.6)
    for i in excluded:cv2.polylines(changes,[np.rint(np.asarray(rows[i]['points'])).astype(np.int32)],False,(215,70,55),1,cv2.LINE_AA)
    panels=[(Image.open(dest/'source.png').convert('RGB'),'原图'),(draw(read(BASE/f'{ident}_500'/'plan.json')['curves']) if (BASE/f'{ident}_500'/'plan.json').exists() else draw([dict(points=np.asarray(r['points'])) for r in rows]),'已认可500笔基线' if controls else '本轮未筛选输出'),
        (draw(read(BASE/f'{ident}_700'/'plan.json')['curves']) if (BASE/f'{ident}_700'/'plan.json').exists() else draw(selected),'已认可700笔基线' if controls else '本轮筛选后候选'),
        (draw([dict(points=np.asarray(r['points'])) for r in rows]),f'未筛选完整候选 {len(rows)}笔'),
        (draw(selected),f'背景筛选后候选 {len(selected)}笔'),(Image.fromarray(changes),'红：筛选器排除的候选 · 需核查是否误删')]
    board=Image.new('RGB',(1560,1150),'#f4f1e8');d=ImageDraw.Draw(board)
    for j,(im,title) in enumerate(panels):
        x=j%3*520;y=j//3*555;d.text((x+10,y+10),title,font=FONT,fill='#263e34');board.paste(im.resize((496,496),Image.Resampling.LANCZOS),(x+10,y+43))
    d.text((10,1119),'仅为离线候选：未训练新权重、未采用、未实画；排除线条不等于已证实杂线，轮廓遗漏需逐图核查。',font=SMALL,fill='#263e34')
    board.save(dest/'comparison.jpg',quality=95);shutil.copy2(dest/'comparison.jpg',dest/'sequence_comparison.jpg')
    catalog=[dict(id=ident,label=f'{ident} · 筛选后完整候选{len(selected)}笔')]
    budgets=select_budgets(selected,(500,700))
    for budget,selection in budgets.items():
        key=f'{ident}_{budget}';folder=ROOT/key;folder.mkdir(exist_ok=True)
        rr=[copy.deepcopy(selected[i]) for i in selection['selected_ids']]
        bp=make_plan(ident,rr,modules,[dict(points=np.asarray(r['points'])) for r in rows],{**meta,'drawing_detail_selection':selection});bp['id']=key;save(folder/'plan.json',bp)
        for name in ('source.png','modules.png','comparison.jpg','sequence_comparison.jpg'):shutil.copy2(dest/name,folder/name)
        catalog.append(dict(id=key,label=f'{ident} · 可选绘画候选{len(rr)}笔（预算{budget}）'))
    print('SCREENED',ident,json.dumps(summary,ensure_ascii=False),flush=True)
    return catalog,summary

def main(ids):
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(2)
    meta=read(MODEL/'training_summary.json');threshold=float(meta['threshold'])
    save(ROOT/'protocol.json',dict(context_model_sha256=sha(MODEL/'model.pt'),threshold=threshold,threshold_source=str(MODEL/'training_summary.json'),
        posthoc_stage=True,no_threshold_tuning_on_full_images=True,no_new_training=True,no_reference_labels_at_inference=True,old_baselines_not_used_for_selection=True,
        prior_full_image_failures_preserved_at=str(RAW),independent_accuracy_not_established=True))
    model=ContextCurveNet().cuda();model.load_state_dict(torch.load(MODEL/'model.pt',map_location='cpu',weights_only=False)['model']);model.eval()
    catalog=[];results=[]
    for ident in ids:
        if (ROOT/ident/'summary.json').exists():
            frozen=read(ROOT/ident/'plan.json')
            assert frozen['raw_input_plan_sha256']==sha(RAW/ident/'plan.json')
            assert frozen['context_model_sha256']==sha(MODEL/'model.pt') and frozen['context_threshold']==threshold
            r=read(ROOT/ident/'summary.json');c=[]
            for key in (ident,ident+'_500',ident+'_700'):
                pp=read(ROOT/key/'plan.json');c.append(dict(id=key,label=f'{key} · 候选{len(pp["curves"])}笔'))
        else:c,r=run(ident,model,threshold)
        catalog+=c;results.append(r);gc.collect();torch.cuda.empty_cache()
        save(ROOT/'catalog.json',dict(scenes=catalog,default='G03_01_500',robot_connected=False));save(ROOT/'results.json',results)
    page=(RAW/'index.html').read_text('utf8')
    page=page.replace('输入细段 → 连续候选','本轮输入候选 → 当前显示选笔').replace('原图支持的断段连接','当前选笔中的模型支持配对')
    page=page.replace('<small>按块完成 · 块内分层 · 连续笔画候选</small>','<small>实验候选 · 整幅质量未通过 · 非正式绘画版</small>')
    (ROOT/'index.html').write_text(page,encoding='utf8')
    save(ROOT/'execution_binding.json',dict(scene=None,journal=None,plan_sha256=None));save(ROOT/'decision.json',dict(adopted=False,new_training=False,robot_connected=False,accuracy_verified=False))
    html='''<!doctype html><meta charset="utf-8"><title>整幅提取与分块回放</title><style>body{font:16px/1.7 Microsoft YaHei,system-ui;background:#f4f1e8;color:#243d33;max-width:1560px;margin:auto;padding:24px}img{width:100%}section{margin:40px 0}pre{white-space:pre-wrap;font-size:13px;background:white;padding:15px}a{color:#285c47}.note{padding:16px;background:#eee0bf}</style><h1>整幅自动提取 · 分对象与整笔归属实验</h1><p class="note">本轮已完成整幅自动提示、模型归属辅助连续性、整笔背景筛选和分块分层回放。使用已有冻结权重，未新训练。对象与皴线角色仍是候选，不等于正确识别独立山石或具体皴法。未替换认可版本，未连接机械臂。动画不是实时实画。</p><p>源图骨架未经筛选时明显过碎，尤其1254×1254原图出现大量纸纹候选；原失败输出保留。另用既有背景筛选模型及其已保存的0.97阈值试验，排除结果可能包含真实细线，不能以笔数下降声称准确率提高。三幅固定对照加三幅既有生成画，全部列出；不是未见过的新画作独立测试。</p><p><a href="/">打开分块绘画回放（候选，非实画）</a></p>'''
    for r in results:
        i=r['id'];html+=f'<section><h2>{i}</h2><p>未经背景筛选 {r["unfiltered"]["count"]}笔 → 筛选后 {r["screened"]["count"]}笔候选；分块回访 {r["module_reentries"]}次。</p><img src="/{i}/comparison.jpg"><p><a href="/?scene={i}">全部候选回放</a> · <a href="/?scene={i}_500">500笔预算回放</a> · <a href="/?scene={i}_700">700笔预算回放</a></p><pre>{json.dumps(r,ensure_ascii=False,indent=2)}</pre></section>'
    (ROOT/'report.html').write_text(html,encoding='utf8')
    (ROOT/'source_snapshot').mkdir(exist_ok=True)
    for n in ('full_image_ownership_core_20261105.py','build_full_image_ownership_20261105.py','screen_full_image_ownership_20261105.py'):shutil.copy2(Path(__file__).with_name(n),ROOT/'source_snapshot'/n)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+',default=list(SCENES));main(p.parse_args().ids)
