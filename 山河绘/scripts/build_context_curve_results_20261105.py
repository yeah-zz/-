# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,shutil
import cv2,numpy as np,torch
from PIL import Image,ImageFont,ImageDraw
from scipy.spatial import cKDTree
from context_curve_data_20261105 import ROOT,read,write,source_records
from context_curve_apply_20261105 import load,select,audit,scores
from context_relation_20261105 import load_relation,suppress,actual_audit
from module_stroke_core_20261104 import samples,length
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
BASE=LAB/'source_evidence_selection_20261104'

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def draw(shape,paths,origin,color=(0,0,0),base=None):
    out=np.full((shape[0]*3,shape[1]*3,3),255,np.uint8) if base is None else base.copy()
    for p in paths:cv2.polylines(out,[np.rint((np.asarray(p)-origin)*3).astype(np.int32)],False,color,2,cv2.LINE_AA)
    return out

def panel(folder,key,title,rgb,before,after,removed,added,box):
    dest=folder/key;dest.mkdir(exist_ok=True);x0,y0,x1,y1=box;shape=(y1-y0,x1-x0);origin=np.array([x0,y0])
    old=draw(shape,before,origin);new=draw(shape,after,origin)
    changes=draw(shape,removed,origin,(215,55,35),base=new);changes=draw(shape,added,origin,(15,150,80),base=changes)
    source=Image.fromarray(rgb[y0:y1,x0:x1]);source.save(dest/'source.png')
    for name,im in [('before.png',old),('after.png',new),('changes.png',changes)]:Image.fromarray(im).save(dest/name)
    board=Image.new('RGB',(1640,460),'#f4f1e8');d=ImageDraw.Draw(board)
    for k,(im,label) in enumerate([(source,'原图'),(Image.fromarray(old),'上轮保守结果'),(Image.fromarray(new),'上下文筛选候选'),(Image.fromarray(changes),'红：排除；绿：源图补提')]):
        f=min(400/im.width,335/im.height);im=im.resize((max(1,round(im.width*f)),max(1,round(im.height*f))),Image.Resampling.LANCZOS)
        board.paste(im,(k*410+3,43));d.text((k*410+3,9),label,font=FONT,fill='black')
    d.text((8,394),title+'｜中心线前后同一显示宽度，保留曲线不改形；颜色是修改说明，不是皴法类别。',font=FONT,fill='black')
    d.text((8,426),'神经筛选与源图保护联合输出，仅离线候选；新增/排除尚未逐笔确认，不代表毛笔实画。',font=FONT,fill='black')
    board.save(dest/'comparison.jpg',quality=96)

def native_additions(model,rgb,paths):
    d=read(LAB/'ribbon_topology_refinement_20261104/T02_native_candidates.json')
    pool=[r for r in d['fine_diagnostics']['pending'] if r['source_bilateral_support']>=.70 and r['observed_width_fraction']>=.65
        and r['tree_route_fraction']>=.85 and r['source_width_px']<=3.2 and r['novel_fraction']>=.65]
    if not pool:return [],[]
    candidates=[np.asarray(r['points'],float) for r in pool];p=scores(model,rgb,paths+candidates)[len(paths):]
    accepted=[];records=[];tree=cKDTree(np.concatenate([samples(q,.5) for q in paths]))
    for row,q,prob in zip(pool,candidates,p):
        novel=float((tree.query(samples(q,.4))[0]>1.35).mean())
        take=bool(prob[0]>=.995 and novel>=.65)
        if take and accepted:
            near=cKDTree(np.concatenate([samples(a,.4) for a in accepted])).query(samples(q,.4))[0]
            if (near<1.).mean()>.55:take=False
        records.append(dict(proposal_id=row['id'],visible_probability=float(prob[0]),native_novel_fraction=novel,
            accepted=take,source_checks=row,expert_truth=False))
        if take:accepted.append(q)
    return accepted,records

def build_scene(ident,model,threshold,allowed,relation_model,relation_threshold,relation_allowed):
    folder=ROOT/ident;folder.mkdir(exist_ok=True);parent=BASE/ident/'candidates.json';doc=read(parent)
    source=Path(doc['source']);rgb=np.array(Image.open(source).convert('RGB'))
    ids=doc['selection']['selected'];rows=[doc['input_curves'][i] for i in ids]
    old_decisions=[doc['selection']['decisions'][i] for i in ids]
    selection=select(model,threshold,rgb,rows,old_decisions)
    context_chosen=selection['selected'] if allowed else list(range(len(rows)))
    relation=suppress(relation_model,relation_threshold,rgb,[rows[i] for i in context_chosen]) if relation_allowed else dict(selected=list(range(len(context_chosen))),removed=[],events=[],candidate_pairs=0)
    chosen=[context_chosen[i] for i in relation['selected']]
    before=[np.asarray(r['points'],float) for r in rows];after=[before[i] for i in chosen]
    rejected=[before[i] for i in range(len(before)) if i not in set(chosen)]
    added=[];new_records=[]
    if ident=='T02' and allowed:added,new_records=native_additions(model,rgb,before)
    after+=added
    save(folder/'candidate.json',dict(source=str(source),source_sha256=sha(source),parent=str(parent),parent_sha256=sha(parent),
        input_rows=rows,upstream_input_ids=ids,context_selection=selection,selected_input_ids=chosen,
        relation_selection=relation,context_selected_input_ids=context_chosen,
        output_paths=after,added_native_paths=added,native_addition_records=new_records,
        model_applied=allowed,guarded_fallback=not allowed,model_sha256=sha(ROOT/'model.pt'),
        relation_model_applied=relation_allowed,relation_model_sha256=sha(ROOT/'relations/model.pt'),
        original_input_geometry_archived=True,coordinate_system='image pixels only',
        labels_read_at_inference=False,physical_pressure_mm=None,ready_for_robot=False))
    stat=dict(id=ident,input_curves=len(before),neural_reject_requests=sum(r['neural_reject_requested'] for r in selection['decisions']),
        source_guards_restored=selection['contour_guards_restored'],excluded_curves=len(rejected),native_added=len(added),
        relation_candidate_pairs=relation['candidate_pairs'],relation_duplicates_removed=len(relation['removed']),
        output_curves=len(after),model_applied=allowed,expert_pixel_accuracy=None,physical_improvement_verified=False)
    save(folder/'statistics.json',stat)
    panel(folder,'full',ident,rgb,before,after,rejected,added,[0,0,rgb.shape[1],rgb.shape[0]])
    if ident=='G03_01':
        for key,title,box in [('rock','山石主轮廓与皴线',[178,285,326,389]),('pavilion','亭子屋檐梁柱',[108,198,231,299])]:
            panel(folder,key,title,rgb,before,after,rejected,added,box)
    if ident=='T02':panel(folder,'pine','原生松枝细线',rgb,before,after,rejected,added,[859,766,1254,961])
    print('CONTEXT_RESULT',stat,flush=True);return stat

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);model,threshold=load()
    auditfile=ROOT/'actual_prediction_audit.json'
    actual=audit(model,threshold)
    relation_model,relation_threshold=load_relation()
    relation_actual=actual_audit(model,threshold,relation_model,relation_threshold)
    relation_test=read(ROOT/'relations/fresh_pipeline_test.json')
    training=read(ROOT/'training_summary.json')
    allowed=actual['passes_all_actual_output_coverage_guards'] and training['evaluation']['positive_removed']==0 and threshold<=1.
    relation_allowed=allowed and relation_actual['passes_all_actual_output_coverage_guards'] and relation_threshold<=1.
    results=[build_scene(i,model,threshold,allowed,relation_model,relation_threshold,relation_allowed) for i in ['G03_01','G05_03','T02']]
    # Show a genuine prediction-vs-reference example; references are evaluation
    # overlays only and never copied into inferred curves.
    r04=next(r for r in source_records() if r['id']=='R04');a=next(r for r in relation_actual['records'] if r['id']=='R04')
    after=[r04['paths'][i] for i in a['selected']]
    folder=ROOT/'R04_actual';folder.mkdir(exist_ok=True)
    panel(folder,'full','R04：实际预测回查',r04['rgb'],r04['paths'],after,[r04['paths'][i] for i in range(len(r04['paths'])) if i not in set(a['selected'])],[],[0,0,r04['rgb'].shape[1],r04['rgb'].shape[0]])
    ref=r04['rgb'].copy();tree=cKDTree(np.concatenate([samples(p,.5) for p in after]))
    for p in r04['refs']:
        q=samples(p,.5);hit=tree.query(q)[0]<=2
        for v,h in zip(q,hit):cv2.circle(ref,tuple(np.rint(v).astype(int)),1,(25,150,160) if h else (230,40,30),-1)
    Image.fromarray(ref).save(folder/'reference_coverage.png')
    save(ROOT/'results.json',dict(records=results,network_applied=allowed,actual_prediction_audit=str(auditfile),
        relation_applied_to_offline_candidates=relation_allowed,relation_has_known_heldout_error=True,
        actual_prediction_relation_audit=str(ROOT/'actual_prediction_relation_audit.json'),
        new_weights_trained=True,new_updates=training['completed_updates'],robot_connected=False,
        independent_painting_accuracy=False,generated_scene_candidates_not_expert_verified=True))
    table=''.join(f'<tr><td>{r["id"]}</td><td>{r["input_curves"]}</td><td>{r["excluded_curves"]}</td><td>{r["source_guards_restored"]}</td><td>{r["native_added"]}</td><td>{r["output_curves"]}</td></tr>' for r in results)
    auditrows=''.join(f'<tr><td>{r["id"]} · {r["split"]}</td><td>{r["before"]["mean_reference_coverage"]:.1%}</td><td>{r["after"]["mean_reference_coverage"]:.1%}</td><td>{r["before"]["trace_length_inside_explicit_background_px"]:.1f} → {r["after"]["trace_length_inside_explicit_background_px"]:.1f}px</td></tr>' for r in relation_actual['records'])
    page='''<!doctype html><meta charset="utf-8"><title>整笔上下文筛选 · 实际预测对照</title><style>body{font:17px/1.65 system-ui;max-width:1700px;margin:auto;padding:24px;background:#f4f1e8;color:#263d33}img{width:100%;background:white}td,th{padding:9px;border:1px solid #b8c4b6}table{border-collapse:collapse}.note{padding:16px;background:#e1e8dd}a{color:#155f85}</style><h1>整条曲线＋局部图像上下文筛选</h1>'''
    page+=f'<p class="note">新模块：66,962参数；RTX 5060 Laptop GPU完成1000次更新，保留第{training["retained_step"]}次权重。模型分别查看整条曲线和沿线三处图像、候选掩膜及邻近路径。仅判断可见墨迹候选与明确背景错误；重复墨带仍由源图亮缝规则处理，不宣称识别皴法类别。当前保护门槛通过：{allowed}。</p>'
    rel_summary=read(ROOT/'relations/training_summary.json')
    page+=f'<p>另训笔画关系模块1000次更新，区分连续碎片、同墨带重复与独立笔画；只在一个较长代表完整覆盖候选、没有延伸亮缝时提出去重。原留出检查存在{rel_summary["test"]["separate_or_continuous_wrongly_accepted"]}次错误重复判断；后续新种子与保留A06曲线组合检查，联合规则仍有{relation_test["joint_false_duplicate"]}次误判。因此关系模块只用于本页离线候选对照，不认定已可靠，不用于实机。实际旧参考覆盖保护通过：{relation_allowed}。</p>'
    page+='<h2>当前已见生成画 · 离线开发候选</h2><table><tr><th>图</th><th>上轮</th><th>本轮排除</th><th>源图保护恢复</th><th>原图补提</th><th>本轮候选</th></tr>'+table+'</table>'
    for title,prefix in [('山石轮廓与内部皴线','G03_01/rock'),('亭子结构','G03_01/pavilion'),('原生松枝','T02/pine')]:
        page+=f'<h2>{title}</h2><img src="{prefix}/comparison.jpg"><p><a href="{prefix}/changes.png">放大红色排除／绿色补提</a></p>'
    page+='<h2>真实预测与旧复核线稿对齐</h2><p>以下输入是旧版存档提取器实际输出，源图像素已逐一核对，不是把参考线直接送进筛选器。当前生成画没有完整真值，不能将这里的覆盖率推广给它们。误差容差为原生2像素，仅评分已明确描绘的部分。</p><table><tr><th>图</th><th>上轮参考覆盖</th><th>候选参考覆盖</th><th>明确背景中的预测长度</th></tr>'+auditrows+'</table><img src="R04_actual/full/comparison.jpg"><p><a href="R04_actual/reference_coverage.png">青：已覆盖参考；红：仍遗漏参考</a></p>'
    ev=training['evaluation']
    page+=f'<p class="note">混合样本留出检查：{ev["known_positive"]}个已标正例中排除{ev["positive_removed"]}个，{ev["known_negative"]}个已标负例中排除{ev["negative_removed"]}个。数据含合成样本和部分真实参考，同一出版来源／画作可能跨图片，不能解释为独立整幅画准确率。未标区域没有当负例。对新增细线和排除候选尚无逐笔专家真值，无法证明每一处都正确。纯筛选不会找回尚未生成的山石线，仍遗漏的位置在参考覆盖图中标红。</p>'
    page+='<p>保留曲线没有拉直或改形，源文件和排除列表可追溯。全部图像为等宽中心线预览，不表示实际笔宽、浓淡或晕染；本轮未连接机械臂。</p><p><a href="actual_prediction_relation_audit.json">实际预测与关系筛选评估</a> · <a href="training_summary.json">图像筛选训练</a> · <a href="relations/training_summary.json">关系训练</a> · <a href="relations/fresh_pipeline_test.json">关系误判回查</a> · <a href="protocol.json">监督与划分</a> · <a href="results.json">候选统计</a></p>'
    (ROOT/'index.html').write_text(page,encoding='utf8')
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['context_curve_model_20261105.py','context_curve_data_20261105.py','train_context_curve_model_20261105.py','context_curve_apply_20261105.py','context_relation_20261105.py','build_context_curve_results_20261105.py']:
        shutil.copy2(Path(__file__).with_name(name),snap/name)
    print('REPORT',ROOT/'index.html',flush=True)

if __name__=='__main__':main()
