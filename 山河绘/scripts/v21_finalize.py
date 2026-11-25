# -*- coding: utf-8 -*-
"""Audit saved predictions and package actual offline results, including failed trials."""
from v21_stroke_learning import *
from v21_background_regions import EVAL_BACKGROUND

def explicit_background(r,rgb,refs,fallback):
    key=r['id'].rsplit('_',1)[1] if r['id'].startswith('sina') else r['id'];h,w=rgb.shape[:2]
    if key not in EVAL_BACKGROUND:return fallback,'previous_partial_explicit_regions'
    bg=np.zeros((h,w),bool)
    for box in EVAL_BACKGROUND[key]:
        x0,y0,x1,y1=[round(v*(w-1 if k%2==0 else h-1)/100) for k,v in enumerate(box)];bg[y0:y1,x0:x1]=1
    _,_,dist=v16.geometry(refs,(h,w));bg[dist<6]=False
    return bg,'assistant_explicit_paper_added_during_development_not_blind'

def finalize():
    variants=['init','control','improved','ribbon_improved'];case_data={r['id']:(r,rgb,ref,valid,bg,target) for r,rgb,ref,valid,bg,target in evaluation_cases()};allrows={};summary={};precision=[]
    for variant in variants:
        data=read(ROOT/variant/'evaluation'/'summary.json');rows=[]
        for row in data['records']:
            key=row['id'];r,rgb,ref,valid,bg,_=case_data[key];bg,provenance=explicit_background(r,rgb,ref,bg);bg&=valid>0;pred=np.load(ROOT/variant/'evaluation'/key/'prediction.npz')['prediction'];paths=read(ROOT/variant/'evaluation'/key/'trajectory.json')['paths'];sk=raster(paths,rgb.shape[:2]);count=int(bg.sum());pc=dict(id=key,variant=variant,background_pixels=count,background_source=provenance,ink_false_positive_fraction=float((pred[1][bg]>.5).mean()) if count else None,center_false_positive_fraction=float((pred[0][bg]>.3).mean()) if count else None,trajectory_background_density=float(sk[bg].mean()) if count else None);precision.append(pc);metric=dict(row['metrics'],**{k:v for k,v in pc.items() if k not in ['id','variant']});effective='legacy_related_probe' if key in ['stage_018','stage_029'] else row['split'];rows.append(dict(row,effective_split=effective,metrics=metric))
        allrows[variant]={r['id']:r for r in rows};summary[variant]={}
        for split in ['development_holdout','legacy_related_probe','fixed_external_development','user_reviewed_old_development']:
            group=[r for r in rows if r['effective_split']==split];g={name:float(np.mean([r['metrics'][name] for r in group])) for name in ['point_coverage_3px','mean_best_single_path_coverage','center_ref_recall_3px','mean_reference_fragments']};g['images']=len(group);g['mixed_reference_paths_flagged']=sum(r['metrics']['mixed_reference_paths_flagged'] for r in group);bgc=sum(r['metrics']['background_pixels'] for r in group);g['explicit_background_pixels']=bgc;g['ink_background_false_positive_weighted']=sum((r['metrics']['ink_false_positive_fraction'] or 0)*r['metrics']['background_pixels'] for r in group)/max(1,bgc);summary[variant][split]=g
    jsave(ROOT/'precision_audit.json',dict(records=precision,background_labels_added_after_development_inspection=True,precision_is_on_selected_paper_not_complete_image=True));jsave(ROOT/'metrics_summary.json',dict(aggregate=summary,records=allrows,not_artwork_accuracy=True,legacy_related_probes_separated=['stage_018','stage_029']))
    # Compact, entirely actual result board: no reference trace is presented as a prediction.
    selected=['stage_065','stage_049','sina_6586334590_31','X01','X04','R04'];board=Image.new('RGB',(1800,360*len(selected)+45),'#f5f3ec');draw=ImageDraw.Draw(board);draw.text((5,3),'v21 实际离线预测：颜色只区分路径；未规划物理笔顺/压力，未在机械臂执行',font=FONT,fill='black')
    for i,key in enumerate(selected):
        _,rgb,_,_,_,_=case_data[key];control=read(ROOT/'control'/'evaluation'/key/'trajectory.json')['paths'];new=read(ROOT/'ribbon_improved'/'evaluation'/key/'trajectory.json')['paths'];pred=np.load(ROOT/'ribbon_improved'/'evaluation'/key/'prediction.npz')['prediction'];y=45+i*360
        for j,(title,a) in enumerate([('原图',rgb),('控制组实际轨迹',line_view(rgb.shape[:2],control,True)),('修正监督预测墨迹',np.uint8(255*(1-pred[1]))),('修正监督实际轨迹',line_view(rgb.shape[:2],new,True))]):draw.text((450*j+5,y),key+' '+title,font=SMALL,fill='black');paste(board,a,450*j+5,y+28,440,322)
    board.save(ROOT/'RESULT_COMPARISON.jpg',quality=95)
    # Scientific validation of artifacts, not an accuracy claim.
    manifest=read(ROOT/'manifest.json');foot=read(ROOT/'footprint_v2_manifest.json');checks={};checks['frozen_reference_and_images_unchanged']=all(digest(r['labels'])==r['label_sha256'] and digest(r['image'])==r['image_sha256'] for r in manifest['records']);checks['first_trial_data_hashes_match']=all(digest(r['data'])==r['data_sha256'] for r in manifest['records']);checks['footprint_data_hashes_match']=all(digest(r['data'])==r['sha256'] for r in foot['records']);checks['initial_weights_unchanged']=digest(manifest['init_weights'])==manifest['init_sha256'];checks['original_raster_parent_unchanged']=digest(PARENT)==read(BASE20/'training_manifest.json')['parent_sha256'];checks['user_review_source_unchanged']=digest(V19/'user_review_snapshot.json')==read(V19/'review_audit.json')['user_review_sha256'];checks['reference_free_entrypoint_completed']=read(ROOT/'reference_free_smoke'/'summary.json')['reference_labels_used'] is False
    checks['weights_finite']=True
    for v in variants[1:]:
        state=torch.load(ROOT/v/'final.pt',map_location='cpu',weights_only=False)['model'];checks['weights_finite']&=all(bool(torch.isfinite(t).all()) for t in state.values())
    checks['trajectories_finite_and_within_image']=True
    for v in variants:
        for key,(_,rgb,_,_,_,_) in case_data.items():
            for p in read(ROOT/v/'evaluation'/key/'trajectory.json')['paths']:
                a=np.asarray(p);checks['trajectories_finite_and_within_image']&=bool(np.isfinite(a).all() and (a>=0).all() and (a[:,0]<rgb.shape[1]).all() and (a[:,1]<rgb.shape[0]).all())
    newtrain={r['id'] for r in manifest['records'] if r['split']=='train'};checks['new_training_excludes_calibration_and_fixed_external']=not newtrain&{r['id'] for r in manifest['records'] if r['split']!='train'} and not newtrain&{'X01','X02','X03','X04'};checks['uncertain_user_pixels_stay_unsupervised']=True
    for r in read(V19/'manifest_v2.json')['records']:
        z=np.load(r['data']);u=z['uncertain_exclusion']>0
        if u.any():checks['uncertain_user_pixels_stay_unsupervised']&=bool((z['mask'][:,u]==0).all())
    jsave(ROOT/'verification.json',dict(checks=checks,passed=all(checks.values()),robot_connected=False,extractor_updates=3000,endpoint_artificial_updates=600,real_endpoint_training_skipped_for_insufficient_pairs=True))
    decision=dict(weights_promoted=False,default_operational_baseline='v16_protected_refinement',research_candidate='ribbon_improved',endpoint_enabled=read(ROOT/'ribbon_improved'/'endpoint_calibration.json')['enabled'],reason='Ink footprint and several coverage measures improve, but fragmentation/mixed-path proxies do not improve consistently; small correlated development set',robot_connected=False);jsave(ROOT/'deployment_gate.json',decision)
    def n(v,s,k):return summary[v][s][k]
    table='\n'.join('| '+label+' | '+' | '.join(f'{n(v,s,"mean_best_single_path_coverage"):.3f}' for s in ['development_holdout','fixed_external_development','user_reviewed_old_development'])+' |' for v,label in [('init','初始 v17'),('control','v21 控制组'),('improved','补线参考试验'),('ribbon_improved','修正墨带与空白监督')])
    bg_table='\n'.join('| '+label+' | '+f'{100*n(v,"development_holdout","ink_background_false_positive_weighted"):.2f}% |' for v,label in [('init','初始 v17'),('control','控制组'),('improved','首次补线试验'),('ribbon_improved','修正墨带监督')])
    report=f'''# v21：分阶段诊断、逐笔辅助监督与连接试验

日期 2026-10-25。全部训练、模型与大型输出在 E 盘；未安装依赖，未连接机械臂。

## 本轮实际完成

整理 24 个已有清晰示例、156 条主要可见线链，分为 11 张新增训练图、5 张校准图、8 张开发留出图。旧有 9 张训练图以相同日程回放。24 张主要来自三个教学资料系列，并非 24 个独立作者的新作品；有些是相近构图或同一作品不同步骤。辅助描线也不是完整专家真值，未描墨迹和不确定位置保持未知，裁剪边界不当作真实起收笔。stage_018/stage_029 与旧训练资料存在关联风险，已单列为旧资料相关探针，不计入下面 6 张主要开发图。

控制组、首次补线组、修正墨带组各真实训练 1000 次更新，共 3000 次；使用同一初始化、5 个输出头、相同优化器与损失，不叠加上轮失败的结构损失。实验组采样时以 35% 概率替换为 11 张新辅助监督样本。最终一组为墨迹范围增加横截面扫描和约 38.8 万个明确空白像素。宽墨带边界、交叉等仍有自动估计误差。

## 主要发现

初始模型在 X04 参考线附近墨迹召回接近 1，中心召回约 0.695，最终单路径覆盖约 0.131。首次补线试验把中心召回提高到约 0.938，路径覆盖仍约 0.195，且空白被预测成宽墨块。该失败记录保留在 improved/，不把高召回当作准确提取。

修正墨带/明确空白监督后，概率图的大面积空白误检显著减少，实际墨迹更贴近笔画；复杂画面的中心仍有侧边重复、漏线、碎段及错误连接。见 RESULT_COMPARISON.jpg 和 ribbon_improved/STAGE_DIAGNOSTIC.jpg。图上的彩色仅用于区分实际预测路径，不是已确认的物理笔顺。

| 模型 | 6 张开发图单路径覆盖 | 原 X01–X04 固定开发图 | 用户 R03/R04 复核参考 |
|---|---:|---:|---:|
{table}

这些是 3px 容差的局部参考指标，不是整幅准确率。R03/R04 此轮按完整输入画布预测，与早期原生裁剪协议的数字不能直接比较。固定 X 图也已经多次查看过，属于开发回归集，不是新盲测。

为了查出高召回掩盖的误检，新增开发图明确空白区域检查，结果按像素加权如下。区域由我结合原图确定，在查看开发结果之后补充，只用于诊断；不能当作独立专家测试。

| 模型 | 6 张开发图明确空白被判为墨迹的比例 |
|---|---:|
{bg_table}

该比例只针对圈定空白，不是整图精确率。路径背景密度、中心误检和逐图结果在 precision_audit.json。

## 连接模块为什么仍未启用

从辅助线链构造同线断口和邻线反例，训练了 600 次更新的端点网络。校准人工断口上接受 558 个正例、误接 1 个反例；这不能代表真实断点准确率。真实解码训练图中，在图像支持、距离、方向与参考一致性筛选后，只得到 3 个确定同线、2 个确定邻线的候选；校准图为 0/0。因此没有强行训练“真实断口”网络，审计见 endpoint_real_fragments/data_audit.json。

另对已接触但被分成两段的节点做有限几何拼接。它在五张校准图上只把平均单路径覆盖从约 0.653 提至 0.656，未达到预设最低增益，最终 endpoint_enabled=false。joined_evaluation 中同时保留未启用的真实输出与可视化试验候选，不能把候选图当作已接受升级。

## 当前结论与下一步

本轮取得的是“墨迹范围更受约束，部分图的中心/路径覆盖改善”，没有解决复杂笔画拓扑。新权重仍为研究候选，未替换 v16 保留基线；没有完成各种皴法分类、真实笔宽/压力标定或机械臂实画验证。

下一步应集中建立分叉/接触节点的逐笔关系监督与解码，特别是两条相接线在拐角处如何连续、三叉处应如何分笔；不是只找相距几像素的断口。还需要真正不同作者和来源、完整范围的留出线稿。当前辅助数据的来源与完整性有限，不能继续把它自动扩成专家金标。宽墨与淡线继续同时看覆盖、重复中心和空白误检，避免单独追求某一分数。

## 产物

- RESULT_COMPARISON.jpg：原图、控制轨迹、修正墨带概率、修正轨迹，均为真实离线输出。
- ribbon_improved/STAGE_DIAGNOSTIC.jpg：原图→墨迹概率→中心概率→轨迹→辅助参考，末栏非预测。
- REFERENCE_AUDIT_*.jpg / FOOTPRINT_AUDIT_*.jpg：标注审计，不是模型效果。
- metrics_summary.json / precision_audit.json：分组与逐图指标。
- verification.json / deployment_gate.json：完整性、有限数值、未知像素及晋级决定。
- 三组 final.pt 与 training.jsonl：实际权重和训练记录。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8');snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['v21_stroke_learning.py','v21_visible_references.py','v21_background_regions.py','v21_path_learning.py','v21_finalize.py','v21_predict.py']:
        p=Path(__file__).with_name(name)
        if p.exists():shutil.copy2(p,snap/name)
    print('V21_FINAL_METRICS',json.dumps(summary),flush=True);print('V21_VERIFICATION',checks,flush=True)

if __name__=='__main__':finalize()
