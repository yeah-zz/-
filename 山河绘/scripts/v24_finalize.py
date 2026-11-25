# -*- coding: utf-8 -*-
"""Freeze reproducible v24 results, integrity checks and the rejection decision."""
from v24_train import ROOT,V21
from v24_eval import cases
from rock_v15_common import read,jsave,digest,FONT,SMALL,LAB
from PIL import Image,ImageDraw
from pathlib import Path
import numpy as np,torch,shutil,collections
import v21_stroke_learning as v21

def main():
    manifest=read(ROOT/'training_manifest.json');evaluation=read(ROOT/'evaluation.json');cal=read(ROOT/'joint'/'calibration.json');aux=read(ROOT/'auxiliary_trial'/'evaluation.json');selected=read(ROOT/'selected.json')['records'];local=read(ROOT/'local_relations.json')['records'];trainlocal=[p for p in local if p['split']=='train'];configs={v:read(ROOT/v/'config.json') for v in ['control','joint']};summaries={v:read(ROOT/v/'training_summary.json') for v in ['control','joint']};checks={}
    checks['parent_weights_unchanged']=digest(manifest['init_weights'])==manifest['init_sha256']
    checks['training_data_matches_frozen_hashes']=all(digest(r['data'])==r['data_sha256'] for r in manifest['records'])
    checks['all_original_images_and_labels_unchanged']=all(digest(r['image'])==r['image_sha256'] and digest(r['labels'])==r['label_sha256'] for r in manifest['records'])
    checks['new_source_originals_unchanged']=all(digest(r['source_image'])==r['source_image_sha256'] for r in selected)
    checks['new_reference_definition_unchanged_after_freeze']=digest(ROOT/'visible_references_before_training.py')==digest(Path(__file__).with_name('v24_visible_references.py'))
    checks['training_manifest_and_local_relationships_unchanged']=all(c['training_manifest_sha256']==digest(ROOT/'training_manifest.json') and c['local_relations_sha256']==digest(ROOT/'local_relations.json') for c in configs.values())
    checks['both_models_completed_1200_updates']=all(s['updates']==1200 for s in summaries.values())
    checks['matched_sample_usage']=summaries['control']['usage']==summaries['joint']['usage']
    checks['checkpoint_hashes_match']=all(digest(ROOT/v/'final.pt')==summaries[v]['checkpoint_sha256'] for v in summaries)
    checks['checkpoint_parameters_finite']=all(all(torch.isfinite(t).all().item() for t in torch.load(ROOT/v/'final.pt',map_location='cpu',weights_only=False)['model'].values()) for v in summaries)
    train_groups={r['work_group'] for r in selected if r['split']=='train'};nontrain_groups={r['work_group'] for r in selected if r['split']!='train'}
    checks['new_work_groups_separated']=not bool(train_groups&nontrain_groups)
    trainids=set(configs['joint']['training_ids']);checks['new_calibration_and_holdout_ids_excluded']=not bool(trainids&{'C01','H01','H02'}) and all(p['image_id'] not in {'C01','H01','H02'} for p in trainlocal)
    checks['all_pair_coordinates_finite']=all(np.isfinite(np.asarray([p['p'],p['q'],p['center']])).all() for p in local)
    protected=True;uncertain_pixels=0
    old=read(LAB/'autonomous_review_v19_20261025'/'manifest_v2.json')['records'];lookup={r['id']:r for r in manifest['records']}
    for r in old:
        z=np.load(r['data'])
        if 'uncertain_exclusion' not in z:continue
        excluded=z['uncertain_exclusion']>0;uncertain_pixels+=int(excluded.sum());current=np.load(lookup['replay_'+r['id']]['data']);protected=bool(protected and (current['valid'][excluded]==0).all() and (current['mask'][:,excluded]==0).all() and (current['node'][:,excluded]==0).all())
    checks['explicit_user_uncertainty_excludes_all_losses']=protected
    user=LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json';checks['original_user_cross_sections_unchanged']=digest(user)=='8cf074c41402a6f897c92d04db3788ac61186449b63adc514343bba996ff8210'
    checks['original_raster_parent_unchanged']=digest(LAB/'clean_strokes_v12_20261024'/'final.pt')=='8c0988f03a0686fab85a75f4b6b9a82cf21012f4b7a802ea0a90d55d3a316fc6'
    smoke=read(ROOT/'reference_free_smoke'/'summary.json');checks['reference_free_entrypoint_completed']=smoke['reference_labels_used'] is False and smoke['new_reference_module_loaded'] is False
    shapes={r['id']:rgb.shape[:2] for r,rgb,refs,valid,bg in cases('development')};bounds=True;finite_predictions=True
    for variant in ['parent','control','joint']:
        for key,(h,w) in shapes.items():
            pred=np.load(ROOT/'predictions'/variant/(key+'.npz'))['prediction'];finite_predictions=bool(finite_predictions and np.isfinite(pred).all())
            for p in read(ROOT/'evaluation'/variant/key/'trajectory.json')['paths']:
                q=np.asarray(p);bounds=bool(bounds and np.isfinite(q).all() and (q[:,0]>=0).all() and (q[:,0]<w).all() and (q[:,1]>=0).all() and (q[:,1]<h).all())
    checks['all_predictions_finite']=finite_predictions;checks['all_actual_paths_finite_and_in_image']=bounds
    jsave(ROOT/'verification.json',dict(checks=checks,passed=all(checks.values()),uncertain_pixels_protected=uncertain_pixels,optimizer_updates=2400,robot_connected=False))
    gate=dict(official_model_promoted=False,current_default='v16_protected_refinement',new_data_control_promoted=False,joint_candidate_promoted=False,auxiliary_decoder_enabled=False,reason='Joint candidate regressed both held-out new-work partial references and fixed/user-reviewed development references; auxiliary calibration gain gate failed.',calibration_config=cal['selected']['config'],robot_connected=False,ready_for_robot=False,physical_width_calibrated=False)
    jsave(ROOT/'deployment_gate.json',gate)
    snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['v24_sources.py','v24_curate.py','v24_visible_references.py','v24_data.py','v24_train.py','v24_eval.py','v24_predict.py','v24_finalize.py','rock_v15.ps1']:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    # A compact comparison of the two new-work groups, with no reference panel
    # that could be mistaken for model output.
    board=Image.new('RGB',(1600,780),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,6),'v24 新作品对照：原图 / 原研究基线 / 新数据对照组 / 联合训练候选（均为真实离线预测）',font=FONT,fill='black')
    for i,key in enumerate(['H01','H02']):
        r,rgb,refs,valid,bg=next(c for c in cases('development') if c[0]['id']==key)
        panels=[('原图',rgb)]
        for variant,title in [('parent','原研究基线'),('control','新数据对照组'),('joint','联合训练候选')]:
            paths=read(ROOT/'evaluation'/variant/key/'trajectory.json')['paths'];panels.append((title,v21.line_view(rgb.shape[:2],paths,True)))
        for j,(title,a) in enumerate(panels):d.text((j*400+6,45+365*i),key+' '+title,font=SMALL,fill='black');v21.paste(board,a,j*400+6,75+365*i,388,315)
    board.save(ROOT/'NEW_WORK_COMPARISON.jpg',quality=95)
    names=[('new_work_development','两组本轮新作品'),('development_holdout','6 张原主要开发图'),('fixed_external_development','固定 X01–X04'),('user_reviewed_old_development','用户 R03/R04')];rows=[]
    for key,name in names:
        g=evaluation['groups'][key];rows.append('| '+name+' | '+' | '.join(f"{g[v]['mean_best_single_path_coverage']:.3f}" for v in ['parent','control','joint'])+' | '+f"{aux['groups'][key]['mean_best_single_path_coverage']:.3f}"+' |')
    newtrust='\n'.join('| '+r['id']+' | '+r['split']+' | '+f"{r['reference_center_trust_fraction']:.1%}"+' |' for r in manifest['records'] if r['kind']=='new_visible_reference')
    provenance=dict(collections.Counter(p['provenance'] for p in trainlocal));newcount=sum(p['image_id'].startswith('U') for p in trainlocal);groups=evaluation['groups'];cg=cal['candidates'][0];best=cal['trial'];chaintrain=sum(r['reference_chains'] for r in manifest['records'] if r['kind']=='new_visible_reference' and r['split']=='train')
    report=f'''# v24：中心线、转折与关系的联合训练

日期：2026-10-26。实际完成匹配对照与联合组各 1200 次更新，共 2400 次。本轮未形成可靠升级；两个新权重及辅助解码均未替换项目默认 v16 保留版。所有下载、数据、权重、缓存和大型输出位于 E 盘，全程离线、未连接或移动机械臂。

## 数据与监督

采集 51 张公开页面候选图片，经内容、尺寸、已有目录感知哈希和人工视觉筛查，选 9 张来源图：6 张训练、1 张校准、2 张作品留出。并非 51 张都用于训练，也不是数百幅独立新作品。同一作品的步骤图归在同一作品组；仅下载但未选的步骤和复杂整幅图没有进入训练。感知哈希只覆盖若干既有来源目录，不能完全排除裁剪或重印版本的重复。

新图共 124 条辅助可见线链提案，其中 {chaintrain} 条属于训练图；描线在看到 v24 预测前完成。粗墨带与描线不一致、中心横截面偏移超过 2px、测得墨带宽超过 20px、水印遮挡等位置，中心、方向和宽度监督保持未知；只给可信墨带范围和明确纸面负例。未描出的真实墨线没有整块当作背景。标签是助手参考和图像几何估计，不能称为专家完整逐笔真值。

新来源仅占本轮训练的一部分，另回放 v21 的 11 张清晰标注与 v19 的 9 张已有复核数据。最终选择 **337 个训练局部关系**：200 个接续、137 个分开。来源构成为 {provenance}；其中 {newcount} 个选中关系来自本轮 U01–U06 新图，其余来自既有参考和审计节点。337 是局部关系数，不是独立作品或图片数。

新辅助参考中被接受用于中心监督的线带比例如下；其余没有硬作中心真值。低比例说明宽墨、描线近似与可见证据仍限制训练数据精度。

| 图 | 分组 | 提案中心线带内获信任的像素比例 |
|---|---|---:|
{newtrust}

H01/H02 的量化只覆盖已描且通过上述一致性检查的局部参考，不是整幅轮廓完整度。SOURCE/REFERENCE 审计图与原始下载均保留，没有用户新一轮人工校正。

资料页面：[山石起手与三面示范](https://www.sohu.com/a/412487984_120749958)、[山石步骤与斧劈示范](https://www.sohu.com/a/277347183_100275234)、[王兴堂山石教学文章](https://www.culturechina.cn/m/167620.html)、[小斧劈步骤页面](https://www.sohu.com/a/993519974_100169461)。页面数或转载来源数不等于独立作者数，作者/画谱关系没有全部核实；图片仅保留本地研究。

## 真正训练了什么

从同一个 v21 修正墨带检查点初始化，并增加转折热图与 8 维关系特征输出，共享原有图像主干。对照组只训练既有中心、墨带、方向、图像笔宽；联合组另加转折、同线/异线关系和局部可见线连续性损失。未知像素仅以原模型概率弱约束，原用户明确不确定区域同时从该约束排除。

两组使用相同采样、旋转、翻转、学习率、训练长度和初始主干。原图、目标方向、坐标关系及掩膜同步变换；抽样计数核对一致。训练每组 1200 步，checkpoint、config.json、source_at_start.py 和每 100 步日志已保存。推理入口后来仅移除无用的标注模块导入，前向网络结构没有改动；训练时源码仍保存在各模型目录。

## 实际结果

下表为 3px 容差的平均最佳单路径覆盖，是局部参考代理指标，并非整图准确率。旧图已被多轮查看，是固定开发回归集；两组新作品未参与本轮优化器更新与门槛选择，但也只有助手部分参考，不能宣称充分独立的专家盲测。

| 评估组 | 原 v21 研究基线 | 新数据对照 | 联合训练当前解码 | 未启用辅助候选 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

联合组在 X01–X04 的中心邻域召回约 {groups['fixed_external_development']['parent']['center_ref_recall_3px']:.1%} → {groups['fixed_external_development']['joint']['center_ref_recall_3px']:.1%}，但单路径覆盖 {groups['fixed_external_development']['parent']['mean_best_single_path_coverage']:.3f} → {groups['fixed_external_development']['joint']['mean_best_single_path_coverage']:.3f}。输出检出的中心像素更多，最终笔画却没有更完整。两组新作品、R03/R04 和复杂 X 图均未形成稳定收益。

对照组在简单 stage_065 的覆盖从 0.709 提至 0.912，联合组仅约 0.722；这说明不能把新增数据的局部收益归功于联合损失。联合组在 stage_049 略有改善，但其他图有退化，完整结果见 evaluation.json。

## 辅助解码为什么没启用

只用原 5 张校准图加 C01，共 6 张，比较 12 种节点保留、关系接续和强异线否决设置。最高分候选为 {best['config']}；单路径覆盖 {cg['single']:.5f} → {best['single']:.5f}，未达到预设 0.005 的最低增益要求，实际输出继续使用原解码。

为避免漏掉可用方向，另对这个在开发评估前已经冻结的最高分候选做离线探针。它把简单 stage_065 提到约 0.941，但新作品均值降至约 0.492，R 组降至约 0.560，并使 R 组混线参考标记均值上升到 6。探针结果更差，证明不应因一个简单成功图放开门槛。auxiliary_trial/ 明确是未启用候选，不能当作已接受效果。

文字、纸纹、水印和复杂皴擦的误检仍能在整图预测中看到；局部参考与空白区域指标不能覆盖所有杂线。完整可视化没有达到用户所要求的精准完整提取。

## 结论与下一步

本轮联合目标在这批小规模、部分可信监督上没有显示泛化收益。继续增加相同样本轮次不能证明改善，应先建立更精确、范围完整且作品分组独立的清晰单石参考，再做直接输出连续线段/曲线实例的表示对照，减少对像素细化后再拼接的依赖。这是从本轮结果提出的下一步实验方向，尚未实现或验证。

所有新权重保留为实验候选。没有完成整幅工笔山水的稳定提取、皴法完整分类、真实笔宽/压力标定或纸面实画。彩色路径只区分线条，显示宽度不是机械臂毛笔宽度；轨迹 physical_z 为空，ready_for_robot=false。

## 产物与检查

- RESULT_COMPARISON.jpg：原图、基线、匹配对照、联合组真实预测。
- NEW_WORK_COMPARISON.jpg：两组新作品的紧凑对照。
- evaluation.json：逐图覆盖、碎段、混线代理、粗线多中心与淡线覆盖诊断。
- joint/calibration.json 与 auxiliary_trial/：门槛选择及未启用候选的完整证据。
- control/、joint/：两组真实训练日志与检查点。
- reference_free_smoke/：仅输入图片的实际推理记录，本轮参考标注模块未加载；输入已经在开发评估见过，是接口检查而非新盲测。
- verification.json：冻结数据、原权重、用户校正、样本隔离、随机日程、有限数值和路径范围检查。
- deployment_gate.json：拒绝晋级的具体决定。source_snapshot/ 保存本轮最终脚本，各模型 source_at_start.py 保存真实训练源码。

完整性检查通过：{all(checks.values())}。用户原始校正未改动，保护 {uncertain_pixels} 个明确不确定像素不参与任何损失。缓存、下载和输出均遵守 E 盘约定，机械臂未连接。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8');print('V24_FINAL',checks,'report',str(ROOT/'REPORT.md'),flush=True)
    if not all(checks.values()):raise RuntimeError('v24 integrity check failed')

if __name__=='__main__':main()
