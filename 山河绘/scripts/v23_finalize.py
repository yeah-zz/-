# -*- coding: utf-8 -*-
"""Freeze v23 evidence, verification and a balanced visual comparison."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
import v21_stroke_learning as v21
from v23_junction_relations import ROOT,BASE,WEIGHT
from v23_assistant_nodes import SPECS

def main():
    audit=v21.read(ROOT/'data_audit.json');training=v21.read(ROOT/'training_summary.json');pairs=v21.read(ROOT/'pair_classification.json');rel=v21.read(ROOT/'evaluation.json');band=v21.read(ROOT/'center_band'/'evaluation.json');bridge=v21.read(ROOT/'center_bridges'/'calibration.json')
    manifest=v21.read(BASE/'manifest.json');checks={}
    checks['extractor_weights_unchanged']=v21.digest(WEIGHT)==audit['input_hashes']['weights']
    checks['v21_manifest_unchanged']=v21.digest(BASE/'manifest.json')==audit['input_hashes']['manifest']
    checks['original_source_images_and_labels_unchanged']=all(v21.digest(r['image'])==r['image_sha256'] and v21.digest(r['labels'])==r['label_sha256'] for r in manifest['records'])
    user=v21.LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json'
    checks['user_cross_sections_unchanged']=v21.digest(user)=='8cf074c41402a6f897c92d04db3788ac61186449b63adc514343bba996ff8210'
    checks['original_raster_parent_unchanged']=v21.digest(v21.LAB/'clean_strokes_v12_20261024'/'final.pt')=='8c0988f03a0686fab85a75f4b6b9a82cf21012f4b7a802ea0a90d55d3a316fc6'
    checks['all_2400_updates_recorded']=training['total_updates']==2400 and sum(v21.read(ROOT/('seed_'+str(r['seed']))/'training.json')['updates'] for r in training['models'])==2400
    checks['trained_checkpoint_hashes_match']=all(v21.digest(ROOT/('seed_'+str(r['seed']))/'final.pt')==r['sha256'] for r in training['models'])
    checks['trained_checkpoints_finite']=all(all(torch.isfinite(t).all().item() for t in torch.load(ROOT/('seed_'+str(r['seed']))/'final.pt',map_location='cpu',weights_only=False)['model'].values()) for r in training['models'])
    tr=np.load(ROOT/'train_pairs.npz');ca=np.load(ROOT/'calibration_pairs.npz')
    checks['pair_inputs_finite_and_unknowns_excluded']=bool(np.isfinite(tr['x']).all() and np.isfinite(ca['x']).all() and np.isin(tr['y'],[0,1]).all() and np.isin(ca['y'],[0,1]).all())
    checks['relation_calibration_not_in_training']=not bool(set(tr['ids'])&set(ca['ids']))
    holdout_ids={r['id'] for r in rel['records']};checks['relation_evaluation_ids_not_in_training']=not bool(set(tr['ids'])&holdout_ids)
    checks['original_graph_baseline_reconstructed_exactly']=audit['baseline_reconstruction_matches']
    checks['reference_free_entrypoint_completed']=v21.read(ROOT/'reference_free_smoke'/'summary.json')['reference_labels_used'] is False
    shape={r['id']:rgb.shape[:2] for r,rgb,refs,valid,bg,t in v21.evaluation_cases()};valid_paths=True
    for folder in [ROOT/'evaluation',ROOT/'center_band'/'evaluation',ROOT/'center_bridges'/'evaluation']:
        for key,(h,w) in shape.items():
            file=folder/key/'trial_trajectory.json'
            if not file.exists():continue
            for p in v21.read(file)['paths']:
                q=np.asarray(p);valid_paths=bool(valid_paths and np.isfinite(q).all() and (q[:,0]>=0).all() and (q[:,0]<w).all() and (q[:,1]>=0).all() and (q[:,1]<h).all())
    checks['all_candidate_paths_finite_and_in_bounds']=valid_paths
    copies=ROOT/'source_snapshot';copies.mkdir(exist_ok=True)
    for name in ['v23_junction_relations.py','v23_assistant_nodes.py','v23_center_band.py','v23_finalize.py','rock_v15.ps1']:shutil.copy2(Path(__file__).parent/name,copies/name)
    v21.jsave(ROOT/'verification.json',dict(checks=checks,passed=all(checks.values()),robot_connected=False,backbone_updated=False,updates=2400,current_data_hashes={s:v21.digest(ROOT/(s+'_pairs.npz')) for s in ['train','calibration']}))
    # Calibration passed for a global center-band decoder but its subsequent
    # fixed regression results declined, so the operational gate remains closed.
    gate=dict(official_model_promoted=False,current_default='v16_protected_refinement',relation_enabled=False,global_center_band_enabled=False,short_bridge_enabled=False,reasons=['Relation calibration gain is zero and classifier does not generalize to the small known calibration sample.','Center-band retention improves one simple peak but regresses fixed X and reviewed R references.','Short bridges provide no calibration benefit; baseline remains selected.'],robot_connected=False,ready_for_robot=False,physical_width_calibrated=False)
    v21.jsave(ROOT/'deployment_gate.json',gate)
    lookup={r['id']:r for r in band['records']};board=Image.new('RGB',(1500,1140),'#f5f3ec');d=ImageDraw.Draw(board)
    d.text((10,8),'v23 诊断对照：局部可改善，但仍未达到整体升级要求',font=v21.FONT,fill='black')
    d.text((10,37),'彩色只区分轨迹；右栏是未采用的中心带候选，线宽不代表毛笔实画宽度。',font=v21.SMALL,fill='#555555')
    for i,key in enumerate(['stage_065','sina_6586334590_31','R04']):
        row=lookup[key];before=row['before']['mean_best_single_path_coverage'];after=row['trial']['mean_best_single_path_coverage'];d.text((10,70+i*350),f'{key}  单路径覆盖参考：{before:.1%} → {after:.1%}',font=v21.FONT,fill='black')
        if key.startswith('R'):
            case=next(c for c in v21.evaluation_cases() if c[0]['id']==key);rgb=case[1]
        else:rgb=np.asarray(Image.open(BASE/'images'/(key+'.png')).convert('RGB'))
        old=v21.read(BASE/'ribbon_improved'/'evaluation'/key/'trajectory.json')['paths'];new=v21.read(ROOT/'center_band'/'evaluation'/key/'trial_trajectory.json')['paths']
        for j,(title,a) in enumerate([('原图',rgb),('v21 原轨迹',v21.line_view(rgb.shape[:2],old,True)),('中心带候选：未采用',v21.line_view(rgb.shape[:2],new,True))]):
            d.text((500*j+7,100+i*350),title,font=v21.SMALL,fill='black');v21.paste(board,a,500*j+7,125+i*350,485,280)
    board.save(ROOT/'DIAGNOSIS_COMPARISON.jpg',quality=95)
    train_base=[r for r in audit['records'] if r['split']=='train' and r['policy']==['center',.65,.30]]
    manual_positive=sum(len(v.get('positive',[])) for v in SPECS.values());manual_negative=sum(len(v.get('negative',[])) for v in SPECS.values())
    groups=['development_holdout','fixed_external_development','user_reviewed_old_development']
    table='\n'.join('| '+name+' | '+f"{rel['groups'][g]['before']['mean_best_single_path_coverage']:.3f}"+' | '+f"{rel['groups'][g]['trial']['mean_best_single_path_coverage']:.3f}"+' | '+f"{band['groups'][g]['trial']['mean_best_single_path_coverage']:.3f}"+' |' for g,name in zip(groups,['6 张主要开发图','固定 X01–X04','用户 R03/R04 复核参考']))
    text=f'''# v23：真实节点关系训练与骨架断裂诊断

日期：2026-10-26。全程离线，所有新实验数据、权重和大输出在 E 盘；无依赖安装和下载，无机械臂连接、使能、运动或实画。本轮对照使用 v21 修正墨带研究模型，项目默认仍为 v16 保留版。

## 实际完成的工作

从真实图片的 v21 预测骨架中采集相接节点的分支对，按分支内部 4–22px 与既有参考线链的匹配关系建立正反例。未知和相邻参考难区分的分支保持未知，未硬判为负例。训练图额外比较三个解码门槛生成相关图视图，这些增强不是新作品。

最初自动对应得到训练 30 个正例、24 个负例，校准只有 1 个正例、3 个负例。我检查了 57 个未知节点的原图放大图，针对 22 个较清楚节点补充 {manual_positive} 个几何接续、{manual_negative} 个分开关系；密集、含糊节点继续未知。这些是助手局部视觉判断，并非画家专家标注，更不是真实历史起收笔的证明。原始用户描线与横截面文件没有改写。

最终训练 46 正例、43 负例，共 89 个带标签的分支对/图视图，涉及 8 张图；其中固定门槛的原始图视图有 {sum(r['positive']+r['negative'] for r in train_base)} 对，其余是相关增强。校准有 3 正例、7 负例，来自 3 张有可用节点标签的图。资料仍主要为既有 Sina 和 Sohu 教学系列；本轮没有新增来源，不能称为跨作者大规模数据集。

三个 26 输入特征的小型关系网络，各训练 800 次更新，合计 **2400 次真实优化更新**。输入为局部形状、方向、预测中心/墨带和原图明暗，不含参考线链 ID；v21 主干权重固定。模型权重、逐次训练摘要与种子均已保存。

## 学习是否有效

在选定 0.7 门槛上，训练关系样本命中 43/46 正例，误接 0/43 负例；小校准集命中 0/3 正例，误接 1/7 负例。门槛后的几何约束挡住了这个负例进入最终连接。这明确暴露出小样本局部分类的过拟合与覆盖不足，训练表现不能当成实际准确率。详见 pair_classification.json。

关系解码校准的平均单路径覆盖 0.65303 → 0.65303，没有通过最低增益要求。因此关系模型保留为研究候选。开发图候选中的部分接续无完整参考支持，不能以“未命中已知负例”宣称正确。

## 骨架构造对照

另在固定网络输出上比较原方向非极大值抑制、两种高置信中心带保留、完整中心带细化，共 4 种构造。仅在原 5 张校准图选择候选，得到混合中心带 0.65；校准单路径覆盖约 0.653 → 0.658，混线参考标记 8 → 5。随后在固定开发图上检查，结果如下。表内都是局部参考、3px 容差的平均单路径覆盖，不是整图准确率。

| 评估组 | v21 原路径 | 局部关系候选 | 中心带候选 |
|---|---:|---:|---:|
{table}

简单峰形 stage_065 的单路径覆盖 0.709 → 0.936，说明部分断裂与骨架生成有关。但 Sina31、X 组和 R 组多处变差；R 组混线参考标记均值 3 → 4.5。6 张开发图的均值提高主要受一个简单样本推动，不能宣称整体变好。原图及正反结果见 DIAGNOSIS_COMPARISON.jpg，完整对照见 center_band/RESULT_COMPARISON.jpg。

最后试验只补高置信 3/5/7px 短断口，并保留原骨架位置、限制方向、竞争分支和闭环。在这批校准图上均没有增益，选择仍为原解码；center_bridges/evaluation 是“最终选择基线”的结果，不代表三个短桥候选已经在开发集全部单独得到验证。

## 决定与下一步

本轮没有可靠的整体升级。局部关系模型未启用；全局中心带虽通过小校准集，但在固定回归图退化，也未采用。deployment_gate.json 已明确关闭三项新功能，项目默认未替换。

接下来应把“节点处中心线是否保留、如何转折、哪两支接续”的监督一起加入中心线网络，先形成约 200–500 个来自不同清晰真实图片的局部转角/分叉样本，再做联合训练。当前瓶颈包含骨架构造和监督覆盖；单独追加这 89 个相关关系样本的轮次，容易继续拟合现有小样本。新增数据应保留明确墨迹、中心、边侧与未知区域，按原始作品分组留出，不能把自动生成线稿当成专家真值循环训练。

本轮没有完整皴法分类升级、真实笔宽/压力标定或机械臂纸面验证。所有轨迹坐标均为图像像素，physical_z 为空，ready_for_robot=false。

## 复现和完整性

- source_snapshot/：本轮脚本与进程级 E 盘环境包装器。
- seed_231026/、seed_231027/、seed_231028/：三个训练检查点与日志。
- data_audit.json、node_audit/：样本来源、标签、未知关系与原图局部审计。
- calibration.json、evaluation.json：关系解码校准和逐图候选结果。
- center_band/、center_bridges/：骨架构造对照与否决证据。
- reference_free_smoke/：只提供图片路径的实际推理接口运行记录；是接口检查，输入已在开发阶段见过，不能当作盲测。
- verification.json：原权重、图像、用户标注、参考文件、检查点哈希、有限数值、训练/校准 ID 隔离及轨迹范围检查。

完整性检查通过：{all(checks.values())}。全部开发/回归材料均有既往查看或上游模型接触风险，不是新盲测。
'''
    (ROOT/'REPORT.md').write_text(text,encoding='utf-8')
    print('V23_FINALIZED',checks,'report',str(ROOT/'REPORT.md'),flush=True)
    if not all(checks.values()):raise RuntimeError('Verification failed')

if __name__=='__main__':main()
