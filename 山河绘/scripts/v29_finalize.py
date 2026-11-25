# -*- coding: utf-8 -*-
from rock_v15_common import LAB,read,jsave,digest,FONT,SMALL,PARENT
from pathlib import Path
import json,shutil
import numpy as np,torch
from PIL import Image,ImageDraw
import v29_targeted_relations as exp

ROOT=exp.ROOT
GROUPS=['development_holdout','fixed_external_development','new_work_development','user_reviewed_old_development','legacy_real_faint_development','legacy_related_probe']
NAMES={'development_holdout':'既有清晰山石6张','fixed_external_development':'复杂外部图4张','new_work_development':'v24新作品留出2张','user_reviewed_old_development':'用户复核R03/R04','legacy_real_faint_development':'历史真实淡线N10/N11','legacy_related_probe':'历史同源探针2张'}

def main():
    supervision=read(ROOT/'supervision.json');node_manifest=read(ROOT/'node_manifest.json');evaluation=read(ROOT/'evaluation.json');groups=evaluation['groups'];cal=read(ROOT/'calibration.json');safe=read(ROOT/'safety_selection.json');budget=read(ROOT/'matched_error_budget.json');summaries={n:read(ROOT/n/'training_summary.json') for n in ['control','targeted']};old_policy=read(exp.prior.ROOT/'split_policy.json');old_data=np.load(exp.prior.ROOT/'trusted_train_pairs.npz');old_fit=~np.isin(old_data['ids'],exp.prior.VALIDATION_IDS);new_data=np.load(ROOT/'targeted_train.npz');new_val=np.load(ROOT/'targeted_validation.npz');baseids=set(old_data['ids'][old_fit]);newids=set(new_data['ids']);dev={r['id'] for r,rgb,refs,valid,bg in exp.prior.exp.cases('development')};heldout={r['id'] for r,rgb,refs,valid,bg in exp.prior.validation_cases()};fit_ids={str(i).replace('replay_','') for i in baseids|newids}
    checks=dict(new_annotation_definition_frozen=digest(ROOT/'visible_labels_before_training.py')==supervision['annotation_sha256']==digest(Path(__file__).with_name('v29_visible_labels.py')),selected_node_manifest_frozen=digest(ROOT/'node_manifest.json')==supervision['node_manifest_sha256'],all_selected_original_graphs_unchanged=all(digest(p)==h for p,h in node_manifest['all_parent_graphs_sha256'].items()),parent_data_split_unchanged=digest(exp.prior.ROOT/'split_policy.json')==supervision['parent_split_sha256'],old_fit_relationships_unchanged=digest(exp.prior.ROOT/'trusted_train_pairs.npz')==old_policy['trusted_train_pair_sha256'],new_relationship_arrays_frozen=all(digest(ROOT/('targeted_'+name+'.npz'))==supervision['counts'][name]['sha256'] for name in ['train','validation']),expanded_validation_frozen=digest(ROOT/'combined_validation.npz')==supervision['combined_validation_sha256'],development_images_excluded_from_fit=not (fit_ids&dev),validation_images_excluded_from_fit=not (fit_ids&{str(i).replace('replay_','') for i in heldout}),all_6_models_completed_600_updates=all(r['completed_updates']==600 for s in summaries.values() for r in s['models']),actual_training_used_gpu=all(r['actual_device'].startswith('cuda') for s in summaries.values() for r in s['models']),matched_old_sample_schedule=all(a['old_sampler_sha256']==b['old_sampler_sha256'] for a,b in zip(summaries['control']['models'],summaries['targeted']['models'])),matched_extra_sample_schedule=all(a['targeted_sampler_sha256']==b['targeted_sampler_sha256'] for a,b in zip(summaries['control']['models'],summaries['targeted']['models'])),all_new_checkpoint_hashes_match=all(digest(ROOT/n/('seed_'+str(r['seed'])+'.pt'))==r['checkpoint_sha256'] for n,s in summaries.items() for r in s['models']),original_strict_gate_not_relaxed=not cal['enabled'] and not budget['original_strict_gate_passed'] and budget['strict_original_gate_unchanged'] and safe['original_gate_not_relaxed_or_relabelled'],original_user_cross_sections_unchanged=digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json')=='8cf074c41402a6f897c92d04db3788ac61186449b63adc514343bba996ff8210',original_raster_parent_unchanged=digest(PARENT)=='8c0988f03a0686fab85a75f4b6b9a82cf21012f4b7a802ea0a90d55d3a316fc6',base_pixel_extractor_unchanged=digest(exp.prior.exp.prev.INIT)==read(exp.prior.ROOT/'manifest.json')['base_raster_sha256'])
    parameters=True;norms=True;initial_kept=True
    for name,summary in summaries.items():
        for rec in summary['models']:
            file=ROOT/name/('seed_'+str(rec['seed'])+'.pt');s=torch.load(file,map_location='cpu',weights_only=False);parent_file=exp.prior.ROOT/'visual'/('seed_'+str(rec['seed'])+'.pt');p=torch.load(parent_file,map_location='cpu',weights_only=False);parameters&=all(bool(torch.isfinite(a).all()) for a in s['model'].values());norms&=np.array_equal(s['mean'],p['mean']) and np.array_equal(s['std'],p['std']) and s['initial_checkpoint_sha256']==digest(parent_file) and s['supervision_sha256']==digest(ROOT/'supervision.json')
            if rec['retained_step']==0:initial_kept&=all(torch.equal(s['model'][k],p['model'][k]) for k in p['model'])
    checks['parameters_finite']=parameters;checks['normalization_and_initialization_preserved']=bool(norms);checks['retained_step_zero_is_actual_initial_model']=initial_kept
    duplicates={r['key'] for r in supervision['duplicate_nodes_excluded']};checks['duplicate_annotation_nodes_not_trained']=not (duplicates&{r['key'] for r in supervision['records']});lookup=exp.source_lookup();parent_manifest=read(exp.prior.ROOT/'manifest.json');graphs={(r['id'],r['view']):r['graph'] for r in parent_manifest['graphs'] if r['partition']=='train'};graphs.update({(r['id'],'base'):r['graph'] for r in parent_manifest['graphs'] if r['partition']=='calibration'});cache={};safe_labels=True;feature_match=True
    for row in supervision['records']:
        file=graphs[row['image'],row['view']]
        if file not in cache:cache[file]=exp.prior.load_graph(file)
        g=cache[file];c=g['candidates'][row['candidate']];safe_labels&=exp.safe_candidate(g,c,lookup[row['image']][1]);feature_match&=bool(np.array_equal(c['feature'],np.asarray(row['feature'],np.float32)))
    checks['new_labels_exclude_invalid_and_user_uncertain_areas']=bool(safe_labels);checks['new_features_match_frozen_graph_candidates']=feature_match
    path_valid=True;footprint=True;documents=0
    for r,rgb,refs,valid,bg in exp.prior.exp.cases('development'):
        h,w=rgb.shape[:2];base=read(ROOT/'trajectories'/'unpaired'/(r['id']+'.json'));baseline=exp.prior.exp.prev.raster(base['paths'],(h,w))
        for variant in evaluation['records'][0]['metrics']:
            doc=read(ROOT/'trajectories'/variant/(r['id']+'.json'));ends=[]
            for event in doc['selected_relations']:ends.extend([tuple(event['a']),tuple(event['b'])])
            path_valid&=len(ends)==len(set(ends)) and doc['raw_edges_used_once'] and doc['physical_z'] is None and not doc['ready_for_robot'] and not doc['actually_drawn'] and not doc['reference_labels_used_at_inference']
            for p in doc['paths']:
                p=np.asarray(p);path_valid&=bool(p.ndim==2 and p.shape[1]==2 and np.isfinite(p).all() and (p[:,0]>=0).all() and (p[:,0]<w).all() and (p[:,1]>=0).all() and (p[:,1]<h).all())
            footprint&=bool(np.array_equal(baseline,exp.prior.exp.prev.raster(doc['paths'],(h,w))));documents+=1
    checks['unique_endpoints_and_valid_offline_trajectories']=path_valid;checks['all_joins_preserve_exact_raster_footprint']=footprint;smoke=read(ROOT/'reference_free_interface_check'/'summary.json');checks['reference_free_entrypoint_completed']=not smoke['reference_labels_used'] and not smoke['reference_label_matching_called'] and not smoke['new_annotation_module_loaded'] and smoke['actual_device'].startswith('cuda')
    jsave(ROOT/'verification.json',dict(checks=checks,passed=bool(all(checks.values())),offline_trajectory_documents=documents,robot_connected=False))
    aggregate={}
    for variant in ['v28','control_conservative','targeted_conservative','control_zero_error','targeted_zero_error','control','targeted']:
        aggregate[variant]={key:sum(r['relationships'][variant][key] for r in evaluation['records']) for key in ['accepted','known_positive','known_negative','unknown']}
    jsave(ROOT/'deployment_gate.json',dict(default_replaced=False,operational_default='v16 protected refinement',strict_continuity_and_error_gate_passed=cal['enabled'],conservative_threshold_is_validation_only=True,development_relationship_counts=aggregate,robot_ready=False,expert_gold=False))
    counts=supervision['counts'];types={}
    for category in exp.TYPES:
        rows=[r for r in supervision['records'] if r['category']==category and r['partition']=='train'];types[category]=dict(positive=sum(r['label']==1 for r in rows),negative=sum(r['label']==0 for r in rows),direct=sum(r['view']=='base' for r in rows),transferred=sum(r['view']!='base' for r in rows))
    lines=['# v29 三类交接处的定向关系监督','',
      '**已完成标注审计、两组GPU关系模型续训及开发集对照。原严格门槛未通过；本轮仍为离线研究候选，没有替换默认模型。**','',
      f"审计40个交接节点（训练30、验证10），针对近平行、粗墨交汇、长轮廓/短纹理三类候选。新增训练关系{counts['train']['positive']}个应续接、{counts['train']['negative']}个不应续接，来自{counts['train']['images']}个图像文件；其中{counts['train']['direct_base_relationships']}个直接查看原图后标注，{counts['train']['transferred_correlated_views']}个为唯一几何匹配的相关视图。验证另外增加{counts['validation']['positive']}正例/{counts['validation']['negative']}负例，扩展为21正例/41负例。",'',
      f"有{len(supervision['uncertain_pairs_omitted'])}个待判组合未新增标签，{len(supervision['duplicate_nodes_excluded'])}个重复局部排除，{len(supervision['conflicting_existing_labels_preserved'])}个与已有可信参考冲突的观察未覆盖原标签。这些是助手的非专家可见关系观察，不能当作画家真实提笔金标准；图像文件数、相关视图数和独立画作数不能混为一谈。",'',
      '负例统一表示不应续接：既包括不同可见笔画，也包括粗墨内部没有独立墨线支撑的小环/毛刺。未明确的宽墨中心关系继续未知。已有可信关系保留；旧开发图没有被新增标注后加入训练或阈值选择。数据、标注定义和验证集均在本轮训练前冻结。','',
      f"GPU为{summaries['targeted']['gpu']} / PyTorch {summaries['targeted']['torch_version']}。普通对照与定向监督各3个模型、每个完成600步小关系网更新，共3600步；像素提取网络保持冻结。两组从对应v28图像关系网初始化，使用同一旧样本及额外样本随机顺序、相同扩展验证集。普通组不反传新增关系损失；定向组额外加入0.5倍新增关系损失。标准化参数固定为v28旧拟合集参数。",'',
      f"普通组保留步数{[r['retained_step'] for r in summaries['control']['models']]}，定向组{[r['retained_step'] for r in summaries['targeted']['models']]}。保留0步表示续训没有胜过初始权重，实际使用的是初始模型，并非把600步末尾结果冒充最佳模型。",'',
      '## 标注类型分布（含相关视图）','', '| 类型候选 | 应续接 | 不应续接 | 直接标注 | 几何转移 |','|---|---:|---:|---:|---:|']
    for category in exp.TYPES:
        x=types[category];lines.append(f"| {exp.LABELS[category]} | {x['positive']} | {x['negative']} | {x['direct']} | {x['transferred']} |")
    lines+=['','## 校准结果与保守对照','',
      '新增验证标签暴露了原先未知连接中的错误。按原有“单笔覆盖至少增加0.005、已知误连为零”等门槛，本轮普通组和定向组均未通过。这个失败记录保留在calibration.json，未放宽门槛或改写成成功。其最高分候选保留为诊断试验，文件中的control/targeted/calibrated_selected轨迹不是已获准默认输出。','',
      f"同分数阈值0.90下，普通模型对验证62个关系单独分类为6个真阳性/2个假阳性，定向模型为4个真阳性/0个假阳性，伴随召回下降。不能只凭同一个分数阈值就宣称模型更准，因为概率分数的校准发生了变化。",'',
      f"因此另在开发图评估前冻结“相同已知验证误连预算”对照：普通组阈值{budget['variants']['control']['threshold']:.6f}，定向组{budget['variants']['targeted']['threshold']:.6f}，竞争间隔均0.05，均高于各自已知验证负例的最高分。普通组实际选3个已知正例、0个已知负例、3个未知，定向组选5个已知正例、0个已知负例、3个未知。其验证单笔覆盖仅分别为{budget['variants']['control']['single']:.3f}与{budget['variants']['targeted']['single']:.3f}，尚未达到原连续性增益门槛。",'',
      'zero_error文件夹名称仅指验证阈值预算，不代表开发集或真实图片零误连。验证集参与了检查点和阈值选择，不是独立准确率测量。','',
      '## 单笔覆盖：匹配验证误连预算后的开发结果','', '| 图组 | v28原续接 | 普通续训对照 | 定向监督保守候选 |','|---|---:|---:|---:|']
    for g in GROUPS:
        x=groups[g];lines.append('| '+NAMES[g]+' | '+' | '.join(f"{x[v]['mean_best_single_path_coverage']:.3f}" for v in ['v28','control_zero_error','targeted_zero_error'])+' |')
    lines+=['','## 所有18张开发图的关系统计','', '| 候选 | 选中连接 | 已知同参考关系 | 已知不应续接 | 仍未知 |','|---|---:|---:|---:|---:|']
    for v,title in [('v28','v28原续接'),('control_zero_error','普通续训，匹配验证预算'),('targeted_zero_error','定向监督，匹配验证预算')]:
        x=aggregate[v];lines.append(f"| {title} | {x['accepted']} | {x['known_positive']} | {x['known_negative']} | {x['unknown']} |")
    lines+=['','## 结论','',
      '定向关系监督改善了扩展验证损失，并提供了更保守的候选，但开发结果没有证明整体判断能力已稳定提高。减少误连也伴随着少接正确笔画；普通对照在同样验证误连预算下也能减少误连，因此不能把保守筛选带来的全部变化归功于新增监督。已有未标注连接仍不足以计算整图连接准确率。','',
      '下一步应优先把分支本身是否真实、同一交汇的全部分支配对与保留未连接选项一起判断。当前模型独立打分每一对分支，对局部图结构和粗墨中的伪分支描述有限；继续对同一小数据集增加轮数尚无稳定收益证据。','',
      '本轮所有轨迹复用了同一像素提取与去重图边，续接前后的栅格墨迹覆盖逐一验证保持一致。颜色表示曲线ID，不表示实际笔顺、机械臂笔宽或墨色。原人工标注、不确定区域与历史权重保持保护；所有开发图都是历史开发样本，不是新的盲测集。没有连接、使能或移动机械臂，没有纸面落笔记录。','',
      f"完整性检查{sum(checks.values())}/{len(checks)}通过。详细数据见supervision.json、classification.json、evaluation.json与verification.json。"]
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['v29_targeted_relations.py','v29_visible_labels.py','v29_finalize.py','rock_v29.ps1']:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    print('V29_VERIFICATION',checks,flush=True);print('V29_DEVELOPMENT_COUNTS',aggregate,flush=True)
    for g in GROUPS:print('V29_GROUP',g,{v:round(groups[g][v]['mean_best_single_path_coverage'],4) for v in ['v28','control_zero_error','targeted_zero_error']},flush=True)
    print('V29_DONE',flush=True)

if __name__=='__main__':main()
