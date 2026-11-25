# -*- coding: utf-8 -*-
from rock_v15_common import LAB,read,jsave,digest,FONT,SMALL,PARENT
from pathlib import Path
import json,shutil,hashlib
import numpy as np,torch
from PIL import Image,ImageDraw
import v28_stroke_relations as exp

ROOT=exp.ROOT
GROUPS=['development_holdout','fixed_external_development','new_work_development','user_reviewed_old_development','legacy_real_faint_development']
NAMES={'development_holdout':'既有清晰山石6张','fixed_external_development':'复杂外部图4张','new_work_development':'v24新作品留出2张','user_reviewed_old_development':'用户复核R03/R04','legacy_real_faint_development':'历史真实淡线N10/N11'}

def main():
    manifest=read(ROOT/'manifest.json');split=read(ROOT/'split_policy.json');evaluation=read(ROOT/'evaluation.json');choice=read(ROOT/'calibration.json');groups=evaluation['groups'];summaries={n:read(ROOT/n/'training_summary.json') for n in ['geometry','visual']};z=np.load(ROOT/'trusted_train_pairs.npz');fit=~np.isin(z['ids'],exp.VALIDATION_IDS)
    checks=dict(base_pixel_weights_unchanged=digest(exp.exp.prev.INIT)==manifest['base_raster_sha256'],all_original_training_images_and_labels_unchanged=all(digest(r['data'])==r['data_sha256'] and digest(r['labels'])==r['labels_sha256'] for r in manifest['inputs']),raw_relationship_arrays_unchanged=all(digest(ROOT/(name+'_pairs.npz'))==manifest['counts'][name]['sha256'] for name in ['train','calibration']),trusted_training_relationships_frozen=digest(ROOT/'trusted_train_pairs.npz')==split['trusted_train_pair_sha256'],validation_relationships_frozen=digest(ROOT/'relation_validation_pairs.npz')==split['validation_pair_sha256'],manifest_matches_split_policy=digest(ROOT/'manifest.json')==split['manifest_sha256'],all_validation_views_excluded_from_optimizer=not (set(z['ids'][fit])&set(exp.VALIDATION_IDS)),all_18_development_ids_excluded_from_optimizer=not ({r['id'] for r,rgb,refs,valid,bg in exp.exp.cases('development')}&{str(x).replace('replay_','') for x in z['ids'][fit]}),six_models_completed_800_updates=all(r['completed_updates']==800 for s in summaries.values() for r in s['models']),actual_training_used_gpu=all(r['actual_device'].startswith('cuda') for s in summaries.values() for r in s['models']),all_checkpoint_hashes_match=all(digest(ROOT/n/('seed_'+str(r['seed'])+'.pt'))==r['checkpoint_sha256'] for n,s in summaries.items() for r in s['models']),matched_sampling_between_two_relation_models=all(a['schedule_sha256']==b['schedule_sha256'] for a,b in zip(summaries['geometry']['models'],summaries['visual']['models'])),original_user_review_unchanged=digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json')=='8cf074c41402a6f897c92d04db3788ac61186449b63adc514343bba996ff8210',original_raster_parent_unchanged=digest(PARENT)=='8c0988f03a0686fab85a75f4b6b9a82cf21012f4b7a802ea0a90d55d3a316fc6')
    norms=True;finite=True
    for name,summary in summaries.items():
        n=26 if name=='geometry' else len(exp.FEATURES);mu=z['x'][fit,:n].mean(0);sd=np.maximum(z['x'][fit,:n].std(0),.05)
        for r in summary['models']:
            checkpoint=torch.load(ROOT/name/('seed_'+str(r['seed'])+'.pt'),map_location='cpu',weights_only=False);norms&=np.allclose(checkpoint['mean'],mu) and np.allclose(checkpoint['std'],sd) and checkpoint['split_policy_sha256']==digest(ROOT/'split_policy.json');finite&=all(bool(torch.isfinite(p).all()) for p in checkpoint['model'].values())
    checks['feature_normalization_uses_fit_images_only']=bool(norms);checks['all_relation_parameters_finite']=finite
    valid_paths=True;footprint=True;documents=0
    for r,rgb,refs,valid,bg in exp.exp.cases('development'):
        h,w=rgb.shape[:2];base=read(ROOT/'trajectories'/'unpaired'/(r['id']+'.json'));base_raster=exp.exp.prev.raster(base['paths'],(h,w))
        for variant in ['unpaired','geometric','learned_geometry','learned_visual','learned_selected']:
            doc=read(ROOT/'trajectories'/variant/(r['id']+'.json'));used=[]
            for event in doc['selected_relations']:used.extend([tuple(event['a']),tuple(event['b'])])
            valid_paths&=len(used)==len(set(used)) and doc['raw_edges_used_once'] and not doc['ready_for_robot'] and not doc['actually_drawn'] and doc['physical_z'] is None and not doc['reference_labels_used_at_inference']
            for p in doc['paths']:
                p=np.asarray(p);valid_paths&=bool(p.ndim==2 and p.shape[1]==2 and np.isfinite(p).all() and (p[:,0]>=0).all() and (p[:,0]<w).all() and (p[:,1]>=0).all() and (p[:,1]<h).all())
            footprint&=bool(np.array_equal(base_raster,exp.exp.prev.raster(doc['paths'],(h,w))));documents+=1
    checks['unique_endpoint_pairing_and_valid_offline_paths']=valid_paths;checks['joining_preserves_exact_raster_footprint']=footprint
    # Verify that the explicitly excluded user uncertainty supplies no fitted
    # branch relation. A context patch may still contain unlabelled background.
    bad_maps={};old=read(LAB/'autonomous_review_v19_20261025'/'manifest_v2.json')['records']
    for r in old:
        a=np.load(r['data'])
        if 'uncertain_exclusion' in a and (a['uncertain_exclusion']>0).any():bad_maps['replay_'+r['id']]=a['uncertain_exclusion']>0
    violations=[]
    for r in manifest['graphs']:
        if r['partition']!='train' or r['id'] not in split['fit_ids'] or r['id'] not in bad_maps:continue
        g=exp.load_graph(r['graph']);bad=bad_maps[r['id']]
        for i,c in enumerate(g['candidates']):
            if c['label'] is None:continue
            q=[np.asarray(c['node'])[None]]
            for endpoint in [c['a'],c['b']]:q.append(exp.exp.rel.endpoint_segment(g['chains'][endpoint[0]],endpoint[1])[1])
            xy=np.rint(np.concatenate(q)).astype(int);xy[:,0]=np.clip(xy[:,0],0,bad.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,bad.shape[0]-1)
            if bad[xy[:,1],xy[:,0]].any():violations.append(dict(image=r['id'],view=r['view'],candidate=i))
    checks['no_relation_target_on_explicit_user_uncertainty']=not violations
    smoke=read(ROOT/'reference_free_interface_check'/'summary.json');checks['reference_free_entrypoint_completed']=not smoke['reference_labels_used'] and not smoke['reference_label_matching_called'] and smoke['device'].startswith('cuda')
    jsave(ROOT/'verification.json',dict(checks=checks,passed=bool(all(checks.values())),uncertainty_relation_violations=violations,trajectory_documents=documents,robot_connected=False))
    primary=[r for r in evaluation['records'] if r['group'] in GROUPS];all_rows=evaluation['records'];negative=sum(r['relationships']['learned_selected']['known_negative'] for r in primary);unknown=sum(r['relationships']['learned_selected']['unknown'] for r in primary);positive=sum(r['relationships']['learned_selected']['known_positive'] for r in primary)
    jsave(ROOT/'deployment_gate.json',dict(default_replaced=False,operational_default='v16 protected refinement',calibration_gate_passed=choice['enabled'],primary_development_known_positive_joins=positive,primary_development_known_negative_joins=negative,primary_development_unknown_joins=unknown,ready_for_further_validation=negative==0 and choice['enabled'],reference_chain_labels_not_artist_stroke_gold=True,robot_ready=False))
    lines=['# v28 去重图边的同笔关系训练','',
      '**本轮训练目标是图边之间的同笔关系。像素提取网络保持冻结；未修改正式默认模型。**','',
      f"训练了几何特征关系网与图像特征关系网，两组各3个随机种子、每个完成800次GPU更新，共4800次小关系网更新。使用RTX 5060 / PyTorch {summaries['visual']['torch_version']}，不是对像素提取主干重新训练4800步。各模型保留内部验证损失最低的100步检查点，实际保留步数见下文及训练日志。",'',
      f"原始关系池来自21张有可用关系的训练图，共186正例/175负例，含同一图的3种相关提取视图。严格筛除短毛刺、墨迹支撑不足的关系后保留331个相关关系；最终优化使用18张图的{split['fit_positive']}正例/{split['fit_negative']}负例。3张图U03、stage_035、replay_R02的所有视图都不进入关系网优化。扩展验证集由原6张校准图加这3张图组成，其中5张图提供{split['validation_positive']}正例/{split['validation_negative']}负例。没有新增独立画作，也没有新增专家提笔金标准。",'',
      '同一参考链两侧、位置方向明确的分支作为正例；对应不同参考链的分支作为负例；靠近多条线、标注不确定或缺乏墨迹支撑的关系不进入损失，也不当负例。数据原始记录与质量过滤前划分均保留。参考链身份只是当前线稿任务的代理，并不代表历史画家的真实分笔。','',
      '几何网使用26项长度、弯折、方向和墨迹证据；图像网另外使用冻结像素网络在局部图像中的分支及节点特征。图像网没有把参考线或参考链ID作为输入。拟合样本与内部验证图按图像隔离，验证图只选择检查点和阈值；父级像素网络历史上见过部分这些图，因此它们不是全链路盲测。','',
      '续接只发生在共享节点的去重图边之间，每个端点最多连接一次，每条原始图边只使用一次；不添加跨空白连接，也不重新使用已经绘制的图边。新增闭环连接被阻止，原有闭合单链保留。推理选择和轨迹拼接先完成，再用参考线作事后评估。','',
      f"验证选择：{choice['selected']['variant']}关系网，概率≥{choice['selected']['threshold']:.2f}，竞争间隔≥{choice['selected']['margin']:.2f}。验证单笔覆盖由{choice['selected']['baseline_single']:.3f}变为{choice['selected']['single']:.3f}；选中连接里已知正例{choice['selected']['known_positive']}、已知负例{choice['selected']['known_negative']}、未知{choice['selected']['unknown']}。验证门槛通过不等于实际连接都正确。",'',
      f"**开发集发现：主要5个图组上，模型选中{positive}个已知同参考链连接、{negative}个已知异参考链连接，另有{unknown}个连接缺乏明确参考。出现已知误连，因此本轮不替换正式默认。**",'',
      '## 单笔覆盖（局部参考指标）','', '| 图组 | 去重后不续接 | 几何规则续接 | 几何关系网 | 图像关系网（校准选择） |','|---|---:|---:|---:|---:|']
    for g in GROUPS:
        x=groups[g];lines.append('| '+NAMES[g]+' | '+' | '.join(f"{x[v]['mean_best_single_path_coverage']:.3f}" for v in ['unpaired','geometric','learned_geometry','learned_selected'])+' |')
    lines+=['','## 误连与重复约束','', '| 图组 | 不续接混合链代理 | 规则续接混合链代理 | 学习续接混合链代理 | 不续接重叠代理 | 学习续接重叠代理 | 学习连接已知负例 | 学习连接未知 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        x=groups[g];u=x['unpaired'];geom=x['geometric'];learn=x['learned_selected'];lines.append(f"| {NAMES[g]} | {u['mixed_reference_paths_flagged']:.2f} | {geom['mixed_reference_paths_flagged']:.2f} | {learn['mixed_reference_paths_flagged']:.2f} | {u['duplicate_point_fraction']:.1%} | {learn['duplicate_point_fraction']:.1%} | {learn['known_negative_joins']} | {learn['unknown_joins']} |")
    lines+=['','## 关系分类检查','',
      '保留验证关系仅38个，不能用其比例宣称专家准确率。被选图像模型在这些关系上单独按概率阈值分类：6/16个正例过阈值、0/22个负例过阈值；再经过竞争和拓扑约束后实际选择5个已知正例。开发集仍出现已知误连，说明有限验证样本尚不能保证泛化。','',
      '## 局限与下一步依据','',
      '续接后的图像位置覆盖应保持不变，本轮检验实际保存轨迹的栅格覆盖是否完全一致；增加的是连续性，不是补画未检测到的墨线。图中颜色只区分曲线ID，不代表实际笔顺、笔宽或墨色。','',
      '学习续接恢复部分连续性，但整体没有稳定超过既有几何规则；不能把启用续接本身的收益全部归功于新关系网络。后续优先补足当前错误连接和未知连接所在类型的可信关系监督，特别是相邻近平行笔画、粗墨交汇和主轮廓与皴线交接；需要继续按图像隔离验证，而不是继续对这219个拟合关系无限加轮数。','',
      '所有开发图均为已使用的开发样本；N10/N11是历史真实淡线探针，不是新的盲测。没有采集画家动作或真实提笔记录。原始人工标注、不确定像素、父模型及历史实验保持保护。未连接、使能或移动机械臂，也没有实纸落笔。','',
      f"完整性检查：{sum(checks.values())}/{len(checks)}通过。详见verification.json；逐图连接的已知正、负、未知数量见evaluation.json。"]
    retained={name:[r['retained_step'] for r in summary['models']] for name,summary in summaries.items()}
    retention_text=f"每100步检查一次内部验证损失并保留最佳检查点；几何组保留步数为{retained['geometry']}，图像组为{retained['visual']}。"
    lines=[line.replace('各模型保留内部验证损失最低的100步检查点，实际保留步数见下文及训练日志。',retention_text) for line in lines]
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['v28_stroke_relations.py','v28_finalize.py','rock_v28.ps1']:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    lookup={r['id']:rgb for r,rgb,refs,valid,bg in exp.exp.cases('development')};keys=['H02','R04','X03','N11'];board=Image.new('RGB',(1600,310*len(keys)+40),'#f4f2eb');d=ImageDraw.Draw(board);d.text((5,5),'v28 同笔连接：输入／去重后未续接／几何规则／关系模型候选（离线，尚未替换默认）',font=FONT,fill='black')
    for i,key in enumerate(keys):
        rgb=lookup[key];images=[('输入',rgb)]
        for name,title in [('unpaired','去重后不续接'),('geometric','几何规则续接'),('learned_selected','关系模型续接')]:images.append((title,exp.exp.prev.v21.line_view(rgb.shape[:2],read(ROOT/'trajectories'/name/(key+'.json'))['paths'],True)))
        for j,(title,im) in enumerate(images):d.text((400*j+5,40+310*i),key+' '+title,font=SMALL,fill='black');exp.exp.prev.v21.paste(board,im,400*j+5,65+310*i,390,240)
    board.save(ROOT/'OVERVIEW.jpg',quality=95);print('V28_VERIFICATION',checks,flush=True)
    for g in GROUPS:print('V28_GROUP',g,{v:round(groups[g][v]['mean_best_single_path_coverage'],4) for v in ['unpaired','geometric','learned_selected']},'wrong',groups[g]['learned_selected']['known_negative_joins'],'unknown',groups[g]['learned_selected']['unknown_joins'],flush=True)
    print('V28_DONE',positive,negative,unknown,flush=True)

if __name__=='__main__':main()
