# -*- coding: utf-8 -*-
from rock_v15_common import LAB,read,jsave,digest,FONT,SMALL,PARENT
from pathlib import Path
import json,shutil,hashlib
import numpy as np,torch
from PIL import Image,ImageDraw
import v27_separation_faint as exp

ROOT=exp.ROOT
NAMES={'development_holdout':'既有清晰山石6张','fixed_external_development':'复杂外部图4张','new_work_development':'v24新作品留出2张','user_reviewed_old_development':'用户复核R03/R04','legacy_real_faint_development':'历史真实淡线N10/N11'}

def main():
    m=read(ROOT/'manifest.json');e=read(ROOT/'evaluation.json');cal=read(ROOT/'calibration.json');groups=e['groups'];a=read(ROOT/'control'/'summary.json');b=read(ROOT/'faint'/'summary.json')
    checks=dict(
      base_weights_unchanged=digest(exp.prev.INIT)==m['init_sha256'],
      training_data_unchanged=all(digest(r['data'])==r['data_sha256'] for r in m['records']),
      original_labels_unchanged=all(not r['labels'] or digest(r['labels'])==r['labels_sha256'] for r in m['records']),
      frozen_focus_supervision_unchanged=all(digest(r['focus'])==r['focus_sha256'] for r in m['records']),
      original_user_cross_sections_unchanged=digest(exp.REVIEW/'expert_cross_sections'/'reviewed.json')==m['user_review_sha256']=='8cf074c41402a6f897c92d04db3788ac61186449b63adc514343bba996ff8210',
      both_models_completed_600_updates=a['updates']==b['updates']==600,
      same_training_schedule=a['schedule_sha256']==b['schedule_sha256'] and a['usage']==b['usage'],
      actual_training_device_is_gpu=a['actual_device'].startswith('cuda') and b['actual_device'].startswith('cuda'),
      both_checkpoint_hashes_match=all(digest(ROOT/n/'final.pt')==s['checkpoint_sha256'] for n,s in [('control',a),('faint',b)]),
      calibration_and_evaluation_excluded_from_training=not ({r['id'].replace('replay_','') for r in m['records']}&{r['id'] for r in m['evaluation']}),
      original_raster_parent_unchanged=digest(PARENT)=='8c0988f03a0686fab85a75f4b6b9a82cf21012f4b7a802ea0a90d55d3a316fc6',
      predictions_finite=all(np.isfinite(np.load(p)['prediction']).all() for p in (ROOT/'predictions').rglob('*.npz')),
      checkpoint_parameters_finite=all(bool(torch.isfinite(p).all()) for n in ['control','faint'] for p in torch.load(ROOT/n/'final.pt',map_location='cpu',weights_only=False)['model'].values()),
      strict_center_targets_unchanged=all(digest(exp.prev.V25/'strict_targets'/(r['id']+'.npz'))==r['target_sha256'] for r in read(exp.prev.V25/'reference_audit.json')['records'])
    )
    config_checks=[]
    for n in ['control','faint']:
        cfg=read(ROOT/n/'config.json');config_checks.append(cfg['manifest_sha256']==digest(ROOT/'manifest.json') and cfg['init_sha256']==m['init_sha256'])
    checks['model_configs_match_frozen_sources']=all(config_checks)
    trajectory_ok=True;documents=0
    trajectory_files=list((ROOT/'trajectories').rglob('*.json'))+list((ROOT/'union_trial'/'trajectories').rglob('*.json'))
    for p in trajectory_files:
        doc=read(p);variant=p.parent.name;model='control' if variant.startswith('control') else 'faint' if variant.startswith('faint') else 'base';h,w=np.load(ROOT/'predictions'/model/(p.stem+'.npz'))['prediction'].shape[1:]
        for q in doc['paths']:
            q=np.asarray(q);trajectory_ok&=bool(q.ndim==2 and q.shape[1]==2 and np.isfinite(q).all() and (q[:,0]>=0).all() and (q[:,0]<w).all() and (q[:,1]>=0).all() and (q[:,1]<h).all())
        trajectory_ok&=doc['physical_z'] is None and not doc['ready_for_robot'] and not doc['actually_drawn'] and not doc['reference_labels_used_at_inference'];documents+=1
    checks['all_offline_trajectories_valid']=trajectory_ok
    protected=True;count=0;lookup={r['id']:r for r in m['records']}
    for r in read(LAB/'autonomous_review_v19_20261025'/'manifest_v2.json')['records']:
        z=np.load(r['data'])
        if 'uncertain_exclusion' not in z:continue
        bad=z['uncertain_exclusion']>0;count+=int(bad.sum());rec=lookup['replay_'+r['id']];data=np.load(rec['data']);focus=np.load(rec['focus'])['weight'];protected&=bool(((data['mask']*data['valid'][None])[:,bad]==0).all() and (focus[bad]==0).all())
    checks['uncertain_user_pixels_excluded_from_training_weights']=protected
    jsave(ROOT/'verification.json',dict(checks=checks,passed=bool(all(checks.values())),uncertain_pixels_protected=count,offline_trajectory_documents=documents,robot_connected=False))
    outcomes={}
    for group in exp.GROUPS:
        x=groups[group];old=x['previous_adaptive'];new=x['separation_dedup'];f=x['faint_graph'];control=x['control_graph'];outcomes[group]=dict(point_coverage_delta=new['point_coverage_3px']-old['point_coverage_3px'],single_path_delta=new['mean_best_single_path_coverage']-old['mean_best_single_path_coverage'],mixed_reference_delta=new['mixed_reference_paths_flagged']-old['mixed_reference_paths_flagged'],overlap_proxy_delta=new['duplicate_point_fraction']-old['duplicate_point_fraction'],faint_vs_control_single_delta=f['mean_best_single_path_coverage']-control['mean_best_single_path_coverage'])
    union=read(ROOT/'union_trial'/'evaluation.json') if (ROOT/'union_trial'/'evaluation.json').exists() else None
    union_outcomes={}
    if union:
        for g in exp.GROUPS:
            old=groups[g]['previous_adaptive'];new=union['groups'][g]['base'];union_outcomes[g]=dict(point_coverage_delta=new['point_coverage_3px']-old['point_coverage_3px'],single_path_delta=new['mean_best_single_path_coverage']-old['mean_best_single_path_coverage'],mixed_reference_delta=new['mixed_reference_paths_flagged']-old['mixed_reference_paths_flagged'],overlap_proxy_delta=new['duplicate_point_fraction']-old['duplicate_point_fraction'])
    jsave(ROOT/'deployment_gate.json',dict(default_replaced=False,operational_default='v16 protected refinement',calibration_gate_passed=cal['gate_passed'],clipping_trial_group_outcomes=outcomes,union_trial_group_outcomes=union_outcomes,robot_ready=False,expert_gold_claimed=False))
    weak_summary={}
    for g in exp.GROUPS:
        rows=[r for r in e['records'] if r['group']==g];total=sum(r['metrics']['baseline_graph']['low_contrast_reference_points'] for r in rows);weak_summary[g]=dict(reference_points=total,variants={})
        for variant in ['baseline_graph','control_graph','faint_graph','faint_refined']:
            covered=sum(r['metrics'][variant]['low_contrast_reference_points']*(r['metrics'][variant]['low_contrast_ref_coverage_3px'] or 0) for r in rows);weak_summary[g]['variants'][variant]=dict(covered_points=int(round(covered)),coverage=float(covered/total) if total else None)
    weak_summary['per_image']=[dict(id=r['id'],reference_points=r['metrics']['baseline_graph']['low_contrast_reference_points'],base_coverage=r['metrics']['baseline_graph']['low_contrast_ref_coverage_3px'],control_coverage=r['metrics']['control_graph']['low_contrast_ref_coverage_3px'],faint_coverage=r['metrics']['faint_graph']['low_contrast_ref_coverage_3px']) for r in e['records']]
    jsave(ROOT/'faint_point_summary.json',dict(summary=weak_summary,definition='Point-weighted coverage on original low-contrast reference points; adjacent sampled points are correlated, not independent strokes.',old_evaluation_group_metric_is_image_macro_average=True))
    labels=sum(r['selected_real_faint_center_pixels'] for r in m['records']);support_images=sum(r['selected_real_faint_center_pixels']>0 for r in m['records']);lines=['# v27 交汇分笔、局部去重与真实淡线重点训练','',
      '**已完成两组各600步GPU训练及固定开发集对照。所有结果均是离线实验，没有修改正式默认模型。**','',
      f"GPU：{a['gpu']}，PyTorch {a['torch_version']}。匹配对照训练{a['elapsed_seconds']:.1f}秒，真实淡线加权训练{b['elapsed_seconds']:.1f}秒。两组从同一v25候选初始化，使用相同原图、采样、裁剪、学习率和更新次数；区别是淡线局部损失权重。",'',
      f"训练仍用既有26张图，没有新增独立画作。复用用户已确认的{m['confirmed_faint_cross_sections']}处淡墨横截面，并在{support_images}张训练图已有有效中心监督中筛出{labels}个低对比度中心像素。重点区域只改变采样与损失权重，不改变原来的中心、墨带或背景标签；未标注、不确定区域继续排除。用户确认淡墨存在，不代表初始宽墨中心猜测成为精确专家几何。",'',
      '本轮不使用合成减淡作为新监督。两组均重点采样真实淡线；淡线模型另外把这些区域的监督权重提高到最多3倍。因此普通对照本身也包含真实淡线重点采样，二者比较用于隔离损失加权的作用。','',
      '解码从v26连续轨迹开始：先在墨线交汇处检查轨迹的两侧走向是否对应图像骨架上允许连接的分支，不确定组合先分笔；局部去重只移除与已保留曲线位置接近且方向一致的连续重叠段。相邻但方向不同的交叉线保留。新算法不把端点跨空白强行相连。所有判断只使用模型输出、图像和几何，不读取参考线。','',
      f"解码参数仅在6张原图校准集上选择。分笔加去重：{cal['selected']['config']}，校准门槛通过={cal['gate_passed']}；仅去重：{cal['dedup_trial']['config']}。两种参数均在开发图评估前冻结；即使门槛未通过，仍保存为明确的研究试验。",'',
      '## 分笔与去重：覆盖和代价一起看','', '| 组 | v26点覆盖 | v27点覆盖 | v26单笔覆盖 | v27单笔覆盖 | v26混合链代理 | v27混合链代理 | v26重叠代理 | v27重叠代理 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in exp.GROUPS:
        old=groups[g]['previous_adaptive'];new=groups[g]['separation_dedup'];lines.append(f"| {NAMES[g]} | {old['point_coverage_3px']:.3f} | {new['point_coverage_3px']:.3f} | {old['mean_best_single_path_coverage']:.3f} | {new['mean_best_single_path_coverage']:.3f} | {old['mixed_reference_paths_flagged']:.2f} | {new['mixed_reference_paths_flagged']:.2f} | {old['duplicate_point_fraction']:.1%} | {new['duplicate_point_fraction']:.1%} |")
    lines+=['','## 真实淡线训练：匹配控制与原图结果','', '| 组 | v25骨架基线 | 普通匹配训练+骨架 | 淡线加权+骨架 | 普通匹配训练+新解码 | 淡线加权+新解码 |','|---|---:|---:|---:|---:|---:|']
    for g in exp.GROUPS:
        x=groups[g];lines.append('| '+NAMES[g]+' | '+' | '.join(f"{x[v]['mean_best_single_path_coverage']:.3f}" for v in ['baseline_graph','control_graph','faint_graph','control_refined','faint_refined'])+' |')
    lines+=['','## 低对比度参考点覆盖代理（按点数加权）','',
      '这里按参考点总数加权，避免只有1个淡线参考点的图与有95个点的图占相同权重。evaluation.json原分组值仍保留按图平均以便复现；本表及faint_point_summary.json使用按点加权。相邻点来自同一参考链，不能当作独立样本或独立笔画。','',
      '| 组 | 低对比度参考点数 | 骨架基线 | 普通训练+骨架 | 淡线加权+骨架 | 淡线加权+新解码 |','|---|---:|---:|---:|---:|---:|']
    for g in exp.GROUPS:
        x=weak_summary[g];values=[x['variants'][v]['coverage'] for v in ['baseline_graph','control_graph','faint_graph','faint_refined']];lines.append('| '+NAMES[g]+' | '+str(x['reference_points'])+' | '+' | '.join('无该类点' if v is None else f'{v:.3f}' for v in values)+' |')
    n11=next(r for r in e['records'] if r['id']=='N11');n=n11['metrics']['baseline_graph']['low_contrast_reference_points'];basecovered=round(n*(n11['metrics']['baseline_graph']['low_contrast_ref_coverage_3px'] or 0));faintcovered=round(n*(n11['metrics']['faint_graph']['low_contrast_ref_coverage_3px'] or 0));lines+=['',f'N11上这项局部淡线代理从{basecovered}/{n}个点被覆盖提高到{faintcovered}/{n}，表明真实淡线重点训练对部分弱线有作用。但总体单笔连续性没有同步改善，且N10仅有1个此类参考点，不能用按图平均的高比例宣称淡线问题已解决。']
    if union:
        lines+=['','## 共享图边去重补充试验','',
          '直接裁剪重叠段会损伤另一条完整曲线，因此另做图结构试验：把连续轨迹的线段并集细化成图，每条图边只分配一次，再处理交汇连接。这个试验保留在独立union_trial目录。参数只在同一校准集选择，随后冻结，未以开发结果选参数。', '',
          f"校准选择：{union['choice']['selected']['config']}，门槛通过={union['choice']['gate_passed']}。pair=false表示在交汇处先分笔，尚不代表已经识别真实画家的提笔点。",'',
          '| 组 | v26点覆盖 | 图结构点覆盖 | v26单笔覆盖 | 图结构单笔覆盖 | v26混合链代理 | 图结构混合链代理 | v26重叠代理 | 图结构重叠代理 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for g in exp.GROUPS:
            old=groups[g]['previous_adaptive'];new=union['groups'][g]['base'];lines.append(f"| {NAMES[g]} | {old['point_coverage_3px']:.3f} | {new['point_coverage_3px']:.3f} | {old['mean_best_single_path_coverage']:.3f} | {new['mean_best_single_path_coverage']:.3f} | {old['mixed_reference_paths_flagged']:.2f} | {new['mixed_reference_paths_flagged']:.2f} | {old['duplicate_point_fraction']:.1%} | {new['duplicate_point_fraction']:.1%} |")
        lines+=['','| 组 | 图结构：原权重 | 图结构：普通匹配训练 | 图结构：淡线加权训练 |','|---|---:|---:|---:|']
        for g in exp.GROUPS:
            x=union['groups'][g];lines.append('| '+NAMES[g]+' | '+' | '.join(f"{x[n]['mean_best_single_path_coverage']:.3f}" for n in ['base','control','faint'])+' |')
    lines+=['','## 如何解读','',
      '上述点覆盖和单笔覆盖只针对已有局部参考；混合参考链是错接代理，近邻轨迹点占比是重叠代理，均不是整幅图准确率、实际笔顺或实纸重复率。少量相邻真实笔画也可能被重叠代理计入。分笔可能降低单笔覆盖，因此必须连同点覆盖、碎片数量和可视化判断，不能用删线来制造好看的去重数字。','',
      'N10/N11是真实图片开发探针，未参与本轮训练，曾在历史实验中用于开发评估；没有新的专家完整标注，也不是新的盲测集。既有图也已反复查看。所有自动筛选只来自冻结图像和原有有效监督，没有用待测模型输出生成评判参考。','',
      '原始人工标注、128个不确定像素、v25初始权重与既有实验文件均保持原样。训练、校准和开发图按样本隔离；同源作品关联仍可能存在。未连接、使能或移动机械臂；图中颜色是曲线ID，未映射为笔顺、压笔量或实际墨色。','',
      f"完整性检查：{sum(checks.values())}/{len(checks)}通过。详细逐图指标、轨迹、训练日志和参数见evaluation.json、trajectories、control/faint目录及calibration.json。"]
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['v27_separation_faint.py','v27_finalize.py','v27_union_trial.py','rock_v27.ps1']:shutil.copy2(Path(__file__).with_name(name),snap/name)
    print('V27_VERIFICATION',checks,flush=True)
    for g in exp.GROUPS:
        print('V27_GROUP',g,'OUTCOME',outcomes[g],flush=True)
    print('V27_UNION_OUTCOMES',union_outcomes,flush=True);print('V27_FINISHED',flush=True)

if __name__=='__main__':main()
