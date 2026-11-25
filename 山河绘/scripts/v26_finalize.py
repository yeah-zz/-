# -*- coding: utf-8 -*-
from rock_v15_common import read,jsave,digest,LAB,FONT,SMALL,PARENT
from pathlib import Path
import json,shutil,hashlib
import numpy as np,torch
from PIL import Image,ImageDraw
import v26_tone_coverage as exp

ROOT=exp.ROOT
GROUPS=['development_holdout','fixed_external_development','new_work_development','user_reviewed_old_development']
NAMES={'development_holdout':'既有清晰山石（6张）','fixed_external_development':'固定复杂外部图（4张）','new_work_development':'v24新作品开发留出（2张）','user_reviewed_old_development':'用户复核R03/R04（2张）'}
VARIANTS=[('baseline','v25原权重+骨架'),('decode_only','只修正解码覆盖'),('control_graph','同预算普通微调+骨架'),('control_adaptive','同预算普通微调+新解码'),('tone_only','只加浓淡训练+骨架'),('tone_and_decode','浓淡训练+新解码')]

def main():
    manifest=read(ROOT/'manifest.json');evaluation=read(ROOT/'evaluation.json');groups=evaluation['groups'];control=read(ROOT/'control'/'summary.json');tone=read(ROOT/'tone'/'summary.json');device=read(ROOT/'device.json')
    checks={
      'initial_v25_checkpoint_unchanged':digest(exp.INIT)==manifest['init_sha256'],
      'all_frozen_training_data_unchanged':all(digest(r['data'])==r['sha256'] for r in manifest['records']),
      'all_frozen_label_files_unchanged':all(not r['label'] or digest(r['label'])==r['label_sha256'] for r in manifest['records']),
      'both_models_completed_600_updates':control['updates']==tone['updates']==600,
      'matched_original_sample_schedule':control['sampler_sha256']==tone['sampler_sha256'] and control['sample_usage']==tone['sample_usage'],
      'actual_training_device_is_gpu':control['actual_device'].startswith('cuda') and tone['actual_device'].startswith('cuda'),
      'gpu_forward_and_backward_verified':device['forward_backward_verified'],
      'both_checkpoint_hashes_match':all(digest(ROOT/n/'final.pt')==s['checkpoint_sha256'] for n,s in [('control',control),('tone',tone)]),
      'calibration_and_development_ids_not_in_training':not ({r['id'] for r in manifest['records']}&{r['id'] for r in manifest['evaluation']}),
      'all_predictions_finite':all(np.isfinite(np.load(p)['prediction']).all() for p in (ROOT/'predictions').rglob('*.npz')),
      'original_user_review_unchanged':digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json')=='8cf074c41402a6f897c92d04db3788ac61186449b63adc514343bba996ff8210',
      'original_raster_parent_unchanged':digest(PARENT)=='8c0988f03a0686fab85a75f4b6b9a82cf21012f4b7a802ea0a90d55d3a316fc6',
      'strict_v25_center_targets_unchanged':all(digest(exp.V25/'strict_targets'/(r['id']+'.npz'))==r['target_sha256'] for r in read(exp.V25/'reference_audit.json')['records'])
    }
    paths_valid=True;trajcount=0
    for p in (ROOT/'trajectories').rglob('*.json'):
        doc=read(p);condition=p.parents[1].name;variant=p.parent.name;id=p.stem;pred=np.load(ROOT/'predictions'/('base' if variant in ['baseline','decode_only'] else 'control' if variant.startswith('control') else 'tone')/condition/(id+'.npz'))['prediction'];h,w=pred.shape[1:]
        for a in doc['paths']:
            a=np.asarray(a);paths_valid&=bool(a.ndim==2 and a.shape[1]==2 and np.isfinite(a).all() and (a[:,0]>=0).all() and (a[:,0]<w).all() and (a[:,1]>=0).all() and (a[:,1]<h).all())
        paths_valid&=doc['physical_z'] is None and not doc['ready_for_robot'] and not doc['actually_drawn'] and not doc['reference_labels_used_at_inference'];trajcount+=1
    checks['all_offline_trajectories_finite_bounded_and_not_robot_ready']=paths_valid
    uncertain=0;protected=True;lookup={r['id']:r for r in manifest['records']}
    for r in read(LAB/'autonomous_review_v19_20261025'/'manifest_v2.json')['records']:
        z=np.load(r['data'])
        if 'uncertain_exclusion' not in z:continue
        bad=z['uncertain_exclusion']>0;uncertain+=int(bad.sum());current=np.load(lookup['replay_'+r['id']]['data']);protected&=bool(((current['mask']*current['valid'][None])[:,bad]==0).all())
    checks['user_uncertainty_excluded_from_all_supervision_and_consistency']=protected
    jsave(ROOT/'verification.json',dict(checks=checks,passed=bool(all(checks.values())),uncertain_pixels_protected=uncertain,trajectory_documents=trajcount,actual_gpu=device['gpu'],robot_connected=False))
    # Eligibility is a research check; no candidate replaces the operational v16.
    eligibility={}
    for variant in ['tone_only','decode_only','tone_and_decode']:
        failures=[]
        for g in GROUPS:
            b=groups['original'][g]['baseline'];a=groups['original'][g][variant]
            if a['point_coverage_3px']<b['point_coverage_3px']-.01:failures.append(g+': point coverage regressed')
            if a['mean_best_single_path_coverage']<b['mean_best_single_path_coverage']-.005:failures.append(g+': single-path coverage regressed')
            if a['mixed_reference_paths_flagged']>b['mixed_reference_paths_flagged']+.25:failures.append(g+': mixed-reference proxy increased')
            if a['paper_path_density'] is not None and b['paper_path_density'] is not None and a['paper_path_density']>b['paper_path_density']+.0005:failures.append(g+': known-paper false lines increased')
        eligibility[variant]=dict(eligible_for_more_validation=not failures,failures=failures)
    jsave(ROOT/'deployment_gate.json',dict(variants=eligibility,new_default_enabled=False,operational_default='v16 protected refinement',robot_ready=False,expert_gold_available=False))
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['v26_tone_coverage.py','v26_finalize.py','rock_v26.ps1']:
        shutil.copy2(Path(__file__).with_name(name),snap/name)
    lines=['# v26 浓淡一致性训练与复杂图覆盖实验','',
      '**本轮完成：GPU上两组各600次优化；匹配相同原始样本与裁剪顺序；固定权重解码对照、浓淡训练对照、两项组合对照。未修改正式默认模型。**','',
      f"实际设备：{device['gpu']}；PyTorch {device['torch_version']} / CUDA {device['cuda_runtime']}。对照训练 {control['elapsed_seconds']:.1f} 秒，浓淡训练 {tone['elapsed_seconds']:.1f} 秒。已有兼容运行库位于E盘；v25的CPU回退是旧PyTorch抢先导入导致，本轮修正了导入顺序，没有安装依赖。",'',
      '训练使用既有26张训练图，其中6张为v24新增的局部山石图，另有11张清晰图和9张复核回放图。本轮没有新增独立画作。参考线仍是部分人工复核与非专家辅助参考；未描线区域保持未知。用户原始标注和128个不确定像素保持保护。','',
      '浓淡输入在估计纸面颜色基础上压缩墨迹对比，训练比例0.30/0.55/0.80，保持参考几何不变。监督损失在已知中心、墨带和方向掩码内计算；一致性只覆盖已知区域。另用未用于训练的比例0.40作合成减淡压力测试。合成减淡效果不等同真实淡墨准确率。','',
      '新解码沿同一预测中心概率与方向场积分，起点按64像素网格分配，每格最多16个起点，并抑制重复；取消全图40起点限制。每方向安全预算由图像对角线确定，命中次数单独记录。阈值只用6张原图校准集和基础权重选择，随后全部模型使用相同阈值。它仍然是方向场积分，没有训练整笔控制点回归网络。','',
      '**主要发现：新解码改善了复杂组的单笔覆盖，但混合参考链和重叠轨迹代理增加，不能直接视为精度提升。浓淡训练单独使用没有稳定胜过匹配的普通微调，在清晰原图组还有退步。三种实验候选均未通过全组门槛，本轮不采用新默认。**','',
      '## 原图结果（部分参考线指标）','',
      '| 组 | 原基线单笔覆盖 | 只改解码 | 普通微调 | 浓淡训练 | 两项结合 |','|---|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        x=groups['original'][g];lines.append('| '+NAMES[g]+' | '+' | '.join(f"{x[v]['mean_best_single_path_coverage']:.3f}" for v in ['baseline','decode_only','control_graph','tone_only','tone_and_decode'])+' |')
    lines+=['','## 统一减淡压力测试（合成，比例0.40）','', '| 组 | 基线单笔覆盖 | 普通微调 | 浓淡训练 | 两项结合 |','|---|---:|---:|---:|---:|']
    for g in GROUPS:
        x=groups['synthetic_fade_040'][g];lines.append('| '+NAMES[g]+' | '+' | '.join(f"{x[v]['mean_best_single_path_coverage']:.3f}" for v in ['baseline','control_graph','tone_only','tone_and_decode'])+' |')
    lines+=['','## 原图低对比度参考点覆盖代理','', '| 组 | 基线 | 普通微调 | 浓淡训练 | 两项结合 |','|---|---:|---:|---:|---:|']
    for g in GROUPS:
        x=groups['original'][g];values=[x[v]['low_contrast_ref_coverage_3px'] for v in ['baseline','control_graph','tone_only','tone_and_decode']];lines.append('| '+NAMES[g]+' | '+' | '.join('无该类点' if z is None else f'{z:.3f}' for z in values)+' |')
    lines+=['','## 原图覆盖与误连、重叠代价','', '| 组 | 基线点覆盖 | 两项结合点覆盖 | 基线混合链代理 | 两项结合混合链代理 | 基线轨迹重叠代理 | 两项结合重叠代理 |','|---|---:|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        b=groups['original'][g]['baseline'];a=groups['original'][g]['tone_and_decode'];lines.append(f"| {NAMES[g]} | {b['point_coverage_3px']:.3f} | {a['point_coverage_3px']:.3f} | {b['mixed_reference_paths_flagged']:.2f} | {a['mixed_reference_paths_flagged']:.2f} | {b['duplicate_point_fraction']:.1%} | {a['duplicate_point_fraction']:.1%} |")
    lines+=['','轨迹重叠代理统计输出曲线点落到前面曲线周围约1–2像素范围内的比例；它也可能计入真实相邻笔画，不能当作实纸重复率。此处与混合参考链、可视化共同判断。H02的基础权重新解码使用477个起点，X04使用221个起点，二者安全预算均未命中，证明此前全图40起点确实限制了覆盖；增加起点之后仍需要解决分笔和去重。','',
      '## 下一步依据','',
      '保留本轮冻结的结果。后续优先处理重复路径的局部去重与交汇处分笔，让单笔覆盖的收益不伴随误连增多；浓淡侧应补充图像证据明确的真实淡线局部监督，按原图低对比度分层检验。仅继续压低像素阈值或增加当前小数据集训练步数，尚无稳定收益证据。','',
      '## 解读与局限','',
      '单笔覆盖衡量参考链有多少能由同一条输出轨迹覆盖；点覆盖只要求附近有输出。二者均不是整幅图准确率。低对比度按原图局部墨迹对比划分，是代理指标，不能替代真实淡墨专家标注。错接用混合参考链代理检查；未标注墨线处的全图精度无法由这些局部参考计算。','',
      '原图与合成减淡使用相同参考和相同有效区域。固定开发图已反复查看，不是新的盲测集；同一画作的相似局部可能存在来源关联。模型输出在解码时不读取参考线；参考只用于训练或事后评估。','',
      '候选门槛要求四个主要原图组的点覆盖、单笔覆盖不明显退步，混合参考链与已知纸面误线不增；详细失败项见deployment_gate.json。无论门槛结果如何，本轮均只保留离线研究权重，未替换正式默认。','',
      '全部颜色仅标识曲线ID，未验证笔顺；笔宽是图像上的表观宽度，未映射为机械臂压笔量。没有连接、使能、移动机械臂，也没有实纸落笔记录。','',
      f"完整性检查：{sum(checks.values())}/{len(checks)}通过；GPU训练、原始标注保护、权重哈希、留出排除和轨迹边界均记录在verification.json。"]
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    # Compact preview uses actual saved predicted paths only.
    lookup={r['id']:rgb for r,rgb,refs,valid,bg in exp.old_eval.cases('development')};keys=['H01','R04','X04'];board=Image.new('RGB',(2000,345*len(keys)+35),'#f4f2eb');d=ImageDraw.Draw(board);d.text((5,4),'v26 原图对照：输入／基线／只改解码／只改浓淡训练／两项结合',font=FONT,fill='black')
    for i,key in enumerate(keys):
        rgb=lookup[key];images=[('输入',rgb)]
        for variant,title in [('baseline','基线'),('decode_only','只改解码'),('tone_only','只改浓淡训练'),('tone_and_decode','两项结合')]:
            paths=read(ROOT/'trajectories'/'original'/variant/(key+'.json'))['paths'];images.append((title,exp.v21.line_view(rgb.shape[:2],paths,True)))
        for j,(title,image) in enumerate(images):d.text((400*j+5,35+345*i),key+' '+title,font=SMALL,fill='black');exp.v21.paste(board,image,400*j+5,60+345*i,390,275)
    board.save(ROOT/'OVERVIEW.jpg',quality=95)
    print('V26_VERIFICATION',checks,flush=True);print('V26_ELIGIBILITY',eligibility,flush=True)
    for c in ['original','synthetic_fade_040']:
        for g in GROUPS:print('V26_RESULT',c,g,{k:round(groups[c][g][k]['mean_best_single_path_coverage'],4) for k,_ in VARIANTS},flush=True)

if __name__=='__main__':main()
