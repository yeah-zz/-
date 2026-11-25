# -*- coding: utf-8 -*-
"""V19 final audit/report: learned screening, fixed uncertainty exclusion, no false promotion."""
from auto_extract_v19 import *


def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);manifest=read(ROOT/'manifest_v2.json');current='reviewed_v2';control='control_v2';versions={};results={}
    for name,file,side in [('v16_baseline',ROOT/current/'evaluation'/'summary.json','before'),('v17',OLD/'aligned'/'evaluation'/'summary.json','after'),('v18',V18/'profiles'/'evaluation'/'summary.json','after'),('v19_control',ROOT/control/'evaluation'/'summary.json','after'),('v19_reviewed',ROOT/current/'evaluation'/'summary.json','after')]:
        rows={r['id']:r for r in read(file)['records']};versions[name]=(rows,side);rr=[rows[k] for k in ['R01','R02','R03','R04']];dev=[rows[k] for k in ['R03','R04']];diag=side+'_diagnostics';n=sum(r[diag]['faint_samples'] for r in rr);bn=sum(r[diag]['bold_samples'] for r in rr);results[name]=dict(R_mean_single=float(np.mean([r[side]['mean_best_single_path_coverage'] for r in rr])),R03_R04_mean_single=float(np.mean([r[side]['mean_best_single_path_coverage'] for r in dev])),R_faint_weighted=sum((r[diag]['faint_reference_coverage'] or 0)*r[diag]['faint_samples'] for r in rr)/max(1,n),R_faint_samples=n,R_bold_double_weighted=sum((r[diag]['bold_multiple_center_fraction'] or 0)*r[diag]['bold_samples'] for r in rr)/max(1,bn),N10_single=rows['N10'][side]['mean_best_single_path_coverage'],N11_single=rows['N11'][side]['mean_best_single_path_coverage'])
    jsave(ROOT/'metrics_summary.json',results);qcmodel=reviewer.load_reviewer();cases={r['id']:r for r in prior_eval.evaluation_records()};ids=['R03','R04','N06','N10'];board=Image.new('RGB',(1600,360*len(ids)+75),'#f5f3ec');d=ImageDraw.Draw(board);d.text((6,4),'v19 真实图对照：R03/R04 本轮未训练；N06 训练；N10 开发图。新提取权重未采用。',font=FONT,fill='black');d.text((6,30),'自动检查点：绿＝局部一致；橙＝待定保留；红＝疑似背景。颜色不是人工真值。',font=SMALL,fill='black');qcs={}
    for i,k in enumerate(ids):
        rgb=np.asarray(Image.open(cases[k]['image']).convert('RGB'));h,w=rgb.shape[:2];oldp=read(OLD/'aligned'/'evaluation'/k/'trajectory.json')['paths'];newp=read(ROOT/current/'evaluation'/k/'trajectory.json')['paths'];qc=review_paths(rgb,newp,qcmodel);qcs[k]=qc['counts'];(ROOT/'automatic_checks').mkdir(exist_ok=True);jsave(ROOT/'automatic_checks'/(k+'.json'),qc);Image.fromarray(overlay(rgb,qc)).save(ROOT/'automatic_checks'/(k+'.png'))
        for j,(title,a) in enumerate([('原图',rgb),('v17 模型轨迹',line_view((h,w),oldp,True)),('v19 校正训练后的候选',line_view((h,w),newp,True)),('自动复核（不改原轨迹）',overlay(rgb,qc))]):
            d.text((j*400+6,75+i*360),k+' '+title,font=SMALL,fill='black');paste(board,a,400*j+5,105+i*360,390,320)
    board.save(ROOT/'REAL_COMPARISON.jpg',quality=95)
    # Rebuild borrowed captions so artifacts name their actual experiment/version.
    for variant in [control,current]:
        folder=ROOT/variant/'evaluation';obj=read(folder/'summary.json');obj.update(experiment='v19',selected_variant=variant,reference_at_inference=False,expert_truth_replaced_by_model=False,weights_promoted=False,uncertain_excluded_all_losses=True);jsave(folder/'summary.json',obj)
        for r in obj['records']:
            k=r['id'];rgb=np.asarray(Image.open(cases[k]['image']).convert('RGB'));h,w=rgb.shape[:2];doc=read(cases[k]['labels']);reference=[s['points'] for s in doc['strokes']];oldp=read(legacy.V16/'protected_refinement'/k/'trajectory.json')['paths'];newp=read(folder/k/'trajectory.json')['paths'];page=Image.new('RGB',(1600,430),'#f5f3ec');pd=ImageDraw.Draw(page)
            for j,(title,a) in enumerate([('原图',rgb),('局部参考（非预测）',line_view((h,w),reference)),('v16 保留基线',line_view((h,w),oldp,True)),('v19 '+variant+' 候选轨迹',line_view((h,w),newp,True))]):pd.text((j*400+5,5),k+' '+title,font=SMALL,fill='black');paste(page,a,400*j+6,38,388,383)
            page.save(folder/k/'comparison.jpg',quality=94)
        for title,group in [('reviewed_rocks',['R01','R02','R03','R04']),('new_rocks',['N01','N06','N10','N11']),('transfer',['T01','T02','U_A06'])]:
            page=Image.new('RGB',(1600,430*len(group)+35),'#f5f3ec');pd=ImageDraw.Draw(page);pd.text((8,4),'v19 '+variant+'；R01/R02 训练；R03/R04 开发留出（非全新盲测）',font=FONT,fill='black')
            for i,k in enumerate(group):page.paste(Image.open(folder/k/'comparison.jpg'),(0,35+430*i))
            page.save(folder/(title+'.jpg'),quality=94)
    # Verify actual artifacts and explicit exclusion in both variants.
    data_ok=True;uncertain_count=0;uncertain_excluded=True;source_ok=True
    for r in manifest['records']:
        data_ok &= digest(r['data'])==r['data_sha256'];source_ok &= digest(r['image'])==r['image_sha256'] and digest(r['labels'])==r['label_sha256'];z=np.load(r['data']);u=z['uncertain_exclusion']>.5;uncertain_count+=int(u.sum());uncertain_excluded &= bool((z['mask'][:,u]==0).all() and (z['control_mask'][:,u]==0).all());unknown=(z['mask'][0]<.1)&((z['teacher_center']>.9)|(z['teacher_center']<.1))&~u;uncertain_excluded &= not bool(unknown[u].any())
    model_finite={};plans_ok=True;unchanged_colors=0
    for variant in [control,current]:
        checkpoint=torch.load(ROOT/variant/'final.pt',map_location='cpu',weights_only=False);model_finite[variant]=all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values());uncertain_excluded &= checkpoint['config']['uncertain_excluded_from_all_losses'] is True
        for row in read(ROOT/variant/'evaluation'/'summary.json')['records']:
            k=row['id'];p=read(ROOT/variant/'evaluation'/k/'trajectory.json');h,w=np.asarray(Image.open(cases[k]['image'])).shape[:2]
            for path in p['paths']:
                a=np.asarray(path);plans_ok &= bool(np.isfinite(a).all() and (a[:,0]>=0).all() and (a[:,0]<w).all() and (a[:,1]>=0).all() and (a[:,1]<h).all())
            plans_ok &= p['ready_for_robot'] is False and p['actually_drawn'] is False
            if not row['monochrome_specialist_used']:
                unchanged_colors+=1;plans_ok &= p['paths']==read(legacy.V16/'protected_refinement'/k/'trajectory.json')['paths']
    checkpoint=torch.load(ROOT/'reviewer'/'final.pt',map_location='cpu',weights_only=False);model_finite['automatic_reviewer']=all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values());audit=read(ROOT/'review_audit.json');review_sha=digest(ROOT/'user_review_snapshot.json');smoke=read(ROOT/'smoke_new_input'/'summary.json');smoke_ok=smoke['user_review_required'] is False and smoke['robot_connected'] is False and all(Path(r['automatic_review']).exists() for r in smoke['records']);old=results['v17'];new=results['v19_reviewed'];passed=new['R03_R04_mean_single']>=old['R03_R04_mean_single'] and new['R_faint_weighted']>=old['R_faint_weighted'] and new['N10_single']>=old['N10_single'];promote=False
    gate=dict(weights_promoted=promote,promotion_metric_checks_pass=passed,reason='No consistent gain on reviewed development and complex-rock examples; retain v16 operational baseline',new_user_annotation_requested=False,automatic_reviewer_is_conservative_filter_not_expert_replacement=True,uncertain_policy='retain baseline, do not force background',robot_connected=False);jsave(ROOT/'deployment_gate.json',gate)
    verify=dict(user_snapshot_unchanged=review_sha==audit['user_review_sha256'],latest_user_file_matches_snapshot=digest(V18/'expert_cross_sections'/'reviewed.json')==review_sha,original_training_images_and_labels_unchanged=source_ok,dataset_hashes_match=data_ok,uncertain_exclusion_pixels=uncertain_count,uncertain_excluded_from_supervised_and_distillation_losses=uncertain_excluded,parent_checkpoint_unchanged=digest(PARENT)==manifest['original_parent_sha256'],init_checkpoint_unchanged=digest(manifest['init_weights'])==manifest['init_sha256'],model_weights_finite=model_finite,trajectories_finite_in_bounds_and_offline=plans_ok,color_branch_identical_cases=unchanged_colors,development_ids_not_in_training=set(manifest['training_ids']).isdisjoint(manifest['calibration_ids']+manifest['development_ids']),reference_free_batch_entrypoint_completed=smoke_ok,final_extraction_updates=1600,first_extraction_trial_updates=1600,reviewer_final_updates=600,reviewer_cross_group_updates=800,initial_reviewer_trial_updates=1400,robot_connected=False);jsave(ROOT/'verification.json',verify)
    snapshots=ROOT/'source_snapshot';snapshots.mkdir(exist_ok=True)
    names=['audit_review_v19.py','autonomous_review_v19.py','recheck_reviewer_v19.py','review_guided_train_v19.py','exclude_uncertain_v19.py','auto_extract_v19.py',Path(__file__).name]
    for name in names:shutil.copy2(Path(__file__).parent/name,snapshots/name)
    jsave(snapshots/'source_sha256.json',{name:digest(snapshots/name) for name in names})
    cv=read(ROOT/'reviewer'/'cv_report.json');auto_total=sum(r['automatic_total'] for r in manifest['records']);auto_pass=sum(r['automatic_pass'] for r in manifest['records']);table='\n'.join('| '+k+' | '+f"{r['R_mean_single']:.3f} | {r['R03_R04_mean_single']:.3f} | {r['R_faint_weighted']:.3f} | {r['N10_single']:.3f} |" for k,r in results.items())
    text=f'''# v19：用户校正引导的自动复核与提取训练

日期：2026-10-25。E 盘保存全部模型、数据与缓存；F 盘保存脚本。离线实验，未连接、使能或移动机械臂。没有安装新依赖。

## 结论

已按照本次请求使用用户校正，建立自动横截面筛查和不确定回退流程，不再要求逐图人工校正。**自动筛查能提供局部一致性判断，尚不能作为专家真值。新提取权重没有获得一致提升，因此未替换现有基线。**

## 用户意见的使用

冻结保存 42 项意见：17 清楚墨线、21 淡墨、3 背景、1 不确定。6 项修改了几何坐标，其余为用户检查过的草稿位置，不能当成 36 项新画出的精确边缘。

- 已确认中心与背景用于局部监督；淡墨意见用于保持弱线正样本，不把“淡墨”类别直接换算为含墨量或下压高度。
- 用户指出粗边界仍存在歧义，因此宽墨带中未修改的草稿边缘不作为精确边缘真值；交叉、多峰、太宽或落在图外的横截面自动降低权重或待定。
- 明确不确定的位置（{uncertain_count} 个像素）从人工监督和旧模型保留约束两项损失中共同排除，不作为背景。
- 原始保存文件未改写。校正快照与派生数据分别保留，可追溯。

## 自动学习和筛选

独立 1D 复核网络从确认的横截面及已有绿色背景中学习局部中心/墨迹存在性，同时受墨带几何约束。第一次无几何限制的网络出现高分错误，已保留失败记录，未把它用于自动定真值。

最终模型在两组来源交叉留出的开发检查中，38 个有墨参考位置有约 {cv['positive_2px_fraction']:.1%} 落在用户中心 2 像素内；18 个自动接受位置中 16 个满足此条件。判断规则在查看开发结果后调整过，**这些数字不是独立盲测准确率，更不能推算整幅画或各种皴法的准确率。** 背景参考只有 3 个，不能据此证明不会误检。

自动规则还拒绝远距离跳线、多峰交叉及过宽的歧义墨带。训练图共筛查 {auto_total} 个中心附近横截面，{auto_pass} 个通过一致性检查，剩余位置降低权重或保持未知；这些是自动弱标签，不是新增专家标注。

## 提取训练和真实图对照

最终两组均从 v17 对齐监督权重开始，各 800 次更新；保持原 5 个输出头，并在明确未知而非不确定区域保留旧模型预测。控制组与修正组采用相同输入日程。

初次提取试验发现保留约束仍可能覆盖“不确定”位置，因此另存为问题试验，并用 `data_v2` / `control_v2` / `reviewed_v2` 重算。下面采用修正后的结果。

| 版本 | R01–R04 平均单路径覆盖 | R03/R04 留出开发图覆盖 | 淡线覆盖代理（28 个位置，加权） | 组合山石 N10 覆盖 |
|---|---:|---:|---:|---:|
{table}

以上是局部参考线附近 3 像素的覆盖指标，不是整幅画准确率。R03/R04/N10 等是多次查看过的开发图，不是全新盲测；N 系列参考线仍是辅助局部描线，不是专家完整线稿。彩色工笔图继续使用原分支，本轮并没有改善整幅彩色山水的语义提取或完成皴法类别识别。

新提取模型在训练图和部分示例上有改进，跨图表现不稳定，不能只看训练图升级模型。实际对照图保留成功与失败例子。

## 新图片如何自动复核

`auto_extract_v19.py` 接收一张图片或一个图片文件夹，只使用图像和预测轨迹，不读取参考线或用户逐图意见。当前默认使用保留的 v16 提取基线，然后运行 v19 自动复核。

- 局部中心与墨带一致：自动通过。
- 高置信度且几乎没有可见墨迹：输出疑似背景诊断。
- 宽墨、交叉或判断不一致：保留原轨迹和待定标记，不据此生成新的硬标签，也不要求用户逐项校正。

该入口已经完成真实文件运行，见 `smoke_new_input`。**它现在输出质量诊断，不承诺修复每个疑点；自动复核不会等同于专家真值。** 所有轨迹仍是像素坐标，未标定实际毛笔下压高度，`ready_for_robot=false`。

运行方式：

```powershell
& 'F:\\山河绘\\山河绘\\scripts\\rock_v15.ps1' -Script 'auto_extract_v19.py' -ScriptArgs @('--input','E:\\你的图片文件夹','--output','E:\\你的输出文件夹')
```

## 可检查文件

- `REAL_COMPARISON.jpg`：原图、v17、v19 候选轨迹、自动检查点；绿色检查点不是人工真值。
- `USER_REVIEW_AUDIT.jpg`：你的原始校正叠图。
- `reviewer/cv_report.json`：自动复核模型开发检查及限制。
- `data/*_qc.json`：逐项自动筛选与用户校正如何进入监督。
- `metrics_summary.json`、两组 `evaluation/summary.json`：聚合及逐图结果。
- `verification.json`：源文件、模型、训练划分、不确定损失屏蔽和无需参考线的批处理检查。
- `deployment_gate.json`：新提取权重未推广的依据。

后续工作由自动流程承担常规筛选，我会结合原图和保留开发图判断改动是否有效。难以判断的位置保持不确定，不再依靠模型自己的高分把它们反复训练成确定标签。
'''
    (ROOT/'REPORT.md').write_text(text,encoding='utf-8');print(json.dumps(dict(metrics=results,verification=verify,automatic_checks=qcs),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
