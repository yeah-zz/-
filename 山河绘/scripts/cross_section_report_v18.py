# -*- coding: utf-8 -*-
"""Summarize actual v18 outcomes, including regression and annotation limitations."""
from cross_section_v18 import *
import urllib.request,urllib.error


def main():
    manifest=read(ROOT/'manifest.json');versions={}
    for name,file in [('v16_baseline',ROOT/'profiles'/'evaluation'/'summary.json'),('v17_aligned',OLD/'aligned'/'evaluation'/'summary.json'),('v18_control',ROOT/'control'/'evaluation'/'summary.json'),('v18_profiles',ROOT/'profiles'/'evaluation'/'summary.json')]:
        rows=read(file)['records'];versions[name]={r['id']:r for r in rows}
    summary={}
    for name,rows in versions.items():
        side='before' if name=='v16_baseline' else 'after';diag=side+'_diagnostics';rr=[rows[k] for k in ['R01','R02','R03','R04']];dev=[rows[k] for k in ['R03','R04']];faint_n=sum(r[diag]['faint_samples'] for r in rr);bold_n=sum(r[diag]['bold_samples'] for r in rr)
        summary[name]=dict(R_mean_single=float(np.mean([r[side]['mean_best_single_path_coverage'] for r in rr])),R03_R04_mean_single=float(np.mean([r[side]['mean_best_single_path_coverage'] for r in dev])),faint_reference_proxy_weighted=sum((r[diag]['faint_reference_coverage'] or 0)*r[diag]['faint_samples'] for r in rr)/max(1,faint_n),faint_n=faint_n,bold_double_proxy_weighted=sum((r[diag]['bold_multiple_center_fraction'] or 0)*r[diag]['bold_samples'] for r in rr)/max(1,bold_n),bold_n=bold_n)
    jsave(ROOT/'metrics_summary.json',summary)
    cases={r['id']:r for r in prior_eval.evaluation_records()}
    ids=['R03','R04','N06','N10'];board=Image.new('RGB',(1600,370*len(ids)+40),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,4),'v18 真实图轨迹对照：R03/R04 本轮未训练；N06 训练；N10 开发样本。颜色为不同轨迹。',font=FONT,fill='black')
    for i,k in enumerate(ids):
        r=cases[k];full=np.asarray(Image.open(r['image']).convert('RGB'));shape=full.shape[:2];v16p=read(legacy.V16/'protected_refinement'/k/'trajectory.json')['paths'];v17p=read(OLD/'aligned'/'evaluation'/k/'trajectory.json')['paths'];newp=read(ROOT/'profiles'/'evaluation'/k/'trajectory.json')['paths']
        for j,(title,a) in enumerate([('原图',full),('v16 基线轨迹',line_view(shape,v16p,True)),('v17 对齐监督',line_view(shape,v17p,True)),('v18 横截面监督（未采用）',line_view(shape,newp,True))]):
            d.text((400*j+7,40+370*i),k+' '+title,font=SMALL,fill='black');paste(board,a,400*j+7,70+370*i,386,330)
    board.save(ROOT/'REAL_COMPARISON.jpg',quality=95)
    # Correct borrowed v17 captions and add unambiguous source-role metadata.
    for variant in ['control','profiles']:
        folder=ROOT/variant/'evaluation';obj=read(folder/'summary.json');obj.update(experiment='v18',weights_promoted=False,physical_robot_validation=False,derived_cross_section_labels_not_expert=True)
        jsave(folder/'summary.json',obj)
        for r in obj['records']:
            k=r['id'];full=np.asarray(Image.open(cases[k]['image']).convert('RGB'));doc=read(cases[k]['labels']);shape=full.shape[:2];baseline=read(legacy.V16/'protected_refinement'/k/'trajectory.json')['paths'];paths=read(folder/k/'trajectory.json')['paths'];page=Image.new('RGB',(1600,430),'#f5f3ec');pd=ImageDraw.Draw(page)
            for j,(title,a) in enumerate([('原图',full),('局部参考（非预测）',line_view(shape,[s['points'] for s in doc['strokes']])),('v16 基线轨迹',line_view(shape,baseline,True)),('v18 '+variant+' 实验输出',line_view(shape,paths,True))]):pd.text((400*j+7,5),k+' '+title,font=SMALL,fill='black');paste(page,a,400*j+7,38,386,383)
            page.save(folder/k/'comparison.jpg',quality=94)
        for title,group in [('reviewed_rocks',['R01','R02','R03','R04']),('new_rocks',['N01','N06','N10','N11']),('transfer',['T01','T02','U_A06'])]:
            page=Image.new('RGB',(1600,430*len(group)+35),'#f5f3ec');pd=ImageDraw.Draw(page);pd.text((8,4),'v18 '+variant+'；R01/R02 已参与本轮训练，其余为开发/保留对照。',font=FONT,fill='black')
            for i,k in enumerate(group):page.paste(Image.open(folder/k/'comparison.jpg'),(0,35+430*i))
            page.save(folder/(title+'.jpg'),quality=94)
    # Source, checkpoints and generated trajectories sanity checks.
    source_ok=all(digest(r['image'])==r['image_sha256'] and digest(r['labels'])==r['label_sha256'] for r in manifest['records'])
    finite={};plan_ok=True;no_train_leak=set(manifest['training_ids']).isdisjoint(set(manifest['calibration_ids'])|set(manifest['development_ids']))
    colors=0
    for variant in ['control','profiles']:
        state=torch.load(ROOT/variant/'final.pt',map_location='cpu',weights_only=False)['model'];finite[variant]=all(bool(torch.isfinite(v).all()) for v in state.values())
        for r in read(ROOT/variant/'evaluation'/'summary.json')['records']:
            k=r['id'];p=read(ROOT/variant/'evaluation'/k/'trajectory.json');h,w=np.asarray(Image.open(cases[k]['image'])).shape[:2]
            for path in p['paths']:
                a=np.asarray(path);plan_ok &= bool(np.isfinite(a).all() and (a[:,0]>=0).all() and (a[:,0]<w).all() and (a[:,1]>=0).all() and (a[:,1]<h).all())
            plan_ok &= p['ready_for_robot'] is False and p['actually_drawn'] is False
            if not r['monochrome_specialist_used']:
                colors+=1;plan_ok &= p['paths']==read(legacy.V16/'protected_refinement'/k/'trajectory.json')['paths']
    ui={};saved=ROOT/'expert_cross_sections'/'reviewed.json';oldbytes=saved.read_bytes() if saved.exists() else None
    with urllib.request.urlopen('http://127.0.0.1:8781/data',timeout=5) as response:data=json.load(response);ui['candidate_count']=len(data['rows'])
    with urllib.request.urlopen('http://127.0.0.1:8781/',timeout=5) as response:ui['html_ok']='笔触横截面复核' in response.read().decode('utf-8')
    key=data['rows'][0]['id']
    with urllib.request.urlopen('http://127.0.0.1:8781/context/'+key,timeout=5) as response:ui['context_png_ok']=response.read().startswith(b'\x89PNG\r\n\x1a\n')
    request=urllib.request.Request('http://127.0.0.1:8781/save',data=json.dumps({'annotations':[{'id':'invalid','label':'ink'}]}).encode(),headers={'Content-Type':'application/json'})
    try:urllib.request.urlopen(request,timeout=5);ui['invalid_save_rejected']=False
    except urllib.error.HTTPError as e:ui['invalid_save_rejected']=e.code==400
    ui['invalid_save_did_not_create_expert_labels']=(saved.read_bytes() if saved.exists() else None)==oldbytes
    verify=dict(source_images_and_original_labels_unchanged=source_ok,parent_checkpoint_unchanged=digest(PARENT)==manifest['parent_sha256'],init_checkpoint_unchanged=digest(manifest['init_weights'])==manifest['init_sha256'],checkpoints_finite=finite,calibration_and_development_not_in_new_training=no_train_leak,trajectories_finite_in_bounds_offline=plan_ok,color_branch_identical_cases=colors,ui=ui,expert_review_count=0 if not saved.exists() else read(saved)['reviewed_count'],successful_updates_total=2000,robot_connected=False)
    jsave(ROOT/'verification.json',verify)
    jsave(ROOT/'deployment_gate.json',dict(promoted=False,reason='Real reviewed development coverage regressed; faint-line improvement not established',active_baseline='v16 protected refinement',candidate='v18 profiles',metrics=summary,expert_cross_section_feedback_missing=True,ready_for_robot=False))
    folder=ROOT/'source_snapshot';folder.mkdir(exist_ok=True)
    for name in ['cross_section_v18.py','cross_section_review_v18.py','cross_section_review_candidates_v18.py',Path(__file__).name]:shutil.copy2(Path(__file__).parent/name,folder/name)
    table='\n'.join('| '+name+' | '+f"{r['R_mean_single']:.3f} | {r['R03_R04_mean_single']:.3f} | {r['faint_reference_proxy_weighted']:.3f} | {r['bold_double_proxy_weighted']:.3f} |" for name,r in summary.items())
    audit_total=sum(r['audit']['total'] for r in manifest['records']);audit_accept=sum(r['audit']['accepted'] for r in manifest['records']);train=manifest['training_ids']
    text=f'''# v18 真实笔触横截面监督实验

日期：2026-10-25。全程离线，未连接或驱动机械臂。文件、权重和临时缓存均在 E 盘；F 盘仅保存脚本。没有下载或安装新依赖。

## 本轮实施

1. 使用局部两侧纸色估计，扣除纸色明暗斜坡；噪声来自已有明确背景区域。淡墨置信度根据相对噪声和沿笔画连续性计算，不再仅因颜色浅就降低权重。
2. 在参考线附近查找连续墨带，以相邻横截面约束中心跳动；相邻笔画难以区分的截面降低权重，无法判断的区域仍保持未知。
3. 派生中心、墨带、两侧边界、表观墨色和明确背景目标。表观墨色是照片对比度，不等于真实含墨量，也不能直接变成机械臂下压高度。
4. 新模型增加两侧边缘、对比度和背景辅助监督。控制模型沿用 v17 标签。两组使用相同初始化、输入顺序和采样，分别完成 **1000** 次更新，共 **2000** 次。新模型的差异同时包含新目标和辅助监督，不能把差异全部归因于某一个组件。

数据来自 16 个局部标注记录，并不是 16 幅独立新画作。共检查 {audit_total} 个沿线位置，接受 {audit_accept} 个派生横截面。训练使用 {', '.join(train)}。N07/N08 只用于阈值校准，R03/R04 和 N01/N10/N11 本轮不参与梯度更新。这些开发图已在过去多次查看，并不是全新盲测。

**所有新横截面都是程序派生标签，尚未获得用户对两侧边缘的复核。** 用户原有描线仍作为参考依据，原文件没有改写。

## 真实图片结果

| 版本 | R01–R04 单路径平均覆盖 | R03/R04 平均覆盖 | 淡线覆盖代理（28 个位置，加权） | 粗线双中心代理（加权，越低越好） |
|---|---:|---:|---:|---:|
{table}

覆盖以局部参考线附近 3 像素为容差；参考线不完整且有定位误差，所以这些值不是整幅画准确率。淡/粗分类和双中心项也是自动估计，28 个淡线代理位置过少，不能代表各类皴法的表现。

**新横截面模型在 R03/R04 的覆盖退步，不能采用。** N06（训练图）的粗轮廓有所改善，但新模型还重新出现了照片右侧边缘误检，不能据此替代开发集结论。对比图同时保留成功和失败案例。旧模型保留，两个新检查点均为实验权重。

## 文件

- `REAL_COMPARISON.jpg`：原图、v16、v17、v18 实际预测对照，未把训练标签冒充提取结果。
- `LABEL_AUDIT.jpg`：自动派生训练目标审计图，不是模型输出。
- `metrics_summary.json`：聚合诊断，逐图原始数据在两组 `evaluation/summary.json`。
- `verification.json`：源文件哈希、权重有限值、训练划分、轨迹边界、保留彩色分支和本地复核工具检查。
- `deployment_gate.json`：新模型不推广的依据。

## 已准备好的少量人工校准

本地工具：**http://127.0.0.1:8781/**，36 个笔触横截面和 6 个纸纹/边框候选。可集中调整红色中心和橙色两侧边缘，并标记清楚墨线/淡墨/背景/不确定；可展开原图位置确认。只把勾选“已检查”的项目当作用户复核意见。

先检查 R01、R02、N06、N09 的 24 个笔触横截面和末尾 6 个背景候选即可，不必重描整张画。一次性确定这批边缘与淡墨标准后，才有条件校准程序的阈值、置信度和损失权重；不是让用户逐张复核全部训练图。

点击保存会直接写入本目录 `expert_cross_sections/reviewed.json`，并保留带唯一时间戳的修订副本，不通过浏览器下载。复核意见不会悄悄加入已经完成的本轮训练；后续训练应先审计并冻结新标注版本。
'''
    (ROOT/'REPORT.md').write_text(text,encoding='utf-8');print(json.dumps(dict(metrics=summary,verification=verify),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

