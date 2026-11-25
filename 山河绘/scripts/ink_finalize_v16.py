# -*- coding: utf-8 -*-
"""Complete v16 audit and report. Failed real-image validation blocks promotion."""
from ink_eval_v16 import *

def weighted(rows,side,key,count_key):
    usable=[r[side] for r in rows if r[side].get(key) is not None and r[side].get(count_key,0)>0];count=sum(r[count_key] for r in usable);return sum(r[key]*r[count_key] for r in usable)/count if count else None

def finish():
    torch.set_num_threads(4);root=ROOT/'stage_03';fusion=root/'observed_fusion';base_manifest=read(ROOT/'manifest.json');assert digest(PARENT)==base_manifest['raster_parent_sha256'];benchmark=read(V15/'benchmark_manifest.json')
    for r in benchmark['records']:assert digest(r['image'])==r['image_sha256'] and digest(r['labels'])==r['label_sha256']
    for r in base_manifest['records']:
        assert digest(r['image'])==r['image_sha256']
        if r.get('labels'):assert digest(r['labels'])==r['label_sha256']
        z=np.load(r['file']);assert np.isfinite(z['target']).all() and np.isfinite(z['mask']).all()
    summaries=[]
    for stage in [1,2,3]:
        folder=ROOT/f'stage_{stage:02d}';state=torch.load(folder/'final.pt',map_location='cpu',weights_only=False);assert state['steps']==(1200 if stage<3 else 800);assert state['parent_sha256']==digest(PARENT);assert all(bool(torch.isfinite(x).all()) for x in state['model'].values());summaries.append(read(folder/'training_summary.json'))
    manifest_ids={r['id'] for r in base_manifest['records']};assert not manifest_ids&{'R02','R03','R04','A05','N01','N10','N11','T01','T02','U_A06'}
    # The final stages explicitly ignore artificial native-content padding.
    native=next(r['native'] for r in reference_records() if r['id']=='R01');left,top,w,h=native;source=np.load(next(r['file'] for r in base_manifest['records'] if r['id']=='R01'));valid=np.zeros(source['rgb'].shape[:2],np.float32);valid[top:top+h,left:left+w]=1;assert not np.any((source['mask']*valid[None])[:,valid==0])
    controlled=read(root/'evaluation'/'controlled_test.json')['records'];fused_controlled=read(fusion/'evaluation'/'controlled_test.json')['records'];real=read(root/'evaluation'/'real_summary.json')['records'];fused_real=read(fusion/'evaluation'/'real_summary.json')['records'];by={r['id']:r for r in real};fby={r['id']:r for r in fused_real};stress=read(fusion/'stress'/'known_controls.json')['records'];faded=read(fusion/'stress'/'faded_real_summary.json')['records'];metrics={}
    for title,rows,side in [('v15',controlled,'before'),('v16_neural',controlled,'after'),('v16_observed_fusion',fused_controlled,'after'),('analytic_no_neural',fused_controlled,'analytic_no_neural')]:
        metrics[title]=dict(faint_coverage=weighted(rows,side,'faint_reference_coverage','faint_samples'),bold_coverage=weighted(rows,side,'bold_reference_coverage','bold_samples'),multiple_center_fraction=weighted(rows,side,'bold_multiple_center_fraction','bold_samples'),mean_f1=float(np.mean([r[side]['f1'] for r in rows])))
    jsave(ROOT/'controlled_summary.json',dict(records=metrics,all_are_programmed_stroke_controls_not_natural_artwork_accuracy=True))
    ids=['R01','R02','R03','R04'];old_mean=float(np.mean([by[k]['before']['mean_best_single_path_coverage'] for k in ids]));new_mean=float(np.mean([by[k]['after']['mean_best_single_path_coverage'] for k in ids]));fused_mean=float(np.mean([fby[k]['after']['mean_best_single_path_coverage'] for k in ids]));color=[]
    for folder,rows in [(root/'evaluation',real),(fusion/'evaluation',fused_real)]:
        for r in rows:
            plan=read(folder/r['id']/'trajectory.json');assert plan['ready_for_robot'] is False and plan['actually_drawn'] is False;assert all(np.isfinite(np.asarray(q,float)).all() for q in plan['paths'])
            if not r['monochrome_specialist_used']:
                old_paths=read(V15/'round_04'/'evaluation'/r['id']/'trajectory.json')['paths'];assert plan['paths']==old_paths;color.append(r['id'])
    blanks=[r for r in stress if r['kind']=='known_blank_control'];parallel=[r for r in stress if r['kind']=='known_two_stroke_control'];gate=dict(status='rejected_for_promotion',promoted=False,official_parent_unchanged=True,reviewed_R_mean_single_path_coverage=dict(v15=old_mean,v16_neural=new_mean,v16_observed_fusion=fused_mean),reason='Controlled bold/faint performance improved, but real reviewed-stroke continuity regressed; paper noise and crossing strokes remain unresolved',future_independent_artwork_validation_required=True);jsave(ROOT/'deployment_gate.json',gate)
    # A fixed set includes a simple training-source gain and a harder held-out
    # combination; full R and all new-source comparisons are also preserved.
    board=Image.new('RGB',(1600,430*4+40),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),'v16 真实图：包含改善与退步；新模型未通过整体验收',font=FONT,fill='black')
    for i,k in enumerate(['R01','R02','N06','N10']):board.paste(Image.open(root/'evaluation'/k/'comparison.jpg'),(0,40+430*i))
    board.save(ROOT/'REAL_COMPARISON.jpg',quality=95)
    fmt=lambda x:'—' if x is None else f'{100*x:.1f}%'
    text=f'''# v16 粗墨线双边与淡线漏检实验（2026-10-25）

本轮确实训练了新的线条定位网络，不是继续只训练断点连接。三阶段共 3200 次更新（1200＋1200＋800）；现有 v12/v15 文件与正式绘画流程保持不变，机械臂未连接。

结果：可控粗淡墨线上的表现提高，但真实图片的平均单笔完整度下降。新模型以及墨迹约束版本都没有通过整体验收，deployment_gate.json 明确拒绝自动替换。不能称“粗线双边和天然淡线漏检已经解决”。

## 实现与监督

- 独立小型 U-Net，输入原始 RGB、局部纸色归一化、细/粗对比度、饱和度共 7 通道。输出中心线、墨迹区域、方向和像素表观宽度；宽度不等于毛笔下压量，没有物理标定。
- 第一阶段学习 1～23 像素宽度、0.035～0.88 墨色遮盖率的程序样本，同时使用既有用户局部标注、助手局部草标和弱墨迹标签。
- 第二阶段加亮度减淡增强，第三阶段加入更强的颗粒、空白和边框干扰；没有把校准图 R02 的背景拿去训练。
- 共 {len(base_manifest['records'])} 份真实训练条目，包括同图的人工/弱监督条目，不能当成独立画作数。所有资料复用 E 盘现有数据，没有要求用户补做逐张复核；弱标签从未当成人工金标准。
- 用户局部线稿未描部分仍为未知，只有描线附近和明确绿色背景提供强监督。训练只使用 R01、A01–A04 和原先分配为 train 的助手草标/弱数据；R02、R03、R04、A05、N01、N10、N11、T01、T02、A06 本轮不参加梯度更新。原画和形态仍有来源重叠，且这些图被反复查看，因此只是开发对照，不是独立盲测。
- 第二、三阶段明确屏蔽 R01 的人工补边。第一阶段沿用字段掩码；这一阶段的补边没有额外裁掉。代码与阶段输出均保留，未回写原始标注。

## 两种解码与失败记录

主分支比较墨迹区域骨架与中心线脊线，第三阶段增加墨迹概率约束。配置由程序校准集 162000–162019 和 R02 局部参考选择，并处罚绿色背景中的线像素密度。

另做 observed_fusion 对照：以实际可见的黑灰墨迹约束候选，较宽区域只取中心；弱墨候选由网络与细长结构共同决定。早期局部高值/闭运算纸色估计把灰纸与白页边混淆，失败参数保存在 closing_trial_calibration.json；后改为局部 61 像素中值估计。拥挤交点无法唯一分笔时隔离，避免把纸纹节点全部连在一起。

纯图像法 analytic_no_neural 也独立测试，没有调用训练权重提供方向或弱线种子。粗线归中改善不能全部归功于深度学习；弱标签和图像约束本身可能包含偏差。

## 可控测试：40 张未参与训练/校准的程序图

这里有明确的真实中心线、墨色和宽度，因此可以诊断双边和淡线。它们不是工笔山水画准确率。粗线定义为程序宽度≥7 且遮盖率≥0.10，淡线为遮盖率 0.02～0.18；沿中心法线出现多条轨迹计为多中心诊断。必须同时看覆盖率，删除墨线也会降低多中心比例。

| 方法 | 粗线参考点覆盖 | 多中心诊断比例（越低越好） | 淡线参考点覆盖 | 中心线 F1 |
|---|---:|---:|---:|---:|
'''
    for name,m in metrics.items():text+=f"| {name} | {fmt(m['bold_coverage'])} | {fmt(m['multiple_center_fraction'])} | {fmt(m['faint_coverage'])} | {fmt(m['mean_f1'])} |\n"
    text+='''
## 真实图片：局部参考下的单笔覆盖

容差 3 像素，衡量每条参考线被一条候选路径覆盖的比例；不是准确率，也不证明没有错误串线。N 开头参考是助手局部草标，不能称专家评价。未标注区域无法评价整图误检。

| 图 | v15 | v16 网络 | v16 墨迹约束对照 |
|---|---:|---:|---:|
'''
    for k in ids+['T01','T02','U_A06']+['N'+str(i).zfill(2) for i in range(1,13)]:
        a,b=by[k],fby[k];text+=f"| {k} | {fmt(a['before']['mean_best_single_path_coverage'])} | {fmt(a['after']['mean_best_single_path_coverage'])} | {fmt(b['after']['mean_best_single_path_coverage'])} |\n"
    text+=f'''
四张 R 图平均：v15 为 {fmt(old_mean)}，新网络为 {fmt(new_mean)}，墨迹约束对照为 {fmt(fused_mean)}。R01 和部分简单新图有所提高，但 R02/R04 及组合石仍有退步，所以不能只挑改善例子宣布成功。青绿山水 A01–A05、A06 的旧分支直接保留，{len(set(color))} 张彩色对照的计划路径逐项相同；本轮没有改善整幅青绿山水的提取能力。

真实图的粗淡子集只是按图像局部对比度和自动宽度估计划分，结果保存在各图 before_diagnostics/after_diagnostics；受纸色、照片纹理、粗略描线影响，只能作诊断，不能当作天然皴法/墨色标签。

## 空白、相邻线与人工减淡诊断

- 6 张已知空白控制（不同颗粒幅度）：约束对照最大线像素比例 {fmt(max(r['after_line_pixel_fraction'] for r in blanks))}。
- 4 组已知相邻双线：约束对照有 {sum(r['after']['both_separate'] for r in parallel)}/4 组同时保持两条独立线并达到严格位置覆盖要求。详见 stress/known_controls.json，不能直接等同复杂皴线分离率。
- R03、N10 原图按纸色人工减淡到 30%，原参考几何不变。这只是压力测试，不是新收集的天然淡墨作品：

| 人工减淡输入 | v15 参考点覆盖 | 墨迹约束对照 |
|---|---:|---:|
'''
    for r in faded:text+=f"| {r['id']} | {fmt(r['before']['point_coverage_3px'])} | {fmt(r['after']['point_coverage_3px'])} |\n"
    text+='''
这些结果表明，程序淡墨提高没有稳定转化为真实照片淡线提高。纸纹和淡线的区分、宽墨交汇区域的逐笔连续性仍是瓶颈。下一轮应优先让真实训练资料的纸纹、光照、墨迹区域与中心线监督一致，并保留整条参考曲线的连续性约束；继续只增加简单程序样本或单独降阈值不能视为解决方案。可以继续利用已有绿色背景自动挖掘难负例，无需用户每张重新描线。

## 文件与验证

- REAL_COMPARISON.jpg：固定展示 R01、R02、N06（训练来源）、N10（组合开发图），同时包含改善与退步。
- stage_03/evaluation/reviewed_rocks.jpg、new_rocks.jpg、transfer.jpg：全部复核图、新来源开发图和迁移对照。
- stage_03/evaluation/controlled_comparison.jpg：可控程序图的真实模型输出，明确标为程序诊断。
- stage_03/observed_fusion/evaluation：墨迹约束对照，另含纯图像法的可控测试。
- stage_03/observed_fusion/stress：空白、相邻线和人工减淡图对照。
- controlled_summary.json、deployment_gate.json、verification.json：汇总、拒绝接管正式模型的原因与检查结果。
- stage_01/02/03/final.pt：各阶段候选权重，均未部署。source_snapshot 保留脚本；阶段的训练日志与输入来源哈希另存。

离线入口：通过 rock_v15.ps1 调用 ink_predict_v16.py，使用 --input 指定图片或目录；默认只输出 v15/v16 对照、候选轨迹与自动质检，--neural-only 可查看不加墨迹约束的候选。输出统一留在 E 盘。所有计划均 ready_for_robot=false，未连接、使能或移动机械臂。
'''
    (ROOT/'REPORT.md').write_text(text,encoding='utf-8');verification=dict(all_three_weights_finite=True,total_successful_updates=sum(r['successful_updates'] for r in summaries),stable_parent_unchanged=True,reviewed_benchmark_hashes_unchanged=True,source_images_and_labels_unchanged=True,current_training_excludes_calibration_and_development_ids=True,colored_branch_paths_exactly_preserved=True,all_saved_trajectories_finite=True,all_saved_trajectories_offline=True,robot_connected=False,new_candidate_not_promoted=True);jsave(ROOT/'verification.json',verification)
    snapshots=ROOT/'source_snapshot';snapshots.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*v16.py'):shutil.copy2(path,snapshots/path.name)
    print('V16_FINALIZED',json.dumps(dict(controlled=metrics,reviewed_single_path_before=old_mean,reviewed_single_path_neural=new_mean,reviewed_single_path_fusion=fused_mean,promoted=False)),flush=True)

if __name__=='__main__':finish()
