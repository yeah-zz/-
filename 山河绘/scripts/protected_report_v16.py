# -*- coding: utf-8 -*-
"""Report the bounded correction separately from the rejected replacement net."""
from protected_refine_v16 import *

def finish():
    folder=ROOT/'protected_refinement';rows=read(folder/'summary.json')['records'];by={r['id']:r for r in rows};before=float(np.mean([by[k]['before']['mean_best_single_path_coverage'] for k in ['R01','R02','R03','R04']]));after=float(np.mean([by[k]['after']['mean_best_single_path_coverage'] for k in ['R01','R02','R03','R04']]));worst_drop=max(r['before']['mean_best_single_path_coverage']-r['after']['mean_best_single_path_coverage'] for r in rows);faint_added=sum(r['faint_recovery'].get('faint_paths_added',0) for r in rows);dedup=sum(r['correction'].get('redundant_paths_removed',0) for r in rows);ref=reference_records()+read(V15/'new_labels'/'manifest.json')['records'];original={r['id']:r for r in ref};oldr={r['id']:r for r in read(LAB/'clean_strokes_v12_20261024'/'manifest.json')['records']};color_preserved=0
    for r in rows:
        plan=read(folder/r['id']/'trajectory.json');assert not plan['ready_for_robot'] and not plan['actually_drawn'];rr=original.get(r['id'],oldr.get(r['id']));im=Image.open(rr['image']);h,w=im.height,im.width
        for p in plan['paths']:
            q=np.asarray(p);assert np.isfinite(q).all() and (q[:,0]>=0).all() and (q[:,0]<w).all() and (q[:,1]>=0).all() and (q[:,1]<h).all()
        if not r['monochrome_route']:
            assert plan['paths']==read(V15/'round_04'/'evaluation'/r['id']/'trajectory.json')['paths'];color_preserved+=1
    assert read(ROOT/'verification.json')['source_images_and_labels_unchanged'];assert digest(PARENT)==read(ROOT/'manifest.json')['raster_parent_sha256']
    # Highlight only the extra hypotheses, so absence of a real ink line is visible.
    added=[r for r in rows if r['faint_recovery'].get('quarantined_dark_edge_candidates')];board=Image.new('RGB',(1400,420*len(added)+40),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),'已隔离的补线候选：疑似粗墨第二条边，未加入计划轨迹',font=FONT,fill='black')
    for i,r in enumerate(added):
        rgb=np.asarray(Image.open(original[r['id']]['image']).convert('RGB'));overlay=rgb.copy()
        for proposal in r['faint_recovery']['quarantined_dark_edge_candidates']:cv2.polylines(overlay,[np.rint(proposal['points']).astype(np.int32)],False,(235,25,20),2,cv2.LINE_AA)
        d.text((8,40+i*420),r['id']+' 原图',font=SMALL,fill='black');d.text((708,40+i*420),'被拦截的候选（红色未进入计划轨迹）',font=SMALL,fill='black');paste(board,rgb,8,72+i*420,684,375);paste(board,overlay,708,72+i*420,684,375)
    board.save(folder/'faint_candidates.jpg',quality=95)
    gate=dict(status='experimental_protected_correction',promoted_to_official=False,raw_replacement_net_rejected=True,mean_R_single_path_before=before,mean_R_single_path_after=after,worst_single_image_drop=worst_drop,extra_faint_candidate_paths=faint_added,entire_redundant_paths_removed=dedup,reason='Small development improvement within old-path boundaries; doubled strokes are not fully removed and faint candidates lack independent expert confirmation',independent_artwork_validation=False);jsave(folder/'deployment_gate.json',gate);jsave(folder/'verification.json',dict(trajectories_finite_in_bounds=True,all_plans_offline=True,colored_paths_exactly_preserved=color_preserved,parent_unchanged=True,existing_source_verification_passed=True))
    old_report=ROOT/'REPORT.md'
    if not (ROOT/'REPORT_NEURAL_EXPERIMENT.md').exists():shutil.copy2(old_report,ROOT/'REPORT_NEURAL_EXPERIMENT.md')
    fmt=lambda x:f'{100*x:.1f}%'
    text=f'''# v16 最终记录：新定位网络未接管，受限校正有小幅收益

本轮针对粗墨线双边和淡线漏检，确实完成了新定位网络三阶段训练，共 3200 次参数更新。直接替换定位网络后，程序样本改善没有稳定转化到真实山石，故已拒绝替换。全部失败对照保留在 REPORT_NEURAL_EXPERIMENT.md，未改动正式模型或原标注，也没有连接机械臂。

随后在旧版轨迹上完成一组更保守的候选：只校正同一条可见宽墨带内部的位置，不全盘重生成轨迹；只补充原模型仍有弱响应、两档曝光提取一致、新网络与可见墨迹共同支持的长淡线候选。

## 真实图结果

以下是平均单笔参考覆盖（3 像素容差），不是准确率或无误检率。参考多为局部描线，图像也已被反复查看，不能称独立盲测。

| 图 | v15 | 受限校正 |
|---|---:|---:|
'''
    for k in ['R01','R02','R03','R04','T01','T02','U_A06','N01','N05','N06','N10','N11','N12']:
        r=by[k];text+=f"| {k} | {fmt(r['before']['mean_best_single_path_coverage'])} | {fmt(r['after']['mean_best_single_path_coverage'])} |\n"
    text+=f'''
R01–R04 平均从 {fmt(before)} 到 {fmt(after)}。完整 24 张参考图中，最大单图下降约 {100*worst_drop:.2f} 个百分点；彩色分支有 {color_preserved} 张对照的路径逐项保持相同。N 开头仍是助手局部草标，未当作专家金标准。

经最终检查，自动补入的淡线为 {faint_added} 条。初版在 N05、N11 提出过两条候选，但它们靠近深墨带，存在把粗线另一侧重复描出的风险，已被规则拦截、隔离，没有加入最终计划。因此本轮没有可确认的可靠淡线补全收益。红色叠加图展示的是被拒绝的候选。

粗墨带中的轨迹向中间作了小幅校正，但整条冗余路径实际删除数为 {dedup}。因此**本轮尚不能声称粗线双边已经消除**；物理落笔重复和复杂交汇处的逐笔连续性仍未解决。此次收益主要是少量定位调整，不是全面突破。

## 保护规则和证据

- 校正强度仅在 R02 上校准，选定 0.35。沿旧轨迹法线检查可见墨带，要求宽度 6～24 像素、足够墨色、与旧线相邻及新网络墨迹概率支持；单点位移最多 2.1 像素，不跨白色间隙跳向另一笔。
- 淡线候选要求至少 32 像素长，两档对比度推理一致，原模型原图中已有弱响应，且远离现有路径和图像边框；未通过的候选直接保留为拒绝项，不灌入人工标准。
- 推理不读取参考线或用户背景多边形，参考只用于训练/校准/评测。青绿山水不启用这组黑灰墨线校正，旧输出保留。
- 原始复核、父模型哈希、三个阶段权重有限性、离线标志和坐标范围检查均通过。所有图和缓存位于 E 盘，代码位于 F 盘。

## 查看结果

- protected_refinement/comparison.jpg：R01、R02、N06、N10 的原图→参考→v15→受限校正，包含改善很小/未改善的例子。
- protected_refinement/faint_candidates.jpg：N05、N11 被隔离的候选叠加，它们未进入最终轨迹。
- protected_refinement/summary.json：全部真实图的前后指标与修改数量。
- protected_refinement/verification.json、deployment_gate.json：验证结果及仍仅作为实验候选的原因。
- REPORT_NEURAL_EXPERIMENT.md、stage_01/02/03：3200 次更新的新网络、可控测试、真实图退步、空白/相邻笔画控制及人工减淡测试。

程序测试上，新网络的粗线多中心诊断比例显著降低、淡线参考点覆盖明显提高，但不能当作天然山水画识别率。真实图表现证明：只用弱墨迹标签和程序样本还不足以可靠区分纸纹、淡墨和相邻笔画。正式绘画流程继续保留旧模型；所有新计划均 ready_for_robot=false。
'''
    old_report.write_text(text,encoding='utf-8');snap=ROOT/'source_snapshot'
    for name in ['protected_refine_v16.py','protected_report_v16.py']:shutil.copy2(Path(__file__).with_name(name),snap/name)
    print('PROTECTED_REPORT_COMPLETE',json.dumps(gate),flush=True)

if __name__=='__main__':finish()
