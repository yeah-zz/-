# -*- coding: utf-8 -*-
"""Freeze reviewed benchmark, verify experiments, publish complete comparisons."""
from rock_v15_common import *
import shutil,math
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from human_mountain_v11 import fields,paste
from rock_pairs_v15 import reference_records,connect,PairNet
from rock_eval_v15 import quality_audit
from evaluate_clean_strokes_v12 import line_view

def run():
    torch.set_num_threads(4);records=reference_records();pair_manifest=read(ROOT/'pairs'/'manifest.json')['records'];bench=[]
    for r in records:
        matched=next(x for x in pair_manifest if x['id']==r['id']);assert digest(r['image'])==matched['image_sha256'];assert digest(r['labels'])==matched['label_sha256'];doc=read(r['labels']);assert doc['strokes'];bench.append(dict(**r,image_sha256=digest(r['image']),label_sha256=digest(r['labels']),strokes=len(doc['strokes']),complete=False))
    for r in read(PREVIOUS/'manifest.json')['records']:
        live=LAB/'stroke_topology_v13_20261024'/'user_review'/'labels'/(r['id']+'.json');assert digest(live)==r['original_user_label_sha256']
    benchmark=dict(records=bench,derived_reviewed_strokes=sum(r['strokes'] for r in bench),original_raw_r_strokes=50,derived_r_strokes=48,unmarked='unknown',human_selected_robot_strokes_not_historical_pen_lifts=True,source_families=['heye_motif','pima_motif','qianli_jiangshan'],shared_artwork_calibration=True,not_independent_accuracy=True)
    if (ROOT/'benchmark_manifest.json').exists():assert read(ROOT/'benchmark_manifest.json')==benchmark
    else:jsave(ROOT/'benchmark_manifest.json',benchmark)
    rounds=[]
    for k in range(1,5):
        folder=ROOT/f'round_{k:02d}';summary=read(folder/'training_summary.json');state=torch.load(folder/'pair_net.pt',map_location='cpu',weights_only=False);assert state['steps']==650;assert summary['parent_sha256']==digest(PARENT);assert all(bool(torch.isfinite(v).all()) for v in state['model'].values());rounds.append(summary)
    assert digest(ROOT/'round_04'/'pseudo_pairs_at_training.npz')==rounds[-1]['pseudo_dataset_sha256']
    saved=np.load(ROOT/'round_04'/'pseudo_pairs_at_training.npz');current=np.load(ROOT/'pseudo_pairs.npz');assert all(np.array_equal(saved[k],current[k]) for k in ['patch','features','targets'])
    sources=read(ROOT/'pseudo_pair_sources.json');curated=read(ROOT/'curated_manifest.json');lookup={r['id']:r for r in curated['records']};assert len(sources)==len(current['targets']);assert all(lookup[identity]['split']=='train' for identity in sources);assert all((current['targets']==1))
    new=read(ROOT/'new_labels'/'manifest.json')['records'];assert all(digest(r['image'])==r['image_sha256'] and digest(r['labels'])==r['label_sha256'] for r in new)
    extra=read(ROOT/'additional_real_pairs_audit.json');assert all(r['id'] not in ['R02','R03','R04','A05'] for r in extra)
    # Meaningful safety checks: a forced high score must not resolve a tie,
    # and a closed curve must not be opened to join a nearby unrelated path.
    class ForcedHigh(torch.nn.Module):
        def forward(self,patch,feat):return torch.full((len(patch),),12.,device=patch.device)
    p=np.ones((80,80),np.float32)*.8;rgb=np.full((80,80,3),230,np.uint8)
    paths=[np.array([[5.,40],[35,40]]),np.array([[40.,39],[70,39]]),np.array([[40.,41],[70,41]])];_,events,uncertain=connect(paths,p,rgb,ForcedHigh().cuda(),.9);assert not events and uncertain
    theta=np.linspace(0,2*math.pi,90);circle=np.stack([30+10*np.cos(theta),30+10*np.sin(theta)],1);_,events,_=connect([circle,np.array([[42.,30],[65,30]])],p,rgb,ForcedHigh().cuda(),.9);assert not events
    qc=quality_audit([circle],p,rgb,[]);assert qc['not_expert_certified'] and not qc['ready_for_robot']
    # Validate actual standalone inference artifacts, not only unit inputs.
    smoke=read(ROOT/'smoke_inference'/'summary.json');assert len(smoke['records'])==1 and not smoke['robot_connected'];smoke_folder=next(p for p in (ROOT/'smoke_inference').iterdir() if p.is_dir());plan=read(smoke_folder/'trajectory.json');assert plan['ready_for_robot'] is False and (smoke_folder/'trajectory.svg').exists()
    for folder in [ROOT/'round_04'/'evaluation',ROOT/'round_04'/'new_source_evaluation',ROOT/'batch_results']:
        for path in folder.glob('*/trajectory.json'):
            plan=read(path);assert not plan['ready_for_robot'];assert all(np.isfinite(np.asarray(q,float)).all() for q in plan['paths'])
    evaluation=read(ROOT/'round_04'/'evaluation'/'summary.json')['records'];by={r['id']:r for r in evaluation};rids=['R01','R02','R03','R04'];old_mean=float(np.mean([by[k]['before']['mean_best_single_path_coverage'] for k in rids]));new_mean=float(np.mean([by[k]['after']['mean_best_single_path_coverage'] for k in rids]));dev_ids=['T01','T02','U_A06'];max_drop=max(by[k]['before']['mean_best_single_path_coverage']-by[k]['after']['mean_best_single_path_coverage'] for k in dev_ids)
    gate=dict(status='experimental_only',promoted=False,criteria=dict(min_reviewed_mean_gain=.015,max_development_drop=.02,max_calibration_false_join_fraction=.01),measured=dict(reviewed_mean_gain=new_mean-old_mean,max_development_drop=max_drop,calibration_false_join_fraction=rounds[-1]['calibration']['different_stroke_false_join_fraction']),reason='Reviewed-stroke completeness gain is too small; independent complete artwork validation is unavailable',formal_raster_unchanged=True)
    jsave(ROOT/'deployment_gate.json',gate)
    batch=read(ROOT/'batch_results'/'summary.json');new_evaluation=read(ROOT/'round_04'/'new_source_evaluation'/'summary.json')['records'];new_by={r['id']:r for r in new_evaluation}
    # Selected examples are fixed: both rock-motif types, a different new-source
    # outline and the old complex landscape. Full sets remain linked below.
    board=Image.new('RGB',(1600,430*4+40),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),'v15：模型判断接线，仍有断线与误连风险；参考与预测分栏',font=FONT,fill='black')
    entries=[ROOT/'round_04'/'evaluation'/k/'comparison.jpg' for k in ['R01','R04']]
    for i,path in enumerate(entries):board.paste(Image.open(path),(0,40+i*430))
    n=next(r for r in new if r['id']=='N01');cr=lookup[n['source_id']];rgb=np.asarray(Image.open(cr['content_image']).convert('RGB'));doc=read(n['labels']);reference=[s['points'] for s in doc['strokes']];from human_mountain_eval_v11 import extract
    pred=np.load(ROOT/'auto_predictions'/(n['source_id']+'.npz'))['prediction'];_,old=extract(pred);newplan=read(ROOT/'round_04'/'new_source_evaluation'/'N01'/'trajectory.json')['paths']
    for j,(title,a) in enumerate([('N01 新来源原图',rgb),('助手草标（非用户复核）',line_view(rgb.shape[:2],reference)),('冻结定位原追踪',line_view(rgb.shape[:2],old)),('本轮候选轨迹',line_view(rgb.shape[:2],newplan,True))]):d.text((j*400+7,900),title,font=SMALL,fill='black');paste(board,a,j*400+7,933,386,377)
    board.paste(Image.open(ROOT/'round_04'/'evaluation'/'U_A06'/'comparison.jpg'),(0,40+3*430));board.save(ROOT/'RESULTS_OVERVIEW.jpg',quality=95)
    fmt=lambda x:f'{x*100:.1f}%'
    report=f'''# v15 自动接线、分笔训练与质检（2026-10-25）

用户已授权把复核样本作为持续训练依据，减少逐张人工复核。本轮已实现批量提取、学习接线/分笔、自动质检和伪标签隔离；但没有证据说明模型已具备画家水平的自主复核能力。候选模型未替换正式版，所有操作离线。

## 固定用户评判依据

benchmark_manifest.json 记录 {benchmark['derived_reviewed_strokes']} 条已复核参考线及图像/标注 SHA256，包括 A01–A05 与修正后的 R01–R04；原始 R 文件 50 条线仍保留，派生训练标准 48 条（按用户确认替换两条重叠旧草稿，旧线余段仅监督几何）。未描部分保持未知，不将其认定为背景。

为避免全部用于训练后再声称盲测，接线网络训练使用 R01、A01–A04，阈值校准使用 R02、A05，开发对照使用 R03、R04、旧 T01/T02/A06。它们存在同画作/同形态来源重叠，且被反复查看，均不能称独立泛化准确率。R01/R02 与 R03/R04 各为一个山石形态的步骤图。未来换来源测试也不能拿自动标签当专家金标准。

## 新数据与自动标注

- 本轮下载 {len(read(ROOT/'sources.json'))} 张候选图片，筛出 {curated['raw_image_count']} 张山石示范图，来自 {curated['source_pages']} 个教程页面；整理 {curated['crop_count']} 个裁块。裁块和步骤图不能算独立画作，底层画谱家族未完全核实。
- 对 12 张已查看新图补充 49 条助手局部草标，保存在 new_labels。它们是非专家、非完整标注，与用户复核分开保存。文字、标志、二维码和明显不相关图片未纳入精筛集。
- 自动候选通过亮度/翻转一致性、可见墨迹、边框、近邻重叠边线及已知示意箭头排除规则筛选。仅将通过检查的开放线段作为软伪标签；伪标签只用于同一段内部的人为断口，不把邻近伪线直接标成“不同笔”。
- 第四组训练使用 {rounds[-1]['auto_pseudo_pair_samples']} 个自动正片段对。相关性高的自一致检查不能证明艺术语义正确；初版伪标签中出现的边框/粗线两侧问题已经记录，收紧后的版本单独与初版保存。
- kuhw_29483 整个新来源页面及组合石 27/28 未加入伪标签训练；pseudo_pair_sources.json 可逐项追溯。来源独立性仍受转载/同书影响。

## 真正训练了什么

原有 v12 定位网络冻结不动，训练三层卷积的局部图像网络结合几何 MLP，判断两个端点能否接成用户指定的一笔。推理只使用图片、定位概率与候选轨迹，不读取参考线或背景标注。拒绝低置信、非相互最优和竞争分数接近的接线，保留无法判断的情况。

共 4 组，每组 650 次参数更新，总计 2600 次。第一组学习真实复核线的人为断口和邻线负例；第二组增加难负例与扰动；第三组加入真实预测片段、助手草标和初版软伪标签；第四组从第二组重新开始，使用更严格的软标签进行对照。并非连续在同一模型上训练 2600 次。第一组训练完成后发生报告 JSON 序列化错误，已恢复已保存的 650 步权重并重算报告，没有重复算作额外训练。

第四组包含 {rounds[-1]['reference_pair_samples']} 个用户/助手参考及真实预测片段训练对，另有 {rounds[-1]['auto_pseudo_pair_samples']} 个软伪标签训练对。阈值由 R02/A05 的 {rounds[-1]['calibration']['admissible_negative']} 个可比较异笔负例和 {rounds[-1]['calibration']['admissible_positive']} 个同笔断口校准。当前阈值 {rounds[-1]['threshold']:.3f}，校准异笔误接比例 {fmt(rounds[-1]['calibration']['different_stroke_false_join_fraction'])}，同笔人为断口接回比例 {fmt(rounds[-1]['calibration']['same_cut_recall'])}。这些是相关画作的片段指标，不能理解为整图误接概率保证。

| 组 | 校准人为断口接回比例 | 校准异笔误接比例 | 自动片段对 |
|---|---:|---:|---:|
'''
    for r in rounds:report+=f"| {r['round']} | {fmt(r['calibration']['same_cut_recall'])} | {fmt(r['calibration']['different_stroke_false_join_fraction'])} | {r['auto_pseudo_pair_samples']} |\n"
    report+='''
## 完整图上的实际结果

“单笔覆盖率”指每条参考线被一条候选路径在 3 像素内覆盖的平均比例。它不是准确率，错误串线也可能提高覆盖；部分参考无法用于统计全部误检。旧版与本轮还含追踪器差异，另有不接线图追踪对照 unpaired。

| 参考图 | 旧版流程 | 第四组 | 自动接线数 |
|---|---:|---:|---:|
'''
    for k in rids+dev_ids:
        r=by[k];report+=f"| {k} | {fmt(r['before']['mean_best_single_path_coverage'])} | {fmt(r['after']['mean_best_single_path_coverage'])} | {r['automatic_joins']} |\n"
    report+=f'''
四张 R 图平均由 {fmt(old_mean)} 到 {fmt(new_mean)}，提升有限，未达到预设 1.5 个百分点的平均完整度提升门槛。因此保持 experimental_only，不替换正式绘画模型。相较 v14 的大幅迁移退步，本轮冻结定位避免了参数遗忘，但这不等于提取已经完整。

新增来源开发图：N01 为助手局部草标，单笔覆盖从 {fmt(new_by['N01']['before']['mean_best_single_path_coverage'])} 到 {fmt(new_by['N01']['after']['mean_best_single_path_coverage'])}；组合石 N10/N11 分别为 {fmt(new_by['N10']['after']['mean_best_single_path_coverage'])}、{fmt(new_by['N11']['after']['mean_best_single_path_coverage'])}，仍然较差。新图对照全部保存，没有只展示提升样本。粗墨线产生双边、浓墨区域分支、真实淡线缺失依然是问题，接线网络无法补回未被定位网络找到的全部线。

## 自动复核能做与不能做的事

已自动处理全部 {len(batch['records'])} 张精筛图。模型复核接线候选；程序检查弱定位支撑、可疑小闭环、边框、近邻重叠线和竞争接线，并把不确定项写入 JSON。通过检查只表示“未触发这些几何检查”，不等于艺术语义正确。无需你每张点击确认；隔离项由后续程序诊断和实验处理，不自动提升为人工正确标注。

尚不能自动确认每一种皴法名称、原画家的真实落笔/抬笔、物理笔宽浓淡或实机可画程度。没有训练完整专家复核器的负例覆盖，也没有独立、完整人工金标准，因此不能声称已经完全替代专家。当前最明显瓶颈是定位对浓墨宽笔与淡墨细线的不一致；后续可用保留的自动隔离集改进清晰山石定位分支，再做独立来源验证，而不是不断重复这几张复核图。

## 使用与文件

- RESULTS_OVERVIEW.jpg：固定展示 R01、R04、N01、A06，参考与真实预测分栏。
- round_04/evaluation/reviewed_rocks.jpg、transfer.jpg：全部 R 图与三个旧开发对照。
- round_04/new_source_evaluation/new_source_comparison.jpg：新增来源的全部留出局部参考对照。
- batch_results：21 张图的轨迹、预测图、自动质检记录；summary.json 汇总。
- auto_labels：程序软标注及被隔离的候选原因；auto_gate_v1 保存初版，禁止当作人工金标准。
- new_labels/label_audit.jpg：新增助手草标叠加图。
- benchmark_manifest.json、deployment_gate.json、verification.json：评判依据、拒绝自动替换的原因和验证记录。
- 向 E:\\Codex_Workspace_Data\\Gongbi_Line_Model_20261022\\autonomous_strokes_v15_20261025\\new_inputs 放图片，再运行 F:\\山河绘\\山河绘\\scripts\\run_offline_rock_v15.cmd，可批量生成新的实验轨迹和质检结果。此入口使用候选第四组并明确标记为实验，不改变正式绘画程序。

输出轨迹为像素坐标，SVG 为单线预览；ready_for_robot=false。没有实机连接、使能、运动、Z 下压或墨水压力标定。

新来源链接：[小斧劈皴分步示范](https://www.kuhw.com/hhjc/sshjc/2020-02-12/29483.html)、[山石基础示范](https://k.sina.com.cn/article_6586334590_188937d7e00100ip0p.html)、[北宗山石图例](https://www.kuhw.com/hhjc/sshjc/2019-09-12/2256.html)。sources.json 保存每张图来源及下载问题；未主张这些网页图片有开放再发布许可。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8');verification=dict(original_user_labels_unchanged=True,benchmark_hashes_verified=True,raster_frozen_unchanged=True,all_four_weights_finite=True,ambiguous_competing_joins_abstained=True,closed_curve_not_opened_for_join=True,pseudo_train_source_exclusion_verified=True,pseudo_numeric_payload_matches_training=True,assistant_drafts_separate_from_user_labels=True,standalone_inference_svg_and_json_verified=True,planned_coordinates_finite=True,robot_connected=False);jsave(ROOT/'verification.json',verification)
    (ROOT/'new_inputs').mkdir(exist_ok=True)
    snapshots=ROOT/'source_snapshot';snapshots.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*v15*'):
        if path.suffix in ['.py','.ps1','.cmd']:shutil.copy2(path,snapshots/path.name)
    print('V15_COMPLETE',json.dumps(dict(reviewed_strokes=benchmark['derived_reviewed_strokes'],raw_candidates=len(read(ROOT/'sources.json')),curated_images=curated['raw_image_count'],crops=curated['crop_count'],accepted_auto_paths=curated['accepted_auto_paths'],reviewed_mean_before=old_mean,reviewed_mean_after=new_mean,model_promoted=False)),flush=True)

if __name__=='__main__':run()
