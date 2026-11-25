# -*- coding: utf-8 -*-
"""Resume completed v12: verify, fairly rescore cached outputs, write report."""
import sys,json,shutil,inspect
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
import numpy as np,torch,cv2
from PIL import Image
from clean_strokes_v12 import ROOT,jsave,digest
from evaluate_clean_strokes_v12 import evaluate_records,oracle_checks
from stroke_link_v12 import load,connect

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));records=manifest['records'];weights_before=digest(ROOT/'final.pt');links_before=digest(ROOT/'link_model.pt')
    old=ROOT/'evaluation'/'summary.json';backup=ROOT/'evaluation'/'summary_before_native_content_fix.json'
    if not backup.exists():shutil.copy2(old,backup)
    for r in records:
        r['prediction']=str(ROOT/'evaluation'/r['id']/'probabilities.npz')
        with np.load(r['prediction']) as z:assert all(np.isfinite(z[k]).all() for k in ('before','after'))
        assert digest(r['image'])==r['image_sha256'] and digest(r['labels'])==r['label_sha256']
    torch.set_num_threads(4);cv2.setNumThreads(2);model,threshold=load();rows=evaluate_records(records,model,threshold);known=oracle_checks(model,threshold)
    def mean(items,side,key):return float(np.mean([r[side][key] for r in items]))
    groups={}
    for name,items in [('clean_train',[r for r in rows if r['id'].startswith('C')]),('clean_different_source_test',[r for r in rows if r['id'].startswith('T')]),('user_train',[r for r in rows if r['id'].startswith('U_') and r['split']=='train']),('user_reused_development',[r for r in rows if r['id']=='U_A06'])]:
        groups[name]=dict(images=len(items),per_image_means={side:{key:mean(items,side,key) for key in ('point_coverage_3px','mean_best_single_path_coverage','mean_reference_fragments','single_path_90pct_fraction')} for side in ('before','new_pixels_old_tracer','new_pixels_learned_links')})
    diagnostic={side:{key:mean(known,side,key) for key in ('point_coverage_3px','mean_best_single_path_coverage','mean_reference_fragments','single_path_90pct_fraction')} for side in ('before','after')}
    # Concrete no-join controls: unrelated parallel strokes and perpendicular
    # endpoints should be rejected by the generic direction gates.
    yy,xx=np.mgrid[:128,:128];prob=np.full((128,128),.8,np.float32);rgb=np.full((128,128,3),245,np.uint8)
    parallel=[np.array([[20.,50.],[100.,50.]]),np.array([[20.,54.],[100.,54.]])];perpendicular=[np.array([[20.,60.],[75.,60.]]),np.array([[78.,62.],[78.,100.]])]
    for paths in (parallel,perpendicular):
        out,events=connect(paths,prob,rgb,model,threshold);assert not events and len(out)==2
    assert digest(ROOT/'final.pt')==weights_before and digest(ROOT/'link_model.pt')==links_before
    config=json.loads((ROOT/'config.json').read_text());assert digest(config['initialization'])==config['initialization_sha256']
    for file in ('final.pt','link_model.pt'):
        state=torch.load(ROOT/file,map_location='cpu',weights_only=False)['model'];assert all(bool(torch.isfinite(x).all()) for x in state.values())
    checked_images=0
    for path in list((ROOT/'evaluation').rglob('*.png'))+list((ROOT/'evaluation').rglob('*.jpg'))+list((ROOT/'diagnostics').glob('*.jpg')):
        with Image.open(path) as im:im.verify()
        checked_images+=1
    for r in records:
        trajectory=json.loads((ROOT/'evaluation'/r['id']/'trajectory.json').read_text());assert not trajectory['ready_for_robot'] and trajectory['physical_z'] is None
        with Image.open(r['image']) as im:w,h=im.size
        for points in trajectory['paths']:
            a=np.asarray(points);assert np.isfinite(a).all() and (a>=0).all() and (a[:,0]<w).all() and (a[:,1]<h).all()
    summary=dict(status='experimental_not_promoted',neural_successful_updates=1000,endpoint_classifier_successful_updates=500,new_clean_train_crops=6,previous_user_train_crops=5,different_page_test_crops=2,test_source_illustrations=1,
      new_real_reference_provenance='assistant_initial_visual_trace_not_expert',semantic_cunfa_recognition_trained=False,physical_width_or_pressure_trained=False,underlying_source_family_independence_not_verified=True,raw_predictions_not_modified_using_user_masks=True,
      native_content_extraction_applied_to_both_models=True,unlinked_and_linked_metrics_use_matching_initial_resampling=True,groups=groups,known_reference_diagnostic_scene_mean=diagnostic,
      source_images_not_procedural_scenes_counted_as_real=True,link_threshold_validation_admissible_negative_pairs=5,actual_drawing=False,robot_connected=False)
    jsave(ROOT/'RESULTS_SUMMARY.json',summary)
    t1=next(r for r in rows if r['id']=='T01');t2=next(r for r in rows if r['id']=='T02');user=next(r for r in rows if r['id']=='U_A06')
    report=f'''# 清晰山石与逐笔连接实验 v12

本轮已经完成：主线中心/方向网络 1000 次有效更新，端点连接评分网络 500 次更新。没有重复执行中断前已经完成的训练。主网络训练循环 82.55 秒；本轮为实验候选，未替换旧绘画模型、未连接机械臂。

## 数据的实际范围

采集了 25 张候选图（在本次有限的旧清单中没有发现字节完全相同的图片；不是全库近似去重保证）。筛选 7 张来源插图，制作 8 个清晰山石局部：6 个训练局部、2 个不同网页来源的测试局部。6 个训练局部主要是两种山石形态的绘制阶段；2 个测试局部来自同一张插图，不能视为两幅独立画作。披麻示例与之前用户示例有相似形态，不作为新颖独立画作计数。

新增参考线共 46 条用于训练，由助手逐条初描，尚未经用户或画家复核；测试参考也属于助手部分草标。教程上的皴法名只作来源提示，没有训练或验证具体皴法分类准确率。

另外混合了原先用户复核的 A01–A05 共 5 个局部、80 个已知轨迹程序样本。程序样本不算真实画作。A06 保留为重复查看的开发对照，不参与本轮梯度训练，不能称新的盲测。

训练来源： [山水画基础教程：山石的画法](https://www.sohu.com/a/277347183_100275234)。不同来源对照：[王兴堂谈山水画临摹：山石程序的五点画法](https://www.culturechina.cn/m/167620.html)。底层书籍、转载关系未完全核实，不能声称严格独立来源泛化。

## 如何处理一笔被拆开

端点分类器用局部切线、间距、曲率稳定性、墨迹和模型概率等 14 项特征，学习两段是否应连接。正例来自已标参考路径的人为切段，负例来自不同路径；实际推理不读取参考路径身份。共使用 160 个程序场景和 11 个真实参考图的配对，训练 500 次；35 个程序场景用于阈值校准。

推理还要求端点方向一致、概率支持，以及双向最佳候选且无近似平分候选。所有补接在 trajectory.json 中记录桥段、分数，并标为未经画家确认的连接假设；不据此宣称还原了原作者真实落笔、抬笔和笔顺。

校准经过几何筛选后只剩 5 个负配对，零误连没有充分统计意义，不能当成真实画作上可靠的错接率。已通过无关平行线和垂直端点不应连接的明确控制案例，但复杂重叠笔画仍需要真实标注检验。

## 实际对照

指标为部分参考线附近 3 像素内，最佳一条预测路径覆盖参考线点的比例。不是全图精确率或专家正确率；下表的两张测试局部来自同一张插图。

| 不同来源局部 | 上一版＋原追踪 | 本轮中心线＋原追踪 | 本轮中心线＋学习连接 |
|---|---:|---:|---:|
| T01 稀疏轮廓 | {t1['before']['mean_best_single_path_coverage']:.1%} | {t1['new_pixels_old_tracer']['mean_best_single_path_coverage']:.1%} | {t1['new_pixels_learned_links']['mean_best_single_path_coverage']:.1%} |
| T02 较密皴线 | {t2['before']['mean_best_single_path_coverage']:.1%} | {t2['new_pixels_old_tracer']['mean_best_single_path_coverage']:.1%} | {t2['new_pixels_learned_links']['mean_best_single_path_coverage']:.1%} |

T01 改善主要来自像素/方向预测，连接器没有补接；T02 虽有局部接续，完整笔画仍不理想。不能把联合结果的改善全部归给连接分类器。

重复开发图 A06 的最佳单路径覆盖 {user['before']['mean_best_single_path_coverage']:.1%} → {user['new_pixels_learned_links']['mean_best_single_path_coverage']:.1%}；平均参考碎片数 {user['before']['mean_reference_fragments']:.2f} → {user['new_pixels_learned_links']['mean_reference_fragments']:.2f}，仍未证明整体连续性改善。mixed_reference_paths_flagged 只是部分参考路径混合的检查提示，不是真实错接准确率。

## 追踪器单独诊断

16 个未用于训练的程序场景，输入已知正确参考概率和方向，另含人为低置信缺口。这是追踪器诊断，不是神经模型的真实山石提取成果。逐场景均值：最佳单路径覆盖 {diagnostic['before']['mean_best_single_path_coverage']:.1%} → {diagnostic['after']['mean_best_single_path_coverage']:.1%}，参考碎片数 {diagnostic['before']['mean_reference_fragments']:.2f} → {diagnostic['after']['mean_reference_fragments']:.2f}。有成功连接，也有缺口未接续，完整记录没有筛掉失败案例。

## 补边与公平对照

宽度不足的图片曾为训练批次补边。核验时发现这些人工边界可能成为伪轮廓，现统一只在原始内容矩形内追踪，两版规则一致。没有利用人工描线或绿色/紫色标签擦除预测线。原始概率保留在 evaluation/*/probabilities.npz。

为消除重采样带来的微小指标差异，未连接对照和连接结果使用相同次数的初始路径重采样。旧的评分记录保留为 evaluation/summary_before_native_content_fix.json。此修正没有重新训练网络或更改权重。

## 查看结果与后续

- evaluation/clean_comparison.jpg：预先固定的两张训练图与全部两张异来源对照；参考与预测分栏。
- evaluation/T01/comparison.jpg、evaluation/T02/comparison.jpg：两张不同来源对照详情。
- evaluation/U_A06/comparison.jpg：旧青绿山水开发图。
- diagnostics/known_reference_comparison.jpg：明确标注为已知参考输入的诊断。
- evaluation/summary.json、RESULTS_SUMMARY.json：全部指标与范围。

清晰、稀疏的石块轮廓已有局部改善；密集皴线的逐笔归属、断笔和错接仍是主要问题。该结果不支持扩大到整幅画或认为各种皴法已经识别成功。下一步应复核少量新样本的逐笔边界，并补充真正容易误接的端点对照，保留不同画作/教程来源的完整留出。

权重、原始概率、数据快照和轨迹已校验；所有实验数据与缓存均在 E 盘。轨迹只有像素坐标，未标定物理 Z、速度或实际毛笔效果。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    verification=dict(completed_model_updates_preserved=True,model_sha256=weights_before,link_model_sha256=links_before,parent_unchanged=True,source_labels_unchanged=True,test_labels_not_trained=True,all_weights_finite=True,all_probability_arrays_finite=True,verified_images=checked_images,trajectories_finite_and_in_bounds=True,native_content_only=True,no_join_parallel_control=True,no_join_perpendicular_control=True,robot_connected=False)
    jsave(ROOT/'final_verification.json',verification)
    for name in ('evaluate_clean_strokes_v12.py','finish_clean_strokes_v12.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/('verified_'+name))
    print('V12_FINISHED',json.dumps(dict(T01=[t1['before']['mean_best_single_path_coverage'],t1['new_pixels_learned_links']['mean_best_single_path_coverage']],T02=[t2['before']['mean_best_single_path_coverage'],t2['new_pixels_learned_links']['mean_best_single_path_coverage']],known_reference=diagnostic,verified_images=checked_images),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
