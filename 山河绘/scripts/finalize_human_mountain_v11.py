# -*- coding: utf-8 -*-
"""Readable experiment record and compact presentation of unchanged outputs."""
import sys,json,shutil
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
from PIL import Image,ImageDraw
from human_mountain_v11 import ROOT,FONT,paste,jsave,digest

def run():
    final=ROOT/'refinement_sparse';m=json.loads((final/'evaluation'/'metrics.json').read_text());held=json.loads((final/'evaluation'/'A06_same_work_holdout'/'metrics.json').read_text());manifest=json.loads((final/'manifest.json').read_text());first=json.loads((ROOT/'training_summary.json').read_text());second=json.loads((final/'training_summary.json').read_text());rows=m['rows'];bg=sum(r['before']['explicit_background_pixels'] for r in rows)
    bg_before=sum((r['before']['explicit_background_high_probability_fraction'] or 0)*r['before']['explicit_background_pixels'] for r in rows)/bg;bg_after=sum((r['after']['explicit_background_high_probability_fraction'] or 0)*r['after']['explicit_background_pixels'] for r in rows)/bg
    # Layout only: predictions remain exactly the PNGs saved by inference.
    folder=final/'evaluation'/'same_work_full_scene';scene=Image.new('RGB',(2000,500),'#f5f3ec');d=ImageDraw.Draw(scene);d.text((10,5),'同一画作全图推理：含已训练局部及未标注内容，非独立画作验证；候选线非实画',font=FONT,fill='black')
    for j,(label,name) in enumerate([('原图','original'),('训练前候选线','before'),('训练后候选线','after'),('训练后叠加','overlay')]):d.text((j*500+8,45),label,font=FONT,fill='black');paste(scene,Image.open(folder/(name+'.png')),j*500+8,85,484,400)
    scene.save(folder/'comparison.jpg',quality=95)
    conclusion=dict(status='experimental_not_promoted',total_successful_updates=first['successful_updates']+second['successful_updates'],user_marked_paths=339,human_background_regions=5,purple_ignore_regions=3,train_images=['A01','A02','A03','A04','A05'],internal_holdout='A06',source_work_count=1,independent_artwork_test=False,
      partial_training_reference_coverage_3px=[m['stroke_weighted_coverage_3px_before'],m['stroke_weighted_coverage_3px_after']],partial_training_reference_coverage_6px=[m['stroke_weighted_coverage_6px_before'],m['stroke_weighted_coverage_6px_after']],
      internal_holdout_reference_coverage_3px=[held['before']['partial_reference_coverage_3px'],held['after']['partial_reference_coverage_3px']],internal_holdout_reference_coverage_6px=[held['before']['partial_reference_coverage_6px'],held['after']['partial_reference_coverage_6px']],
      marked_background_high_response_fraction=[bg_before,bg_after],unmarked_is_unknown=True,purple_all_losses_zero=True,raw_predictions_not_masked_by_annotations=True,original_model_unchanged=True,actual_drawing=False,robot_connected=False,
      conclusion='Marked-background responses improve substantially; narrow-tolerance training reference coverage improves modestly, wider-tolerance and A06 coverage decline. Completeness/generalization improvement not established.')
    jsave(ROOT/'EXPERIMENT_STATUS.json',conclusion)
    report=f'''# 人工山体主线训练结果（2026-10-24）

已按用户本次明确确认的标注训练 A01–A05：339 条山体描线、5 个绿色背景区域、3 个紫色忽略区域。同一幅画作的局部，不是五幅独立画作。A01 文件仍保存为 draft，以本次聊天中用户确认完成作为使用依据；没有修改源标签状态。

累计完成 1400 次有效更新：初版 600 次，修订 800 次。初版将宽片响应骨架化后产生多边形伪线，失败结果仍保存在主目录 evaluation/；最终候选保存在 refinement_sparse/。两版对照均采用相同的法向非极大值抑制、阈值和路径处理；不能把后处理更换的收益全算成训练收益。

绿色背景只用其内部，边界留 2px 余量；少量与描线冲突的位置保持未知。紫色区域完全排除直接训练损失、稀疏先验、局部峰值约束的采样以及评分。没有画出的其他内容保持未知，不当作用户确认的负例。修订训练在其余未标注区域使用了弱稀疏先验和旧模型软正则；它们不是人工背景真值，可能压低未标注的有效细线。

预测时只输入原始 RGB 图片，没有读入绿色、紫色区域去擦除或修饰输出，也没有把橙色人工描线混入模型输出。粗细、浓淡、具体皴法类别和机械臂动作没有在本轮训练。

| 对照指标 | 旧模型 | 本轮最终候选 |
|---|---:|---:|
| A01–A05 部分参考点覆盖，3px容差，按标线数加权 | {m['stroke_weighted_coverage_3px_before']:.1%} | {m['stroke_weighted_coverage_3px_after']:.1%} |
| A01–A05 部分参考点覆盖，6px容差，按标线数加权 | {m['stroke_weighted_coverage_6px_before']:.1%} | {m['stroke_weighted_coverage_6px_after']:.1%} |
| A06 同画作内部留出部分参考点覆盖，3px容差 | {held['before']['partial_reference_coverage_3px']:.1%} | {held['after']['partial_reference_coverage_3px']:.1%} |
| A06 同画作内部留出部分参考点覆盖，6px容差 | {held['before']['partial_reference_coverage_6px']:.1%} | {held['after']['partial_reference_coverage_6px']:.1%} |
| 用户绿色背景内部高响应比例，概率≥0.5 | {bg_before:.3%} | {bg_after:.3%} |

A06 没有参与本轮训练，也没有用于选择检查点；评分排除了与 A03 同一扫描图已知重叠区及 16px 余量、紫色区。但它仍属于同画作，其他参考扫描间重叠尚未排尽，父模型也可能见过该画，不能称独立画作验证。绿色背景指标是训练区域内的表现，不是全图精确率。

## 采用判断

背景杂纹抑制明显改善；主线定位的提升有限且不一致。A03、A05 的 3px 覆盖下降，A06 也略降，6px 指标下降，尚未证明完整性或泛化改善。组合山石仍有断裂、重复和额外线条。本轮保留为实验候选，不替换实际绘画模型。

## 查看文件

- refinement_sparse/evaluation/first_preview.jpg：预先固定展示 A01、A04、A05。
- refinement_sparse/evaluation/all_reviewed.jpg：全部五张训练参考图。
- refinement_sparse/evaluation/A06_same_work_holdout/comparison.jpg：内部留出对照。
- refinement_sparse/evaluation/same_work_full_scene/comparison.jpg：同画全图推理。
- refinement_sparse/evaluation/metrics.json：逐图指标及部分参考路径覆盖。
- refinement_sparse/final.pt：最终实验权重。

已验证权重有限、PNG可读、快照哈希一致、旧权重未改。回归检查确认常数宽片响应不会凭空生成中轴多边形、已知直线脊仍保留；紫色区域在初版和修订版均不产生损失梯度。所有训练、缓存及结果在 E 盘，未连接机械臂。
'''
    (ROOT/'EXPERIMENT_STATUS.md').write_text(report,encoding='utf-8');(final/'REPORT.md').write_text(report,encoding='utf-8')
    for name in ('human_mountain_eval_v11.py','human_mountain_holdout_v11.py'):shutil.copy2(Path(__file__).with_name(name),final/('final_'+name))
    # Validate raw artifacts without rerunning or altering predictions.
    import torch
    state=torch.load(final/'final.pt',map_location='cpu',weights_only=False)['model'];assert all(bool(torch.isfinite(v).all()) for v in state.values())
    for r in manifest['records']:assert digest(r['snapshot_labels'])==r['label_sha256'] and digest(r['snapshot_image'])==r['image_sha256']
    verification=json.loads((final/'verification.json').read_text());verification.update(all_state_tensors_finite=True,holdout_did_not_train=True,source_group_count=1,experimental_not_promoted=True);jsave(final/'verification.json',verification)
    print(json.dumps(conclusion,ensure_ascii=False),flush=True)
if __name__=='__main__':run()
