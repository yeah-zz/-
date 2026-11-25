# -*- coding: utf-8 -*-
"""Regression probes and reproducible report for reviewed rock fitting."""
import sys, json, shutil
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure, LAB
configure()
import numpy as np
import torch, cv2
from PIL import Image, ImageDraw
from reviewed_rocks_v14 import ROOT, REVIEW, PARENT, model_new, direct, fields, digest, jsave, FONT, SMALL, paste
from evaluate_reviewed_v14 import extract
from evaluate_clean_strokes_v12 import path_metrics, line_view
STAGE = 'main'


def probes():
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    folder = ROOT / STAGE / 'development_probes'
    folder.mkdir(parents=True, exist_ok=True)
    old = LAB / 'clean_strokes_v12_20261024'
    records = json.loads((old / 'manifest.json').read_text(encoding='utf-8'))['records']
    records = [next(r for r in records if r['id'] == name) for name in ['T01', 'T02', 'U_A06']]
    model = model_new().cuda()
    model.load_state_dict(torch.load(ROOT / STAGE / 'final.pt', map_location='cuda', weights_only=False)['model'])
    model.eval()
    parent = model_new().cuda().eval()
    board = Image.new('RGB', (1600, 435 * len(records) + 45), '#f5f3ec')
    d = ImageDraw.Draw(board)
    d.text((8, 5), '重复开发对照：本轮不训练；参考为局部标注，不能计算整图准确率', font=FONT, fill='black')
    summaries = []
    for i, r in enumerate(records):
        rgb = np.asarray(Image.open(r['image']).convert('RGB'))
        doc = json.loads(Path(r['labels']).read_text(encoding='utf-8'))
        _, _, mask, _ = fields(doc)
        pred = direct(model, rgb)
        left, top = r.get('padding', [0, 0])
        box = r.get('source_box', [0, 0, rgb.shape[1], rgb.shape[0]])
        w, h = box[2] - box[0], box[3] - box[1]
        paths, info, _ = extract(pred[:, top:top+h, left:left+w])
        paths = [p + np.array([left, top]) for p in paths]
        parent_pred = direct(parent, rgb)
        parent_paths, _, _ = extract(parent_pred[:, top:top+h, left:left+w])
        parent_paths = [p + np.array([left, top]) for p in parent_paths]
        before = json.loads((old / 'evaluation' / r['id'] / 'trajectory.json').read_text(encoding='utf-8'))['paths']
        reference = [s['points'] for s in doc['strokes']]
        if r['id'] == 'U_A06':
            mask[:, :86, :96] = 0
        row = dict(id=r['id'], baseline=path_metrics(reference, before, mask[3]),
                   current=path_metrics(reference, paths, mask[3]),
                   parent_same_decoder=path_metrics(reference, parent_paths, mask[3]),
                   independent_test=False, trained_this_round=False,
                   previously_reused_development=True,
                   background_high_response=float(np.mean(pred[0][mask[1]>0] >= .5)) if mask[1].any() else None)
        summaries.append(row)
        out = folder / r['id']
        out.mkdir(exist_ok=True)
        jsave(out / 'metrics.json', row)
        jsave(out / 'trajectory.json', dict(paths=[p.round(2).tolist() for p in paths], ready_for_robot=False, actually_drawn=False))
        np.savez_compressed(out / 'probabilities.npz', prediction=pred)
        for j, (label, a) in enumerate([('原图', rgb), ('局部参考（非预测）', line_view(rgb.shape[:2], reference)),
                                       ('旧版候选轨迹', line_view(rgb.shape[:2], before)),
                                       ('本轮预测轨迹', line_view(rgb.shape[:2], paths, True))]):
            d.text((j*400+8, 45+i*435), r['id']+' '+label, font=SMALL, fill='black')
            paste(board, a, j*400+8, 80+i*435, 384, 390)
        print('V14_PROBE', r['id'], row['baseline']['mean_best_single_path_coverage'], row['current']['mean_best_single_path_coverage'], 'parent_same_decoder',row['parent_same_decoder']['mean_best_single_path_coverage'],flush=True)
    board.save(folder / 'comparison.jpg', quality=95)
    jsave(folder / 'metrics.json', dict(records=summaries, no_model_selection_on_these_probes=True))
    return summaries


def report(rows):
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    ev = json.loads((ROOT / STAGE / 'evaluation' / 'metrics.json').read_text(encoding='utf-8'))['records']
    summary = json.loads((ROOT / STAGE / 'evaluation' / 'summary.json').read_text(encoding='utf-8'))
    # User-facing preview contains model output only alongside source and reference.
    # The oracle diagnostic remains a separate file and is never called prediction.
    board = Image.new('RGB', (1600, 435*4+45), '#f5f3ec')
    d = ImageDraw.Draw(board)
    d.text((8, 5), '本轮四张复核训练图：颜色区分候选笔画；仍有断线/串线，未经实机验证', font=FONT, fill='black')
    for i, r in enumerate(manifest['records']):
        out = ROOT / STAGE / 'evaluation' / r['id']
        for j, (name, title) in enumerate([('original', '原图'), ('user_reference', '你的参考（非预测）'),
                                          ('before', '旧版候选轨迹'), ('after', '本轮模型预测轨迹')]):
            d.text((400*j+8, 45+435*i), r['id']+' '+title, font=SMALL, fill='black')
            paste(board, Image.open(out/(name+'.png')), j*400+8, 80+i*435, 384, 390)
    board.save(ROOT / STAGE / 'training_comparison.jpg', quality=95)
    state = torch.load(ROOT / STAGE / 'final.pt', map_location='cpu', weights_only=False)
    assert all(bool(torch.isfinite(v).all()) for v in state['model'].values())
    assert digest(PARENT) == state['config']['initialization_sha256']
    for r in manifest['records']:
        assert digest(REVIEW/'labels'/(r['id']+'.json')) == r['original_user_label_sha256']
        assert digest(Path(r['labels'])) == r['label_sha256']
        assert digest(Path(r['image'])) == r['image_sha256']
        doc = json.loads(Path(r['labels']).read_text(encoding='utf-8'))
        assert len(doc['strokes']) == r['strokes']
        z = np.load(r['target'])
        t, m = z['target'], z['mask']
        assert np.isfinite(t).all() and np.isfinite(m).all()
        left, top, w, h = r['native_content']
        outside = np.ones(t.shape[1:], bool); outside[top:top+h, left:left+w] = False
        assert not np.any(m[:, outside])
        if r['id'] == 'R01':
            assert len(doc['replacement_log']) == 2 and len(doc['geometry_only_fragments']) == 1
            assert not {'draft_1', 'draft_2'} & {s['id'] for s in doc['strokes']}
        plan = json.loads((ROOT/STAGE/'evaluation'/r['id']/'trajectory.json').read_text(encoding='utf-8'))
        assert plan['ready_for_robot'] is False and plan['actually_drawn'] is False
        for p in plan['paths']:
            p = np.asarray(p)
            assert np.isfinite(p).all()
            assert (p[:,0]>=left).all() and (p[:,0]<left+w).all()
            assert (p[:,1]>=top).all() and (p[:,1]<top+h).all()
    verification = dict(weights_finite=True, source_user_labels_unchanged=True, parent_unchanged=True,
                        derived_replacements_audited=True, artificial_padding_unsupervised=True,
                        planned_coordinates_finite_in_bounds=True, robot_connected=False)
    jsave(ROOT/('verification.json' if STAGE=='main' else 'verification_'+STAGE+'.json'), verification)
    fmt = lambda x: f'{100*x:.1f}%'
    text = '''# v14 用户复核山石实验（2026-10-24）

本轮接入 R01–R04 的用户修改。原始标注共 50 条线、22 处绿色背景。用户确认“新线替换对应的重叠旧草稿”后，仅在训练副本中处理 R01 两条对应旧线，得到 48 条参考笔画。部分替换的旧线余段另存为 1 条 geometry_only_fragments：只监督位置与方向，不把切割点当成用户指定的落笔/抬笔。原始文件和 v12 权重的校验和保持不变。

## 训练与边界

- 600 次更新：R01–R03 小样本真实图拟合诊断，R04 未参与这一步更新。
- 800 次更新：从 v12 初始化，使用全部四张复核图；未改动正式绘画流程。
- 未描区域保持未知，仅用户描线附近和明确绿色背景参与监督；原生尺寸训练，人工补边不参与监督。
- 四张图来自两个山石形态的步骤图，均被父模型见过。所有复核状态为 checked_partial，因此不代表完整金标准，更不是四幅独立画作的泛化测试。
- 模型输出中心线、方向及交点/端点候选；没有训练皴法名称分类、真实笔宽/墨浓度或压力映射。

## 四张复核图上的拟合

以下“单笔覆盖率”是每条参考线被某一条候选路径在 3 像素容差内覆盖的平均比例，不能当作整体准确率；串线也可能覆盖参考。旧版与本轮包含追踪器差异，不能把差值完全归因于网络训练。

| 图 | 旧版单笔覆盖率 | 本轮单笔覆盖率 | 本轮参考点覆盖率 |
|---|---:|---:|---:|
'''
    for r in ev:
        text += f"| {r['id']} | {fmt(r['before']['mean_best_single_path_coverage'])} | {fmt(r['direct_paired']['mean_best_single_path_coverage'])} | {fmt(r['direct_paired']['point_coverage_3px'])} |\n"
    text += f"\n本轮平均参考点覆盖率 {fmt(summary['mean_reference_point_coverage'])}，平均单笔覆盖率 {fmt(summary['mean_best_single_path_coverage'])}。这两个数字仅衡量对已描线的覆盖，不能证明没有多余线，也不能证明分笔正确。\n"
    text += '''
## 重复开发图的迁移检查

T01、T02 的参考是助手草标；A06 是用户局部标注。三者本轮不参与梯度训练，但已被多次查看，属于开发对照，不能叫独立盲测。A06 的已知同扫描重叠和紫色区域仅在评分时排除，不用于修饰预测。

| 图 | 旧版流程单笔覆盖率 | 父权重采用本轮追踪器 | 本轮单笔覆盖率 |
|---|---:|---:|---:|
'''
    for r in rows:
        text += f"| {r['id']} | {fmt(r['baseline']['mean_best_single_path_coverage'])} | {fmt(r['parent_same_decoder']['mean_best_single_path_coverage'])} | {fmt(r['current']['mean_best_single_path_coverage'])} |\n"
    text += '''
## 结论与后续

本轮用于检验少量用户复核是否能纠正中心线和分笔。仍有断线、相邻线连接错误的风险；不能据此称整幅复杂山水或各种皴法已能可靠提取。分块推理与原尺寸推理也有差异，完整数字已分别记录，展示图使用原尺寸预测，未按参考去擦除多余线。

后续应保留当前真实标注，扩充不同来源的清晰单石及组合石，并单独标注容易误连的接点“接/不接/无法判断”，避免只增加同类步骤图更新次数。再固定整幅图的分块推理规则与独立来源测试，之后才训练皴法类别及宽度浓淡。当前缺乏独立来源且完整复核的测试集，无法给出泛化准确率。

## 结果文件

- main/training_comparison.jpg：原图、用户参考、旧版、本轮真实模型预测，四张全部展示。
- main/development_probes/comparison.jpg：全部三张重复开发对照。
- main/evaluation/comparison.jpg：另含“正确参考输入追踪诊断”；该栏是参考驱动的追踪器诊断，**不是神经模型提取结果**。
- main/evaluation/summary.json、各图 metrics.json：原尺寸/分块/不配对/已知参考诊断分别记录。
- manifest.json、data/*_original_user.json：原始快照、派生标注及逐项替换记录。
- main/final.pt：实验权重；未部署到机械臂。
- verification.json：源文件/父权重未变、权重有限、边界及轨迹数值检查。

所有轨迹均为离线像素坐标，ready_for_robot=false，未校准物理 Z、速度、压力与墨水，也没有纸面实画。
'''
    if STAGE=='replay':
        text=text.replace('800 次更新：从 v12 初始化，使用全部四张复核图；未改动正式绘画流程。','800 次更新：从 v12 初始化，使用全部四张复核图，并循环回放原训练集 C01、C06、U_A01–U_A05 的 v12 软概率与方向。软教师用于限制遗忘，不是新专家标注；T01、T02、A06 没有参加回放。未改动正式绘画流程。').replace('main/', 'replay/').replace('verification.json','verification_replay.json')
        text+='\n本回放对照是在 main 阶段开发图出现退步后开展的预设 800 步训练。没有根据这些对照图挑选中途权重；开发图已经影响了实验方向，不能用作独立验证。\n'
    (ROOT/('REPORT.md' if STAGE=='main' else 'REPORT_'+STAGE+'.md')).write_text(text, encoding='utf-8')
    for name in ['reviewed_rocks_v14.py', 'evaluate_reviewed_v14.py', 'reviewed_v14_final_report.py']:
        shutil.copy2(Path(__file__).with_name(name), ROOT/('final_'+name))
    print('V14_REPORT_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',nargs='?',choices=['main','replay'],default='main');args=parser.parse_args();STAGE=args.stage
    report(probes())
