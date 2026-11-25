# -*- coding: utf-8 -*-
"""Report both experiments, including the unsuccessful transfer outcomes."""
import sys, json
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw
from reviewed_rocks_v14 import ROOT,FONT,SMALL,paste,jsave
from evaluate_clean_strokes_v12 import line_view


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def run():
    fits={k:read(ROOT/k/'evaluation'/'metrics.json')['records'] for k in ['main','replay']}
    devs={k:read(ROOT/k/'development_probes'/'metrics.json')['records'] for k in ['main','replay']}
    board=Image.new('RGB',(1600,435*3+45),'#f5f3ec');draw=ImageDraw.Draw(board)
    draw.text((8,5),'训练拟合与迁移退步同时展示：颜色仅区分候选路径，非笔宽/压力',font=FONT,fill='black')
    old=LAB/'clean_strokes_v12_20261024'
    oldr={r['id']:r for r in read(old/'manifest.json')['records']}
    for i,identity in enumerate(['R02','T01','U_A06']):
        isfit=identity.startswith('R');kind='evaluation' if isfit else 'development_probes'
        if isfit:
            original=Image.open(ROOT/'main'/kind/identity/'original.png');before=Image.open(ROOT/'main'/kind/identity/'before.png');a=Image.open(ROOT/'main'/kind/identity/'after.png');b=Image.open(ROOT/'replay'/kind/identity/'after.png')
        else:
            original=Image.open(oldr[identity]['image']).convert('RGB');shape=(original.height,original.width)
            before=Image.fromarray(line_view(shape,read(old/'evaluation'/identity/'trajectory.json')['paths']))
            a=Image.fromarray(line_view(shape,read(ROOT/'main'/kind/identity/'trajectory.json')['paths'],True))
            b=Image.fromarray(line_view(shape,read(ROOT/'replay'/kind/identity/'trajectory.json')['paths'],True))
        for j,(title,im) in enumerate([('原图：'+('本轮训练图' if isfit else '本轮不训练的开发图'),original),('旧版流程',before),('仅四张复核图训练',a),('复核图＋旧图软教师回放',b)]):
            draw.text((j*400+8,45+i*435),identity+' '+title,font=SMALL,fill='black');paste(board,im,j*400+8,80+i*435,384,390)
    board.save(ROOT/'overview.jpg',quality=95)
    fmt=lambda x:f'{100*x:.1f}%'
    text='''# 本轮结果：用户复核已接入，尚无可替换正式模型的版本

已按用户确认，仅在训练副本中用新线替换 R01 对应重叠草稿。原始 50 条标注保留；派生数据 48 条参考笔画、1 条仅监督位置/方向的旧线余段、22 处绿色背景。其他未匹配旧线保留。原始文件与 v12 父权重校验和未变，机械臂未连接。

## 实际完成的实验

1. 600 次更新的真实三图小拟合诊断（36 条参考笔画，R04 本阶段不训练）。
2. 800 次更新的全部四张复核图训练（main）。
3. 800 次更新的回放对照（replay）：仍用四张复核图，另外循环使用 7 张旧训练图的 v12 软概率/方向，限制遗忘。这些软教师输出不是新人工标注，T01/T02/A06 不在回放中。

下面数字是每条参考线被一条候选路径覆盖的平均比例，容差 3 像素；不能当作准确率，也不证明分笔无误。未标注区域无法评价完整精度。旧流程和本轮的追踪器不同，差值包含网络与追踪方法两部分；各明细报告另列了相同追踪器下的父权重结果。

| 复核训练图 | 旧版流程 | 四图训练 | 加入回放 |
|---|---:|---:|---:|
'''
    for a,b in zip(fits['main'],fits['replay']):
        text+=f"| {a['id']} | {fmt(a['before']['mean_best_single_path_coverage'])} | {fmt(a['direct_paired']['mean_best_single_path_coverage'])} | {fmt(b['direct_paired']['mean_best_single_path_coverage'])} |\n"
    text+='\n| 重复开发图（本轮不训练） | 旧版流程 | 四图训练 | 加入回放 |\n|---|---:|---:|---:|\n'
    for a,b in zip(devs['main'],devs['replay']):
        text+=f"| {a['id']} | {fmt(a['baseline']['mean_best_single_path_coverage'])} | {fmt(a['current']['mean_best_single_path_coverage'])} | {fmt(b['current']['mean_best_single_path_coverage'])} |\n"
    text+='''
## 结论

四图训练让目标参考点的平均覆盖达到 99.3%，但平均单笔覆盖只有 80.8%，长线仍有拆分/错误连接；换旧开发图则明显退步。说明“画到线条附近”与“正确分成完整笔画”仍有较大差距，而且本轮四图训练过拟合。

回放后，平均训练图单笔覆盖为 62.8%；T02 与 A06 大致回到旧版水平，T01 仍从 66.2% 降至 45.5%。回放能部分缓解退步，但没有得到全面更好的版本。两组权重都仅保留为实验结果，正式绘画流程继续保持原版。

四张是两个山石形态的步骤图，且父模型见过；开发图也被反复检查，不能声称独立泛化准确率。原尺寸与分块推理存在明显差异，尚未解决整幅推理稳定性。皴法类别、真实笔宽与墨色浓淡本轮均未训练，更没有实机压力/速度校准。

## 接下来应做的具体工作

优先把 R01–R04 的高置信断点/接点做成“必须连接、不得连接、不确定”的局部训练样本，在冻结中心线网络的条件下单独改进追踪器。正确参考输入追踪器时也会出现分笔不足，说明这一问题不只来自图像噪声。保持旧图回放，避免继续只对同样四张图加轮数。然后引入不同原图来源的清晰单石与组合石，固定一批完整复核、按原画来源隔离的测试图；通过后再扩大复杂整幅图和皴法名称/宽度浓淡监督。

## 图与明细

- overview.jpg：固定展示 R02、T01、A06，同时呈现训练改善与迁移退步。
- main/training_comparison.jpg：四张全部训练图，含你的参考与真实模型输出。
- replay/training_comparison.jpg：回放后的四张全部结果。
- main/development_probes/comparison.jpg、replay/development_probes/comparison.jpg：全部三张迁移开发对照。
- REPORT.md、REPORT_replay.md：训练配置、逐图指标、诊断与文件说明。
- verification.json、verification_replay.json：源标注未变、父模型未变、权重有限、轨迹坐标有效等检查均通过。

所有候选轨迹都是离线像素路径，ready_for_robot=false。用于诊断的“正确参考输入追踪”单独标记，未当成神经模型预测展示。
'''
    (ROOT/'RESULTS.md').write_text(text,encoding='utf-8')
    jsave(ROOT/'RESULTS.json',dict(stages=['small','main','replay'],training_updates={'small':600,'main':800,'replay':800},reviewed_strokes=48,raw_user_strokes=50,background_regions=22,original_files_preserved=True,promoted_to_painting=False,robot_connected=False,style_classifier_trained=False,fit_improved_but_transfer_not_accepted=True))
    print('COMPARATIVE_REPORT_COMPLETE',str(ROOT/'RESULTS.md'))


if __name__=='__main__':run()
