"""Continuation-run report: actual counts, provenance, comparable metrics."""
import hashlib
import json
from collections import Counter
from .runtime import LAB


def write_report(run,review_note='Pending visual review; metrics are not expert painting accuracy.'):
    summary=json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    manifest=json.loads((run/'dataset_manifest_snapshot.json').read_text(encoding='utf-8'))
    comparison=json.loads((run/'comparison_audit.json').read_text(encoding='utf-8'))
    challenge_path=run/'synthetic_challenge.json'
    challenge=json.loads(challenge_path.read_text(encoding='utf-8')) if challenge_path.exists() else None
    inference=json.loads((run/'inference'/'summary.json').read_text(encoding='utf-8'))
    inference_steps=sorted({r['checkpoint_step'] for r in inference})
    baseline=LAB/'runs'/comparison['baseline']
    previous=json.loads((baseline/'training_summary.json').read_text(encoding='utf-8'))
    real=Counter(r['split'] for r in manifest['real']);synthetic=Counter(r['split'] for r in manifest['synthetic'])
    audit=dict(training_summary=summary,previous_summary=previous,config=config,
               source_snapshot_verified=comparison['training_code_snapshot_matches'],
               dataset_snapshot_sha256=hashlib.sha256((run/'dataset_manifest_snapshot.json').read_bytes()).hexdigest(),
               checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('*.pt')},
               real_counts=dict(real),synthetic_counts=dict(synthetic),human_dense_gold_labels=0,
               visual_review=review_note,model_ready_for_robot=False,
               logged_nonfinite_gradient_iterations=comparison.get('logged_nonfinite_gradient_iterations',[]),
               all_checkpoint_model_tensors_finite=comparison.get('all_checkpoint_model_tensors_finite',{}))
    audit['synthetic_challenge']=challenge
    assert audit['dataset_snapshot_sha256']==config['data_manifest_sha256']
    (run/'evaluation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[]
    for domain,label in [('synthetic','合成几何验证'),('real_weak','与原自动弱标签的一致度（非国画精度）')]:
        a=previous['validation'][domain];b=summary['validation'][domain]
        lines.append(f"| {label} | {a['f1']:.4f} | {b['f1']:.4f} |")
    sources='\n'.join('- '+r['id']+' | '+r['split']+' | '+str(r['source'].get('page_url','用户提供参考图')) for r in manifest['real'])
    challenge_text='未运行。'
    if challenge:
        a=challenge['metrics'][comparison['baseline']];b=challenge['metrics'][run.name]
        challenge_text=(f"16个未参与训练的合成种子（900000–900015），施加褪色与已知分隔线：F1 {a['f1']:.4f} → {b['f1']:.4f}；"
                        f"已知分隔线像素误提比例 {a['frame_false_positive_fraction']:.2%} → {b['frame_false_positive_fraction']:.2%}。"
                        "这只说明该合成干扰测试的变化，不代表真实屏风、真实国画准确率。见synthetic_challenge.json和synthetic_challenge.png。")
    card=f'''# 工笔山水结构提线：第二轮续训

## 已实际完成
- 从 {config['initialization_checkpoint']} 初始化权重，新建优化器；本轮完成 {summary['completed_steps']} 步，父权重位于第 {summary['parent_checkpoint_step']} 步。
- 本轮耗时 {summary['elapsed_seconds']:.1f} 秒；{config['gpu']}；峰值模型分配显存 {summary['max_gpu_memory_bytes']/1024**3:.2f} GiB。
- 当前预览使用best.pt，处于本轮第 {inference_steps} 次迭代；last.pt保留本轮最后一步。选择规则未因逐图观感而更改。
- 本轮实际抽取样本次数：{summary['consumed_training_samples']}，其中重复随机裁块不计为新增画作。
- 模型权重平均更新量 {summary['head_mean_absolute_parameter_update']:.6f}；文件哈希见 evaluation.json。
- 步数指训练迭代，不保证每一步都更新优化器。FP16混合精度中，非有限梯度更新由GradScaler跳过；抽样日志中出现的迭代为 {comparison.get('logged_nonfinite_gradient_iterations',[])}。只每25步记录一次，无法据此统计准确跳步总数。保存权重和运行缓冲区有限性检查：{comparison.get('all_checkpoint_model_tensors_finite',{})}。
- 原始真实图依然是23张（用户7张 + 馆藏16张）：训练{real['train']}张、验证{real['validation']}张。没有新下载图，也没有新增专家线稿。
- 合成训练{synthetic['train']}张、合成验证{synthetic['validation']}张。按画作/册页分组隔离。

## 训练变化
1. 只重建训练组弱标签：增加与蓝绿色无关的深色曲线候选，补淡墨树干、岩石主线。拒绝过短、团块及大部分近水平水纹。新增候选置信度较低，仍会漏线和误提。
2. 记录少量助手草拟的装裱/书法排除区域，以及参考图02的天空/水域排除区域。这不是专家审核的标准线稿。
3. 真实图采样概率由0.4提高到{config['real_probability_after_warmup']}；随机尺度0.65–1.6，灰度/褪色/色调混合；合成分隔线和画框负样本。
4. 从首轮权重开始微调，编码器学习率 {config['learning_rates']['encoder']}、解码器 {config['learning_rates']['decoder']}，余弦衰减；没有改网络结构。
5. 训练前检查24个增强配对样本的尺寸、范围与有限数值；验证记录逐项比较确认未变，原始manifest及标签保留。

## 对照指标
| 范围 | 首轮最后一步 F1 | 本轮最后一步 F1 |
| --- | ---: | ---: |
{chr(10).join(lines)}

这些是最后一步的诊断结果；best.pt 由75%合成损失+25%弱标签损失选择，可能与 last.pt 不同步。它们不能代表实际画作的人工准确率。自动验证标签本身很稀疏，补出的真实线条也可能被旧标签记为假阳性。不存在经过专家审核的完整真值，也没有全新最终测试集；这三幅留出图已经用于开发观察和模型选择。

### 合成干扰诊断
{challenge_text}

## 本轮目视复核
{review_note}

## 预测对照与复现
- baseline_comparison/：每张图的“原图 / 第一版 / 第二版”。两版同尺寸、同概率阈值0.35/0.55、同12像素最短路径规则。
- reference_comparison.jpg：七张训练参考图对比；不可当作泛化证明。
- heldout_comparison.jpg：三组独立留出图对比；同样展示失败样本。
- inference/：本轮 best.pt 的实际概率图、结构PNG、SVG、像素路径JSON。
- comparison_audit.json：阈值、尺寸、权重初始化和代码快照检查。路径更多或更少不等于更好。
- source_snapshot/、dataset_manifest_snapshot.json：本轮训练开始时的代码与数据清单。
- 所有数据、缓存、权重均在E盘；未连接或移动机器人；未更换生产提线引擎。

在 F:\\山河绘\\山河绘 运行：
```powershell
.\\scripts\\gongbi_lab.ps1 infer --run {run.name}
.\\scripts\\gongbi_lab.ps1 compare --run {run.name} --baseline {baseline.name}
```

训练参数在config.json。复现训练需使用新运行名，避免覆盖结果：
```powershell
.\\scripts\\gongbi_lab.ps1 train --name {run.name}_reproduce --steps 3000 --init-from "{config['initialization_checkpoint']}" --manifest "{config['data_manifest_path']}" --augmentation v2 --real-probability 0.6 --warmup 0 --encoder-lr 0.00002 --decoder-lr 0.00015
```

## 下一步数据工作
模型仍受弱标签限制。优先对多幅独立青绿工笔画逐像素复核山脊、主皴线、树干、屋宇和负样本；完整作品保持分组。标注工具仍可通过 gongbi_lab.ps1 annotate 打开，不通过浏览器默认下载目录保存文件。

## 来源
{sources}
'''
    (run/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    print('CONTINUATION_REPORT',str(run/'MODEL_CARD.md'),flush=True)
