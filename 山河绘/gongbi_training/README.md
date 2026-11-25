# 工笔山水结构线训练实验

实际首轮成果、数据与权重：`E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\runs\mobilenet_unet_v01`。
阅读该目录 MODEL_CARD.md，其中包含全部预测、失败现象和来源索引。

第一版已实际 CUDA 训练1200步，但独立画作预测严重漏线，尚未达到稳定换图使用的标准。
训练参考图上的可见山形不代表泛化能力。无专家完整线稿；弱标签不能作为真实精度证明。

在项目根目录运行（包装脚本会将缓存与临时目录指向 E 盘）：

```powershell
.\scripts\gongbi_lab.ps1 infer --run mobilenet_unet_v01
.\scripts\gongbi_lab.ps1 report --run mobilenet_unet_v01
.\scripts\gongbi_lab.ps1 check
.\scripts\gongbi_lab.ps1 annotate
```

标注地址 http://127.0.0.1:8766 。直接保存 E 盘；未知像素不当背景，原始弱标签不覆盖，重存有备份。
优先校正山脊/主岩线、树干与屋宇；给绢纹、装裱、文字、印章、色彩渐变标负样本。
完整画作与同册页始终在同一数据分组，验证组的标注保留为验证用途。

下一轮增量标注后的运行：

```powershell
.\scripts\gongbi_lab.ps1 prepare --manual-only
.\scripts\gongbi_lab.ps1 train --name mobilenet_unet_manual_v03 --steps 1200
.\scripts\gongbi_lab.ps1 infer --run mobilenet_unet_manual_v03
```

首轮准备/采集可由 collect、prepare 复现，但重复 prepare 会重新生成弱标签和manifest，已有人工标注应再运行 prepare --manual-only 合并。
当前代码在首轮结束后补充了手工标注合并、检查和避免覆盖旧运行的功能；首轮实际训练代码在 source_snapshot/，哈希在 config.json。
两通道为结构与可选细节。SVG/JSON 仍是像素路径，不包含纸面标定、压力、速度、干湿浓淡参数。此实验不连接机械臂，不修改现有绘画流程默认引擎。

## 第二版续训（已完成）

`E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\runs\mobilenet_unet_v02` 保存第二版：从第一版best.pt初始化，再运行3000次迭代。未新增专家标签；在训练组补充与颜色无关的曲线弱候选、少量助手草拟背景排除区域、缩放/褪色增强、合成画框/分隔线负样本。真实图采样概率0.6。原验证图、验证标签不变；第一版数据未覆盖。

注意FP16 GradScaler会跳过溢出梯度更新，所以迭代数不等于保证成功的优化器更新次数。最终权重有限性与抽样日志审计在comparison_audit.json；best.pt与last.pt可处于不同迭代。

```powershell
.\scripts\gongbi_lab.ps1 infer --run mobilenet_unet_v02
.\scripts\gongbi_lab.ps1 compare --run mobilenet_unet_v02 --baseline mobilenet_unet_v01
.\scripts\gongbi_lab.ps1 challenges --run mobilenet_unet_v02
.\scripts\gongbi_lab.ps1 check --run mobilenet_unet_v02
```

阅读第二版MODEL_CARD.md与baseline_comparison/。报告区分合成诊断、旧自动标签一致度与实际目视判断，不能将其混为真实国画准确率。复现参数及数据快照都在该目录，需使用新运行名。默认manifest.json保留第一版；第二版训练使用datasets/manifest_v02b.json。

## 第三轮：简单青绿山石（已完成）

成果：`E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\runs\mobilenet_unet_narrow_v03`。
新数据：`E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\narrow_corpus_v03`。

新发现203个目录页面、2208张公开预览地址。成功下载2207个文件，完全去重后2206张；风格/清晰度/内容与近重复筛选后保留761张照片。照片含同一画作的整图、分段与局部，不是761幅独立画作。
实际制备616个山石裁块，来自240张照片、84个页面：475训练、62验证、79测试。按作者、已知册页、近重复关系保守分组35/5/3；《千里江山图》全部归训练。另有1024张简单几何合成训练配对与128张合成验证配对，单独统计。真实标签仍为自动弱标签，无专家完整线稿。

本轮从第一版best.pt初始化，再使用新山石数据进行3000次BF16迭代，成功更新3000次、跳过0次；耗时约697秒。本轮没有混入旧的细密水墨图片。父权重继承此前已有训练知识。网络结构不变。

```powershell
.\scripts\gongbi_lab.ps1 infer --run mobilenet_unet_narrow_v03
.\scripts\gongbi_lab.ps1 narrow_evaluate --run mobilenet_unet_narrow_v03
```

新数据流程为narrow_collect、narrow_select、narrow_prepare；训练入口仍是train，必须指定datasets/manifest_narrow_v03.json、新运行名和对应config.json里的参数。保留旧数据和旧运行，勿用默认manifest覆盖本轮配置。实际来源、排除理由、自动标签、所有测试预览和权重均在E盘。详细结论见本轮MODEL_CARD.md，不能把弱标签一致度称作真实提线准确率。

## 第四轮：单石、石块组合与皴法墨线

数据目录：`E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\rock_cunfa_v04`。
运行目录：`E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\runs\mobilenet_unet_rock_cunfa_v04_clean`。

本轮改为**墨迹恢复**：通道0是包含轮廓及皴笔的墨迹覆盖，通道1是连续墨色密度，不能沿用前几版“结构/可选细节”的解释。不将墨线的两侧边缘当成两根笔画。仅适用于已裁出的单石及皴法示范，不声称已学会从整幅山水中理解全部山石语义。

241张去重网页下载中筛选163张示范照片，裁成179个示例；用户17张参考页裁出52个可用示例，另6个因纸纹弱标签仍有明显残留而未纳入训练。共231个示例，214训练、4验证、13最终测试；同一教程、连续步骤和近重复关系保守分组，来源组7/2/3。另有1284训练、8验证的配对退化版本，不是新增画作。仍有相似步骤和局部，不能将裁图数称作独立画作数。

标签为纸底归一化、纸纹估计、连通墨迹筛选获得的自动弱标签，没有专家逐笔真值，也没有披麻/斧劈等语义分类标签。网络从ImageNet编码器加新解码器开始训练，不续用青绿色块权重。训练采用BF16与墨线专用损失，灰度密度输出保留笔画宽度、浓淡和部分飞白。

```powershell
.\scripts\gongbi_lab.ps1 cunfa_collect --additional-only
.\scripts\gongbi_lab.ps1 cunfa_prepare --variants 6
.\scripts\gongbi_lab.ps1 train --name NEW_ROCK_RUN --manifest 'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\datasets\manifest_rock_cunfa_v04.json' --augmentation ink --precision bf16 --steps 3000 --batch 6 --tile 320 --real-probability .55 --warmup 120 --validation-real-patches 24
.\scripts\gongbi_lab.ps1 cunfa_evaluate --run NEW_ROCK_RUN
```

首次完整采集请去掉`--additional-only`；该选项只追加已保存的公开图片地址。原始参考图仍读取会话附件路径，附件失效时应改为E盘references中保存的副本。正式数据完整性检查入口`scripts/cunfa_verify.py`。复现请使用新的运行名，保留已有实验。

实际训练结果以运行目录的`training_summary.json`为准。`evaluation/all_heldout_tests.jpg`展示全部最终测试图；`ink_*_probability.png`是模型原始概率，`ink.png`只应用所有图共用的显示规则。中心线SVG和路径JSON仅是像素骨架片段，宽度、密度不等于已标定的毛笔压力、速度或笔顺；不连接机械臂。
