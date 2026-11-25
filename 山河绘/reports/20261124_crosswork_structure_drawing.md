# 跨作品结构、皴法笔组与数字绘画

2026-11-24。继承上一轮 [多源视觉与数字专项](20261124_multisource_digital.md)、冻结的 `SceneObjects` 四类分区权重、候选原图与标注、按作者分折的 40 条皴法参考以及数字画布回放。实验在独立输出目录运行。

## A 多源候选关联

沿用暗墨、亮脊、双侧和 Lab 色带检测，先按骨架端点与交汇拆分路径。候选以同一原图坐标下的距离、方向重叠与宽度匹配，记录各来源和未匹配尾部。连续双侧证据同时检查方向、边界切线、宽度变化及 Lab 连续性。岩纹反证来自整图与细尺度分区的冲突。交汇处只尝试在方向和语义证据相容的同一来源路径之间传播局部证据。

| 方案 | 12 点候选覆盖 | 正确树木归属 | 粗标岩区判成树 | 粗标岩区待核 | 40 窗口的候选中心像素 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 原暗墨 | 9 | 9 | 2,246 | 1,535 | 11,308 |
| A1 直接融合 | 12 | 11 | 3,860 | 3,099 | 23,237 |
| A2 匹配去重 | 12 | 11 | 3,820 | 3,044 | 22,885 |
| A3 连续双侧和整段证据 | 12 | 11 | 2,608 | 4,241 | 22,885 |
| A4 岩纹反证 | 12 | 11 | 2,516 | 4,333 | 22,885 |
| A5 交汇语义证据 | 12 | 11 | 2,524 | 4,325 | 22,885 |

40 个窗口来自历史 20 张归档图。计数是候选图上实际占用的像素，测量对象中心线负担；粗标可包含真实树根。A4 相对 A1 的树木归属点不降，粗标岩区树木候选下降 34.8%，其中大部分转为待核。粗标岩像素里，待核占已有候选的比例由 13.4% 升到 19.1%。A4 仍比暗墨多收 270 个岩区树木候选。原已知贴岩斜干的第三点在 A4 仍落入岩面归属。A5 新放行 8 个岩区树木像素，树身参考无增益。

交界重点图上，A1 到 A4 的粗标岩区树木候选 294 → 159，待核 296 → 427。根部有无同实例连续证据、真实遮挡和岩脊各自归属尚待人工核对。分支原始粗标树木候选分别为暗墨 2,246、亮脊 1,431、配对 332，有跨分支重叠。A4 对赵佶归档一处窗口相较 A0 仍多收 128 个候选，对唐寅一处多收 72 个。后续按条带岩纹、折转、浅色岩面、阴影和树根邻接分别复核。

[原图与 A0–A5 对照](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_structure_20261124/candidate_comparison.png>) · [逐图逐结构计数](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_structure_20261124/structure.json>) · [原图坐标和候选关联](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_structure_20261124/paths.json>) · [困难岩纹核查](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_structure_20261124/rock_negative_audit.json>)

## B 皴法分组与未知拒识

沿用原 Gabor 的 40 局部、12 作品、9 位作者分折。每类 20 个。审核了原图坐标、尺寸、对比度、文件哈希、来源及标签等级，完整作品在同一折。15 个参考来自同一位教学示例作者。许多其余标签来自作品级描述，仍缺逐笔组复核。SVM 比较单笔统计、加笔组统计、HOG／结构张量／Hessian 的裁片结构代理，以及特征融合。裁片没有完整岩面标签，第三层仍只能使用结构代理。背景对照只读高明度像素的颜色统计。

| 特征 | 对照标签一致 | Macro-F1 | 已知正确且接受 | 已知被拒 | 未知被拒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 Gabor | 17／40 | 0.422 | — | — | — |
| 单笔 | 17／40 | 0.394 | 10／40 | 15／40 | 4／5 |
| 单笔与笔组 | 18／40 | 0.427 | 10／40 | 15／40 | 3／5 |
| HOG、结构张量、Hessian | 16／40 | 0.341 | 12／40 | 9／40 | 3／5 |
| 全部结构特征 | 14／40 | 0.335 | 7／40 | 14／40 | 3／5 |
| 高明度颜色对照 | 20／40 | 0.499 | 见逐例结果 | 见逐例结果 | 见逐例结果 |

笔组验证混淆矩阵为 `[[5,15],[7,13]]`，披麻与斧劈召回分别为 25% 和 65%。各折训练内合计矩阵为 `[[159,1],[7,153]]`，训练项在多折重复，训练内 Macro-F1 约 0.975。固定猜一个类别为 20／40；每折用训练多数类会随留出作者比例改变，本轮为 9／40。200 次随机平均约 50%。40 次标签置乱用于全特征诊断，平均 41.8%，95 百分位约 50.8%。笔组的 18／40 仅比 Gabor 多 1 个，尚无稳定优势。拒识用内层作者折的已知类原型距离定阈值，五个范围外示范来自同一教学资料，结果属于压力检查。笔组接受了 15 个错类并拒了 15 个已知样本，未知拒识尚未达到接入条件。

[混淆矩阵与拒识对照](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_cunfa_20261124/cunfa_comparison.png>) · [数据审查和逐折结果](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_cunfa_20261124/cunfa.json>)

## C 树木跨作品学习

7 组已复核树干截面来自 5 件归档作品。每折整件作品留出，其余作品分别形成归一化宽度和侧移曲线，再用稳健中位数拟合。T3 的平均宽度绝对误差约 0.118，等宽约 0.146，线性渐细约 0.122。在排除上一轮训练来源的同六段上，单样本、等宽、多作品估计的宽度误差分别约 0.247、0.112、0.094。侧移 T3 约 0.049，直干基线约 0.037。

另整理三件倪瓒作品的局部可见树干、一级枝和二级枝。保存作品号、原图定位、父子关系及遮挡未知，按作品留一比较树枝附着、角度、长度和侧向。关联样本组的三个分枝成本分别约 0.342、0.342、0.238，高于手工规则的 0.234、0.338、0.164。生成样张采用多作品宽度曲线和单件来源的关联分枝整组迁移，构图位置和分枝渐细为人工先验。参考均为局部可见结构，完整无叶树、树根和叶群尚待复核。馆藏登记为三件不同作品，作者相同。[江岸风树](https://www.metmuseum.org/art/collection/search/41154) · [秋林野兴](https://www.metmuseum.org/art/collection/search/45635) · [虞山林壑](https://www.metmuseum.org/art/collection/search/45636)

[原图、规则与样本生成对照](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_drawing_20261124/tree_comparison.png>) · [来源、留出清单与误差](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_drawing_20261124/tree_references.json>)

## D 长披麻笔组

从一件带明确长披麻图注的示范取两个局部，另取具名小斧劈作结构对照。使用距离、双端切线、宽度比及立方曲线沿途暗墨证据，只有两端互为首选且证据持续时才接续。主长披麻局部 145 段形成 118 条轨迹，新增 27 处推断接续，最长由 68.8 px 增至 95.3 px，13 对间隙证据不足保留未接。另一长披麻局部最长由 77.0 px 增至 106.4 px。同一示范的两个局部不能验证跨来源的笔组学习，真实分笔和历史顺序保持未知。

在两种石形上使用同一组参考。R0 用不接续的 18 个片段，R1 用证据接续后选出的 12 条轨迹，R2 在同样的 12 条上加入内部脊线和两侧岩面形变。岩面方向与位置来自手工结构，曲线和墨色来自参考图；只完成了迁移与回放对照。两幅结果仍有局部疏散和形态刻板，需要另外的长披麻来源及逐笔组标注。倪瓒《秋林野兴图》的馆藏描述提及披麻笔触，可作为后续来源线索，局部归属需要单独核查。[馆藏说明](https://www.metmuseum.org/art/collection/search/45635)

以岩面高度归一，两个石形的皴线中位长度分别从 R0 的 0.152、0.139 增至 R2 的 0.262、0.232。12 条源图路线的对应值约 0.243。两组 R2 的归一化最近邻间距约 0.124、0.123，均小于源图的 0.154。轴向一致度分别是 0.865、0.851，接近这组源图的 0.874。这些是长度、疏密和方向的描述统计，视觉对照仍显示岩面缺少自然的浓淡与笔组层叠。[绘画指标](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_drawing_20261124/drawing_results.json>)

[教学原图与接续证据](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_drawing_20261124/hemp_sources.png>) · [两种石形的 R0–R2 对照](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_drawing_20261124/hemp_transfer.png>) · [片段、推断曲线和证据](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_drawing_20261124/hemp_evidence.json>)

## E–G 画布、笔刷与复现

复用现有数字笔触的几何与变宽墨带，增加沿曲线的墨色剖面、起收笔变化、横向干笔留白、边缘衰减及叠层。保存渲染器模式与随机种子。树根安置在山石内部肩部，画面按轮廓、脊线、皴组、树干分枝和地面水纹落笔。上一轮和本轮均为 24 笔。另用本轮同一 24 笔分别跑旧、新笔刷以隔离渲染影响。构图规则和根部位置来自手工先验，表观浓淡来自样本与虚拟渲染参数。

[24 笔构图和笔刷对照](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_drawing_20261124/painting_comparison.png>) · [从空白画布逐笔回放](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_drawing_20261124/tree_rock.gif>) · [完整笔触 JSON](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_drawing_20261124/tree_rock.json>)

在项目目录按 `AGENTS.md` 设置 E 盘缓存和临时目录，CPU 运行建议将 `OPENBLAS_NUM_THREADS` 和 `OMP_NUM_THREADS` 设置为 4。

```powershell
& 'E:/Anaconda/python.exe' -B 'scripts/structure_candidate_round.py'
& 'E:/Anaconda/python.exe' -B 'scripts/cunfa_group_open_round.py'
& 'E:/Anaconda/python.exe' -B 'scripts/crosswork_drawing_round.py'
& 'E:/Anaconda/python.exe' -B 'scripts/verify_crosswork_round.py'
```

实现入口为 [候选融合](../gongbi_training/structure_candidates.py)、[分组与拒识](../scripts/cunfa_group_open_round.py)、[树木参考与对照](../scripts/crosswork_tree_samples.py)、[墨带接续](../gongbi_training/cunfa_continuity.py)、[数字绘画](../scripts/crosswork_drawing_round.py) 和 [笔刷适配](../master_studio/digital_strokes.py)。研究依据见 [参考记录](../RESEARCH_REFERENCES.md)。

## 回归与独立状态

原始模型权重及 500 笔、489 笔计划通过哈希复核。6 套旧计划和 14 套新计划从 JSON 确定性重放，与对应 PNG 逐像素一致。合绘 GIF 有空白起始帧和 24 个逐笔帧。交汇同实例、完整树木与皴法笔组的独立作品尚未封存。当前结果只做开发与跨作品原型对照。网页工具已在浏览器验证原图载入及哈希、落点、区域、关系和 JSON 导出。正式标注应先复核完整作品及历史重复，再封存给阅标者。下一轮优先核对 A4 转入待核的真实树木，补来自不同示范的完整皴笔和无叶整树。

[离线结构标注工具](../tools/structure_reference_review.html) · [浏览器核验](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_structure_20261124/annotation_browser_check.json>) · [回归核验](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/crosswork_drawing_20261124/verification.json>)
