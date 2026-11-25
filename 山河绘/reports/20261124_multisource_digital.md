# 2026-11-24 多源结构发现与数字专项首轮

## 推进目标

推进 G2 的候选发现、G1 的披麻／斧劈辨识、G3／G4 的树石专项和 G5 的可追溯数字笔触。以后沿 [研发总纲](../MASTER_ROADMAP.md) 继续两条算法线。

## 继承与实现

| 已有资产 | 本轮用途 |
| --- | --- |
| `SceneObjects`、`mapping_fixed.pt` 和 256／512 输入的整图分数 | 冻结四类分区模型，用同一树石分数对比不同来源的候选 |
| [上轮语义与墨迹拓扑](20261124_scene_topology.md) | 复查旧失败及对照指标，保留该轮代码、参考和结果 |
| `recognition_trunk_rock_holdout_20261124` 原图、标点、原图尺寸和粗标 | 对照候选是否找到树身，检查岩区候选负担，保留根部争议 |
| Gabor 对照、教材图注局部和复核树干截面 | 对同一 40 例做按作者留出，拟合一个树干中心和宽度样本，从长披麻具名示范取可见墨带 |
| `master_studio/brush.py` 的水墨、图层和连续变宽渲染 | 增加可选逐点宽度，以数字笔触接口复用原有渲染，输出 PNG、逐笔 GIF 和 JSON |

亮脊采用多个尺度的 Hessian 法向和两侧亮度检查。另一支在八个方向、六种半宽上配对双侧轮廓，用 Lab 内部颜色一致性、外侧色差和切向邻域证据产生候选。导出原图位置、配对端点、宽度、响应和冻结语义分数。暗墨候选、亮脊或配对推断轴线、数字画笔轨迹各有独立字段。当前对照未跨无证据的间隙补线。方法参考与条件登记在 [研究资料](../RESEARCH_REFERENCES.md)。

## A 线对照

树身覆盖按同一参考点的 3 像素半径评价。岩区计数来自 20 张归档图的 40 个粗标窗口，共 125,971 个粗标岩面像素。同一分割证据用于各列。归档分区粗标会覆盖部分树根，表中岩区数据是误收核查负担。

| 检查 | 暗墨基线 | 亮脊 | 双侧与色带 | 融合 |
| --- | ---: | ---: | ---: | ---: |
| 12 个树身参考中的候选覆盖 | 9 | 10 | 5 | 12 |
| 12 个参考中同时取得树木归属 | 9 | 9 | 4 | 11 |
| 贴岩斜干候选覆盖 | 1／3 | 3／3 | 3／3 | 3／3 |
| 贴岩斜干取得树木归属 | 1／3 | 2／3 | 2／3 | 2／3 |
| 粗标岩区中取得树木归属的候选像素 | 2,246 | 1,431 | 332 | 3,860 |
| 粗标岩区内待核候选像素 | 1,535 | 1,108 | 718 | 3,099 |

配对区域覆盖了斜干的三个树身参考位置，对前景浅干的三个点未形成双侧区域，亮脊补了其中一个基线漏点。把不同来源组合后覆盖提高，原语义仍把一处斜干留在树木结果之外，粗标岩区同时出现更多树木候选。保存开发清单，含 7 个已圈定岩纹区域、40 个粗标窗口和 4 个有明确类别图注的纹理对照。树根接岩处单列待核。已登记独立作品标注字段及整幅隔离要求，当前还没有新封存作品。

[原图与候选对照](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/visual_structure_round_20261124/structure_comparison.png>) · [逐图结果](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/visual_structure_round_20261124/results.json>) · [困难岩纹清单与标注要求](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/visual_structure_round_20261124/rock_review_registry.json>)

## B 线对照与数字绘画

- **皴法辨识**　用可见笔迹长度、宽度、直线度、方向集中度、密度和浓淡训练 SVM。沿用 Gabor 的同 40 个区域和同一作者划分，笔形统计与来源标签一致 15／40，Gabor 为 17／40。两类混淆矩阵为 `[[9,11],[14,6]]`，五个范围外示范均未拒识。候选不采用。
- **树干**　用一个浅干的 6 处实图参考截面拟合中心侧移和宽度变化，转成两种数字树形。对另一作品中的两段树干，归一化宽度平均绝对误差分别为 0.240 和 0.288，等宽对照为 0.125 和 0.071。生成的分枝来自显式规则，本轮树木学习仅覆盖树干。中心侧移也未优于直干。
- **山石及披麻皴**　以教学图注明确的长披麻局部为来源，提取 18 段可见纹理的几何、宽度及浓淡，迁移到两个给定石形，按轮廓到皴笔绘制。图中皴笔仍偏散碎，同一源图的短线可能属于一笔的一部分，尚缺完整笔组和可靠时序。两种石形、树石布局及干湿参数为手工给定。
- **数字画布**　完成六套可回放笔触序列。树单绘 5 笔，山石单绘 19 笔，皴法单绘 18 笔，树石合绘 24 笔。每笔保存对象、角色、技法、几何、逐点宽度、方向、逐点曲率、数字提按、颜色、透明度、干笔、扩散、先后和来源。用连续变宽的墨带打底，加上低透明的干笔细节，减轻初版的横向断节。

[数字绘画对照](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/digital_specialty_round_20261124/digital_comparison.png>) · [数字笔刷调整前](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/digital_specialty_round_20261124/before_brush_refinement.png>) · [树石合绘逐笔回放](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/digital_specialty_round_20261124/composition.gif>) · [树石合绘笔触记录](<E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/digital_specialty_round_20261124/composition.json>)

## 修改与复现

在项目目录按 AGENTS.md 设置 E 盘缓存和临时目录后运行以下命令。可用 `OPENBLAS_NUM_THREADS=4`、`MKL_NUM_THREADS=4` 和 `OMP_NUM_THREADS=4` 控制 CPU 线程。

```powershell
& 'E:/Anaconda/python.exe' -B 'scripts/visual_structure_round.py'
& 'E:/Anaconda/python.exe' -B 'scripts/digital_specialty_round.py'
```

- [视觉检测](../gongbi_training/visual_structure.py) 与 [A 线评估](../scripts/visual_structure_round.py) 读取原图、导出结构并做消融。
- [数字笔触接口](../master_studio/digital_strokes.py)、[墨笔](../master_studio/brush.py) 和 [B 线入口](../scripts/digital_specialty_round.py) 负责拟合、辨识对照与数字绘画。

## 独立验证与下一步

独立作品验收尚未通过。A 线为归档开发检查，B 线皴法为历史标签的作者分组对照，树干外推为少量已知作品。空白图无结构提取及亮色斜线检测通过合成检查。六套笔触从各自 JSON 重放均与输出 PNG 逐像素一致，GIF 均包含空白起始页及每一笔对应的画面。认可的 500 笔和 489 笔计划与上轮协议均按哈希核验。数字样张作为独立实验输出。

A 线优先验证双侧轮廓沿同一树身的持续证据，按岩纹类型审查误收。B 线补跨作品树干和分枝参考，以及可分辨的完整披麻笔组，分别检验学习参数和手工规则的收益。
