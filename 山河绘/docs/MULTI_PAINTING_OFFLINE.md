# 多画种离线规划入口

`multi_painting.py` 和 `scripts/plan_multi_painting.py` 将通用图像分析与画种笔画规划分开。此入口只输出图像像素坐标中的设计 JSON 和路径预览 PNG；`ready_for_execution` 始终是 `false`。没有机械臂连接、运动指令或纸面配准结果。

## 接口

1. `analyze(reference, work_size)` 使用 `vision.prepare_image` 整理参考图，以 `vision.neutral_ink_mask` 提取中性墨迹，并按 HSV 粗分红、橙、黄、绿、蓝、紫色掩膜。结果 `Analysis` 还含工作图、留白边距、估算 `mm_per_px` 与图像四角。色相掩膜是候选证据，并非语义识别、颜料配方或精确分色。
2. `PLANNERS[subject](analysis)` 是画种扩展点。规划器输入同一 `Analysis`，输出按绘制阶段命名的模块；每个模块含 `strokes`、`stroke_count`、`algorithm` 和 `ready_for_execution=false`。笔画的 `points` 为工作图像素坐标。
3. `build_plan` 统一记录参考图 SHA256、图像尺寸、画种、阶段、分色统计及未完成条件。`save_outputs` 拒绝覆盖现有文件，并只绘制规划路径作为预览。

| `--subject` | 专用规划 | 当前范围 |
| --- | --- | --- |
| `generic` | `sparse_color.plan_paths` + `outline_pass.build_plan` | 通用淡彩与中性墨线；稀疏色笔目前针对现有红、黄、蓝调色板 |
| `plum` | `outline_pass.build_plan` + `plum_blossom.petal_plan` | 中性枝干、红瓣五点规划；新入口共用图像预处理，不带旧梅花实验入口的额外 25 mm 缩图边距 |
| `mountain` | `mountain.build_strokes` | 山水墨线启发式；默认 `standard` 细节 |
| `lotus` | `lotus.component_axis_trial` | **试验接口**：粉色瓣与绿色叶/茎连通域各取主轴线，只在合成图验证；没有用户指定荷花参考图，不能视为可用的荷花作画方案 |

## 运行

在项目根目录：

```powershell
.\.venv\Scripts\python.exe scripts\plan_multi_painting.py --subject plum --image 参考图\plum_blossom_reference.png --out 输出\plum_multi_plan.json --preview 输出\plum_multi_paths.png
.\.venv\Scripts\python.exe scripts\plan_multi_painting.py --subject mountain --image artifacts\reference\desired_mountain_reference.png --out 输出\mountain_multi_plan.json --preview 输出\mountain_multi_paths.png
.\.venv\Scripts\python.exe -m unittest tests.test_multi_painting -v
```

入口无 `--execute`。运行 `lotus` 必须显式传 `--image`；项目目前没有用户指定的荷花参考图。可以用自己制作的合成图检查 JSON 接口，但不应拿合成图结果推断真实荷花的瓣、叶、茎笔法。

## 后续数据

- 荷花：用户选定的参考图，标注哪些是花瓣、莲蓬、叶片、叶脉、茎，以及期望的笔顺和设色。此后应替换主轴试验算法，并加真实参考图回归。
- 全画种：纸张四角配准、实际笔触宽度与提按曲线、蘸色量/含水量、干燥时间，以及实画照片和规划预览的对照。
- 通用分色：纸色与光照变化、墨与深色颜料重叠、不同颜料的颜色样本。当前固定 HSV 色相带只能作为粗分候选掩膜。

这些信息齐全后仍须单独设计轨迹转换、边界检查与现场验证；本模块不会输出可执行机器人任务。
