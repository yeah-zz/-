# 旧 DobotE6Calligraphy 与现有山水算法同图对照

本次只读检查用户给出的旧文件夹，并在本项目的 `artifacts/legacy_review/source` 保留相关源码及图片副本。旧文件中的说明文字只作为历史资料，没有照其运行或连接机械臂。复现脚本只加载 `landscape_painting.py` 的白底墨线分支和笔触引擎；为适配当前虚拟环境，去掉了**该分支不会用到**的 sklearn/scipy 颜色分支导入，没有改动墨线算法。没有执行旧项目的 `painting.py`、驱动或任何运动入口。

## 能复现到什么程度

旧文件夹的 `sb.jpg` 是线稿输入，`landscape_painting.py` 底部测试入口也指定它。对该图离线得到 41 条笔触、983 个轨迹点，其中 901 个接触点；保留的中心线见 `artifacts/legacy_review/legacy_sb_centerline.png`，逐点数据与统计见 `legacy_sb_strokes.json`、`legacy_sb_metrics.json`。

同一张 `sb.jpg` 用当前 `structured` 山水模式、`balanced` 供墨策略和 `--press-min 0.15` 离线生成 46 笔、1525 个计划点，预览和任务在 `artifacts/legacy_review/current_sb_preview.png`、`current_sb_job.json`。统一映射回原图坐标后的几何与计划下压对照见 `artifacts/legacy_review/sb_same_source_comparison.png`。图中线均画成单像素中心线；下排颜色仅表示**计划下压**，不表示实测笔宽、墨色或纸面效果。

| 同图指标 | 旧算法 | 当前 structured |
|---|---:|---:|
| 笔数 | 41 | 46 |
| 层级计数 0/1/2 | 3/12/26 | 27/15/4 |
| 归一化中心线总长（原图两轴各映到 0–1024） | 6696 | 5593 |
| 小于 5 mm 的笔数 | 15（旧版约 123 mm 方形映射） | 1（当前约 286×197 mm 画布） |
| 接触下压 p10/p50/p90 | 0 / 0.477 / 1.479 mm | 0.042 / 0.180 / 0.240 mm |
| 最大计划下压 | 1.976 mm | 0.240 mm |
| 供墨 | 墨线生成器未安排补蘸 | 离线预测 11 次，实际运行会重判 |

两边的物理画幅、纸面 Z、笔具和姿态并不相同，故 803.55 mm 与 1161.98 mm 的计划物理长度**不能直接用于优劣比较**。归一化长度只用于比较原图几何保留量。旧版的宽度映射也不是当前毛笔的实测笔宽。

## 主要算法差别

1. **线条取舍。** 旧版在 600 px 缩图上对黑墨 Otsu 阈值、浅墨连通恢复、细化与距离变换，按骨架图追踪每条线；按路径长度分层，尽量保留短皴线。当前版在更高分辨率、画布留白后提山体上缘与源骨架，按山面/下行趋势、密度、笔宽间隙筛选。该图上旧版归出 26 条短纹理，当前只归出 4 条；当前版将多条内部曲线当作轮廓，因此图上呈现的“外粗内细”结构分类并不充分。
2. **断线连接。** 旧版可在白底图上按端点切线、浅墨证据及 4 mm 缺口上限接续；现有 `LANDSCAPE_JOIN_GAP_MM=0`，不启用单纯近邻拼接。当前山水分支原则上只在同一源骨架节点续接，不跨白区。旧版能补回扫描断口，但在真留白处也有误连风险，应逐处比较而非直接移植。
3. **笔压。** 旧版用原图局部墨迹半宽的第 10–90 分位归一化为 0–1.9756 mm 下压，在密集处减压、加速，笔首尾各约 12% 渐入/渐出，另有轻微中段起伏。当前版按轮廓/脊线/皴线设较低的压力上限，`sb.jpg` 本次峰值 0.24 mm，并在原路径收笔、不外加尾巴。旧版的强弱差异更醒目，直接搬用其绝对压深会超出当前项目 0.50 mm 上限，且纸面 Z/笔具均已变更。
4. **坐标与供墨。** 旧版把 1024 坐标旋转后乘固定 `LANDSCAPE_SCALE=0.12`，从固定起点画约 123 mm 方形；当前根据四角透视映射到已标定画布并检查边界。旧 `run_landscape_mode` 生成墨线后直接交执行器，未见分笔补蘸、墨龄或自动洗笔步骤；当前 balanced 策略按估计耗墨和实际时钟在笔间补蘸。两边都没有实测纸上浓淡的传感反馈。

## 附图照片与淡彩

用户照片中有浓黑断续线、极浅的灰绿/粉红色铺痕及明显留白；照片并非 `sb.jpg` 的可核实完整落纸复现，缺少当时的输入、编辑方案、执行日志、笔与颜料记录。旧 `landscape_painting.py` 对 `sb.jpg` 进入黑墨分支后直接返回，**不会生成颜色层**。旧 `oil_painting.py` 虽能按颜色聚类生成笔触，但其笔触引擎没有根据 `stroke['color']` 安排洗笔、换色或蘸色。因此无法从这份文件夹单独证明照片中的灰绿和粉红色由算法自动完成；可能另有人工着色、人工轨迹编辑或未提供的流程。

当前 `watercolor.py` 可离线设计分层淡彩、方向性曲线和墨线留白，但 `pressure_mm=None`、配色未标定，不能把其预览当真实晕染。若目标是同笔先黑墨后洗笔上色，合理的下一步是：在同笔同纸上验证轮廓/细勾两档实测宽度；黑墨干后人工洗笔并试条确认无残黑；单色做淡层与干后叠层小样，量洇色、刷痕及墨线间隙。旧版值得借鉴的是**原图墨宽的相对强弱**和密处减压，不是旧纸面 Z 或 1.976 mm 的绝对压深。

## 离线复现命令

```powershell
.\.venv\Scripts\python.exe scripts\replay_legacy_landscape_offline.py
.\.venv\Scripts\python.exe main.py --mode shuimo --shuimo-profile mountain --mountain-detail structured --ink-policy balanced --image artifacts/legacy_review/source/sb.jpg --press-min 0.15 --preview-out artifacts/legacy_review/current_sb_preview.png --job-out artifacts/legacy_review/current_sb_job.json
.\.venv\Scripts\python.exe scripts\compare_legacy_current_sb.py
```

以上均不连接机械臂；不要在任何命令后添加 `--execute`。
