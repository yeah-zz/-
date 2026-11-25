# V6：沿原图走势组织山面，先验证小样

> **最新现场：v6 整幅已完成真机绘制（2026-09-23）。** 用户明确要求整幅运行，并确认已换新纸、标定未变后执行。108/108 笔，766.8 秒（约 12 分 47 秒），实际蘸墨 13 次（预测 17 次），墨内停留合计 8.4 秒，实际轨迹 2826 点。最大周期结束墨龄 55.156 秒。最终 RobotMode=5、无报警，User2/Tool0 位姿 `(35.70,80.36,-100.08,-3.0,-0.8,30.8)`，笔离纸 60 mm，保持使能，连接已关闭。日志 `artifacts/mountain/mountain_v6_job_execution_log.json`、实际轨迹 `artifacts/mountain/mountain_v6_job_actual.json`。当前纸上为 v6，不能默认空白；尚未收到本次实画照片，不能据执行成功断言美术效果。下方“仅离线”等记录是此前阶段状态。

本轮仅修改代码并生成离线任务，未连接或驱动机械臂。纸上最近一次实画仍是 v5。用户最新要求：内部走势自然，不用横线代替山体结构。

## 实现

- 新增 `--mountain-detail structured`。轮廓做 0.32 mm 容差简化，保留端点及每段最高峰；内部重新沿原图墨线骨架提取，不生成横向排线、不随机补线、不跨白区连接断线。
- 内部线按邻近轮廓归属山面，保留自然的转折、长短和疏密。每条内部线要求整体向下，垂直落差至少占长度的 48%；分割后的线也检查此条件。源骨架仅在同一节点按切线接续。
- 远、中、近轮廓下压上限分别 0.12 / 0.18 / 0.24 mm（本次 `--press-min .15`）。内部线更轻；起笔在 1.8 mm 内渐入，收笔在原路径末段渐出，不额外延伸尾巴。远近判断基于画面竖向位置和轮廓起伏，是构图启发式，非三维识别。
- v5 内部线 67 条，v6 为 79 条；v6 共 108 笔、1924.8 mm、2844 个计划轨迹点，预测蘸墨 17 次。实际次数仍由真实用时和消耗模型决定。
- 沿用 v5 短蘸/沥墨及运行时供墨判断；未凭照片声称重新标定了供墨容量。新增笔压、蘸墨时长对照任务，供后续实测使用。
- 旧 standard/enriched、classic 及标定配置保持原行为。控制器平滑参数未改；它和毛笔扩散仍可能影响山峰尖锐度。

## 审查文件与小样

- `artifacts/mountain/mountain_v6_comparison.png`：左 v5，右 v6，均为估计接触轨迹，不是实画照片。
- `artifacts/mountain/mountain_v6_preview.png`：源图、分层轨迹、估计接触、额外加宽 0.5 mm 的敏感性图。
- `artifacts/mountain/mountain_v6_sample_job.json` / 同名 PNG：三组山体，32 笔、523.17 mm，预测蘸墨 5 次。保留整幅中的原坐标及实际尺寸，不把小样放大到整张纸。
- `artifacts/mountain/mountain_v6_job.json`：整幅候选，尚未上纸。
- `artifacts/mountain/mountain_v6_validation.json`：任务清单和离线检查记录。

## 对照任务（分别使用新纸，不要叠画）

所有诊断线在已标定的纸面内部，不使用未经确认的试色/擦笔站。不同任务在相同纸面坐标绘制，必须分别试验。诊断用直线仅用于量笔宽和供墨，不属于山体内部线。

1. `mountain_v6_pressure_08/12/18/24.json`：每个文件一条 40 mm 竖线，固定下压量分别 0.08 / 0.12 / 0.18 / 0.24 mm。固定同一初蘸时长 1.2 s、沥墨 1.6 s、速度和几何；关闭刚蘸后减压以免混淆对照。记录稳定段笔宽、起笔墨团、断线与收尾情况。诊断线本身不渐变下压，以便观测定压表现。
2. `mountain_v6_ink_06/09/12.json`：每个文件八条 12 mm 竖线，按从左至右顺序；压力固定 0.12 mm。仅初蘸时长不同（0.6 / 0.9 / 1.2 s），沥墨均 1.6 s。预测每任务只蘸一次，画程 96 mm；若实际变慢触发供墨保护补蘸，保留日志并按墨周期解读，不能当作一次蘸墨的完整续航。

任务间笔头原有含墨量、墨液和纸张会影响结果。需尽量保持一致的笔头起始状态；系统没有自动洗笔/称量或视觉反馈，不能仅凭这些任务给出严格的墨量标定。先观察笔压测试，选出连续且足够细的压力，再做供墨对照和三山小样。

## 复现及后续运行入口

生成整幅（只离线）：

```powershell
uv run python main.py --mode shuimo --shuimo-profile mountain --mountain-detail structured --ink-policy balanced --image artifacts/reference/desired_mountain_reference.png --press-min 0.15 --preview-out artifacts/mountain/mountain_v6_preview.png --job-out artifacts/mountain/mountain_v6_job.json
uv run python review_mountain_v6.py
```

只校验小样（不连接）：

```powershell
uv run python run_mountain_balanced.py --job artifacts/mountain/mountain_v6_sample_job.json
```

后续用户明确要求运行、现场新纸和标定状态确认后，使用同一命令加 `--execute`。必须显式指定 `--job`；脚本默认仍指向历史 v5。不要用旧 v4 固定 70 笔脚本或通用回放器替代，它们不具备这一任务所需的运行时供墨检查。执行器保留墨盘高位水平进出、到位后计蘸墨停留、实际时钟供墨保护和结束抬笔。

## 验证

`uv run python -m unittest -v test_mountain test_ink_control test_mountain_structure`：19 项通过。

覆盖原模式回归、实时供墨判断/到位停留、原图内部线距离与下行方向、峰顶/端点和简化误差、下压层次与起收笔、小样同尺度同坐标、禁止联网的完整离线流程。所有对照任务均通过执行器的标定快照、范围、笔压和预算检查；`main.py --check` 通过。

尚未验证：真实笔宽、墨色稳定性、轻压时连续接触、控制器平滑后的转角及实际审美效果。不能把预览当作实画效果承诺。
