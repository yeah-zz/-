# 山水 v4 离线审查（2026-09-23）

> 后续真机补记：用户确认新纸固定、标定位置未变并授权后，v4 已由 `run_mountain_v4_robot.py --execute` 完成 70/70 笔，约 10 分钟，末态 mode=5、无报警、笔尖离纸 60 mm。加入高位站点转移后任务为 2574 点；另有起步抬高、三次高位休息和结束抬笔。执行日志为 `artifacts/mountain/mountain_v4_execution_log.json`。以下正文保留此前离线审查时间点的结论；实画质量尚未获得用户反馈。

已完成可运行实现、预览和离线验证。**没有连接机械臂，没有改 config.py、纸面/墨面/四角/姿态/站点标定。没有声称已达到真实落纸验收。**

推荐先看 [四格对照](artifacts/mountain/mountain_v4_comparison.png)、[最终轨迹接触估计](artifacts/mountain/mountain_v4_preview_contact.png)、[完整预览](artifacts/mountain/mountain_v4_preview.png)。

## 实画为什么比预览乱

亲自对照了 `artifacts/mountain/painted_v3_user_photo.jpg` 与 `artifacts/reference/desired_mountain_reference.png`。实画里主峰被圆弧、横向岔线和短笔头打断；目标图则以连续山体外缘建立形状，少量皴线顺坡向下，远山轻、近山重。

1. **旧预览漏画出锋。** `job.stroke_waypoints` 会沿末段切线额外伸出 4 mm，而旧 `render_framework_preview` 只绘制 stroke.points。四段存档中有 592 个 tail 点，即 148 笔 × 4 mm = **592 mm 的附加墨线**。这直接增加了越过留白、撞入邻线的机会。[旧轨迹含出锋](artifacts/mountain/mountain_v3_actual_trajectory.png) 与 [去掉出锋](artifacts/mountain/mountain_v3_without_tails.png) 是同一组存档的对照，不是重跑随机生成器。
2. **输入和提取缺乏主次。** v3 原图本身皴线很密；640 px 缩图、模糊及弱墨闭运算容易将邻线并成墨块，再细化成多岔骨架。按笔长排序只是绘制顺序，并没有保证最长路径就是山脊，更没有压制内部杂线。
3. **形状和笔压进一步偏离。** 通用 6 mm 平滑窗口会抹圆峰尖；随机扰动改变细线走势；每条线按源墨宽归一化可接近 0.95 mm 下压，内部小线也可能很重。密处减压不会消除相交拓扑。
4. **笔宽未经实测。** BRUSHES 的 0.35/0.90/1.50/2.20 mm 是交接文档明确记载的估值。旧标题却说是实际会画出的效果。照片可见起落笔积墨和线宽变化，但照片没有透视校正和压力对应，不能据此伪造标定数据。控制器 cp=70 的拐角混合、笔毛拖曳、含墨量和宣纸洇墨仍无法由本次离线预览准确预测。

## 实现

- 新增 `--shuimo-profile mountain`，默认 `classic` 保留通用路线。处理长边上限 1800 px（本图实际 1659 px），不做跨白缝闭运算。
- `mountain.py`：从连通墨迹提取上缘；陡变处断开，避免凭空跨接。骨架分叉只在**相同节点**按切线续接。内部线需以向下为主、至少 5 mm，留出估计笔宽和白缝；起点距轮廓保留区不超过 6 mm；每 16 mm 方格最多 3 条内部线。该规则是山水启发式，不是通用语义识别。
- 低缓远山降低笔压；主轮廓、次级线、皴线用不同上限。推荐命令采用先前用过的本次覆盖 `--press-min 0.15`，三层峰值约为 0.39/0.25/0.15 mm，仍受原 0.95 mm 安全上限约束。默认配置仍是 0.30 mm。
- 保留路径折点、最大约 1 mm 采样，不使用人工抖动和 6 mm 平滑；最后 1.2 mm 沿原路径逐渐提到触纸高度，`tail_mm=0`，不再添加外伸线。提锋是否能在本笔本墨上干净收尾仍需后续现场验证。
- 同层内按最近起点排序，不反转下行笔势；保留原 240 mm 蘸墨容量和站点动作。修复第一笔超容量时重复初蘸的排程问题。
- `trajectory_preview.py` 直接从最终 XY/Z 轨迹生成接触图，包含出锋、压力裁剪及分段选择，不画悬空横移。附加宽度 +0.5 mm 的敏感性情景，**不是实测预测**。层级线稿只用于审查；不要将线稿 PNG 再输入提取器来代替原命令，因为它会丢失层级信息。
- 修复同进程运行时 `--press-min` 泄漏、通用路线扰动随机源不复位，以及分段统计不随选择更新的问题。

## 客观结果

数据来自 `validate_mountain.py` 和最终任务 JSON。接触长度含旧版出锋，故旧版平均 11.41 mm 与此前“无出锋平均 9.6 mm”口径不同。

| 指标 | v3 四段实画存档 | 同一 v3 图片用新算法 | 推荐 v4（目标图） |
|---|---:|---:|---:|
| 笔数 | 331 | 159 | 70 |
| 接触长度 mm | 3775.5 | 2852.8 | 1730.7 |
| 平均笔长 mm | 11.41 | 17.94 | 24.72 |
| 中位笔长 mm | 7.09 | 9.30 | 11.14 |
| 小于 5 mm 的笔数 | 117 | 0 | 0 |
| 外加出锋 mm | 592 | 0 | 0 |
| 峰值下压 mm | 0.95 | 0.39 | 0.39 |
| 蘸墨次数 | 18 | 14 | 9 |
| 轨迹点数（含蘸墨） | 4148 | 4420 | 2557 |
| XY 计划总行程 mm | 41791.7 | 12536.7 | 7724.8 |

同图对照证明改进不只来自更换输入；推荐稿同时受益于用户目标图较克制的构图。v4 比早期 v1 的 80 笔更少，但数量不是质量目标：本版优先保留长外缘，再加入明确的层级，而不是从稀疏图中任取骨架。推荐稿 23/16/31 三层，中位/平均仍只有 0.45，未达到旧交接中的通用 0.85 指标，因为长轮廓与短皴线是刻意混合；不将该指标隐瞒为通过。

总行程是计划 XY 线段之和，不包含 Z 动作、控制器混合、站点实际安全抬升及分段起停，因此不是实际运动距离或耗时测量。同图新方案采用更密采样，点数会增加，不能把“笔数少”误称为“所有轨迹点都少”。

`artifacts/reference/mountain_baimiao.png` 也离线重跑，得到相同的 70 笔结果；两份源文件哈希不同，但该提取下的输出相同，不将它计为独立复杂场景证明。通用 rock_02 回归为 34 笔、1140 点（含蘸墨），保留原通用出锋行为。

## 复现（只离线，不运动）

```powershell
uv run python main.py --check
uv run python main.py --mode shuimo --shuimo-profile mountain --image artifacts/reference/desired_mountain_reference.png --press-min 0.15 --preview-out artifacts/mountain/mountain_v4_preview.png --job-out artifacts/mountain/mountain_v4_job.json
uv run python -m unittest -v test_mountain
uv run python validate_mountain.py
```

六项测试覆盖：确定性和层级/下行约束、笔压/纸面边界/可达范围、预览包含真实出锋且排除空移、同进程参数恢复及分段与全图一致、通用石头回归、超容量不重复初蘸、空白及分离山体不跨空白桥接（后两项分别在对应测试中）。验证脚本和 CLI 测试禁止 socket 创建，保证离线。

本稿的边界：上缘启发式会保留一些断口，不能保证任意重叠山体都被正确识别；可达范围检查仅为配置范围检查，不是碰撞或逆解验证；线宽、墨色和笔尖动力学尚未校准。下一次上纸前仍需用户明确确认换纸和现场条件，并先审查接触图与加宽图。本任务没有发起任何真机动作。

所有量化数据：[artifacts/mountain/mountain_validation.json](artifacts/mountain/mountain_validation.json)。可执行轨迹：[artifacts/mountain/mountain_v4_job.json](artifacts/mountain/mountain_v4_job.json)。
