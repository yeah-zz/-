# 墨线之后的彩色第二遍（离线设计）

这条流程用于先画好的 balanced/structured 山体墨线任务。它从**相同构图、相同原始像素尺寸**的彩色设计图提取红、黄、蓝三类区域，避开源墨线和任务中的墨线笔触，按估计笔宽生成彩色笔触及叠色预览。它不连接机械臂，也不生成可执行的机器人 `waypoints`。

## 输入与命令

准备两张完全对齐的图片：

1. `artifacts/reference/desired_mountain_reference.png`：当初生成墨线任务的黑白线稿。已有 `artifacts/mountain/mountain_v6_job.json` 指向它。
2. `my_colored_design.png`：在**同一张线稿的同一画布上**标出想上色的红、黄、蓝区域。不要裁剪、旋转、移动画面或改变图片像素尺寸。照片和已经生成的独立概念图不能直接套到现有墨线任务上。

在 VS Code 终端运行（只做离线设计）：

```powershell
uv run python main.py --mode color-pass --ink-job artifacts/mountain/mountain_v6_job.json --ink-image artifacts/reference/desired_mountain_reference.png --image my_colored_design.png --color-brush-width 3 --color-ink-gap 1 --job-out my_color_plan.json --preview-out my_color_preview.png
```

若输出文件已有同名文件，换新文件名；程序不会覆盖。输出的 JSON 中 `ready_for_execution` 固定为 `false`，只有像素坐标笔触与人工换色顺序；加 `--execute` 会明确拒绝。`--ink-image` 用于历史任务中的源图绝对路径无效时；省略时程序先尝试任务记录的路径，再尝试任务旁同名文件。

## 现在的计算规则

- 在 Lab 色彩空间按色相归到红、黄、蓝，低彩度的黑墨与白纸不计入彩色区域。
- 用源线稿中的墨像素及墨线任务里的规划路径生成禁画区，保留 `--color-ink-gap` 指定的空隙。笔触中心还会按 `--color-brush-width` 估计半径向色区内部收缩。
- 用色区横向的间隔线作为初版铺色笔触。特别小或容不下笔宽的色块会舍弃。彩色计划先排黄、再红、最后蓝；每次换色都要求人工洗笔/擦笔、确认笔已干净、蘸色及试色。
- 根据彩色设计图的 Lab 亮度给每笔记 `light`、`medium`、`strong` 目标标签（分界为 L* 86 和 72）。**这是设计意图，不是已标定的颜料浓度、纸上灰度或下压力。** 预览也只是示意。

这套横向铺色适合少量、宽敞的色区。荷花花瓣、细梅枝等曲线题材还需要单独的笔势规划。干画后纸张变形、颜料扩散和真实笔宽都可能使彩色越过预留空隙。

## 从设计计划走到实画还需完成

目前 `config.py` 只有一格实测墨，清水盂、擦笔布、试色区坐标仍是占位值。每种新增颜料要在现场测实际容器中心、液面及安全进出高度；换笔时重新核对纸面和液面高度、笔宽。先在同种纸上测颜色、稀释比例和笔宽，再决定 `light/medium/strong` 的实物实现方式。洗笔次数并不能证明笔已干净，换色须看试色条；必要时用不同的笔或换清水。

完成这些测量和小样验证前，勿把这个设计 JSON 交给原有的 `run_mountain_balanced.py` 或把旧 `--mode color` 结果直接叠画在现有墨线上。后者没有本模块的墨线保护和两遍对齐检查。
