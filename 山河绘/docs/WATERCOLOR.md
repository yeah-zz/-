# 参考图式山水淡彩

新增 `watercolor.py` 和 `main.py --mode watercolor`。输入彩色参考图，输出新构图的上色设计JSON、色面预览和真实中心路径预览。不会连接机械臂，不生成运动指令。现有 `color-pass` 对齐墨线叠色入口及已完成的69笔蓝色方案保持独立。

## 与旧蓝色排线的区别

1. 从图像中分离蓝/黄/红色区、黑墨和白色云雾；淡色识别阈值下调，同时排除中性灰和纸白。
2. 用图像局部纹理的方向生成顺山势的曲线路径，弱纹理区域采用向下的坡向；红日单独铺填。
3. 按宽→中→细分级覆盖：最大3mm设计对应3.0/1.56/0.72mm三档，窄山脊使用细档；最大1mm设计对应1.0/0.52/0.30mm。所有数值均为**目标笔宽**，不是已验证的压笔参数。
4. 每条路径保存逐点目标宽度和目标颜色覆盖强度。大部分笔身保持连续宽度，短起笔、渐细收笔；参考图原有的浓淡变化仍保留在逐点目标里。
5. 先蓝山，再黄山，最后红日；各色分wash/body/accent三个目标层次。颜色切换安排人工洗笔、擦笔和检查，深层之前安排现场观察干燥；不编造固定等待秒数、水比例或自动颜料站坐标。
6. 按完整估计笔头半径腐蚀色区，并膨胀墨线禁画区；整段曲线均要留在有效区域，不能只检查端点。

## 使用

```powershell
.\.venv\Scripts\python.exe main.py --mode watercolor --image samples\watercolor_landscape_reference.png --color-brush-width 3 --job-out new_watercolor_plan.json --preview-out new_watercolor_preview.png
```

当前细笔可先离线比较 `--color-brush-width 1`。不覆盖已有方案，请使用新的输出文件名。`--watercolor-ink-gap` 默认0.6mm，仅为**新构图设计**留白，不是当前旧纸的实机允许间隙，也没有修改旧蓝色执行器的8mm限制。

输出预览来自生成笔触的逐点宽度/浓淡包络，只有中性墨线来自参考图。不会把原图彩色像素直接贴回预览。半透明边缘为示意，不是宣纸吸水、洇化、叠色或纸面干湿的物理仿真；预览的光学强度不能直接当作颜料浓度。

## 本次产物

- 参考图：`samples/watercolor_landscape_reference.png`
- 3mm设计：`artifacts/watercolor/watercolor_landscape_plan.json`、`artifacts/watercolor/watercolor_landscape_preview.png`、`artifacts/watercolor/watercolor_landscape_preview_paths.png`。607条笔触，蓝310/黄283/红14，路径约2521.8mm。
- 细笔设计：`artifacts/watercolor/watercolor_fine_brush_plan.json`、`artifacts/watercolor/watercolor_fine_brush_preview.png` 及 `_paths.png`。1009条笔触，路径约5230.4mm，最大目标笔宽1mm。预览更细密，但没有验证这些宽度与实际压笔量的对应关系。
- `watercolor_reference_*`、`watercolor_landscape_v2_*` 是迭代草稿，不是最终建议方案。

## 真正实现参考图效果仍需的现场条件

参考图与目前纸上的山水不是同一构图。不能把参考图缩放后直接覆盖当前墨线；应使用新纸，从此参考构图准备匹配的墨线，再做这套上色，或者单独建立和验证旧画的色区注册。

参考效果需要可控的淡色铺染和渐淡笔触。先用所选毛笔实测“压笔量→笔宽”，换笔后重新标定笔尖/纸面高度；再在同种纸上建立蓝/黄/红各档浓淡色样、单次储色长度以及洗笔后的残色检查。宽软笔较适合铺色，当前细笔方案会产生更多细线，无法仅靠改速度得到纸上自然晕染。程序中的 `pressure_mm`、配色水比例均保留为空，避免把未标定的美术参数误当作真机参数。

`--execute`、`--probe`、运动/标定参数组合以及把旧 `--ink-job` 传入新构图入口均被拒绝。既有蓝色执行器也不会接受该新格式。后续要接真机，必须先把这些现场数据接入执行器；本次没有自动蘸蓝、黄或红。

## 测试

`python -m unittest test_watercolor test_color_pass test_blue_wash -v`：25项通过。覆盖色区/墨线全路径约束、弯曲方向、白色留白不穿越、收笔和浓淡变化、无笔触预览不得复制参考图颜色、换色层次、最细笔宽上限、CLI输出和硬件执行阻断，以及旧上色执行回归。
