# 密集细节与长线分层实验

打开结果：`E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/split_detail_structure_20261028/v4/index.html`。

这版沿用用户认可的原图像素松针方法，对密集短线和长皴线分别处理。它是传统图像处理流程实验，尚未训练新的树木/山石模型，也不推断真实笔顺或皴法类别。

在 PowerShell 运行：

```powershell
& 'F:\山河绘\山河绘\scripts\run_split_detail_structure.ps1' -ImagePath 'E:\你的图片.png' -OpenPreview
```

默认结果写入 E 盘独立目录。当前原型要求图像长边不超过 2600 像素，超出会明确报错，不会自动缩小整图。输入图片在原分辨率下处理。没有新增依赖，CPU 执行 OpenCV/SciPy 图像运算。

输出中 `lines.png` 是本轮筛选中心线；`all_evidence.png` 用灰色显示原图骨架中未被当前中心线覆盖的像素，含真实短线和杂点；`pending.png` 保留这些待定像素。路径 JSON 中的 `pending_paths` 仅包含待定的脊线候选，全部原图未覆盖像素以 `pending.png` 为准。`ink.png` 保存像素墨迹，`raw_skeleton.png` 保存未筛选骨架，`dense_regions.png` 显示几何规则选出的密集区域。

蓝色叠加为长线，紫色为密集细节，两者不是语义金标准。SVG 和路径 JSON 为像素坐标，不含压力、速度或设备坐标；`ready_for_robot=false`。

本轮统一参数验证 3 张生成图和 3 张旧真实图；11 项算法行为检查通过，但这些不是图画识别准确率测试。松针保留，山石部分碎线减少，仍有断笔、漏提和网状骨架；未满足替换稳定模型入口的条件。
