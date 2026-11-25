# 研发参考

| 资料 | 项目用途、条件与限制 |
| --- | --- |
| [Steger，曲线结构检测](https://www.computer.org/csdl/journal/tp/1998/02/i0113/13rRUyfbwrN) | 用曲线法向、局部剖面和尺度说明如何定位线状结构。本轮实现采用 Hessian 法向和亮脊搜索，并另做 Lab 双侧扫描。CPU 可运行，山石线纹也会响应，需要下游归属。 |
| [Lampo Leong，Asian Art Museum 用笔示例](https://education.asianart.org/wp-content/uploads/sites/6/2019/09/Chinese-Brushstrokes-Vocabulary-AAM_0.pdf) | 具名长短披麻及大小斧劈支持类别、笔形和对照范围。单幅图不能恢复真实分笔或运笔时序。同一示例的裁剪保持在同一个组。 |
| [Shit 等，clDice](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html) | 有连续结构像素掩膜时可评估管状覆盖。训练需可核对的结构掩膜和骨架，GPU 成本随整图尺度增加。该指标不能证明交汇处同实例或相互遮挡。 |
| [Xie 等，SegFormer](https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html) | 可作多尺度语义编码器对照，需有标注的国画图、缓存权重和相同的作品划分。当前沿用四类分区权重，等待细边界标签后比较。 |
| [Li 等，DiffVG](https://cseweb.ucsd.edu/~tzli/diffvg/) | 适合有目标图和已有曲线的几何、笔宽、颜色优化。迭代有额外算力开销，仍要单独验证分笔与水墨效果。本轮使用已有 CPU 墨笔渲染。 |
| [Bendale 与 Boult，开放集识别](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html) | 训练类之外应有明示拒识测试。本轮仅采用可在 CPU 留一作者折校准的原型距离，五个探针同一来源。 |
| [Hertzmann，曲线与方向引导的笔触渲染](https://mrl.cs.nyu.edu/publications/painterly98/) | 说明曲线应受方向场约束。本轮以手工岩面脊线约束源图曲线迁移；独立样本仍需检查国画笔组结构。 |
| [大都会馆藏《秋林野兴图》](https://www.metmuseum.org/art/collection/search/45635) | 馆藏说明描述披麻用笔，当前只用于定位潜在补充来源。本轮树木局部来自原图可见结构，笔组没有按整画描述自动赋标签。 |

国画知识与规划的其他来源见 [既有文献核查](research/guohua_system_literature_review_20261110.md)。
