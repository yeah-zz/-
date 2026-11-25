# 2026-11-25 专业数据接入和树石监督训练

本轮在 F 盘项目增量实现专业数据读取、作品分组、监督训练、公开模型对照及等笔数轮廓回放。数据、权重和输出存放在 E 盘。旧树石规则 A0、A1、A4 待作历史对照。

## 来源与实际获取

| 来源 | 实际检查及使用状态 |
| --- | --- |
| [CLPD 发布仓库](https://huggingface.co/datasets/Rane7/CLP_dataset) | 锁定修订 `54cba6131985352b643d11394663a214dc008c51`。索引找到 2,000 对训练和 207 对验证，卡片正文写 2,206。实际下载并校验 98 对图像与单通道类别掩膜，未见读取失败。许可按仓库说明为 CC BY-NC 4.0，限本轮研究。山体 2 与岩石 13 映射到项目山石类，树木 3 映射到树类，0 与 255 保留忽略。标注提供语义区域，无法直接推断枝根实例或精细岩纹。 |
| [ShanShuiSeg-B5 发布页](https://github.com/shengbei4980/ShanShuiSeg/releases/tag/v1.0.0) | 下载了 328,213,785 字节权重以及 6 个配置、源码和许可文件。用 E 盘独立虚拟环境和现有 PyTorch 推理，没有向绘画环境安装依赖。原包使用 Python 3.8、旧版 MMCV 和 MMSeg。本轮按其 B5 配置实现本地原生 Torch 适配，使用权重安全加载和 SHA-256 锁定，尚未与原包逐像素比较。权重仅作非商业研究，不接入绘画。类别与 CLPD 一致，发布方训练样本 ID 未确认。 |
| [古画实例分割发布仓库](https://github.com/zpc481523/res-painting) | 已下载一张验证图和验证 COCO 标注。标注实读 207 图、1,292 个实例，14 类包括山、岩、树，README 的实例估计数与实读有差别。声明署名、非商业、相同方式共享。取样图 `image_0000.png` 解码后的 RGB 与历史 CLPD 图 `Princeton_127` 哈希完全相同，已证实至少一处跨集重合。尚未用于训练。 |
| [RealBrush 墨迹发布仓库](https://huggingface.co/datasets/madorin0130/pg23-ink-painting-dataset) | 数据卡写明 CC BY 4.0、1,818 个画笔样本、路径、手工掩膜及逐笔画布。公开可浏览目录元数据，实际 `1_sumie.json` 请求返回 401，需要登录并经发布方人工审核。实际下载的笔触文件为 0，笔触拟合和真实墨迹评价均待取得配对文件。 |
| 芥子园画谱、树木馆藏 | Met 共取回 15 张公有领域原图或馆藏细节图。原指定三条目主要展示封面、题页、云水版页，补找了含树和岩面的具名图页。CONTENTdm 入口需要 JS，其单条目 API 请求返回 403，本轮取得该站原页 0 张。画谱图页与真实作品的可见结构进入人工审核队列，局部技法待核。 |

截图、另存图和推断笔段均未作为新监督标签。目录卡已记录文件、下载地址、尺寸、SHA-256 和许可。资源核查时使用 i9-14900HX（24 核 32 线程）、16 GB 物理内存、RTX 5060 Laptop（8,151 MiB 显存，时点空闲 6,557 MiB）及 E 盘约 357.6 GB 可用空间。馆藏依据 [60757](https://www.metmuseum.org/art/collection/search/60757)、[60758](https://www.metmuseum.org/art/collection/search/60758)、[60760](https://www.metmuseum.org/art/collection/search/60760)、[画谱图页 CP28](https://www.metmuseum.org/art/collection/search/63225)、[画谱图页 CP9](https://www.metmuseum.org/art/collection/search/63203)、[冬景枯树](https://www.metmuseum.org/art/collection/search/41192)。

[CLPD 样本签名与划分](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/clpd/manifest.json>) · [下载清单](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/source_audit/downloads.json>) · [补充馆藏清单](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/source_audit/new_museum.json>) · [COCO 实读核对](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/source_audit/coco_sample_audit.json>)

## 分组和实际训练

CLPD 首批选择可从文件名解析作品线索的 24 组，每组至多 5 张。编号而缺少可核对作品主键的 1,003 对留在目录清单。先对下载样本做 SHA-256 精确去重及 63 位感知哈希（含镜像）检查，近重复则合组，再以固定种子分为训练 62 张、开发 17 张、评估 19 张，对应 16、4、4 组。98 对内未出现重复或感知哈希触发的跨题名合并。哈希近似无法证明不同裁剪来自不同作品；作品名代理也仍需馆藏编号核对。

另将 98 张与 100 张可实际读取的历史清单图进行同名作品、原文件和感知哈希对照。训练、开发、评估分别检测到 42、17、10 张与历史重合或同作品。其中 10 张评估图与历史文件哈希相同。其余 9 张来自 2 个题名组，保留为「未检测到历史重合」的子组，来源仍在同一套 CLPD。历史 S0 同样曾使用 CLPD，不能视为独立作品测验。该模型及 500 笔、489 笔哈希已锁定。

S2 从现有冻结 ImageNet MobileNetV3-Small 特征初始化语义结构，重置并监督训练多尺度解码头。使用 CUDA、256 输入、批量 6、固定 12 轮、AdamW 及学习率 0.002，训练做左右翻转并按类别频率加权。最后一轮预先选定；评估图未参与 S2 更新或选轮数。训练损失首轮 0.903，末轮 0.253，开发树木 IoU 首轮 64.6%，末轮 68.9%。S0、S2 都以 256 输入，S1 按原包方式使用整图 512 重采样。三者对相同原图和映射后的 4 类比较。

| 队列与测量 | S0 旧冻结 | S1 公开模型 | S2 新训练 |
| --- | ---: | ---: | ---: |
| 19 张评估图树木 IoU | 66.2% | 99.5% | 57.1% |
| 19 张评估图山石 IoU | 76.0% | 99.8% | 68.8% |
| 19 张评估图树木 Precision／Recall／F1 | 72.4%／88.5%／79.7% | 99.7%／99.8%／99.7% | 70.8%／74.7%／72.7% |
| 19 张评估图山石 Precision／Recall／F1 | 92.2%／81.2%／86.3% | 99.9%／99.9%／99.9% | 84.9%／78.4%／81.5% |
| 19 张评估图树石交界树木／山石 IoU | 50.7%／37.8% | — | 42.0%／39.3% |
| 9 张无检测重合，树木／山石 IoU | 59.9%／72.9% | — | 66.0%／72.6% |
| 9 张无检测重合，交界树木／山石 IoU | 47.1%／33.8% | — | 45.6%／39.4% |
| 12 个历史树身参考点的精确树类归属 | 11／12 | — | 12／12 |

S1 与 CLPD 掩膜近乎逐像素相同。训练作品清单未核实，不能把这一成绩用于泛化宣称。S2 整组表现低于旧基线，两个无检测重合作品中树木 IoU 上升而交界树木 IoU 下降。相同 85% 输出覆盖下，整组 S0／S2 的错误风险是 9.6%／10.9%，9 张子组为 12.8%／9.4%。诊断性匹配约 75% 树木召回时，整组粗标山石被判为树为 4.7%／9.5%，9 张子组为 6.7%／4.1%。结果同时给出保留率和风险。12 个点使用整图语义像素计分，保留既有候选模块的独立计分；缺少可靠树根和岩纹精标，S3–S5 未接入。

[训练曲线](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/scene_pilot/training_curve.png>) · [S0 与 S2 原图对照](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/scene_pilot/scene_comparison.png>) · [S1 原图对照](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/scene_s1/shanshui_comparison.png>) · [S0、S2 完整指标](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/scene_pilot/results.json>) · [S1 完整指标](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/scene_s1/results.json>) · [重合审计](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/scene_pilot/overlap_audit.json>)

## 标注、皴法和绘画

把 40 个旧皴法样本按来源分为已定位来源标签 6 个（A级）、需要复核的可靠描述及教学图注裁片 34 个（B级），另有 5 个旧范围外例。A级仍需核查局部像素范围。保留历史按作者验证，笔组 18／40、Gabor 17／40，未知拒识 3／5 且已知拒绝 15／40。未达到新来源拟合条件，本轮仅做数据分级，冻结旧结果。完整树干、树根、分枝和新长披麻笔组尚未人工确认，暂未开始新增专项拟合。

已准备 4 项原图审核任务，先任选 **CP28 中央的一株树**，依次标可见树身、主干中心、根部和分枝连接。遮挡和交接分别标待核。后续候选包括枯树林一株、CP9 山石笔组与吴镇松树的树石交界。同一画谱的两页暂合并为同一来源组。审核队列里的图未用于三组模型指标，也未送入本轮推理。入独立集前仍需可靠组件标注与较完整的跨来源去重。

使用相同持出原图，各从 S0 和 S2 的语义输出取 2 条山石、3 条树木轮廓，以同一数字笔刷回放。几何来自各模型预测，笔墨与选取预算通过规则固定。两张新输出均从笔触 JSON 重放并逐像素核对，GIF 各 6 帧（含空白帧）。这轮完成的是分区轮廓示范，真实笔触、皴法笔组造型及完整无叶树生成都仍待标注。RealBrush 通过发布审批并取得实际文件后才能做具有真实墨迹监督的渲染对照。

[离线结构标注工具](../tools/structure_reference_review.html) · [四项审核任务和原图坐标](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/annotation_queue.json>) · [馆藏预览](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/source_audit/museum_review_contact.png>) · [皴法分级清单](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/historical_cunfa_grade.json>) · [同 5 笔绘画对照](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/scene_pilot/drawing_comparison.png>) · [S2 逐笔 GIF](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/scene_pilot/s2_drawing.gif>)

## 复现与基线

在 F 盘项目目录按 AGENTS.md 将缓存和临时目录放到 E 盘。

```powershell
& 'E:/Anaconda/python.exe' -B scripts/professional_clpd.py
& 'E:/Anaconda/python.exe' -B scripts/train_professional_scene.py
$proData = 'E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125'
& 'E:/Anaconda/python.exe' -B -m venv --system-site-packages --without-pip "$proData/s1_inference_env"
& "$proData/s1_inference_env/Scripts/python.exe" -B scripts/evaluate_shanshuiseg_research.py
```

原 20 套笔触回放经重新核对，与原 PNG 像素一致。500 笔、489 笔以及冻结 S0 权重均通过哈希核验。Git 当前分支为 `main`，仓库无已提交的 HEAD 和 GitHub remote，未提交或推送。独立作品尚未验收，下一轮优先核对第一组人工树石边界并扩充作品 ID 已确认的评估集。在该合集上验证树木召回和岩纹误收之后，再尝试可靠困难参照微调。

[回放验收](<E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125/regression.json>)
