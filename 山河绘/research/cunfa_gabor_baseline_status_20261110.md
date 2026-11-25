# 皴法辨识初始基线（2026-11-10）

本轮未通过，不替换绘画系统，不连接机械臂。

数据：40个来源依据候选区域（披麻皴20、斧劈皴20），9位署名/归属作者，12个作品/示范，3个发布机构。5例其他类型仅作压力检查。69个裁剪中24个在训练前因空白、植被或边框等排除。旧30个弱来源样本未用。正式专家局部标签/测试均为0，未达每类30–50个目标。

来源：AAM用笔图例、NPM万壑松风局部说明、DPM七件藏品描述。AAM全为Lampo Leong署名；李唐同幅画包含两种局部标签，不能按整幅画统一标签。DPM属于作品级技法依据＋人工裁剪，不是专家局部真值。源文件与裁剪SHA-256写入清单。

方法：40个Gabor幅值图、128×128直方图均衡灰度、最多8维PCA、训练投影标准化、RBF SVM；C∈[0.1,1,10]内层作者GroupKFold选择。外层LeaveOneGroupOut按作者9轮。与原2014论文三类任务不同，属于方法族适配。主模型17/40符合来源标签（42.5%），macro-F1=0.4217，混淆矩阵行真列预测[[10,10],[13,7]]。Gabor统计特征45.0%；颜色明暗对照47.5%。未显示结构辨识优势。

完整性：所有69个源/裁剪哈希、作者/作品/精确裁剪不跨折、40个各预测一次、未知探针不进训练、模型存取等价通过。双PCA与独立SVD距离误差1.12e-07。没有将正式未知类别训练完成，“需复核”仅为分类间隔阈值。没有整幅检测、笔触生成或实机动作。

复现：先显式设置E盘TEMP/TMP/PIP_CACHE_DIR/TORCH_HOME/HF_HOME/XDG_CACHE_HOME/CUDA_CACHE_PATH、PYTHONDONTWRITEBYTECODE=1及OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/OMP_NUM_THREADS=4；在项目目录用E:/Anaconda/python.exe -B运行scripts/prepare_cunfa_two_class_20261110.py、scripts/train_cunfa_gabor_20261110.py、scripts/report_cunfa_gabor_20261110.py。既有threadpoolctl在枚举OpenBLAS时不兼容，本轮用启动环境限制线程，没有升级或修改现有环境。

下一步：先复核错误区域标签，补独立作者和清晰皴笔资料；按作者预先封存专家测试。冻结新数据后再比较图像上下文/笔组结构方法，不以本次预测自动扩充标签。原认可版本维持基线。

输出目录：E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\cunfa_gabor_baseline_20261110
清单SHA-256：33468f435e73bd486baa250640a62ceb5631cd524b481b9c2621303e5679877f
