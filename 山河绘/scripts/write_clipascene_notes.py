from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"offline_previews"/"clipascene_20261021"
STORAGE=Path(r"E:\Codex_Workspace_Data\CLIPascene_20261021")
before=json.loads((STORAGE/"disk_space_before.json").read_text(encoding="utf-8-sig"))
after=json.loads((STORAGE/"disk_space_after_cleanup.json").read_text(encoding="utf-8-sig"))
c_before=next(r["Free"] for r in before if r["Name"]=="C")
c_after=next(r["Free"] for r in after if r["Name"]=="C")
summary=dict(c_free_before_bytes=c_before,c_free_after_cleanup_bytes=c_after,
             freed_bytes=c_after-c_before,freed_gib=round((c_after-c_before)/2**30,3),
             archive=str(STORAGE),cleanup_manifest=str(STORAGE/"cleanup_manifest.json"),
             user_environment=str(STORAGE/"user_environment_after.json"))
(OUT/"storage_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
runs=[]
for p in sorted(OUT.glob("layer*/experiment_manifest.json")):
    r=json.loads(p.read_text(encoding="utf-8"))
    runs.append(f"- 特征层 {r['layer']}，初始 {r['initial_paths']} 条曲线，{r['iterations']} 次迭代，seed={r['seed']}，用时 {r['elapsed_seconds']:.1f} 秒。")
body=f"""# CLIPascene 山水画离线实测

输入使用项目现有第一幅“松瀑亭舟”原图。裁切画面，去除题字，再等比例留白至正方形；没有前后景分割或背景补绘。

## 实际运行

使用作者官方 SceneSketch 的位置 MLP、CLIP 特征损失、DiffVG 可微渲染与优化流程。整图模式并非论文展示的全部前后景组合流程，也未完成完整抽象矩阵。

{chr(10).join(runs)}

每个运行目录保留 `best_iter.svg`、`points_mlp.pt`、`config.npy`、逐阶段 SVG 和实验参数。PNG 预览直接从输出 SVG 曲线重绘，不添加毛笔纹理。为了观察路径几何，对照图统一为等宽线。现有提线器保留了更多路径；两者笔数、工作分辨率及表示方式不同，本轮是探索性对照，不能作为公平排名。

## 兼容调整

- 独立 PyTorch 2.8.0 / CUDA 12.8，支持 RTX 5060。
- 原版 DiffVG CPU 后端，4 个工作线程；CLIP 损失在 GPU，位置 MLP 和光栅化在 CPU，保留跨设备自动求导。
- 补齐旧 NumPy/PyTorch API 别名；使用本地官方 CLIP 权重，并检查原始权重 SHA256。
- 为较新的 pybind11 补上 Python Interpreter 查找；避开未使用的 `log2` 同名函数冲突。
- 按微软 STL 的兼容说明定义 `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR`，解决新编译器与已有运行库的互斥锁异常。渲染与非零梯度已单独验证。

这些是环境兼容修改，不是额外的国画笔法学习。输出曲线没有实测笔压、含水量、毛笔姿态和绘画顺序，不能直接作为机器人任务。

## 磁盘与后续约定

实验文件归档在 `{STORAGE}`。本轮迁移与额外缓存整理，共释放 C 盘约 {(c_after-c_before)/10**9:.2f} GB（{(c_after-c_before)/2**30:.2f} GiB）。所有文件在 E 盘保留，迁移前核对大小，大文件核对 SHA256。额外归档 250 个旧缓存文件，17 个在用文件保留。

用户级 TEMP、TMP、PIP_CACHE_DIR、TORCH_HOME、HF_HOME、XDG_CACHE_HOME、CUDA_CACHE_PATH 已指向 E 盘；原值备份在归档目录。当前实验进程显式设置 E 盘环境变量，避免旧终端继承旧值。新终端/应用会逐步继承用户级设置。F 盘项目根部 `AGENTS.md` 已保存长期约定。

日志、迁移清单和清理清单均在 E 盘归档目录；本文件夹保留图片、SVG、参数和结果说明。
"""
(OUT/"README.md").write_text(body,encoding="utf-8")
print(json.dumps(summary,ensure_ascii=True))
