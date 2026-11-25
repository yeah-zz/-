# 项目清理记录（2026-09-24）

## 删除

删除 4 个一次性调试输出文件，共释放 86,040 字节（约 84 KiB）：

- `_ex.txt`：空输出文件。
- `_dbg.txt`、`_dbg.json`、`_dbg.png`：同一次调试运行的报告、任务导出和预览；除输出之间的自引用外，项目代码、测试和说明没有引用它们。

删除前复核了每个绝对路径均位于项目目录内，且大小和修改时间与清理清单一致。

## 保留

保留所有 Python 调试脚本（HANDOVER.md 明确记载其诊断和复现用途）、Dobot CRA 手册、标定/依赖/运行入口、历史任务和预览、回归测试与夹具、用户参考图和实画照片、执行轨迹与日志。保留 `.venv` 和缓存目录。`robot_runs` 由并行启动诊断任务自行管理，本清理未操作该目录；并行任务改动的代码、测试和说明也未被本次清理触碰。

## 离线验证

- `python main.py --help`：退出码 0，CLI 导入并输出帮助。
- `python main.py --check`：退出码 0，配置自检通过；未连接设备或执行运动。

## 按“保留最新版”继续清理

按用户补充要求，又删除 23 个未使用的中间产物/可再生输出，共 9,087,247 字节。结合首轮合计删除 27 个文件，释放 9,173,287 字节（约 8.75 MiB）。这轮包括临时岩石对照图、旧调试结果、白描 v2 与 v3 fine 中间版本、较旧水墨全图预览/任务以及可重新生成的默认预览输出。删除前重新检查了项目内绝对路径、原清单中的大小和修改时间；并对中间产物做了代码/文档引用扫描。

仍保留 v4/v5/v6 的审查链、实际执行日志和轨迹、用户实画照片、当前 v6 任务与预览、回归测试需要的历史任务/图像、源参考图和校准/运行配置。它们仍被审查文档、回归或验证脚本引用，删除会破坏现有说明或复现能力。并行的启动诊断任务仍在运行，其代码、测试和说明未被本次清理触碰。

## 第二轮整理：按目录归类（不删除任何东西）

这一轮只做**移动和改引用**，没有删除业务文件。原则是"被代码按文件名引用的东西一律留在原位"。

| 动作 | 内容 |
|---|---|
| 新建 `docs/` | 14 个历史评审/交接/专项文档从根目录移入（含 `algorithm_handover_evidence.json`）；新增 `docs/README.md` 索引 |
| 新建 `绘画/tools/` | 12 个历史一次性调试脚本 `_*.py` 移入；每个文件顶部插入 3 行 `sys.path` 引导，保证 `python 绘画\tools\<脚本>.py` 仍能 import `绘画/` 模块 |
| 新建 `archive/` | 59 个**零引用**的历史产物按类型归档：`archive/previews/`（35 个预览图）、`archive/jobs/`（9 个旧任务/计划）、`archive/experiments/`（15 个笔压/供墨探针结果与 `pelican_bicycle.html`） |
| 保留在根目录 | 被代码或文档按相对名引用的当前工作集：`artifacts/reference/desired_mountain_reference.png`、`mountain_v{4,5,6}_*.json/png`、`mountain_baimiao*.png`、`mountain_baimiao_v3_part{1..4}_job.json`、`shuimo_*`、`watercolor_*_plan/preview`、用户实画照片、`_cra.pdf` 等 |
| 删除 | 各层 `__pycache__`（含已经不存在模块的孤儿 `.pyc`），以及本轮验证产生的临时文件 |

零引用判定方法：把根目录每个文件名在全部 `*.py` / `*.md` / `*.txt` / `*.bat` / `*.toml` 里做一次全文匹配；
再人工补上**动态拼接**的名字（`mountain_baimiao_v3_part%d_job.json`、`mountain_v6_pressure_%02d.json`、
`mountain_v6_ink_%02d.json`），这几类强制保留或按可再生输出处理。

## 第二轮整理：修掉两处"跑不起来"

1. **`.venv` 的基础解释器丢失**：`.venv/pyvenv.cfg` 的 `home` 原为 `D:\Anaconda3`（已不存在），
   已改为 uv 管理的 `cpython-3.12.14`，原文件备份为 `.venv/pyvenv.cfg.bak-anaconda`。
2. **写死的旧绝对路径**：项目从 `D:\ROBOT\机械臂绘画` 搬到 `G:\机械臂绘画`，
   根目录 33 个仍在用的任务/计划 JSON 里的 `D:\ROBOT\机械臂绘画\...` 已改成相对路径；
   因 `artifacts/mountain/mountain_v6_job.json` 内容变化，蓝色计划里的 `ink_job_sha256` 快照同步重算。
   `robot_runs/`、`绘画/输出/`、`designs/`、`samples/` 中的历史记录保持原样（纯存档）。

文档侧的旧路径和旧脚本名（在根目录直接叫 `prepare_robot.py`、`run_mountain_balanced.py` 等）也已改为
`绘画\` 前缀。详见 [../STARTUP.md](../STARTUP.md) 第 4 节。

## 整理后的回归验证

在项目根目录执行，全部通过：

```powershell
.\.venv\Scripts\python.exe -c "import sys, numpy, cv2; print(sys.version.split()[0], numpy.__version__, cv2.__version__)"  # 3.12.14 2.5.3 5.0.0
.\.venv\Scripts\python.exe -m compileall -q 绘画             # 退出码 0
uv run python main.py --check                                # 配置自检通过
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -p "test_*.py"
.\.venv\Scripts\python.exe launcher.py                       # 菜单正常显示
uv run python main.py --image artifacts/reference/desired_mountain_reference.png --no-preview        # 离线规划跑通
.\.venv\Scripts\python.exe 绘画\tools\_dump.py                 # 迁移后的调试脚本仍可运行
```


## 第三轮整理：`绘画/` 提取为独立项目

按用户要求，把 `绘画/` 从"外层项目里的一个源码目录"提升为**独立、可整体搬走的项目根**。

| 项目 | 内容 |
|---|---|
| 新增 `pyproject.toml` / `uv.lock` / `.venv/` | 用 `uv sync --python 3.12` 在本目录**重建**的独立环境（numpy 2.5.3、opencv-python 5.0.0.93），不再依赖外层的 Anaconda 或旧 `.venv` |
| 新增 `run.bat` | `uv run python launcher.py` |
| 改写 `启动菜单.bat` | 直接用本目录的 `.venv\Scripts\python.exe launcher.py`，不需要 `uv` 在 PATH 里 |
| 改写 `project_paths.py` | 本目录有 `pyproject.toml` 时 `PROJECT_DIR` = 本目录；没有时回退到上一层（兼容旧的二层结构） |
| 新增 `tests/` | 15 个 `test_*.py` 移入，加 `tests/__init__.py`；`test_plum` 改成包内相对 import；测试改为在项目根执行 `-s tests -t .` |
| 新增 `scripts/` | 5 个**无人 import** 的独立脚本：`validate_mountain.py`、`review_mountain_v5.py`、`plan_blue_full.py`、`paint_blue_trial.py`、`run_blue_full.py`，各加 3 行 `sys.path` 引导 |
| 搬入数据 | 55 个任务/预览/参考图（`mountain_*.json/png`、`artifacts/reference/desired_mountain_reference.png`、`shuimo_*`、`watercolor_*_plan/preview`、用户实画照片等）、`samples/`、`robot_runs/`、`designs/`、`docs/`、`archive/`、`_cra.pdf`、`README.md`、`STARTUP.md`、`ROBOT_STARTUP.md`、`requirements.txt` |
| 删除 | 外层只剩兼容转发用的 `main.py`、`launcher.py`、`run_painting_experiment.py`（独立后失去意义）、外层 `pyproject.toml`/`uv.lock`/`.venv`（已被本目录的取代）、外层 `run.bat` |

**为什么顶层模块不分子目录**：`config`、`job`、`robot`、`vision`、`shuimo`、`prepare_robot`、
`run_mountain_v4_robot` 等被十几个文件**扁平互相 import**；要挪进子目录就必须全项目改成包内
相对导入。按"低风险优先"的要求，只把**没有任何模块 import 的叶子文件**分流到
`tests/`、`scripts/`、`tools/`，其余留在顶层。当前顶层模块 38 个，子目录 32 个。

**提取后的回归验证**（在 `G:\机械臂绘画\绘画` 下执行，全部通过）：

```powershell
uv run python main.py --check                                              # 配置自检通过
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -p "test_*.py"  # Ran 86 tests, OK
.\.venv\Scripts\python.exe main.py --image artifacts/reference/desired_mountain_reference.png --no-preview   # 112 个轨迹点
.\.venv\Scripts\python.exe launcher.py                                     # 菜单显示 (G:\机械臂绘画\绘画)
uv run python main.py --check                                              # uv 路径也通，退出码 0
```
## 第四轮整理：历史产物按题材归档 + 输出目录统一

### 1. 山水/淡彩/蓝色的产物搬进 `artifacts/`

54 个历史产物从项目根目录移入 `artifacts/`，按题材分子目录：

| 子目录 | 文件数 | 内容 |
|---|---|---|
| `artifacts/mountain/` | 30 | 山水 v1–v6 的任务、预览、执行记录、对比图、`painted_v*_user_photo.jpg`、`mountain_baimiao_v3_part{1..4}_job.json` |
| `artifacts/blue/` | 9 | 蓝色试笔/整幅/洗笔的计划与预览 |
| `artifacts/watercolor/` | 7 | 淡彩与稀疏上色的计划与预览 |
| `artifacts/shuimo/` | 3 | 水墨默认任务与预览、`shuimo_landscape.png` |
| `artifacts/reference/` | 5 | `desired_mountain_reference.png`、`mountain_baimiao*.png`、`rock02_new.png`、`lanhua_framework.png` |

**这不是单纯搬文件**：其中 30 个被代码当默认路径用（`prepare_robot.py`、`run_mountain_balanced.py`、
`run_mountain_v4_robot.py`、`run_blue_wash.py`、`plan_blue_*.py`、`review_mountain_v6.py`、
`config.py`、`hover_blue_trial.py`，以及 `scripts/` 与 `tests/` 里的 6 个文件）。
所以做了**一次性全局路径替换**（长名优先的单次正则，避免嵌套替换），并同步改掉：

- 代码里的默认参数值；
- `robot_runs/`、`designs/` 历史记录里引用的计划路径；
- `artifacts/` 内 JSON 的 `image` / `ink_job` / `ink_image` 字段；
- `README.md`、`STARTUP.md`、`ROBOT_STARTUP.md`、`docs/*.md`、`使用说明.md` 里的示例命令。

因为 `artifacts/mountain/mountain_v6_job.json` 的 `image` 字段被改写、内容变了，
`artifacts/blue/` 四个计划里的 `ink_job_sha256` 快照已同步重算。

> `archive/` 里的更早历史**没有动**，保持原样作为存档。

### 2. 默认输出改到 `输出/`

原来不指定 `--job-out` / `--preview-out` 时，预览和任务 JSON 会写在 `main.py` 旁边（项目根目录），
导致每跑一次就在根目录堆文件。现在全部落到 `project_paths.OUTPUT_DIR`，也就是 **`输出/`**：

| 模式 | 默认输出（全部在 `输出/`） |
|---|---|
| 色块 `--image X.png` | `paint_preview.png`、`paint_stations.png`、`paint_job.json`、`paint_labels.png`（`--dump-labels`） |
| 水墨 `--mode shuimo` | `shuimo_preview.png`（另带 `_contact/_layers/_lineart/_stress` 四张诊断图）、`shuimo_job.json` |
| 淡彩 `--mode watercolor` | `watercolor_preview.png`、`watercolor_preview_paths.png`、`watercolor_plan.json` |
| 彩色第二遍 `--mode color-pass` | `color_pass_preview.png`、`color_pass_plan.json` |
| 分模块 `--mode painting` | `painting_preview.png` 及各分模块图/JSON |

改的是 `main.py`（4 处默认路径 + 导入 `OUTPUT_DIR`）和 `config.py` 的
`ShuimoConfig.PREVIEW_PNG` / `JOB_JSON`（改回纯文件名，由 `main.py` 拼到 `输出/`）。
`--no-preview` 依然可以完全关掉预览。

### 3. 回归验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -p "test_*.py"   # Ran 86 tests, OK
.\.venv\Scripts\python.exe -m compileall -q .                                  # 退出码 0
.\.venv\Scripts\python.exe main.py --mode shuimo --image artifacts\reference\desired_mountain_reference.png
#   [输出] 预览图: ...\输出\shuimo_preview.png
.\.venv\Scripts\python.exe main.py --image artifacts\reference\desired_mountain_reference.png
#   [输出] 预览图: ...\输出\paint_preview.png
```

整理后项目根目录只剩：38 个业务模块、6 个说明/配置文件、`pigment_measurements.json`，
以及 `参考图/ 输出/ artifacts/ robot_runs/ designs/ samples/ docs/ archive/ scripts/ tests/ tools/ .venv/`。
## 第五轮：`输出/` 历史批次归档

`输出/` 之前积了 114 个文件（10.7 MB），按**运行批次**（文件名里的时间戳）分组后，
只保留最近 3 个批次，其余 90 个文件（7.9 MB）移到 `archive/outputs/`：

| 位置 | 文件数 | 大小 |
|---|---|---|
| `输出/` | 24 | 2.8 MB |
| `archive/outputs/` | 90 | 7.9 MB |

保留的是：最近一次梅花运行（6 个）、本次红梅离线预览（6 个）、`plum_review*`（6 个）、
以及水墨默认输出 `shuimo_job.json` + `shuimo_preview*.png`（6 个）。**没有删除任何文件。**

### 一次运行会生成多少文件（现状，未改）

| 命令 | 预览图 | 任务 JSON |
|---|---|---|
| `--image X.png`（色块） | 2 张（`paint_preview` + `paint_stations`），`--dump-labels` 再 +1 | 1 |
| `--mode shuimo` | **5 张**（主图 + `_contact` `_stress` `_layers` `_lineart`） | 1 |
| `--mode watercolor` | 2 张（`_preview` + `_preview_paths`） | 1 |
| `--mode color-pass` | 1 张 | 1 |
| `--mode painting` | **3 张**（合并 + `_outline` + `_color`） | **3** |
| 菜单 19 梅花 | **3 张**（合并 + `_outline` + `_color`） | **3** |

多出来的图和 JSON 不是多余的：`trajectory_preview.render()` 故意把"标称笔宽"和
"加宽 0.5 mm 敏感性"并排给出（因为 `BRUSHES` 仍是估值）；`painting_workflow.save_outputs()`
按 stage 拆模块，方便只执行黑枝或只执行红瓣。菜单用时间戳命名、不覆盖，所以历史只增不减。
## 第六轮：默认只出一张预览图

按"一张图一张预览就够"的要求，把默认输出从"一次运行 2–5 张图 + 最多 3 个 JSON"压到
**一张预览图 + 一个任务 JSON**。多出来的诊断图改成 `--diagnostics` 按需生成。

| 命令 | 改前 | 改后（默认） |
|---|---|---|
| `--image X.png`（色块） | `paint_preview.png` + `paint_stations.png` + `paint_job.json` | **1 张 + 1 JSON** |
| `--mode shuimo` | `shuimo_preview.png` + `_contact` + `_stress` + `_layers` + `_lineart` + JSON = **6 个** | **1 张 + 1 JSON** |
| `--mode watercolor` | `_preview.png` + `_preview_paths.png` + JSON | **1 张 + 1 JSON** |
| `--mode painting` / 梅花 | 3 张预览 + 3 个 JSON = **6 个** | **1 张 + 1 JSON** |
| `--mode color-pass` | 1 张 + 1 JSON | 不变 |

改动的文件：

| 文件 | 改动 |
|---|---|
| `trajectory_preview.py` | `render(..., diagnostics=False)`；四张分面板图只在 `diagnostics=True` 时另存（它们本来就已经画在拼版图里，**没有丢信息**） |
| `shuimo.py` | `render_framework_preview(..., diagnostics=False)` 透传 |
| `watercolor.py` | 拆出 `render_image()` 返回数组、不写文件；`render()` 只负责落盘。原来 `save_outputs` 要先把色面预览写成 PNG 再读回来做叠加，现在不需要中间文件了 |
| `painting_workflow.py` | `save_outputs(..., split_stages=False)`；默认只写计划 JSON + 一张拼版图，不再写 `<名>_outline/_color.json/png` |
| `main.py` | 新增 `--diagnostics`；站点图、水墨分面板、淡彩纯路径图、分模块文件全部改为按需 |
| `scripts/validate_mountain.py` | 显式传 `--diagnostics`，并补上漏改的路径（见下） |
| `tests/test_watercolor.py` | 断言纯路径图的用例显式加 `--diagnostics` |

### 顺带修掉 3 处上一轮漏改的路径

上一轮全局替换只匹配了**完整文件名**，所以这几个**动态拼接**的名字漏了，会让
`validate_mountain.py` / `review_mountain_v6.py` 找不到文件：

- `scripts/validate_mountain.py`：`mountain_baimiao_v3_part%d_job.json` → 加 `artifacts/mountain/` 前缀；
  四个 `--preview-out` / `--job-out` 的 prefix（`mountain_v4`、`mountain_v3_reprocessed`、
  `mountain_v1_reprocessed`、`rock_classic_regression`）也补成 `artifacts/mountain/...`，
  否则评审产物会重新写回项目根目录。
- `review_mountain_v6.py`：`mountain_v6_pressure_%02d.json`、`mountain_v6_ink_%02d.json` 加前缀。

### 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -p "test_*.py"   # Ran 86 tests, OK
.\.venv\Scripts\python.exe main.py --mode shuimo --image artifacts\reference\desired_mountain_reference.png
#   -> 输出\shuimo_preview.png + shuimo_job.json（2 个文件）
.\.venv\Scripts\python.exe main.py --image artifacts\reference\desired_mountain_reference.png
#   -> 输出\paint_preview.png + paint_job.json（2 个文件）
.\.venv\Scripts\python.exe scripts\plum_preview.py
#   -> 输出\plum_preview_<时间戳>.png + .json（2 个文件）
.\.venv\Scripts\python.exe scripts\validate_mountain.py   # 退出码 0，产物写入 artifacts\mountain\
.\.venv\Scripts\python.exe scripts\review_mountain_v5.py  # 退出码 0
```