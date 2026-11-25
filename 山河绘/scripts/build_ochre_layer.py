"""从参考图提取暖赭(ochre)区域,生成一个**独立的单色上色计划**。

为什么需要它
------------
现有 `*_计划.json` 只含蓝/绿两色。参考图里暖赭(日照坡面)占**彩色面积约 37%**,
却从未被提取成颜色笔触 —— 从最早的 1633 笔密铺版到现在所有版本都是如此。
结果是画面丢掉冷暖对照,整体偏冷发平。

本脚本把暖赭单独做成一个可执行计划,可以在**同一张已勾线的纸上补画**,
不需要重画黑线。

安全性 / 严格性
---------------
* 输出的计划沿用来源计划的 `image_size` / `image_corners` / `mm_per_px` /
  `motion_candidate` / `palette_bgr` 骨架,只替换颜色笔触,因此与配套黑线任务的
  标定快照(纸面 Z、画布四角、RPY、User/Tool、速度倍率、调色盘)天然一致。
* 颜色路径经 `CanvasMap.to_robot` 换算成机器人坐标后逐点检查"距画布边缘 >= 20 mm"
  (`run_color_experiment.module_lines` 会把越界笔触记为 skipped,而
  `run_color_from_job.load_plan` 只要出现 skipped 就直接报错),越界部分被裁掉。
* 单笔长度上限 40 mm(与 `scripts/build_filled_landscape.py` 一致),按**弧长**切分,
  不是只在端点之间切 —— 蛇形的一行可能是一条很长的直线 run。
* **默认不复制配套黑线任务**。复制出来会得到一个和真黑线任务一模一样、名字只差前缀的
  `_黑线任务.json`,谁要是对它跑 `run_mountain_balanced.py`,就会在已经画完的纸上
  **重画整幅黑线**。所以补层计划用 `--ink-job` 显式指向原黑线任务(只做标定核对,
  不执行黑线);不加 `--ink-job` 会直接报错,不会误画。
* **不做任何物理覆盖率或颜料浓度标定**。`target_opacity` 只是记录字段,不驱动机器人;
  纸上浓淡只取决于杯里颜料兑多稀、蘸多深、以及铺得多密。
* 生成后必须自己跑 `--check`;本脚本不连接机械臂。

用法
----
    .\\.venv\\Scripts\\python.exe scripts/build_ochre_layer.py `
        --source-plan "输出\\层峦松水_简约少空走_20260929_170322\\层峦松水_简约少空走_计划.json" `
        --out-name "层峦松水_补赭淡铺" --pitch-mm 1.3 --press-mm 0.16 --opacity 0.35

    # 离线校验(补层计划必须显式给 --ink-job)
    .\\.venv\\Scripts\\python.exe run_color_from_job.py `
        --job "输出\\<输出目录>\\层峦松水_补赭淡铺_计划.json" `
        --ink-job "输出\\层峦松水_简约少空走_20260929_170322\\层峦松水_简约少空走_黑线任务.json" --check
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CanvasConfig                          # noqa: E402
from geometry import CanvasMap                           # noqa: E402
from project_paths import OUTPUT_DIR                     # noqa: E402
from shuimo import press_to_width_mm                     # noqa: E402

MAX_STROKE_MM = 40.0
MIN_MARGIN_MM = 20.0        # module_lines 的硬性边距;低于它会被判为 skipped


def imread_u(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def warm_mask(bgr, hue_max=35, hue_min=145, sat_min=25, val_min=85, ink_max=110,
              min_area=40, close_px=7, open_px=3):
    """暖色(赭/黄/红)区域。hue 以 OpenCV 的 0-179 计: <35 或 >145 为暖侧。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)
    m = ((S >= sat_min) & (V > val_min) & ((H < hue_max) | (H > hue_min))).astype(np.uint8)
    m[V < ink_max] = 0                                   # 排除墨线本身
    if close_px:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    if open_px:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    clean = np.zeros_like(m)
    for i in keep_ids:
        clean[labels == i] = 1
    return clean, labels, stats, keep_ids


def serpentine(comp, y0, x0, hh, pitch_px):
    """把一个连通域按扫描行连成蛇形折线链。返回若干条链(图像像素坐标)。"""
    chains, ends = [], []
    for ri, ry in enumerate(range(0, hh, pitch_px)):
        row = comp[ry]
        d = np.diff(np.concatenate([[0], row, [0]]))
        starts, stops = np.where(d == 1)[0], np.where(d == -1)[0]
        runs = [(int(a), int(b)) for a, b in zip(starts, stops) if b - a >= 2]
        if not runs:
            continue
        if ri % 2:
            runs = runs[::-1]
        new_ends = []
        for a, b in runs:
            entry = np.array([x0 + a, y0 + ry], float)
            exit_ = np.array([x0 + b, y0 + ry], float)
            best, best_d = None, pitch_px * 4
            for ci, ce in ends:
                dd = float(np.hypot(*(ce - entry)))
                if dd < best_d:
                    best, best_d = ci, dd
            if best is None:
                chains.append([entry, exit_])
                new_ends.append((len(chains) - 1, exit_))
            else:
                chains[best] += [entry, exit_]
                new_ends.append((best, exit_))
        ends = new_ends
    return chains


def split_by_length(chain, max_len_px):
    """按**弧长**切分,保证每段不超过 max_len_px。

    不能只在端点之间切:蛇形的一行可能是一段很长的直线 run(只有两个端点),
    那样会切出一条远超上限的笔触(实测会到 48 mm)。
    """
    out = []
    cur = [chain[0]]
    cur_len = 0.0
    for a, b in zip(chain, chain[1:]):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        d = float(math.dist(a, b))
        while cur_len + d >= max_len_px and d > 1e-9:
            need = max_len_px - cur_len
            t = need / d
            mid = a + (b - a) * t
            cur.append(mid)
            if len(cur) >= 2:
                out.append(cur)
            cur = [mid]
            cur_len = 0.0
            a = mid
            d = float(math.dist(a, b))
        cur.append(b)
        cur_len += d
    if len(cur) >= 2:
        out.append(cur)
    return out


def trim_to_margin(pts_px, mapper, polygon, margin=MIN_MARGIN_MM):
    """保留"整条都在边距内"的最长连续段;返回保留的点。"""
    robot = mapper.to_robot(np.asarray(pts_px, float))
    inside = [cv2.pointPolygonTest(polygon, (float(q[0]), float(q[1])), True) >= margin
              for q in robot]
    best_a = best_b = -1
    a = None
    for i, ok in enumerate(list(inside) + [False]):
        if ok and a is None:
            a = i
        elif not ok and a is not None:
            if i - a > best_b - best_a:
                best_a, best_b = a, i
            a = None
    if best_b - best_a < 2:
        return []
    return [pts_px[i] for i in range(best_a, best_b)]


def rotate_for_scan(mask, angle_deg, min_area):
    """把 mask 旋转 angle_deg, 返回 (旋转后的 mask/labels/stats/keep_ids, 反变换矩阵)。

    旋转只是改变"扫描行相对形状的方向"(笔痕方向), 形状本身不变。
    """
    h, w = mask.shape
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), float(angle_deg), 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += (nw - 1) / 2.0 - (w - 1) / 2.0
    M[1, 2] += (nh - 1) / 2.0 - (h - 1) / 2.0
    rot = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rot, 8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return rot, labels, stats, keep, cv2.invertAffineTransform(M)


def unrotate_points(piece, M_inv):
    arr = np.asarray(piece, float)
    homo = np.hstack([arr, np.ones((len(arr), 1))]) @ M_inv.T
    return homo[:, :2]


def build(source_plan, out_name, color, pitch_mm, press_mm, opacity, max_stroke_mm, args):
    source_plan = Path(source_plan)
    plan = json.loads(source_plan.read_text(encoding="utf-8-sig"))
    if plan.get("format") != "painting-workflow-design-v1":
        raise ValueError("请传入完整的 *_计划.json(顶层 format 必须是 painting-workflow-design-v1)")
    module = plan["modules"]["color"]
    if module.get("format") != "sparse-color-design-v1":
        raise ValueError("颜色模块格式不是 sparse-color-design-v1")
    if color not in module.get("palette_bgr", {}):
        raise ValueError("调色盘里没有这个颜色: %s" % color)

    image_size = module["image_size"]
    mm_per_px = float(module["mm_per_px"])
    mapper = CanvasMap(image_size, module["image_corners"])
    polygon = np.asarray(CanvasConfig.ROBOT_CORNERS, np.float32)

    ref_path = Path(plan["reference_image"])
    ref = imread_u(ref_path)
    if ref is None:
        raise FileNotFoundError("读不到参考图: %s" % ref_path)
    W, H = int(image_size[0]), int(image_size[1])
    if ref.shape[:2] != (H, W):
        ref = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA)

    mask, labels, stats, keep_ids = warm_mask(
        ref, args.hue_max, args.hue_min, args.sat_min, args.val_min, args.ink_max,
        args.min_area, args.close_px, args.open_px)
    area_px = int(mask.sum())
    print("暖色区域: %d 个连通域, %d px = 全图 %.2f%%" % (
        len(keep_ids), area_px, 100.0 * area_px / (W * H)))

    mask_bgr = [int(np.median(ref[:, :, c][mask > 0])) for c in range(3)]
    print("参考图该色中位 BGR = %s (仅用于预览配色,不驱动机器人)" % (mask_bgr,))
    M_inv = None
    if abs(float(args.scan_angle)) > 1e-6:
        mask, labels, stats, keep_ids, M_inv = rotate_for_scan(mask, args.scan_angle, args.min_area)
        print("笔痕方向: 旋转 %.1f 度 (0=左右, 90=上下)" % args.scan_angle)

    pitch_px = max(2, int(round(pitch_mm / mm_per_px)))
    max_len_px = max_stroke_mm / mm_per_px
    width_mm = press_to_width_mm(press_mm)
    print("行距 %d px = %.3f mm | 单笔上限 %.0f mm | 下压 %.2f mm -> 模型笔宽 %.2f mm" % (
        pitch_px, pitch_px * mm_per_px, max_stroke_mm, press_mm, width_mm))

    strokes, dropped, seq = [], 0, 0
    for i in keep_ids:
        y0 = stats[i, cv2.CC_STAT_TOP]
        x0 = stats[i, cv2.CC_STAT_LEFT]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        comp = (labels[y0:y0 + hh, x0:x0 + ww] == i).astype(np.uint8)
        for chain in serpentine(comp, y0, x0, hh, pitch_px):
            for piece in split_by_length(chain, max_len_px):
                if M_inv is not None:
                    piece = unrotate_points(piece, M_inv)
                kept = trim_to_margin(piece, mapper, polygon)
                if len(kept) < 2:
                    dropped += 1
                    continue
                robot = mapper.to_robot(np.asarray(kept, float))
                length = float(sum(math.dist(robot[k], robot[k + 1])
                                   for k in range(len(robot) - 1)))
                n = len(kept)
                strokes.append(dict(
                    id="%s-fill-%05d" % (color, seq), color=color, region_id="ochre-%d" % i,
                    points=[[float(p[0]), float(p[1])] for p in kept],
                    recipe="area-fill", layer=0,
                    target_opacity=[opacity] * n,
                    target_width_mm=[round(width_mm, 3)] * n,
                    brush_width_mm_estimate=round(width_mm, 3),
                    length_mm=length, path_style="in-region-serpentine",
                    pressure_mm=None, pigment_recipe_calibrated=False,
                    press=[press_mm] * n))
                seq += 1
    if not strokes:
        raise ValueError("没有生成任何笔触;检查颜色阈值或 --pitch-mm")
    lens = np.array([s["length_mm"] for s in strokes])
    print("保留 %d 笔(边距裁掉 %d 段), 总长 %.0f mm, 均长 %.1f mm, 最长 %.1f mm" % (
        len(strokes), dropped, lens.sum(), lens.mean(), lens.max()))

    out_dir = OUTPUT_DIR / ("%s_%s" % (out_name, datetime.now().strftime("%Y%m%d_%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=False)

    new_plan = dict(plan)
    new_module = dict(module)
    new_module["strokes"] = strokes
    new_module["regions"] = []
    new_module["stroke_count"] = len(strokes)
    new_module["skipped_margin_ids"] = []
    new_module["palette_bgr"] = dict(module["palette_bgr"], **{color: mask_bgr})
    new_plan["modules"] = dict(plan["modules"], color=new_module)
    new_plan["fill_report"] = {color: dict(strokes=len(strokes), mask_area_px=area_px,
                                           estimated_mask_coverage=None)}
    new_plan["simplification"] = dict(
        source_plan=str(source_plan.resolve()),
        note="暖赭补层:参考图暖色区单独成层,与来源计划的蓝/绿分开执行,可在同一张纸上补画。",
        pitch_mm=round(pitch_px * mm_per_px, 4), press_mm=press_mm,
        max_stroke_mm=max_stroke_mm, target_opacity=opacity)
    new_plan["note"] = (
        "%s 单色补层。行距 %.2f mm、下压 %.2f mm、目标不透明度 %.2f(仅记录,不驱动机器人)。"
        "黑线几何与来源计划一致,不重画黑线。未做物理覆盖率或颜料浓度标定 —— "
        "纸上浓淡取决于杯内颜料稀释、蘸入深度与实际铺色密度。"
        % (color, pitch_px * mm_per_px, press_mm, opacity))

    plan_path = out_dir / ("%s_计划.json" % out_name)
    plan_path.write_text(json.dumps(new_plan, ensure_ascii=False, indent=1), encoding="utf8")

    src_ink = source_plan.parent / (source_plan.stem.replace("_计划", "") + "_黑线任务.json")
    if not src_ink.is_file():
        raise FileNotFoundError("找不到配套黑线任务: %s" % src_ink)
    if args.copy_ink_job:
        # 默认关闭:这会留下一个和真黑线任务同名同内容的副本,
        # 误对它跑 run_mountain_balanced.py 会在已完成的纸上重画整幅黑线。
        shutil.copy2(src_ink, out_dir / ("%s_黑线任务.json" % out_name))

    print("\n输出目录: %s" % out_dir)
    print("  计划: %s (%d 笔)" % (plan_path.name, len(strokes)))
    if args.copy_ink_job:
        print("  !! 已复制配套黑线任务(危险):不要对它跑 run_mountain_balanced.py")
    print("\n离线校验(不连机械臂;补层计划必须显式给 --ink-job):")
    print('  .\\.venv\\Scripts\\python.exe run_color_from_job.py \\\n'
          '      --job "%s" \\\n'
          '      --ink-job "%s" --check' % (plan_path, src_ink))
    print("\n上纸前提醒:确认绿色已画完且原上色进程已退出;补层计划与本计划都按【当前标定】校验,"
          "\n改动 config.py(纸面高度/画布/姿态/速度)之后必须重新生成本计划。")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="从参考图提取暖赭区域,生成独立的单色补层计划")
    ap.add_argument("--source-plan", required=True, help="来源 *_计划.json(提供骨架与标定快照)")
    ap.add_argument("--out-name", default="层峦松水_补赭淡铺", help="输出目录/文件前缀")
    ap.add_argument("--color", default="yellow", choices=["yellow", "red", "blue", "green"])
    ap.add_argument("--pitch-mm", type=float, default=1.3, help="扫描行距(mm);越大越疏越淡")
    ap.add_argument("--press-mm", type=float, default=0.16, help="逐点下压(mm),不得超过 profile 的 press_mm")
    ap.add_argument("--opacity", type=float, default=0.35, help="目标不透明度(仅记录)")
    ap.add_argument("--max-stroke-mm", type=float, default=MAX_STROKE_MM)
    ap.add_argument("--hue-max", type=int, default=35)
    ap.add_argument("--hue-min", type=int, default=145)
    ap.add_argument("--sat-min", type=int, default=25)
    ap.add_argument("--val-min", type=int, default=85)
    ap.add_argument("--ink-max", type=int, default=110)
    ap.add_argument("--min-area", type=int, default=40)
    ap.add_argument("--close-px", type=int, default=7)
    ap.add_argument("--open-px", type=int, default=3)
    ap.add_argument("--scan-angle", type=float, default=0.0,
                    help="笔痕方向(度): 0=左右, 90=上下。参考图实测 蓝83 / 绿169 / 暖123")
    ap.add_argument("--copy-ink-job", action="store_true",
                    help="同时复制配套黑线任务(默认不复制;复制后在已完成的纸上误跑会重画黑线)")
    a = ap.parse_args(argv)
    return build(a.source_plan, a.out_name, a.color, a.pitch_mm, a.press_mm,
                 a.opacity, a.max_stroke_mm, a)


if __name__ == "__main__":
    raise SystemExit(main())
