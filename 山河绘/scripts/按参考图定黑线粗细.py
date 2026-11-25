"""按参考图重新设计黑线粗细。

每个黑线点的下压,由"参考图在该位置的墨线实际宽度"决定 —— 不再继承源计划的下压值。

  1. 参考图取墨线蒙版(灰度<阈值) -> 距离变换 = 每处墨线的半宽(px)
  2. 3x3 邻域取最大, 避免采样落在线旁一格而误判为"无线"
  3. 黑线任务的每个 down/draw 点经逆画布变换映射回图像坐标
  4. 采样该处墨线宽度(mm) -> 按 5~95 百分位映射到下压区间

用法:
  python scripts/按参考图定黑线粗细.py --src 输出/层峦松水_色阶版_xxxx
  --mode linear : 保留宽度的相对比例(细处更细, 但一半的点会落在下限)
  --mode rank   : 按排序均匀铺满区间(每笔都压得实, 粗细顺序仍照参考图)
"""
import argparse, json, shutil, sys
from datetime import datetime
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import CanvasMap
from config import ArmConfig


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="含 *_黑线任务.json 与 *_计划.json 的目录")
    ap.add_argument("--press-lo", type=float, default=0.06, help="参考图最细处对应的下压 mm")
    ap.add_argument("--press-hi", type=float, default=0.30, help="参考图最粗处对应的下压 mm")
    ap.add_argument("--mode", choices=("linear", "rank"), default="rank")
    ap.add_argument("--ink-thresh", type=int, default=150)
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args(argv)

    src = Path(a.src)
    ink_p = next(src.glob("*_黑线任务.json"))
    plan_p = next(src.glob("*_计划.json"))
    ink = json.loads(ink_p.read_text(encoding="utf-8-sig"))
    cm = json.loads(plan_p.read_text(encoding="utf-8-sig"))["modules"]["color"]
    mmpp = float(cm["mm_per_px"]); IMG = tuple(cm["image_size"])

    ref = imread_u(Path(ink["meta"]["image"]))
    if (ref.shape[1], ref.shape[0]) != IMG:
        ref = cv2.resize(ref, IMG, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    mask = (gray < a.ink_thresh).astype(np.uint8)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dt = cv2.dilate(dt, np.ones((3, 3), np.uint8))

    mapper = CanvasMap(IMG, cm["image_corners"])
    Minv = np.linalg.inv(mapper.matrix)
    wp = ink["waypoints"]
    xy = np.asarray([[w["x"], w["y"]] for w in wp], float)
    px = (np.hstack([xy, np.ones((len(xy), 1))]) @ Minv.T)[:, :2]
    ix = np.clip(np.rint(px[:, 0]).astype(int), 0, IMG[0]-1)
    iy = np.clip(np.rint(px[:, 1]).astype(int), 0, IMG[1]-1)
    width_mm = dt[iy, ix]*2.0*mmpp

    tags = np.asarray([str(w.get("tag", "")) for w in wp])
    sel = np.array([t.startswith("down:") or t.startswith("draw:") for t in tags])
    # 按 down -> draw* -> lift 切分笔触
    bounds = []; cur = None
    for i, t in enumerate(tags):
        if t.startswith("down:"):
            cur = [i]
        elif cur is not None and t.startswith("draw:"):
            cur.append(i)
        elif cur is not None and t.startswith("lift:"):
            bounds.append(cur); cur = None
    if cur is not None:
        bounds.append(cur)
    # 逐笔聚合: 一笔上取 75 分位宽度, 抵抗几像素的采样偏差
    stroke_w = np.array([np.percentile(width_mm[idx], 75) for idx in bounds])
    nz = stroke_w[stroke_w > 0]
    hit = int((stroke_w > 0).sum())
    print("逐笔采样: %d/%d 笔采到参考图墨线 (%.1f%%)" % (hit, len(bounds), 100.0*hit/max(1, len(bounds))))
    if nz.size:
        p = [np.percentile(nz, q) for q in (10, 25, 50, 75, 90)]
        print("每笔墨线宽度 mm: p10 %.3f p25 %.3f p50 %.3f p75 %.3f p90 %.3f max %.3f" % (
            p[0], p[1], p[2], p[3], p[4], nz.max()))
    lo_w, hi_w = (np.percentile(nz, [10, 90]) if nz.size else (0.0, 1.0))
    if hi_w - lo_w < 1e-6:
        hi_w = lo_w + 1e-6
    t = np.clip((stroke_w - lo_w)/(hi_w - lo_w), 0.0, 1.0)
    if a.mode == "rank":
        t = np.argsort(np.argsort(stroke_w)).astype(float)/max(1.0, len(stroke_w)-1.0)
    press = a.press_lo + (a.press_hi-a.press_lo)*t
    q = [np.percentile(press, x) for x in (10, 50, 90)]
    print("模式 %s: %d 笔, 下压 p10 %.4f p50 %.4f p90 %.4f mm (原: p10 0.072 p50 0.114 p90 0.291)" % (
        a.mode, len(bounds), q[0], q[1], q[2]))
    press_of = {}
    for idx, p_ in zip(bounds, press):
        for i in idx:
            press_of[i] = float(p_)

    new = json.loads(json.dumps(ink))
    k = 0
    for i, w in enumerate(new["waypoints"]):
        if sel[i]:
            p_ = float(press_of.get(i, a.press_lo))
            w["z"] = round(float(ArmConfig.PAPER_Z + p_), 4); k += 1
    new["meta"]["paper_z"] = float(ArmConfig.PAPER_Z)
    new["meta"]["ink_press_source"] = ("reference-image line width -> press %.3f~%.3f mm (%s)"
                                       % (a.press_lo, a.press_hi, a.mode))
    name = a.out_name or ("层峦松水_参考定粗细%s" % a.mode)
    out = src.parent/(name+"_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(exist_ok=False)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, out/f.name)
    ink2 = out/("%s_黑线任务.json" % name)
    plan2 = out/("%s_计划.json" % name)
    (out/ink_p.name).rename(ink2)
    (out/plan_p.name).rename(plan2)
    new["meta"]["image"] = str(out/"参考原图.png")
    ink2.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf8")
    p2 = json.loads(plan2.read_text(encoding="utf-8-sig"))
    p2["reference_image"] = str(out/"参考原图.png")
    plan2.write_text(json.dumps(p2, ensure_ascii=False, indent=1), encoding="utf8")
    print("输出目录:", out)
    print("  %d 个点被按参考图重新定粗细" % k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
