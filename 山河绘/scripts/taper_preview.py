"""把计划按【逐点笔压】画出来 —— 看"一笔之内有没有粗细"。

outline_pass.draw 用的是 cv2.polylines，只能画等宽线，所以梅花预览图里
每条笔触都是常宽的。真正的执行轨迹里 press 是【逐点】数组，Z = PAPER_Z + press
逐点变化（见 config.z_for_press）。本脚本用 shuimo._brush_footprint 的同一套
办法（逐段变宽 + 圆头补点）把逐点宽度画出来，并把几条代表笔触单独放大，
标出起笔 / 中段 / 收笔的笔压与对应笔宽。

    python scripts/taper_preview.py            # 出图到 输出/taper_demo_<时间戳>.png
                                                #     和 输出/taper_strokes_<时间戳>.png

离线，不连机械臂、不发任何指令。
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))

from datetime import datetime

import cv2
import numpy as np

import shuimo
from plum_blossom import build_plan
from project_paths import OUTPUT_DIR, PLUM_REFERENCE_IMAGE

PX_MM = 6.0          # 全图渲染分辨率（像素/毫米）
ZOOM_PX_MM = 24.0    # 单笔放大图分辨率

INK = (40, 40, 40)
RED = (70, 70, 220)


def dense(points_px, presses, scale, step_mm=0.2):
    """把折线按弧长重采样到 step_mm 间隔，并同步插值笔压。"""
    p = np.asarray(points_px, float)
    if len(p) < 2:
        return p, np.asarray(presses, float)
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] * scale
    if d[-1] <= 0:
        return p, np.asarray(presses, float)
    n = max(2, int(d[-1] / step_mm) + 1)
    t = np.linspace(0.0, d[-1], n)
    return (np.column_stack([np.interp(t, d, p[:, 0]),
                             np.interp(t, d, p[:, 1])]),
            np.interp(t, d, np.asarray(presses, float)))


def footprint(canvas, pts_px, presses, color, px_mm):
    """逐段定宽 + 圆头：一笔之内画出粗细变化。"""
    widths = np.array([max(0.15, shuimo.press_to_width_mm(v)) for v in presses]) * px_mm
    for i in range(len(pts_px) - 1):
        x0, y0 = int(round(pts_px[i][0])), int(round(pts_px[i][1]))
        x1, y1 = int(round(pts_px[i + 1][0])), int(round(pts_px[i + 1][1]))
        th = int(max(1, round(float(widths[i] + widths[i + 1]) * 0.5)))
        cv2.line(canvas, (x0, y0), (x1, y1), color, th, cv2.LINE_8)
        cv2.circle(canvas, (x1, y1), max(1, th // 2), color, -1, cv2.LINE_8)


def render_full(plan, scale):
    """整幅：黑枝 + 红瓣，全部按逐点宽度画。"""
    w_mm = plan['image_size'][0] * scale
    h_mm = plan['image_size'][1] * scale
    canvas = np.full((int(h_mm * PX_MM), int(w_mm * PX_MM), 3), 255, np.uint8)
    for key, color in (('outline', INK), ('color', RED)):
        for s in plan['modules'][key]['strokes']:
            pts, pr = dense(s['points'], s['press'], scale)
            footprint(canvas, pts * scale * PX_MM, pr, color, PX_MM)
    return canvas


def render_stroke(s, scale, color):
    """一条笔触单独放大，并返回它的起笔/中段/收笔统计。"""
    pts, pr = dense(s['points'], s['press'], scale)
    w_mm = np.array([shuimo.press_to_width_mm(v) for v in pr])
    n = len(pr)
    stats = dict(press=(float(pr[0]), float(pr[n // 2]), float(pr[-1])),
                 width=(float(w_mm[0]), float(w_mm[n // 2]), float(w_mm[-1])))

    pad_mm = float(w_mm.max()) + 1.5
    pts_mm = pts * scale
    lo = pts_mm.min(axis=0) - pad_mm
    hi = pts_mm.max(axis=0) + pad_mm
    size_mm = np.maximum(hi - lo, 1.0)
    pw = max(60, int(size_mm[0] * ZOOM_PX_MM))
    ph = max(60, int(size_mm[1] * ZOOM_PX_MM))
    canvas = np.full((ph + 70, pw, 3), 255, np.uint8)      # 顶部留 70px 写标注
    footprint(canvas, (pts_mm - lo) * ZOOM_PX_MM + np.array([0.0, 70.0]),
              pr, color, ZOOM_PX_MM)
    cv2.putText(canvas, s['id'], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'len %.0f mm' % float(s.get('length_mm', 0.0)), (6, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'press %.2f > %.2f > %.2f' % stats['press'], (6, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'width %.2f > %.2f > %.2f mm' % stats['width'], (6, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas, stats


def main():
    plan, _work = build_plan(PLUM_REFERENCE_IMAGE)
    scale = float(plan['mm_per_px'])
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    full = render_full(plan, scale)
    out_full = OUTPUT_DIR / ('taper_demo_%s.png' % stamp)
    cv2.imencode('.png', full)[1].tofile(str(out_full))

    ink = sorted(plan['modules']['outline']['strokes'],
                 key=lambda s: -float(s.get('length_mm', 0.0)))
    red = sorted(plan['modules']['color']['strokes'],
                 key=lambda s: -float(s.get('length_mm', 0.0)))
    # 三类笔触各取一条做对照：骨架中线 / 铺毫扫描线 / 花瓣
    skeleton = [s for s in ink if not s.get('source', '').endswith('massfill')]
    massfill = [s for s in ink if s.get('source', '').endswith('massfill')]
    picks = []
    if skeleton:
        picks.append(('trunk', skeleton[0], INK))
    if len(skeleton) > 4:
        picks.append(('branch', skeleton[4], INK))
    if massfill:
        picks.append(('massfill', max(massfill, key=lambda s: float(s.get('length_mm', 0.0))), (120, 120, 120)))
    if red:
        picks.append(('petal', red[0], RED))

    print('%-7s %-16s %7s %7s %7s   %7s %7s %7s' % (
        '图', 'id', '起笔压', '中段压', '收笔压', '起笔宽', '中段宽', '收笔宽'))
    panels = []
    for tag, s, color in picks:
        canvas, st = render_stroke(s, scale, color)
        panels.append(canvas)
        print('%-7s %-16s %7.2f %7.2f %7.2f   %7.2f %7.2f %7.2f' % (
            tag, s['id'], st['press'][0], st['press'][1], st['press'][2],
            st['width'][0], st['width'][1], st['width'][2]))

    hh = max(p.shape[0] for p in panels)
    total_w = sum(p.shape[1] for p in panels) + 12 * (len(panels) + 1)
    row = np.full((hh + 16, total_w, 3), 255, np.uint8)
    x = 12
    for p in panels:
        row[8:8 + p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + 12
    out_zoom = OUTPUT_DIR / ('taper_strokes_%s.png' % stamp)
    cv2.imencode('.png', row)[1].tofile(str(out_zoom))

    print()
    print('全图（逐点宽度）:', out_full)
    print('单笔放大对照  :', out_zoom)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
