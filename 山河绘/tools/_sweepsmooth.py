# -*- coding: utf-8 -*-
"""
平滑强度扫描（跑完即删）—— 挑 PATH_SMOOTH_MM
====================================================================
只算统计 + 出预览，不动机械臂。
"""

# --- path bootstrap: this script now lives in 绘画/tools/, put 绘画/ on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import math

import numpy as np

from config import CanvasConfig, JobConfig, ShuimoConfig
from geometry import CanvasMap
import shuimo
import vision

# 原值 D:\ROBOT\DobotE6Calligraphy\lanhua.jpg 属于旧机器上的旧项目，文件已不存在；
# 换成项目内现有的细线范图。运行时 CWD 请保持项目根目录。
IMG = r"绘画\参考图\plum_blossom_reference.png"
VARIANTS = [0.0, 3.0, 6.0, 10.0, 16.0]


def angles_of(strokes):
    out = []
    for s in strokes:
        pts = np.asarray(s["points"], dtype=np.float64)
        for i in range(1, pts.shape[0] - 1):
            v1 = pts[i] - pts[i - 1]
            v2 = pts[i + 1] - pts[i]
            n1 = float(np.hypot(v1[0], v1[1]))
            n2 = float(np.hypot(v2[0], v2[1]))
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            c = float((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2))
            out.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    return np.asarray(out)


img = vision.load_image(IMG)
work, info = vision.prepare_image(img)
mapper = CanvasMap(info["prepared_size"], info["image_corners"])

print("=" * 92)
print(" PATH_SMOOTH_MM 扫描    (mm/px=%.4f  TRAJ_STEP=%.1f  cp=%d)"
      % (mapper.mm_per_px, ShuimoConfig.TRAJ_STEP_MM,
         ShuimoConfig.CONTINUOUS_PATH_RATIO))
print("=" * 92)
print("%-8s %8s %8s %8s %8s %8s %8s %8s %9s"
      % ("平滑mm", "顶点", "中位°", "均值°", "90分位°", "最大°", ">20°", ">30°", "总笔长mm"))
print("-" * 92)

saved = ShuimoConfig.PATH_SMOOTH_MM
rows = []
for v in VARIANTS:
    ShuimoConfig.PATH_SMOOTH_MM = v
    strokes, st = shuimo.build_framework_strokes(work, mapper.mm_per_px,
                                                 pad=info.get("pad"))
    a = angles_of(strokes)
    n20 = int((a > 20).sum())
    n30 = int((a > 30).sum())
    total = sum(float(s["length_mm"]) for s in strokes)
    rows.append((v, len(strokes), a, total))
    print("%-8.1f %8d %8.2f %8.2f %8.2f %8.2f %8d %8d %9.0f"
          % (v, a.size, np.median(a), a.mean(), np.percentile(a, 90), a.max(),
             n20, n30, total))

print("")
print("出预览（只看平滑强度的视觉差别）：")
for v in (0.0, 6.0, 16.0):
    ShuimoConfig.PATH_SMOOTH_MM = v
    strokes, st = shuimo.build_framework_strokes(work, mapper.mm_per_px,
                                                 pad=info.get("pad"))
    out = "_sm_%d.png" % int(v)
    shuimo.render_framework_preview(work, strokes, mapper.mm_per_px, out)
    print("   PATH_SMOOTH_MM=%-5.1f -> %s" % (v, out))

ShuimoConfig.PATH_SMOOTH_MM = saved
print("")
print("判读：中位角和 >20° 数量应该随平滑增强而下降；")
print("      但如果总笔长明显缩短，说明平滑把笔势吃掉了（过度平滑）。")
