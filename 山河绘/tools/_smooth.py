# -*- coding: utf-8 -*-
"""
笔触平滑度诊断（跑完即删）—— 找出"直来直去"的根源
====================================================================
对每条笔触的轨迹点算：
  · 相邻点间距（采样密度）
  · 每个顶点的【转折角】（方向变化）
  · 有多少个"硬拐角"（转折角 > 10°/20°/30°）
如果硬拐角很多，机械臂走起来就是一段段直线 + 突然拐弯 —— 正是"直来直去"。
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

img = vision.load_image(IMG)
work, info = vision.prepare_image(img)
mapper = CanvasMap(info["prepared_size"], info["image_corners"])
strokes, st = shuimo.build_framework_strokes(work, mapper.mm_per_px,
                                             pad=info.get("pad"))

print("=" * 78)
print(" 笔触平滑度诊断   共 %d 条笔触" % len(strokes))
print("=" * 78)
print("  参数：TRAJ_STEP_MM=%.1f  PATH_EPSILON_PX=%.2f  mm/px=%.4f"
      % (ShuimoConfig.TRAJ_STEP_MM, ShuimoConfig.PATH_EPSILON_PX, mapper.mm_per_px))
print("        CONTINUOUS_PATH_RATIO(cp)=%d   MOVE_DELAY=%.2fs"
      % (JobConfig.CONTINUOUS_PATH_RATIO, ArmConfig.MOVE_DELAY
         if (ArmConfig := __import__("config").ArmConfig) else 0))
print("")

all_angles = []
all_steps = []
hard10 = hard20 = hard30 = 0
total_v = 0

for idx, s in enumerate(strokes):
    pts = np.asarray(s["points"], dtype=np.float64)
    if pts.shape[0] < 3:
        continue
    d = np.diff(pts, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    seg = seg[seg > 1e-9]
    if seg.size:
        all_steps += seg.tolist()
    # 每个顶点的转折角
    for i in range(1, pts.shape[0] - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1 = float(np.hypot(v1[0], v1[1]))
        n2 = float(np.hypot(v2[0], v2[1]))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cosv = float((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, cosv))))
        all_angles.append(ang)
        total_v += 1
        if ang > 10:
            hard10 += 1
        if ang > 20:
            hard20 += 1
        if ang > 30:
            hard30 += 1

a = np.asarray(all_angles)
sp = np.asarray(all_steps)
print("  顶点总数 %d" % total_v)
if a.size:
    print("  转折角: 中位 %.2f°  均值 %.2f°  90分位 %.2f°  最大 %.2f°"
          % (np.median(a), a.mean(), np.percentile(a, 90), a.max()))
    print("  硬拐角: >10° 有 %d 个 (%.0f%%)   >20° 有 %d 个 (%.0f%%)   >30° 有 %d 个 (%.0f%%)"
          % (hard10, 100.0 * hard10 / total_v, hard20, 100.0 * hard20 / total_v,
             hard30, 100.0 * hard30 / total_v))
if sp.size:
    print("  相邻点间距: 中位 %.3f mm  均值 %.3f mm  最大 %.3f mm"
          % (np.median(sp), sp.mean(), sp.max()))

print("")
print("=" * 78)
print(" 前 3 条笔触的转折角序列（°）")
print("=" * 78)
for idx, s in enumerate(strokes[:3]):
    pts = np.asarray(s["points"], dtype=np.float64)
    angles = []
    for i in range(1, pts.shape[0] - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1 = float(np.hypot(v1[0], v1[1]))
        n2 = float(np.hypot(v2[0], v2[1]))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cosv = float((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2))
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosv)))))
    print("  第 %d 条 (%d 点, %.0f mm):" % (idx + 1, len(angles) + 2, s["length_mm"]))
    print("     " + " ".join("%.0f" % x for x in angles[:40]))

print("")
print("=" * 78)
print(" 判读")
print("=" * 78)
print("""
  · 转折角中位数如果只有 1~3°  -> 路径本身是【平滑】的，问题在【运动执行】
      原因：cp（平滑过渡比例）太小，每个顶点都"点对点"停一下再走
      修法：把 JobConfig.CONTINUOUS_PATH_RATIO 调大（水墨建议 60~80）

  · 转折角里如果有一堆 >15° 的硬拐角 -> 路径本身是【折线】
      原因：骨架简化 + 弧长线性插值，折线拐角被保留下来
      修法：加一道【路径平滑】（移动平均 / 样条拟合）

  · 两个都可能有 -> 两处都要改
""")
