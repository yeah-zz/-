# -*- coding: utf-8 -*-
"""
两张 AI 生成图对比（跑完即删）
====================================================================
对每张图跑水墨提取，报告：
  墨迹占比 / 路径数 / 笔触数 / 平均笔长 / 笔压区间
并算出"笔触是否够长、够不够分得开"的判断指标。
"""

# --- path bootstrap: this script now lives in 绘画/tools/, put 绘画/ on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np

from config import ShuimoConfig
from geometry import CanvasMap
import shuimo
import vision

IMAGES = [
    ("rock_01（墨块）", r"samples\ai\rock_01.png"),
    ("rock_02（笔触）", r"samples\ai\rock_02.png"),
]

print("=" * 88)
print(" 两张图对比")
print("=" * 88)
rows = []
for name, path in IMAGES:
    img = vision.load_image(path)
    work, info = vision.prepare_image(img)
    mapper = CanvasMap(info["prepared_size"], info["image_corners"])
    strokes, st = shuimo.build_framework_strokes(work, mapper.mm_per_px,
                                                 pad=info.get("pad"))
    if not strokes:
        print("  %s: 提取不到笔触" % name)
        continue
    lens = [float(s["length_mm"]) for s in strokes]
    avg = sum(lens) / len(lens)
    # 笔触长度的中位数：判断"是不是一堆碎笔"
    med = float(np.median(lens))
    short = sum(1 for x in lens if x < 10.0)
    rows.append((name, st["ink_ratio"] * 100, st["paths"], st["strokes"],
                 sum(lens), avg, med, short, st["bridge_ok"]))
    out = "_cmp_%s.png" % path.split("\\")[-1].replace(".png", "")
    shuimo.render_framework_preview(work, strokes, mapper.mm_per_px, out)
    print("  %-16s -> 预览 %s" % (name, out))

print("")
print("%-18s %7s %6s %6s %9s %9s %9s %7s %6s"
      % ("图", "墨迹%", "路径", "笔触", "总笔长", "平均笔长", "中位笔长", "<10mm", "桥接"))
print("-" * 92)
for r in rows:
    print("%-18s %6.1f%% %6d %6d %9.0f %9.1f %9.1f %7d %6s"
          % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], "开" if r[8] else "关"))

print("")
print("=" * 88)
print(" 判读标准")
print("=" * 88)
print("""
  墨迹占比   5~15%  好      15~30%  偏高但可用      >30%  会塌
  平均笔长   > 25mm 好      < 15mm  偏碎（机械臂会频繁抬笔）
  中位笔长   接近平均值 说明笔触长度均匀；远小于平均值说明有一堆短碎笔
  <10mm 的笔触数量   越少越好（那些画出来就是一个个小点）
""")
