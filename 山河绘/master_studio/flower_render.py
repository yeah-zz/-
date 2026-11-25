# -*- coding: utf-8 -*-
"""独立花鸟小品：当前为程序化花鸟稿，后续可换成花鸟模型。"""
import math
import random
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

from master_studio import brush

OUT = Path(__file__).resolve().parent / "output" / "flowers"
KINDS = {"plum": "梅花", "lotus": "荷花", "orchid": "兰花", "chrysanthemum": "菊花"}
COLORS = {"plum": (168, 52, 62), "lotus": (196, 112, 126), "orchid": (96, 122, 88), "chrysanthemum": (196, 148, 52)}


def render_flower(kind="plum", seed=0, width=700):
    kind = kind if kind in KINDS else "plum"
    rng = random.Random(seed or 0)
    W = H = width
    img = brush.paper_base(W, H, rng).convert("RGBA")
    dr = ImageDraw.Draw(img)
    ink = (36, 42, 36, 235)
    color = (*COLORS[kind], 210)
    # 枝干
    for k in range(3):
        x0 = W * (0.25 + 0.15 * k); y0 = H * 0.88
        x1 = W * (0.35 + 0.20 * k); y1 = H * (0.28 + 0.08 * k)
        for j in range(18):
            t = j / 18
            x = x0 + (x1 - x0) * t + math.sin(t * 4 + k) * 8
            y = y0 + (y1 - y0) * t
            dr.line([(x, y), (x + 2, y - 1)], fill=ink, width=max(2, int(7 * (1 - t * .6))))
        # 小枝
        for j in range(4):
            t = .25 + .15 * j
            bx = x0 + (x1 - x0) * t; by = y0 + (y1 - y0) * t
            dr.line([(bx, by), (bx + rng.uniform(-70, 70), by - rng.uniform(40, 90))], fill=ink, width=2)
    # 花/叶
    for k in range(16 if kind != "orchid" else 8):
        cx = W * rng.uniform(.25, .78); cy = H * rng.uniform(.20, .62)
        if kind == "lotus":
            for q in range(6):
                a = q * math.pi / 3
                dr.ellipse([cx + math.cos(a) * 10 - 14, cy + math.sin(a) * 7 - 20,
                            cx + math.cos(a) * 10 + 14, cy + math.sin(a) * 7 + 20], fill=color, outline=None)
            dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(210, 180, 70, 220))
        elif kind == "chrysanthemum":
            for q in range(14):
                a = q * math.pi / 7
                dr.ellipse([cx + math.cos(a) * 16 - 5, cy + math.sin(a) * 16 - 12,
                            cx + math.cos(a) * 16 + 5, cy + math.sin(a) * 16 + 12], fill=color)
            dr.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(120, 90, 35, 230))
        elif kind == "orchid":
            dr.line([(cx, cy + 40), (cx + rng.uniform(-40, 40), cy - 40)], fill=(70, 105, 70, 220), width=3)
            dr.ellipse([cx - 5, cy - 6, cx + 5, cy + 6], fill=color)
        else:
            for q in range(5):
                a = q * math.pi / 2.5
                dr.ellipse([cx + math.cos(a) * 7 - 6, cy + math.sin(a) * 7 - 6,
                            cx + math.cos(a) * 7 + 6, cy + math.sin(a) * 7 + 6], fill=color)
            dr.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(240, 220, 150, 230))
    # 题字
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(r"C:\Windows\Fonts\STXINGKA.TTF", int(W * .045))
        dr.text((W * .06, H * .06), "花鸟", font=font, fill=ink)
        dr.text((W * .06, H * .12), KINDS[kind], font=ImageFont.truetype(r"C:\Windows\Fonts\STKAITI.TTF", int(W * .028)),
                fill=(80, 85, 72, 220))
    except Exception:
        pass
    img = brush.add_paper_grain(img.convert("RGB"), rng, .05)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "flower-%s-%s.png" % (kind, uuid.uuid4().hex[:8])
    path = OUT / name
    img.save(path, quality=94)
    return path, dict(kind=kind, kind_name=KINDS[kind], url="/flower-preview/" + name)