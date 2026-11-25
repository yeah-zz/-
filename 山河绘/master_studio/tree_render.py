# -*- coding: utf-8 -*-
"""独立树法生成：选择松/杂树/枯枝，生成单棵树稿。"""
import copy
import uuid
from pathlib import Path

from PIL import Image

from master_studio import brush, profiles, render

OUT = Path(__file__).resolve().parent / "output" / "trees"
KINDS = {"pine": "松树", "broadleaf": "杂树", "bare": "枯枝"}


def render_tree(kind="pine", artist="huang_gongwang", seed=0, width=700):
    kind = kind if kind in KINDS else "pine"
    profile = copy.deepcopy(profiles.get_artist(artist))
    profile["inscription"]["text"] = "树法"
    profile["inscription"]["sub"] = KINDS[kind]
    profile["inscription"]["position"] = "top_left"
    profile["inscription"]["vertical"] = True
    layout = {
        "ratio": (1, 1), "mountains": [], "water": 0.94,
        "trees": [dict(x=0.52, base=0.90, height=0.62, depth="near", kind=kind, size=1.0)],
    }
    r = render.MasterRenderer(profile, layout, seed=seed, width=width, include_trees=True)
    r.image = brush.paper_base(r.W, r.H, r.rng)
    r._draw_tree(layout["trees"][0], "near")
    # 地平线/坡脚的一两笔
    r._add([(r.W * .18, r.H * .90), (r.W * .82, r.H * .91)], profile["line"]["width"] * .8,
           profile["line"]["alpha"] * .55, "rock", "near", dry=.4, taper=.4, jitter=.5)
    r.image = r._composite_group("near")
    # 树冠最终压一层墨团（直接压到成图，避免被分组合成吃掉）
    if kind != "bare":
        from PIL import ImageDraw
        ov = Image.new("RGBA", (r.W, r.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        cx, cy = r.W * .52, r.H * .90
        for dx, dy, rx, ry in ((-.13, .38, .20, .11), (.12, .40, .22, .12), (0, .45, .15, .09)):
            d.ellipse([cx + (dx - rx) * r.H, cy - (dy + ry) * r.H,
                       cx + (dx + rx) * r.H, cy - (dy - ry) * r.H],
                      fill=(*profile["palette"]["ink"], 76))
        r.image = Image.alpha_composite(r.image.convert("RGBA"), ov).convert("RGB")
    r.image = brush.add_paper_grain(r.image, r.rng, 0.06)
    r.image = r._draw_inscription(r.image)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "tree-%s-%s.png" % (kind, uuid.uuid4().hex[:8])
    path = OUT / name
    r.image.save(path, quality=94)
    return path, dict(kind=kind, kind_name=KINDS[kind], strokes=len(r.plan), url="/tree-preview/" + name)