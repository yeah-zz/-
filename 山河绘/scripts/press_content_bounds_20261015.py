# -*- coding: utf-8 -*-
"""Upper-half black-ink pressure census: where the ink actually sits per painting."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from web.catalog import ARTWORKS

def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))

for n, (folder, name, _s, rel) in enumerate(ARTWORKS, 1):
    b = read(ROOT / rel)
    w, h = b["ink_job"]["meta"]["image_size"]
    scale = b["ink_job"]["meta"]["ink_execution"]["mm_per_px"]
    pts, prs = [], []
    for s in b["ink_job"]["meta"]["ink_execution"]["base_strokes"]:
        p = np.asarray(s["points"], float); v = np.asarray(s["press"], float)
        if v.size == 1: v = np.full(len(p), float(v))
        pts.append(p); prs.append(v)
    pts = np.concatenate(pts); prs = np.concatenate(prs)
    for s in b["workflow"]["modules"]["color"]["strokes"]:
        p = np.asarray(s["points"], float)
        pts = np.concatenate([pts, p])
    y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
    x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    content_mid = (y0 + y1) / 2.
    canvas_mid = h / 2.
    print("%02d %s  canvas %dx%d (%.0f x %.0f mm)" % (n, name.encode("unicode_escape").decode(), w, h, w*scale, h*scale))
    print("    content y %.0f..%.0f  x %.0f..%.0f  -> content mid %.0f (%.0f mm from top), canvas mid %.0f" %
          (y0, y1, x0, x1, content_mid, content_mid*scale, canvas_mid))
    for label, mid in (("canvas mid", canvas_mid), ("content mid", content_mid)):
        m = pts[:, 1] < mid
        print("    %-11s upper points %5d / %5d" % (label, int(m.sum()), len(pts)))
