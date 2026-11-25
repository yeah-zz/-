# -*- coding: utf-8 -*-
"""How visible is an upper-half press cut? Pixel thickness of black ink."""
import json, sys, importlib.util, io
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2, numpy as np
from web.catalog import ARTWORKS
from ink_control import refresh_stats

spec = importlib.util.spec_from_file_location("bp", ROOT / "scripts/black_upper_press_20261015.py")
bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)

def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))

for n, (folder, name, _s, rel) in enumerate(ARTWORKS[:5], 1):
    bundle = read(ROOT / rel)
    scale = bundle["ink_job"]["meta"]["ink_execution"]["mm_per_px"]
    w, h = bundle["ink_job"]["meta"]["image_size"]
    mid = int(round(bp.content_divider(bundle)))
    print("%02d %s  scale %.4f mm/px  divider y=%d" % (n, name.encode("unicode_escape").decode(), scale, mid))
    for factor in (1.0, .95, .90, .85, .80, .70):
        b = bundle if factor == 1.0 else bp.relieve(bundle, factor)[0]
        strokes = b["ink_job"]["meta"]["ink_execution"]["base_strokes"]
        px, up_px = [], []
        for s in strokes:
            p = np.asarray(s["points"], float)
            t = max(1, round(s.get("width_mm", .3) / scale))
            px.append(t)
            if p[:, 1].mean() < mid:
                up_px.append(t)
        wide = sum(1 for t in up_px if t >= 7)
        print("   factor %.2f  black px-thickness upper: min %d mean %.2f max %d | strokes>=7px: %d/%d"
              % (factor, min(up_px), float(np.mean(up_px)), max(up_px), wide, len(up_px)))
