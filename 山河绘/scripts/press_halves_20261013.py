# -*- coding: utf-8 -*-
"""Read-only: how much brush pressure each painting's upper half carries."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from web.catalog import ARTWORKS

OUT = ROOT / "artifacts/press_halves_20261013"

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def stats(points, press, middle):
    p = np.asarray(points, dtype=float)
    v = np.asarray(press, dtype=float)
    if p.ndim != 2 or len(p) == 0:
        return None
    if v.size == 1:
        v = np.full(len(p), float(v))
    if v.size != len(p):
        v = np.full(len(p), float(np.mean(v)))
    m = p[:, 1] < middle
    out = {}
    for label, mask in (("upper", m), ("lower", ~m)):
        if not mask.any():
            out[label] = None
            continue
        out[label] = dict(pts=int(mask.sum()),
                          press_min=round(float(v[mask].min()), 4),
                          press_max=round(float(v[mask].max()), 4),
                          press_mean=round(float(v[mask].mean()), 4))
    return out

report = {}
for n, (folder, name, _sub, rel) in enumerate(ARTWORKS, start=1):
    b = read(ROOT / rel)
    w, height = b["ink_job"]["meta"]["image_size"]
    middle = height / 2.0
    module = b["workflow"]["modules"]["color"]
    scale = module["mm_per_px"]
    entry = dict(name=name, size=[w, height], mm_per_px=scale,
                 canvas_mm=[round(w * scale, 1), round(height * scale, 1)],
                 color_motion_candidate=module["motion_candidate"].get("press_mm"),
                 color_geometry={k: v for k, v in b["workflow"].get("color_geometry", {}).items()
                                 if "press" in k or "pitch" in k})
    groups = {}
    for s in module["strokes"]:
        groups.setdefault(s.get("color", "?"), []).append(s)
    for color, group in sorted(groups.items()):
        pts = [q for s in group for q in s["points"]]
        prs = [p for s in group for p in (s["press"] if isinstance(s["press"], list) else [s["press"]] * len(s["points"]))]
        entry["color_" + color] = stats(pts, prs, middle)
    ink = b["ink_job"]["meta"]["ink_execution"]["base_strokes"]
    pts = [q for s in ink for q in s["points"]]
    prs = [p for s in ink for p in (s["press"] if isinstance(s["press"], list) else [s["press"]] * len(s["points"]))]
    entry["ink"] = stats(pts, prs, middle)
    report["%02d_%s" % (n, name)] = entry

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
