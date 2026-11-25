# -*- coding: utf-8 -*-
"""Role/region pressure census for the five colour paintings (read-only)."""
import json, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from web.catalog import ARTWORKS
OUT = ROOT / "artifacts/press_halves_20261013"
OUT.mkdir(parents=True, exist_ok=True)

def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def bucket(points, press, middle):
    p = np.asarray(points, float); v = np.asarray(press, float)
    if v.size == 1: v = np.full(len(p), float(v))
    m = p[:, 1] < middle
    return dict(up=dict(n=int(m.sum()), mean=round(float(v[m].mean()), 4) if m.any() else None,
                        mx=round(float(v[m].max()), 4) if m.any() else None),
                lo=dict(n=int((~m).sum()), mean=round(float(v[~m].mean()), 4) if (~m).any() else None,
                        mx=round(float(v[~m].max()), 4) if (~m).any() else None))

lines = []
report = {}
for n, (folder, name, _s, rel) in enumerate(ARTWORKS[:5], start=1):
    b = read(ROOT / rel)
    w, height = b["ink_job"]["meta"]["image_size"]
    middle = height / 2.0
    ink = b["ink_job"]["meta"]["ink_execution"]["base_strokes"]
    rep = {}
    for s in ink:
        key = (s.get("role", "?"), s.get("drawing_region", "?"))
        d = rep.setdefault(key, {"strokes": 0, "pts": 0, "up": 0, "lo": 0, "prs": [], "up_prs": [], "lo_prs": []})
        p = np.asarray(s["points"], float); v = np.asarray(s["press"], float)
        if v.size == 1: v = np.full(len(p), float(v))
        m = p[:, 1] < middle
        d["strokes"] += 1; d["pts"] += len(p)
        d["up"] += int(m.sum()); d["lo"] += int((~m).sum())
        d["prs"] += v.tolist(); d["up_prs"] += v[m].tolist(); d["lo_prs"] += v[~m].tolist()
    lines.append("== %02d %s  size=%s  mid=%.0f" % (n, name, (w, height), middle))
    for (role, region), d in sorted(rep.items(), key=lambda kv: -kv[1]["pts"]):
        mean = float(np.mean(d["prs"])); um = float(np.mean(d["up_prs"])) if d["up_prs"] else float("nan")
        lm = float(np.mean(d["lo_prs"])) if d["lo_prs"] else float("nan")
        lines.append("  %-22s %-14s 笔%3d 点%5d 上%5d 下%5d 平均%.3f 上%.3f 下%.3f" %
                     (role, region, d["strokes"], d["pts"], d["up"], d["lo"], mean, um, lm))
    report[n] = {("%s|%s" % k): v for k, v in rep.items()}
(OUT / "roles.txt").write_text("\n".join(lines), encoding="utf-8")
(OUT / "roles.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n".join(lines).encode("utf-8", "replace").decode("utf-8", "replace"))
