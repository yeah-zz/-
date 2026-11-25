# -*- coding: utf-8 -*-
"""从已筛出的 300 幅 ltfc 山水作品裁干净 patch，排除印章/题字/死黑。"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "corpus" / "ltfc_landscape"
P = 160
MAP = {"黄公望": "pima", "王蒙": "pima", "吴镇": "pima", "沈周": "pima", "文徵明": "pima",
       "倪瓒": "zhedai", "石涛": "fupi", "唐寅": "fupi", "仇英": "fupi", "龚贤": "midian"}


def cls_of(author):
    for k, v in MAP.items():
        if k in author: return v
    return "none"


def score(rgb):
    g = np.asarray(rgb.convert("L"), dtype=np.float32)
    r, gg, b = [np.asarray(rgb.getchannel(c), dtype=np.float32) for c in ("R", "G", "B")]
    red = ((r > 120) & (r > gg * 1.35) & (r > b * 1.35)).mean()
    dark = (g < 105).mean()
    edge = (np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 255.0
    gy, gx = np.gradient(g / 255.0)
    jxx = float(np.mean(gx * gx)); jyy = float(np.mean(gy * gy)); jxy = float(np.mean(gx * gy))
    coh = math.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-6)
    if red > .003 or not (.012 < dark < .26) or edge < .028 or edge > .115 or coh < .05 or g.std() < 15:
        return None
    return coh * 2.2 + edge * 2.5 - abs(dark - .08) * 2.0


def main():
    rows = []
    for d in sorted(BASE.iterdir()):
        if not d.is_dir(): continue
        artist = d.name
        cls = cls_of(artist)
        pd = d / "patches"; pd.mkdir(exist_ok=True)
        for fp in sorted(d.glob("*.jpg")):
            try:
                im = Image.open(fp).convert("RGB")
                w, h = im.size
                if min(w, h) < P * 1.5: continue
                cands = []
                for yy in range(int(h * .14), int(h * .72) - P, P // 2):
                    for xx in range(int(w * .05), w - P, P // 2):
                        c = im.crop((xx, yy, xx + P, yy + P))
                        s = score(c)
                        if s is not None: cands.append((s, xx, yy))
                cands.sort(reverse=True)
                picked = []
                for s, xx, yy in cands:
                    if len(picked) >= 8: break
                    if any(abs(xx - a) < P * .55 and abs(yy - b) < P * .55 for a, b in picked): continue
                    picked.append((xx, yy))
                for k, (xx, yy) in enumerate(picked):
                    name = "%s_%02d.png" % (fp.stem, k)
                    im.crop((xx, yy, xx + P, yy + P)).convert("L").save(pd / name)
                    rows.append(dict(artist=artist, file=name, cls=cls))
            except Exception:
                continue
    (BASE / "patches.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    print("landscape patches", len(rows), Counter(r["cls"] for r in rows))


if __name__ == "__main__":
    main()