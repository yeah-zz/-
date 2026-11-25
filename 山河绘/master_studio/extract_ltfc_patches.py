# -*- coding: utf-8 -*-
"""把中华珍宝馆作品裁成 160×160 山石 patch，并按画家映射四类皴法。"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent / "corpus" / "ltfc"
MAP = {"黄公望": "pima", "王蒙": "pima", "吴镇": "pima", "沈周": "pima", "文徵明": "pima",
       "倪瓒": "zhedai", "石涛": "fupi", "唐寅": "fupi", "仇英": "fupi", "龚贤": "midian",
       "八大山人": "none", "朱耷": "none"}
P = 160


def pick(g):
    # 8 个中上部候选，按边缘/墨色丰富度选，避开题款和前景
    h, w = g.shape
    best = []
    for yy in range(int(h * .12), int(h * .70) - P, max(40, P // 2)):
        for xx in range(int(w * .04), w - P, max(40, P // 2)):
            c = g[yy:yy + P, xx:xx + P]
            dark = (c < 110).mean(); edge = (np.abs(np.diff(c, axis=0)).mean() + np.abs(np.diff(c, axis=1)).mean())
            if .02 < dark < .35 and edge > 8:
                best.append((edge + dark * 60, xx, yy))
    best.sort(reverse=True)
    out = []
    for s, x, y in best:
        if len(out) >= 8: break
        if any(abs(x - a) < P * .5 and abs(y - b) < P * .5 for a, b in out): continue
        out.append((x, y))
    return out


def main():
    rows = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir(): continue
        artist = d.name
        pd = d / "patches"; pd.mkdir(exist_ok=True)
        for fp in sorted(d.glob("*.jpg")):
            try:
                im = Image.open(fp).convert("L")
                w, h = im.size
                if min(w, h) < P * 1.5: continue
                g = np.asarray(im, dtype=np.float32)
                for k, (x, y) in enumerate(pick(g)):
                    im.crop((x, y, x + P, y + P)).save(pd / ("%s_%02d.png" % (fp.stem, k)))
                    rows.append(dict(artist=artist, file="%s_%02d.png" % (fp.stem, k), cls=MAP.get(artist, "none")))
            except Exception:
                continue
    (ROOT / "patches.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ltfc patches", len(rows))
    from collections import Counter
    print(Counter(r["cls"] for r in rows))
    print(Counter(r["artist"] for r in rows))


if __name__ == "__main__":
    main()