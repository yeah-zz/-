# -*- coding: utf-8 -*-
"""清洗版 ltfc patch：排除印章红、树冠黑团、题字和过低对比区域。"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent / "corpus" / "ltfc"
OUT = Path(__file__).resolve().parent / "corpus" / "ltfc_clean"
P = 160
MAP = {"黄公望": "pima", "王蒙": "pima", "吴镇": "pima", "沈周": "pima", "文徵明": "pima",
       "倪瓒": "zhedai", "石涛": "fupi", "唐寅": "fupi", "仇英": "fupi", "龚贤": "midian"}


def stats(rgb):
    g = np.asarray(rgb.convert("L"), dtype=np.float32)
    r, gg, b = [np.asarray(rgb.getchannel(c), dtype=np.float32) for c in ("R", "G", "B")]
    red = ((r > 120) & (r > gg * 1.35) & (r > b * 1.35)).mean()
    dark = (g < 105).mean()
    edge = (np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 255.0
    gy, gx = np.gradient(g / 255.0)
    jxx = float(np.mean(gx * gx)); jyy = float(np.mean(gy * gy)); jxy = float(np.mean(gx * gy))
    coh = math.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-6)
    return g, red, dark, edge, coh


def main():
    rows = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or d.name not in MAP: continue
        artist = d.name
        pd = OUT / artist / "patches"; pd.mkdir(parents=True, exist_ok=True)
        for fp in sorted(d.glob("*.jpg")):
            try:
                im = Image.open(fp).convert("RGB")
                w, h = im.size
                if min(w, h) < P * 1.5: continue
                cands = []
                for yy in range(int(h * .14), int(h * .72) - P, P // 2):
                    for xx in range(int(w * .05), w - P, P // 2):
                        crop = im.crop((xx, yy, xx + P, yy + P))
                        g, red, dark, edge, coh = stats(crop)
                        if red > .003: continue
                        if not (.015 < dark < .24): continue
                        if edge < .035 or edge > .11: continue
                        if coh < .08: continue
                        # 墨色层次：不能是一块死黑或几乎空白
                        if g.std() < 18: continue
                        score = coh * 2.2 + edge * 3.0 - abs(dark - .09) * 2.0
                        cands.append((score, xx, yy))
                cands.sort(reverse=True)
                picked = []
                for s, xx, yy in cands:
                    if len(picked) >= 8: break
                    if any(abs(xx - a) < P * .55 and abs(yy - b) < P * .55 for a, b in picked): continue
                    picked.append((xx, yy))
                for k, (xx, yy) in enumerate(picked):
                    name = "%s_%02d.png" % (fp.stem, k)
                    im.crop((xx, yy, xx + P, yy + P)).convert("L").save(pd / name)
                    rows.append(dict(artist=artist, file=name, cls=MAP[artist]))
            except Exception:
                continue
    (OUT / "patches.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    print("clean patches", len(rows), Counter(r["cls"] for r in rows))


if __name__ == "__main__":
    main()