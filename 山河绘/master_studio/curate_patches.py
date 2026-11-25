# -*- coding: utf-8 -*-
"""生成人工清洗用的 patch 拼版；每个 patch 有编号，人工挑出干净山石。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
OUT = ROOT / "data" / "curation"
CLASSES = ["pima", "zhedai", "fupi", "midian"]
MAP = {"黄公望": "pima", "王蒙": "pima", "吴镇": "pima", "沈周": "pima", "文徵明": "pima",
       "倪瓒": "zhedai", "石涛": "fupi", "唐寅": "fupi", "仇英": "fupi", "龚贤": "midian"}
BASE = {"huang_gongwang": "pima", "ni_zan": "zhedai", "shi_tao": "fupi", "gong_xian": "midian"}


def quality(img):
    g = np.asarray(img.convert("L"), dtype=np.float32)
    dark = (g < 105).mean()
    edge = (np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 255.0
    gy, gx = np.gradient(g / 255.0)
    jxx = float(np.mean(gx * gx)); jyy = float(np.mean(gy * gy)); jxy = float(np.mean(gx * gy))
    coh = math.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-6)
    if not (.01 < dark < .30): return None
    if edge < .025 or edge > .13: return None
    if g.std() < 12: return None
    return coh * 2.0 + edge * 2.0 - abs(dark - .09) * 1.5


def collect(cls):
    rows = []
    # 已有标注 patch
    for aid, c in BASE.items():
        if c != cls: continue
        for p in sorted((CORPUS / aid / "patches").glob("*.png")):
            try:
                im = Image.open(p)
                q = quality(im)
                if q is not None:
                    rows.append(dict(path=str(p), artist=aid, score=q))
            except Exception: pass
    # ltfc patch
    for artist, c in MAP.items():
        if c != cls: continue
        for p in sorted((CORPUS / "ltfc" / artist / "patches").glob("*.png")):
            try:
                im = Image.open(p)
                q = quality(im)
                if q is not None:
                    rows.append(dict(path=str(p), artist=artist, score=q))
            except Exception: pass
    # 每个画家先均衡取，避免全是黄公望
    by = {}
    for r in rows: by.setdefault(r["artist"], []).append(r)
    for a in by: by[a].sort(key=lambda x: -x["score"])
    picked = []
    per = max(6, 240 // max(1, len(by)))
    for a, rs in by.items(): picked += rs[:per]
    # 不够再从剩余补
    rest = sorted([r for r in rows if r not in picked], key=lambda x: -x["score"])
    picked += rest[:max(0, 240 - len(picked))]
    picked.sort(key=lambda x: -x["score"])
    return picked[:240]


def font(size):
    for f in ("C:/Windows/Fonts/STXINGKA.TTF", "C:/Windows/Fonts/STKAITI.TTF", "C:/Windows/Fonts/arial.ttf"):
        try: return ImageFont.truetype(f, size)
        except Exception: pass
    return ImageFont.load_default()


def sheet(cls, rows, size=190, cols=5, per=25):
    d = OUT / cls; d.mkdir(parents=True, exist_ok=True)
    ft = font(18)
    for si in range(0, len(rows), per):
        chunk = rows[si:si + per]
        W = cols * (size + 10) + 10
        H = ((len(chunk) + cols - 1) // cols) * (size + 30) + 20
        canvas = Image.new("RGB", (W, H), (238, 232, 218))
        dr = ImageDraw.Draw(canvas)
        for k, r in enumerate(chunk):
            x = 10 + (k % cols) * (size + 10)
            y = 10 + (k // cols) * (size + 30)
            try:
                im = Image.open(r["path"]).convert("L").resize((size, size), Image.LANCZOS)
                canvas.paste(im.convert("RGB"), (x, y))
            except Exception:
                continue
            dr.rectangle([x, y, x + size, y + size], outline=(150, 145, 125))
            dr.text((x + 4, y + 2), str(si + k), font=font(20), fill=(160, 40, 30))
        canvas.save(d / ("sheet-%02d.jpg" % (si // per)), quality=92)
    (d / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(cls, len(rows), "sheets", (len(rows) + per - 1) // per)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", default="all")
    args = ap.parse_args()
    for c in (CLASSES if args.cls == "all" else [args.cls]):
        rows = collect(c)
        sheet(c, rows)