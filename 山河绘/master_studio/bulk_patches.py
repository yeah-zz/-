# -*- coding: utf-8 -*-
"""从 bulk 图库裁 160×160 patch，供自监督编码器预训练。"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

BULK = Path(__file__).resolve().parent / "corpus" / "bulk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-image", type=int, default=2)
    ap.add_argument("--patch", type=int, default=160)
    args = ap.parse_args()
    out = BULK / "patches"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((BULK / "images").glob("*.jpg")):
        try:
            im = Image.open(p).convert("L")
            w, h = im.size
            if min(w, h) < args.patch:
                continue
            # 中上部，避开题款/前景
            for k in range(args.per_image):
                x = int(w * (0.15 + 0.6 * (k / max(1, args.per_image - 1))))
                y = int(h * 0.28 + (h - args.patch) * 0.25 * (k / max(1, args.per_image - 1)))
                x = max(0, min(w - args.patch, x))
                y = max(0, min(h - args.patch, y))
                crop = im.crop((x, y, x + args.patch, y + args.patch))
                if np.asarray(crop).std() < 6:
                    continue
                crop.save(out / ("%s_%02d.png" % (p.stem, k)))
                n += 1
        except Exception:
            continue
    print("bulk patches", n)


if __name__ == "__main__":
    main()