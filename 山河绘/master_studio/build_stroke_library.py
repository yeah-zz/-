# -*- coding: utf-8 -*-
"""把 StrokeField 的矢量化结果预计算成按画家存放的笔触库。"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from master_studio.stroke_field_infer import trace
from master_studio.train_stroke_field import TinyUNet, S
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "stroke_library"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = TinyUNet()
    ck = torch.load(ROOT / "data" / "stroke_field.pt", map_location="cpu")
    model.load_state_dict(ck["model"]); model.eval()
    for artist in ("huang_gongwang", "ni_zan", "shi_tao", "gong_xian"):
        rows = []
        for p in sorted((ROOT / "corpus" / artist / "patches").glob("*.png")):
            im = Image.open(p).convert("L").resize((S, S), Image.BILINEAR)
            g = np.asarray(im, dtype=np.float32) / 255.0
            with torch.no_grad():
                pred = model(torch.from_numpy(g)[None, None])[0]
            ink = torch.sigmoid(pred[0]).numpy()
            mask = (ink > 0.45).astype(np.uint8)
            sk = skeletonize(mask > 0)
            paths = trace(sk)
            paths.sort(key=len, reverse=True)
            strokes = []
            for path in paths[:40]:
                if len(path) < 6:
                    continue
                pts = [[round(x / S, 4), round(y / S, 4)] for x, y in path]
                # 过滤近乎水平/垂直的极短线；保留有方向的皴线
                xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
                if max(xs) - min(xs) < .03 and max(ys) - min(ys) < .03:
                    continue
                strokes.append(pts)
            if strokes:
                rows.append(dict(file=p.name, strokes=strokes))
        (OUT / (artist + ".json")).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(artist, len(rows), sum(len(r["strokes"]) for r in rows))


if __name__ == "__main__":
    main()