# -*- coding: utf-8 -*-
"""把 StrokeField 输出的墨线场矢量化成笔触序列（自研 image-to-stroke 原型）。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.morphology import skeletonize

from master_studio.train_stroke_field import TinyUNet, S


def trace(skel):
    h, w = skel.shape
    pix = {(x, y) for y, x in zip(*np.nonzero(skel))}
    nbrs = lambda p: [(p[0] + dx, p[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      if (dx or dy) and (p[0] + dx, p[1] + dy) in pix]
    deg = {p: len(nbrs(p)) for p in pix}
    endpoints = [p for p in pix if deg[p] == 1]
    junctions = [p for p in pix if deg[p] >= 3]
    used = set()
    paths = []

    def walk(a, b):
        path = [a, b]
        used.add((a, b)); used.add((b, a))
        prev, cur = a, b
        while True:
            if cur in junctions or deg.get(cur, 0) == 1:
                break
            nx = [q for q in nbrs(cur) if q != prev and (cur, q) not in used]
            if not nx:
                break
            q = nx[0]
            used.add((cur, q)); used.add((q, cur))
            path.append(q)
            prev, cur = cur, q
        return path

    starts = endpoints + junctions
    for s in starts:
        for q in nbrs(s):
            if (s, q) not in used:
                p = walk(s, q)
                if len(p) >= 4:
                    paths.append(p)
    # 剩下的小环/孤立段
    for s in list(pix):
        for q in nbrs(s):
            if (s, q) not in used:
                p = walk(s, q)
                if len(p) >= 4:
                    paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "stroke_field_vector.png"))
    args = ap.parse_args()
    model = TinyUNet()
    ck = torch.load(Path(__file__).parent / "data" / "stroke_field.pt", map_location="cpu")
    model.load_state_dict(ck["model"]); model.eval()
    im = Image.open(args.patch).convert("L").resize((S, S), Image.BILINEAR)
    g = np.asarray(im, dtype=np.float32) / 255.0
    with torch.no_grad():
        pred = model(torch.from_numpy(g)[None, None])[0]
    ink = torch.sigmoid(pred[0]).numpy()
    mask = (ink > 0.42).astype(np.uint8)
    sk = skeletonize(mask > 0)
    paths = trace(sk)
    # 笔画按长度排序
    paths.sort(key=len, reverse=True)
    canvas = np.full((S * 4, S * 4, 3), 245, np.uint8)
    strokes = []
    for p in paths[:80]:
        pts = np.array([[x, y] for x, y in p], np.float32) * 4
        # 简化
        approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), 1.2, False).reshape(-1, 2)
        if len(approx) < 2:
            continue
        cv2.polylines(canvas, [approx.astype(np.int32)], False, (35, 38, 34), 3, cv2.LINE_AA)
        strokes.append({"points": (approx / 4.0 / S).round(4).tolist()})
    cv2.imencode(".png", canvas)[1].tofile(args.out)
    src = cv2.imdecode(np.fromfile(args.patch, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    src = cv2.resize(src, (S * 4, S * 4), interpolation=cv2.INTER_CUBIC)
    cv2.imencode(".png", src)[1].tofile(str(Path(args.out).with_name("stroke_field_source.png")))
    field = (255 * (1 - ink)).astype(np.uint8)
    field = cv2.resize(field, (S * 4, S * 4), interpolation=cv2.INTER_NEAREST)
    cv2.imencode(".png", field)[1].tofile(str(Path(args.out).with_name("stroke_field_pred.png")))
    meta = dict(patch=args.patch, strokes=len(strokes), output=args.out,
                note="StrokeField ink -> skeleton -> polylines；后续接 B 样条/曲率速度/机械臂约束。")
    (Path(args.out).with_suffix(".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()