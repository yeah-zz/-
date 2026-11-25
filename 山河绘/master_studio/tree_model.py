# -*- coding: utf-8 -*-
"""独立树法模型：从名家原作裁树冠/枝干 patch，提取骨架模板并聚类。"""
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
DATA = ROOT / "data"
TREE = CORPUS / "trees"
PATCH = 160


def sources():
    out = []
    for d in ("huang_gongwang", "ni_zan", "shi_tao", "gong_xian", "bada_shanren"):
        out += sorted((CORPUS / d).glob("*.jpg"))
    out += sorted((CORPUS / "bulk" / "images").glob("*.jpg"))
    return out


def features(g):
    gy, gx = np.gradient(g.astype(np.float32) / 255.0)
    mag = np.hypot(gx, gy)
    edge = float((mag > .05).mean())
    dark = float((g < 100).mean())
    rough = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 90.0
    jxx = float(np.mean(gx * gx)); jyy = float(np.mean(gy * gy)); jxy = float(np.mean(gx * gy))
    coh = math.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-6)
    return np.array([dark, edge, min(1, rough), coh], dtype=np.float32)


def trace(skel):
    h, w = skel.shape
    pix = {(x, y) for y, x in zip(*np.nonzero(skel))}
    nbrs = lambda p: [(p[0] + dx, p[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      if (dx or dy) and (p[0] + dx, p[1] + dy) in pix]
    deg = {p: len(nbrs(p)) for p in pix}
    starts = [p for p in pix if deg[p] != 2] or list(pix)
    used = set(); paths = []
    for s in starts:
        for q in nbrs(s):
            if (s, q) in used:
                continue
            path = [s, q]; used.add((s, q)); used.add((q, s)); prev, cur = s, q
            while deg.get(cur, 0) == 2:
                nx = [r for r in nbrs(cur) if r != prev and (cur, r) not in used]
                if not nx: break
                r = nx[0]; used.add((cur, r)); used.add((r, cur)); path.append(r); prev, cur = cur, r
            if len(path) >= 5:
                paths.append(path)
    return paths


def main():
    TREE.mkdir(parents=True, exist_ok=True)
    (TREE / "patches").mkdir(exist_ok=True)
    rows = []
    for p in sources():
        try:
            im = Image.open(p).convert("L")
            w, h = im.size
            if w < PATCH or h < PATCH: continue
            # 树多在中下部；取 2 个高边缘 patch
            for k in range(2):
                x = int(w * (0.10 + 0.55 * k))
                y = int(h * 0.60 + (h - PATCH) * 0.12)
                x = max(0, min(w - PATCH, x)); y = max(0, min(h - PATCH, y))
                crop = im.crop((x, y, x + PATCH, y + PATCH))
                g = np.asarray(crop)
                if g.std() < 8 or (g < 100).mean() < .02:
                    continue
                name = "%s_tree%02d.png" % (p.stem, k)
                crop.save(TREE / "patches" / name)
                rows.append(dict(file=name, source=p.name, feat=features(g).tolist()))
        except Exception:
            continue
    feats = np.array([r["feat"] for r in rows], dtype=np.float32)
    if len(feats) < 6:
        raise SystemExit("tree patches too few")
    feats_n = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    # 简易 k-means 3 类
    rng = np.random.default_rng(3)
    centers = feats_n[rng.choice(len(feats_n), 3, replace=False)]
    for _ in range(30):
        d = ((feats_n[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        for k in range(3):
            if (lab == k).any(): centers[k] = feats_n[lab == k].mean(0)
    # 每类取 4 个骨架模板
    templates = []
    for k in range(3):
        idxs = [i for i, l in enumerate(lab) if l == k][:6]
        for i in idxs:
            g = np.asarray(Image.open(TREE / "patches" / rows[i]["file"]).convert("L"))
            mask = (g < 135).astype(np.uint8)
            sk = skeletonize(mask > 0)
            paths = trace(sk)
            paths.sort(key=len, reverse=True)
            paths = paths[:12]
            norm = []
            for path in paths:
                if len(path) < 6: continue
                pts = np.array([[x / PATCH, y / PATCH] for x, y in path], np.float32)
                # 取端点/拐点简化
                approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), 0.012, False).reshape(-1, 2)
                if len(approx) >= 2:
                    norm.append(approx.round(3).tolist())
            if norm:
                templates.append(dict(cluster=k, file=rows[i]["file"], strokes=norm))
    model = dict(format="tree-skeleton-v1", clusters=3,
                 counts=[int((lab == k).sum()) for k in range(3)],
                 templates=templates,
                 note="树冠/枝干 patch -> skeleton -> 折线模板；渲染时按树型选择并缩放。")
    DATA.mkdir(exist_ok=True)
    (DATA / "tree_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print("tree patches", len(rows), "templates", len(templates), "counts", model["counts"])


if __name__ == "__main__":
    main()