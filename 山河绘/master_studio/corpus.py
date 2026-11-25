# -*- coding: utf-8 -*-
"""公共领域图库、局部山石 patch 与四类皴法模型。"""

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
DATA = ROOT / "data"
MANIFEST = CORPUS / "manifest.json"
MODEL = DATA / "texture_model.json"
CLEAN_MODEL = DATA / "texture_model_clean.json"
CLASS_NAMES = {"pima": "披麻皴", "zhedai": "折带皴", "fupi": "斧劈皴", "midian": "米点皴"}


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def artist_dir(artist_id):
    return CORPUS / artist_id


def images_for(artist_id):
    d = artist_dir(artist_id)
    out = []
    for p in sorted(d.glob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            out.append(p)
    return out


def status():
    m = manifest()
    out = {"artists": {}, "model": None}
    total_img = total_patch = 0
    for aid, info in m["artists"].items():
        imgs = images_for(aid)
        patches = list((artist_dir(aid) / "patches").glob("*.png"))
        total_img += len(imgs)
        total_patch += len(patches)
        meta = {}
        mp = artist_dir(aid) / "meta.json"
        if mp.exists():
            try:
                for row in json.loads(mp.read_text(encoding="utf-8")):
                    meta[row.get("file")] = row
            except Exception:
                meta = {}
        image_rows = []
        for f in imgs[:8]:
            m = meta.get(f.name, {})
            image_rows.append(dict(file=f.name, url="/corpus-image/%s/%s" % (aid, f.name),
                                   title=m.get("title", f.stem), artist=m.get("artist", ""),
                                   source=m.get("source", "")))
        patch_rows = [dict(file=f.name, url="/corpus-patch/%s/%s" % (aid, f.name)) for f in patches[:8]]
        out["artists"][aid] = dict(name=info["name"], texture=info["texture"],
                                   texture_name=info["texture_name"],
                                   images=len(imgs), patches=len(patches),
                                   works=len(info["works"]), image_list=image_rows,
                                   patch_list=patch_rows)
    out["images"] = total_img
    out["patches"] = total_patch
    bulk_imgs = list((CORPUS / "bulk" / "images").glob("*.jpg")) if (CORPUS / "bulk" / "images").exists() else []
    bulk_patches = list((CORPUS / "bulk" / "patches").glob("*.png")) if (CORPUS / "bulk" / "patches").exists() else []
    out["bulk"] = dict(images=len(bulk_imgs), patches=len(bulk_patches))
    ltfc_root = CORPUS / "ltfc"
    ltfc_imgs = list(ltfc_root.glob("*/*.jpg")) if ltfc_root.exists() else []
    ltfc_patches = list(ltfc_root.glob("*/patches/*.png")) if ltfc_root.exists() else []
    out["ltfc"] = dict(images=len(ltfc_imgs), patches=len(ltfc_patches))
    out["total_images"] = total_img + len(bulk_imgs) + len(ltfc_imgs)
    out["total_patches"] = total_patch + len(bulk_patches) + len(ltfc_patches)
    if MODEL.exists():
        try:
            out["model"] = json.loads(MODEL.read_text(encoding="utf-8"))
        except Exception:
            out["model"] = None
    if CLEAN_MODEL.exists():
        try:
            out["clean_model"] = json.loads(CLEAN_MODEL.read_text(encoding="utf-8"))
        except Exception:
            out["clean_model"] = None
    enc = DATA / "encoder_metrics.json"
    if enc.exists():
        try:
            out["encoder"] = json.loads(enc.read_text(encoding="utf-8"))
        except Exception:
            out["encoder"] = None
    tm = DATA / "tree_model.json"
    if tm.exists():
        try:
            m = json.loads(tm.read_text(encoding="utf-8"))
            out["tree"] = dict(clusters=m.get("clusters"), counts=m.get("counts"),
                               templates=len(m.get("templates", [])))
        except Exception:
            out["tree"] = None
    sf = DATA / "stroke_field_metrics.json"
    if sf.exists():
        try:
            out["stroke_field"] = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            out["stroke_field"] = None
    return out


def _features(p):
    g = np.asarray(p.convert("L"), dtype=np.float32)
    if g.std() < 7:
        return None
    dark = float((g < 90).mean())
    mid = float(((g >= 90) & (g < 190)).mean())
    paper = float((g >= 205).mean())
    if dark + mid < 0.03:
        return None
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    edge = float((mag > 12).mean())
    # 结构张量：衡量“有没有一个主导笔向”，比方向直方图熵更稳
    gx = gx[mag > 8]
    gy = gy[mag > 8]
    if len(gx) < 20:
        coh = 0.0
    else:
        jxx = float(np.mean(gx * gx))
        jyy = float(np.mean(gy * gy))
        jxy = float(np.mean(gx * gy))
        coh = float(math.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-6))
    rough = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
    # 方向能量：折带多横线，披麻多竖/斜线
    v_energy = float(np.mean(np.abs(gx)))
    h_energy = float(np.mean(np.abs(gy)))
    orient = (v_energy - h_energy) / (v_energy + h_energy + 1e-6)
    # 8 向加权直方图：比单一 coherence 更能分开披麻/折带/斧劈/米点
    ang = (np.arctan2(gy, gx) + math.pi) % math.pi
    m1 = np.hypot(gx, gy)
    hist, _ = np.histogram(ang, bins=8, range=(0, math.pi), weights=m1)
    hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
    out = dict(dark=dark, mid=mid, paper=paper, edge=edge,
               coherence=min(1.0, coh), rough=min(1.0, rough / 90.0),
               width=min(1.0, dark / max(1e-5, edge)),
               vertical=min(1.0, v_energy / 30.0), orient=orient)
    for i, v in enumerate(hist):
        out["h%d" % i] = float(v)
    return out


def extract_patches(artist_id, per_image=8, patch=240):
    d = artist_dir(artist_id)
    pd = d / "patches"
    pd.mkdir(parents=True, exist_ok=True)
    rows = []
    for img_path in images_for(artist_id):
        try:
            im = Image.open(img_path).convert("RGB")
            im.thumbnail((1600, 1600), Image.LANCZOS)
        except Exception:
            continue
        g = np.asarray(im.convert("L"), dtype=np.float32)
        h, w = g.shape
        if h < patch or w < patch:
            continue
        cands = []
        # 只取中上部山石带，避开题款和前景树
        y0 = int(h * 0.08); y1 = int(h * 0.78)
        x0 = int(w * 0.03); x1 = int(w * 0.97)
        step = max(48, patch // 3)
        for y in range(y0, max(y0 + 1, y1 - patch + 1), step):
            for x in range(x0, max(x0 + 1, x1 - patch + 1), step):
                p = g[y:y + patch, x:x + patch]
                gy, gx = np.gradient(p)
                mag = np.hypot(gx, gy)
                edge = float((mag > 12).mean())
                dark = float((p < 100).mean())
                if edge < 0.045 or dark > 0.36:
                    continue
                gx2 = gx.ravel(); gy2 = gy.ravel()
                jxx = float(np.mean(gx2 * gx2)); jyy = float(np.mean(gy2 * gy2)); jxy = float(np.mean(gx2 * gy2))
                coh = float(math.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-6))
                # 披麻/折带要“有方向的线”，不要树冠那种密集黑团
                score = coh * 2.5 + edge * 1.0 - abs(dark - 0.09) * 2.2
                if score > 0.12:
                    cands.append((score, x, y))
        cands.sort(reverse=True)
        picked = []
        for score, x, y in cands:
            if len(picked) >= per_image:
                break
            if any(abs(x - qx) < patch * 0.55 and abs(y - qy) < patch * 0.55 for _, qx, qy in picked):
                continue
            picked.append((score, x, y))
        cands = picked
        for i, (score, x, y) in enumerate(cands[:per_image]):
            pim = im.crop((x, y, x + patch, y + patch))
            f = _features(pim)
            if not f:
                continue
            name = "%s_%02d.png" % (img_path.stem, i)
            pim.save(pd / name, quality=94)
            f.update(file=name, source=img_path.name, objectID=img_path.stem)
            rows.append(f)
    (d / "patches.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def _vector(f):
    base = [f["dark"], f["edge"], f["coherence"], f["rough"], f["width"],
            f.get("vertical", 0.0), f.get("orient", 0.0)]
    base += [f.get("h%d" % i, 0.0) for i in range(8)]
    return np.array(base, dtype=np.float32)


def apply_model(profile):
    """把训练出的四类皴法质心，回写到画家模型的纹理参数。"""
    if not MODEL.exists():
        return profile
    try:
        path = CLEAN_MODEL if CLEAN_MODEL.exists() else MODEL
        model = json.loads(path.read_text(encoding="utf-8"))
        info = manifest()["artists"].get(profile.get("id"))
        if not info:
            return profile
        cls = info["texture"]
        cen = model.get("raw_centroids", {}).get(cls) or model.get("centroids", {}).get(cls)
        if not cen:
            return profile
        p = copy.deepcopy(profile)
        dark, edge, coh, rough, width = cen[:5]
        p["texture"]["density"] = round(max(0.22, min(0.96, 0.22 + edge * 1.9)), 3)
        p["line"]["width"] = round(max(2.0, min(9.0, profile["line"]["width"] * (0.80 + width * 0.35))), 2)
        p["line"]["dry"] = round(max(0.08, min(0.78, 0.10 + rough * 0.55)), 3)
        p["dots"]["density"] = round(max(0.04, min(0.96, 0.05 + rough * 1.15)), 3)
        p["_texture_model"] = dict(texture=cls, accuracy=model.get("loo_accuracy"))
        return p
    except Exception:
        return profile


def train_model():
    m = manifest()
    xs, ys = [], []
    for aid, info in m["artists"].items():
        cls = info["texture"]
        if cls not in CLASS_NAMES:
            continue
        rows = []
        pd = artist_dir(aid) / "patches"
        if pd.exists():
            for fp in sorted(pd.glob("*.png")):
                try:
                    f = _features(Image.open(fp).convert("RGB"))
                    if f:
                        f["file"] = fp.name
                        rows.append(f)
                except Exception:
                    continue
        for f in rows:
            xs.append(_vector(f))
            ys.append(cls)
    # 中华珍宝馆：按画家映射到四类（只取可比拟的几家）
    ltfc = {"黄公望": "pima", "王蒙": "pima", "吴镇": "pima", "沈周": "pima", "文徵明": "pima",
            "倪瓒": "zhedai", "石涛": "fupi", "唐寅": "fupi", "仇英": "fupi", "龚贤": "midian"}
    for artist, cls in ltfc.items():
        pd = CORPUS / "ltfc" / artist / "patches"
        if not pd.exists():
            continue
        for fp in sorted(pd.glob("*.png")):
            try:
                f = _features(Image.open(fp).convert("RGB"))
                if f:
                    xs.append(_vector(f)); ys.append(cls)
            except Exception:
                continue
    if not xs:
        raise ValueError("还没有 patch。请先下载图库并提取 patch。")
    X_raw = np.stack(xs)
    X = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-6)
    classes = sorted(set(ys))
    centroids = {}
    raw_centroids = {}
    for c in classes:
        idx = [i for i, y in enumerate(ys) if y == c]
        if len(idx) < 2:
            raise ValueError("类别 %s 的 patch 太少（至少 2 个）。" % CLASS_NAMES.get(c, c))
        centroids[c] = X[idx].mean(0)
        raw_centroids[c] = X_raw[idx].mean(0)
    # 留一法准确率
    correct = 0
    for i, y in enumerate(ys):
        best, bestd = None, 1e9
        for c, cen in centroids.items():
            dd = float(np.linalg.norm(X[i] - cen))
            if dd < bestd:
                best, bestd = c, dd
        if best == y:
            correct += 1
    # kNN 留一法：小样本下通常比质心更稳
    best_k, best_acc = 1, 0.0
    for k in (1, 3, 5):
        hit = 0
        for i in range(len(X)):
            d = np.linalg.norm(X - X[i], axis=1)
            d[i] = 1e9
            idx = np.argsort(d)[:k]
            votes = {}
            for j in idx:
                votes[ys[j]] = votes.get(ys[j], 0) + 1
            if max(votes, key=votes.get) == ys[i]:
                hit += 1
        acc = hit / len(X)
        if acc > best_acc:
            best_k, best_acc = k, acc
    model = dict(format="texture-centroid-knn-v1", classes=classes, class_names=CLASS_NAMES,
                 centroids={c: v.tolist() for c, v in centroids.items()},
                 raw_centroids={c: v.tolist() for c, v in raw_centroids.items()},
                 counts={c: ys.count(c) for c in classes},
                 loo_accuracy=round(best_acc, 3), centroid_accuracy=round(correct / len(ys), 3),
                 knn_k=best_k, samples=len(ys),
                 train_vectors=X.tolist(), train_labels=ys,
                 feature_order=["dark", "edge", "coherence", "rough", "width", "vertical", "orient"] + ["h%d" % i for i in range(8)])
    DATA.mkdir(parents=True, exist_ok=True)
    MODEL.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    return model