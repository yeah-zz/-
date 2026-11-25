# -*- coding: utf-8 -*-
"""风格特征学习 v1：统计原作的可解释笔法特征，映射成一套可渲染参数。

这不是神经风格迁移，而是把"墨色覆盖、边缘密度、笔向熵、留白、色度、冷暖"
变成笔触引擎能执行的参数。后续可把 features 向量换成 CNN/CLIP 编码，
profile 的字段与渲染器无需改动。
"""

import base64
import copy
import hashlib
import io
import math

import numpy as np
from PIL import Image

from . import profiles


def _decode(data):
    raw = base64.b64decode(data, validate=True)
    if not 0 < len(raw) <= 12 * 1024 * 1024:
        raise ValueError("单张图片须小于 12 MB。")
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    if min(im.size) < 32 or max(im.size) > 6000:
        raise ValueError("图片边长须在 32 至 6000 像素之间。")
    im.thumbnail((900, 900), Image.LANCZOS)
    return im


def features(images):
    if not images:
        raise ValueError("请至少上传一幅原作。")
    rows = []
    for im in images:
        g = np.asarray(im.convert("L"), dtype=np.float32)
        rgb = np.asarray(im, dtype=np.float32)
        h, w = g.shape
        edge = np.concatenate([g[:4].ravel(), g[-4:].ravel(), g[:, :4].ravel(), g[:, -4:].ravel()])
        paper = float(np.median(edge)) if edge.size else 235.0
        if paper < 90:
            paper = 235.0
        ink = g < paper * 0.62
        dark = g < paper * 0.82
        negative = g > paper * 0.93
        gy, gx = np.gradient(g)
        mag = np.hypot(gx, gy)
        strong = mag > max(8.0, float(np.percentile(mag, 82)))
        edge_density = float(strong.mean())
        ang = (np.arctan2(gy, gx) + math.pi) % math.pi
        hist, _ = np.histogram(ang[strong], bins=12, range=(0, math.pi))
        p = hist.astype(np.float64) / max(1, hist.sum())
        p = p[p > 0]
        ent = float(-(p * np.log(p)).sum() / math.log(12)) if len(p) > 1 else 0.0
        contrast = float(np.clip(g.std() / 58.0, 0, 1.6))
        colorfulness = float(np.clip((np.std(rgb[..., 0] - rgb[..., 1]) + np.std(rgb[..., 1] - rgb[..., 2])) / 38.0, 0, 1.5))
        warmth = float(np.clip((rgb[..., 0].mean() - rgb[..., 2].mean()) / 60.0, -1, 1))
        # 墨线平均粗细的粗略估计：暗像素占边缘像素之比
        thick = float(np.clip(dark.sum() / max(1, strong.sum() * 4.0), 0, 1.5))
        rows.append(dict(ink=float(ink.mean()), dark=float(dark.mean()), negative=float(negative.mean()),
                         edge=edge_density, entropy=ent, contrast=contrast,
                         color=colorfulness, warmth=warmth, thick=thick))
    keys = rows[0].keys()
    avg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    spread = {k: float(np.std([r[k] for r in rows])) for k in keys}
    return avg, spread


def patch_features(images, patch=192, per_image=8):
    """局部山石/皴法学习：只统计有笔墨的 patch，避免整幅构图干扰。"""
    import random
    rows = []
    rng = random.Random(19)
    for im in images:
        g = np.asarray(im.convert("L"), dtype=np.float32)
        h, w = g.shape
        if h < patch or w < patch:
            continue
        for k in range(per_image):
            if k < 6:
                # 先覆盖上/中/下与左/中/右，再随机补
                yy = int((k // 3) * (h - patch) / 2)
                xx = int((k % 3) * (w - patch) / 2)
            else:
                yy = rng.randint(0, h - patch)
                xx = rng.randint(0, w - patch)
            p = g[yy:yy + patch, xx:xx + patch]
            if float(p.std()) < 7:
                continue
            dark = float((p < 90).mean())
            mid = float(((p >= 90) & (p < 190)).mean())
            paper = float((p >= 205).mean())
            if dark + mid < 0.03:
                continue
            gy, gx = np.gradient(p)
            mag = np.hypot(gx, gy)
            edge = float((mag > 12).mean())
            # 笔向一致性：方向直方图熵越低，越像有组织的皴法
            gx2 = gx[mag > 8]; gy2 = gy[mag > 8]
            if len(gx2) < 20:
                coh = 0.0
            else:
                jxx = float(np.mean(gx2 * gx2)); jyy = float(np.mean(gy2 * gy2)); jxy = float(np.mean(gx2 * gy2))
                coh = float(math.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-6))
            rough = float(np.abs(np.diff(p, axis=0)).mean() + np.abs(np.diff(p, axis=1)).mean())
            rows.append(dict(dark=dark, mid=mid, paper=paper, edge=edge,
                             coherence=min(1.0, coh), rough=min(1.0, rough / 90.0),
                             width=min(1.0, dark / max(1e-5, edge))))
    if not rows:
        return None
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}, len(rows)


def _num(v, lo, hi):
    return float(max(lo, min(hi, v)))


def train_from_images(name, base_id, images):
    """批量训练入口：images 为 PIL.Image 列表，可来自上传或本地文件夹。"""
    if not images:
        raise ValueError("没有可训练的原作。")
    if len(images) > 5000:
        images = images[:5000]
    base = profiles.ARTIST_BY_ID.get(base_id) or profiles.ARTISTS[0]
    name = str(name).strip()[:20] or "自定义笔意"

    avg, spread = features(images)
    patch_avg, patch_count = patch_features(images) or (None, 0)
    prof = copy.deepcopy(base)
    prof["id"] = "custom_" + hashlib.sha1((name + str(len(images)) + str(avg)).encode("utf-8")).hexdigest()[:10]
    prof["name"] = name
    prof["style_name"] = "自定义"
    prof["dynasty"] = "自定义"
    prof["years"] = "上传 %d 幅" % len(images)
    prof["work"] = "风格特征学习"
    prof["school"] = "统计蒸馏 · 可解释笔法"
    prof["custom"] = True
    prof["intro"] = ("从 %d 幅原作中统计：墨色覆盖 %.0f%%、边缘密度 %.0f%%、留白 %.0f%%，"
                     "笔向熵 %.0f%%。以此校准线质、皴法密度、点苔与设色。" %
                     (len(images), avg["dark"] * 100, avg["edge"] * 100,
                      avg["negative"] * 100, avg["entropy"] * 100))
    dry = _num(0.10 + avg["contrast"] * 0.34 + spread["edge"] * 0.2, 0.08, 0.72)
    prof["line"]["dry"] = round(dry, 3)
    prof["line"]["width"] = round(_num(base["line"]["width"] * (0.75 + avg["thick"] * 0.7), 1.8, 6.0), 2)
    prof["line"]["alpha"] = int(_num(160 + avg["ink"] * 300, 150, 245))
    prof["line"]["wobble"] = round(_num(base["line"]["wobble"] * (0.8 + avg["entropy"] * 0.6), 0.4, 2.6), 2)
    prof["texture"]["density"] = round(_num(0.15 + avg["edge"] * 2.6, 0.12, 0.98), 3)
    prof["dots"]["density"] = round(_num(0.08 + avg["edge"] * 2.1, 0.05, 0.96), 3)
    prof["dots"]["size"] = round(_num(4.0 + avg["thick"] * 2.8, 3.2, 9.0), 2)
    prof["wash"]["saturation"] = round(_num(avg["color"] * 1.7, 0.02, 0.75), 3)
    prof["wash"]["layers"] = int(_num(1 + avg["dark"] * 8, 1, 5))
    prof["composition"]["empty"] = round(_num(avg["negative"], 0.08, 0.92), 3)
    if patch_avg:
        # 局部 patch 更贴近“山石皴法”，用它微调线质与皴密度
        prof["texture"]["density"] = round(_num(0.18 + patch_avg["edge"] * 4.2, 0.12, 0.98), 3)
        prof["line"]["width"] = round(_num(base["line"]["width"] * (0.72 + patch_avg["width"] * 0.55), 1.8, 9.0), 2)
        prof["line"]["dry"] = round(_num(0.10 + patch_avg["rough"] * 0.55, 0.08, 0.78), 3)
        prof["dots"]["density"] = round(_num(0.05 + patch_avg["rough"] * 1.4, 0.04, 0.96), 3)
        prof["_patch"] = {k: round(v, 4) for k, v in patch_avg.items()}
        prof["_patch_count"] = patch_count
    if avg["color"] < 0.10:
        prof["wash"]["palette"] = "monochrome"
    elif avg["warmth"] > 0.35:
        prof["wash"]["palette"] = "light_ochre"
    else:
        prof["wash"]["palette"] = "wet" if avg["dark"] > 0.22 else "light_ochre"
    prof["inscription"]["sub"] = name + "笔意"
    prof["fingerprint"] = {
        "皴法密度": round(_num(prof["texture"]["density"], 0, 1), 2),
        "墨色层次": round(_num(prof["wash"]["layers"] / 5.0, 0, 1), 2),
        "留白": round(_num(prof["composition"]["empty"], 0, 1), 2),
        "线条骨力": round(_num(0.45 + avg["contrast"] * 0.4, 0, 1), 2),
        "点景密度": round(_num(prof["dots"]["density"], 0, 1), 2),
        "设色": round(_num(prof["wash"]["saturation"], 0, 1), 2),
    }
    prof["_features"] = {k: round(v, 4) for k, v in avg.items()}
    prof["_spread"] = {k: round(v, 4) for k, v in spread.items()}
    prof["_corpus"] = len(images)
    return prof

def train(body):
    name = str(body.get("name", "")).strip()
    if not name or len(name) > 20:
        raise ValueError("请给模型起一个 1–20 字的名称。")
    base_id = body.get("base") or profiles.ARTISTS[0]["id"]
    images = []
    files = body.get("images") or []
    if not isinstance(files, list) or not 1 <= len(files) <= 12:
        raise ValueError("请上传 1–12 幅原作。")
    for f in files:
        images.append(_decode(f.get("data", "")))
    return train_from_images(name, base_id, images)
