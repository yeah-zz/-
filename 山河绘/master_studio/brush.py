# -*- coding: utf-8 -*-
"""笔触底层：纸、墨、水、飞白、笔锋与压力包络。"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normal_array(rng, shape):
    n = int(__import__("numpy").prod(shape))
    vals = __import__("numpy").fromiter((rng.gauss(0, 1) for _ in range(n)), dtype="float32", count=n)
    return vals.reshape(shape)


def catmull_rom(points, samples=10):
    """平滑一条折线；点少时也能得到像写出来的弧。"""
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) < 3:
        return pts
    ext = [pts[0]] + pts + [pts[-1]]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for j in range(samples):
            t = j / samples
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    out.append(pts[-1])
    return out


def resample(points, step=2.0):
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    carry = 0.0
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg = math.hypot(dx, dy)
        if seg <= 1e-6:
            continue
        pos = step - carry
        while pos <= seg:
            t = pos / seg
            out.append((a[0] + dx * t, a[1] + dy * t))
            pos += step
        carry = seg - (pos - step)
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def lerp(a, b, t):
    return a + (b - a) * t


def pressure_envelope(n, lo=0.35, hi=1.0, taper=0.7, rng=None, wobble=0.12):
    if n <= 1:
        return np.array([hi], dtype=np.float32)
    t = np.linspace(0, 1, n)
    # 起笔略顿、行笔中段饱满、收笔出锋；taper 越大，收锋越长。
    enter = np.clip(t / max(0.06, 0.16 * taper), 0, 1) ** 0.55
    exit_ = np.clip((1 - t) / max(0.08, 0.30 * taper), 0, 1) ** 0.85
    body = 0.75 + 0.25 * np.sin(np.pi * t) ** 0.5
    env = enter * exit_ * body
    env = lo + (hi - lo) * env
    if rng is not None and wobble > 0:
        k = max(3, n // 7)
        base = normal_array(rng, (k,))
        base = np.interp(np.linspace(0, k - 1, n), np.arange(k), base)
        env = env * (1 + wobble * base * 0.30)
    return np.clip(env, 0.02, 1.0).astype(np.float32)


def normal_at(pts, i):
    if i <= 0:
        dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
    elif i >= len(pts) - 1:
        dx, dy = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
    else:
        dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
    d = math.hypot(dx, dy) or 1.0
    return -dy / d, dx / d


def draw_stroke(draw, points, base_width, color, alpha, pressure, rng,
                dry=0.0, taper=0.65, bristle=2, end_pool=True):
    """在 RGBA 画布上写一笔。

    points: 已平滑/重采样的点；pressure: 与点等长的 0..1 笔压。
    不是简单等宽线：笔锋有起收、两侧有散锋、干笔有断白。
    """
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) < 2:
        return
    n = len(pts)
    if len(pressure) != n:
        pressure = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(pressure)), pressure)
    dry = clamp(dry, 0, 1)
    # 干笔的断续在整笔上连续发生，而不是逐段随机白点。
    gap = normal_array(rng, (n,))
    gap = np.convolve(gap, np.ones(3) / 3, mode="same")
    gap_threshold = -0.15 + dry * 1.05
    for i in range(n - 1):
        p = float(pressure[i])
        w = max(0.8, base_width * (0.08 + 1.05 * p))
        a = int(clamp(alpha * (0.22 + 0.85 * p), 0, 255))
        if dry > 0.02:
            keep = gap[i] > -gap_threshold * 0.55
            if not keep:
                a = int(a * 0.20)
            else:
                a = int(a * (1.0 - dry * 0.38 * rng.random()))
        if a <= 2:
            continue
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill=(*color, a), width=max(1, int(round(w))))
        if bristle and w >= 3.0:
            nx, ny = normal_at(pts, i)
            off = w * 0.30
            aa = int(a * (0.30 + 0.18 * rng.random()))
            for s in (-1, 1):
                if rng.random() < 0.72:
                    ox, oy = nx * off * s, ny * off * s
                    draw.line([(x1 + ox, y1 + oy), (x2 + ox, y2 + oy)],
                              fill=(*color, aa), width=max(1, int(w * 0.28)))
    if end_pool and base_width >= 2.5:
        p0 = float(pressure[0]); p1 = float(pressure[-1])
        for (x, y), p in ((pts[0], p0), (pts[-1], p1)):
            r = max(1.0, base_width * (0.18 + 0.30 * p))
            a = int(clamp(alpha * (0.45 + 0.4 * p), 0, 255))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(*color, a))


def draw_ink_stroke(target, points, base_width, color, alpha, pressure, rng,
                    dry=0.0, taper=0.5, supersample=2, widths_px=None):
    """连续变宽笔触：用中心线偏移成多边形，避免逐段线造成“串珠”。"""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        return
    n = len(pts)
    if len(pressure) != n:
        pressure = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(pressure)), pressure)
    # 端点略收，但中间保持连续
    t = np.linspace(0, 1, n)
    env = np.clip((0.55 + 0.60 * np.asarray(pressure, dtype=np.float32)), 0.35, 1.35)
    if taper:
        env *= np.clip(np.minimum(t / max(1e-3, taper), (1 - t) / max(1e-3, taper)), 0.25, 1.0) ** 0.28
    widths = np.maximum(1.6, base_width * env)
    if widths_px is not None:
        widths = np.asarray(widths_px, dtype=np.float32)
        if widths.shape != (n,) or not np.isfinite(widths).all() or np.any(widths <= 0):
            raise ValueError("Explicit brush widths must provide one positive width per point")
    W, H = target.size
    S = max(1, int(supersample))
    layer = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    left, right = [], []
    for i, (x, y) in enumerate(pts):
        if i == 0:
            nx, ny = pts[1][0] - x, pts[1][1] - y
        elif i == n - 1:
            nx, ny = x - pts[-2][0], y - pts[-2][1]
        else:
            nx, ny = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        ln = math.hypot(nx, ny) or 1.0
        nx, ny = -ny / ln, nx / ln
        hw = widths[i] * 0.5
        left.append(((x + nx * hw) * S, (y + ny * hw) * S))
        right.append(((x - nx * hw) * S, (y - ny * hw) * S))
    poly = left + right[::-1]
    a = int(max(2, min(255, alpha)))
    d.polygon(poly, fill=(*color, a))
    # 圆头，避免起收笔出现尖断
    for (x, y), w in ((pts[0], widths[0]), (pts[-1], widths[-1])):
        r = max(1.0, w * .5) * S
        d.ellipse([x * S - r, y * S - r, x * S + r, y * S + r], fill=(*color, a))
    layer = layer.resize((W, H), Image.LANCZOS)
    target.alpha_composite(layer)


def stroke_outline(draw, points, base_width, color, alpha, pressure, rng,
                   dry=0.0, taper=0.6, bristle=2):
    draw_stroke(draw, points, base_width, color, alpha, pressure, rng,
                dry=dry, taper=taper, bristle=bristle)


def soft_polygon(size, points, blur=1.5, feather=0):
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.polygon([(float(x), float(y)) for x, y in points], fill=255)
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    if blur:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
    return mask


def mask_array(mask):
    return np.asarray(mask, dtype=np.float32) / 255.0


def composite_mask(base, color, alpha, mode="normal"):
    """把纯色按 alpha 数组叠到 RGB/RGBA 图上。"""
    arr = np.asarray(base.convert("RGB"), dtype=np.float32)
    col = np.asarray(color, dtype=np.float32)
    a = np.clip(alpha, 0, 1)[..., None]
    if mode == "multiply":
        out = arr * (1 - a) + (arr * col / 255.0) * a
    else:
        out = arr * (1 - a) + col * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def gradient_array(mask, axis="y", lo=0.25, hi=0.9, gamma=1.0, invert=False):
    a = mask_array(mask)
    h, w = a.shape
    t = np.linspace(0, 1, h if axis == "y" else w, dtype=np.float32)
    g = lo + (hi - lo) * (t ** gamma)
    if invert:
        g = g[::-1]
    if axis == "y":
        g = np.repeat(g[:, None], w, axis=1)
    else:
        g = np.repeat(g[None, :], h, axis=0)
    return np.clip(a * g, 0, 1)


def paper_base(w, h, rng, tone=(243, 238, 226)):
    """宣纸：低频云斑 + 细纤维 + 边缘旧色。"""
    base = np.ones((h, w, 3), dtype=np.float32) * np.asarray(tone, dtype=np.float32)
    # 低频云斑
    small = normal_array(rng, (max(8, h // 30), max(12, w // 30)))
    cloud = np.array(Image.fromarray(((small - small.min()) / (np.ptp(small) + 1e-6) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), dtype=np.float32)
    base += (cloud[..., None] - 127) * 0.030
    # 纤维
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for _ in range(max(900, w * h // 1400)):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ln = rng.uniform(4, 30)
        ang = rng.gauss(0, 0.12) + (math.pi / 2 if rng.random() < 0.16 else 0)
        dx, dy = math.cos(ang) * ln, math.sin(ang) * ln
        light = rng.random() < 0.45
        c = (255, 252, 244, int(rng.uniform(5, 15))) if light else (198, 188, 168, int(rng.uniform(4, 12)))
        d.line([(x, y), (x + dx, y + dy)], fill=c, width=1)
    base = Image.alpha_composite(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA"), overlay).convert("RGB")
    # 淡旧斑
    arr = np.asarray(base, dtype=np.float32).copy()
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.35, 0.65)
    rr = max(w, h) * rng.uniform(0.35, 0.55)
    vig = np.clip(np.sqrt(((xx - cx) / rr) ** 2 + ((yy - cy) / rr) ** 2), 0, 1)
    arr *= (1 - 0.045 * vig[..., None])
    # 四边轻微压暗
    edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)).astype(np.float32)
    edge = np.clip(edge / (min(w, h) * 0.08), 0, 1)
    arr *= (0.955 + 0.045 * edge[..., None])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def add_paper_grain(img, rng, strength=0.10):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = normal_array(rng, arr.shape[:2])
    noise = np.array(Image.fromarray(((noise - noise.min()) / (np.ptp(noise) + 1e-6) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.35)), dtype=np.float32)
    arr *= (1 + (noise[..., None] - 127) / 127 * strength)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def ink_wash(img, mask, color, alpha, rng, edge_bleed=1.0):
    """一层薄墨/颜色：边缘略深，内部有极轻的颗粒。"""
    a = mask_array(mask)
    if edge_bleed:
        er = np.asarray(mask.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32) / 255.0
        edge = np.clip(a - er * 0.92, 0, 1)
        a = np.clip(a + edge * 0.35, 0, 1)
    grain = 1.0 + 0.06 * normal_array(rng, a.shape)
    a = np.clip(a * grain, 0, 1) * alpha
    return composite_mask(img, color, a)


def seal_image(text, size, rng, color=(168, 54, 42), font_path=None):
    """做一枚有残边的朱文印。"""
    from PIL import ImageFont
    s = max(28, int(size))
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([1, 1, s - 2, s - 2], radius=max(2, s // 16), fill=(*color, 235))
    # 印面留白文字
    try:
        font = ImageFont.truetype(font_path or r"C:\Windows\Fonts\simkai.ttf", int(s * 0.62 / max(1, len(text))))
    except OSError:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((s - tw) / 2 - bbox[0], (s - th) / 2 - bbox[1]), text, font=font, fill=(248, 240, 224, 248))
    # 残边：随机擦掉几个小口
    px = im.load()
    for _ in range(max(20, s // 2)):
        x, y = rng.randrange(s), rng.randrange(s)
        if x < 3 or y < 3 or x > s - 4 or y > s - 4:
            px[x, y] = (0, 0, 0, 0)
    return im
