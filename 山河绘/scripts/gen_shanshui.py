# -*- coding: utf-8 -*-
"""
gen_shanshui.py - 程序化生成水墨山水画（numpy + PIL，不连接机械臂）
==================================================================
运行： python scripts/gen_shanshui.py
输出： 输出/山水画_程序生成/山河绘_水墨山水.png
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1600, 2200
SEED = 20260929

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "\u8f93\u51fa", "\u5c71\u6c34\u753b_\u7a0b\u5e8f\u751f\u6210")
OUT_PNG = os.path.join(OUT_DIR, "\u5c71\u6cb3\u7ed8_\u6c34\u58a8\u5c71\u6c34.png")
OUT_JPG = os.path.join(OUT_DIR, "\u5c71\u6cb3\u7ed8_\u6c34\u58a8\u5c71\u6c34_\u88c5\u88f1.jpg")

INK = np.array([32, 34, 38], dtype=np.float32)
PAPER = np.array([242, 235, 219], dtype=np.float32)

FONTS = [r"C:\Windows\Fonts\STKAITI.TTF", r"C:\Windows\Fonts\simkai.ttf",
         r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
         r"C:\Windows\Fonts\simsun.ttc"]


def font_at(size):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return None


def fractal_noise(w, h, rng, scale, octaves=3):
    out = np.zeros((h, w), dtype=np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        s = max(2, int(scale / (2 ** o)))
        small = np.random.default_rng(rng.randrange(1 << 30)).random(
            (max(2, h // s), max(2, w // s)))
        img = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += amp * (np.asarray(img, dtype=np.float32) / 255.0)
        total += amp
        amp *= 0.55
    return out / total


def spline(ctrl, samples=20, rough=0.0, rng=None):
    pts = [ctrl[0]] + list(ctrl) + [ctrl[-1]]
    out = []
    for i in range(len(pts) - 3):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        for s in range(samples):
            t = s / samples
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                       + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                       + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            if rough and rng is not None:
                y += rng.gauss(0, rough)
            out.append((x, y))
    out.append(ctrl[-1])
    return out


def sample_ridge(ridge, x):
    if x <= ridge[0][0]:
        return ridge[0][1]
    if x >= ridge[-1][0]:
        return ridge[-1][1]
    for i in range(len(ridge) - 1):
        x0, y0 = ridge[i]
        x1, y1 = ridge[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / max(1e-6, x1 - x0)
            return y0 + (y1 - y0) * t
    return ridge[-1][1]


def poly_mask(pts, blur=0.0):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    if blur:
        m = m.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(m, dtype=np.float32) / 255.0


class Scene:
    def __init__(self, rng):
        self.rng = rng
        self.layers = []
        self.sheet = Image.new("L", (W, H), 0)

    def commit(self, arr, weight=1.0):
        self.layers.append((np.clip(arr, 0.0, 1.0).astype(np.float32), float(weight)))

    def brush(self, pts, width, alpha, taper=True, jitter=0.0):
        d = ImageDraw.Draw(self.sheet)
        if len(pts) < 2:
            return
        if jitter:
            pts = [(p[0] + self.rng.uniform(-jitter, jitter),
                    p[1] + self.rng.uniform(-jitter, jitter)) for p in pts]
        n = len(pts)
        for i in range(n - 1):
            t = i / max(1, n - 2)
            if taper:
                shape = math.sin(math.pi * min(1.0, max(0.0, t))) ** 0.4
                w = max(0.7, width * (0.22 + 0.78 * shape))
                a = alpha * (0.28 + 0.72 * shape)
            else:
                w, a = width, alpha
            d.line([pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]],
                   fill=int(max(0, min(255, a))), width=max(1, int(round(w))))

    def flush(self, weight=1.0, blur=0.0, clip=None):
        img = self.sheet
        if blur:
            img = img.filter(ImageFilter.GaussianBlur(blur))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if clip is not None:
            arr = arr * clip
        self.commit(arr, weight)
        self.sheet = Image.new("L", (W, H), 0)
        return arr

    def wash(self, pts, top_y, bot_y, amount, grain=0.4, blur=3.0):
        m = poly_mask(pts)
        ys = np.arange(H, dtype=np.float32)[:, None]
        span = max(1.0, bot_y - top_y)
        fade = np.clip((bot_y - ys) / span, 0.0, 1.0) ** 0.85
        n = fractal_noise(W, H, self.rng, 60, 3)
        v = np.clip(m * fade * amount * (1.0 + grain * (n - 0.5) * 1.8), 0, 1)
        if blur:
            v = np.asarray(Image.fromarray((v * 255).astype(np.uint8))
                           .filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) / 255.0
        self.commit(v)

    def ma_field(self, ridge, top_y, bot_y, mask, amount=0.6, count=1000,
                 lmin=100, lmax=380):
        """披麻皴：沿坡向长皴，按岩体裁掉越界部分。"""
        rng = self.rng
        for _ in range(count):
            x0 = rng.uniform(ridge[0][0] + 20, ridge[-1][0] - 20)
            y0 = sample_ridge(ridge, x0) + rng.uniform(-4, 26)
            depth = (y0 - top_y) / max(1.0, bot_y - top_y)
            length = rng.uniform(lmin, lmax) * (1.0 - 0.6 * depth)
            dy = sample_ridge(ridge, x0 + 14) - sample_ridge(ridge, x0 - 14)
            lean = -dy * 0.20 + rng.uniform(-0.30, 0.30)
            pts = []
            for s in range(10):
                yy = y0 + length * s / 9
                pts.append((x0 + lean * (yy - y0) + math.sin(s * 0.8) * 3.0, yy))
            self.brush(pts, width=rng.uniform(1.1, 2.7),
                       alpha=rng.uniform(55, 145) * amount, jitter=1.0)
        self.flush(0.95, clip=mask)

    def axe(self, cx, cy, sx, sy, count, mask, alpha=(45, 120)):
        rng = self.rng
        for _ in range(count):
            x, y = rng.gauss(cx, sx), rng.gauss(cy, sy)
            ln = rng.uniform(18, 58)
            ang = rng.choice([-1, 1]) * rng.uniform(0.7, 1.35)
            self.brush([(x, y), (x + math.cos(ang) * ln, y + abs(math.sin(ang)) * ln * 0.85)],
                       width=rng.uniform(1.4, 3.0), alpha=rng.uniform(*alpha), taper=False)
        self.flush(0.9, clip=mask)

    def moss(self, ridge_or_none, count, x0, x1, y0, y1, mask,
             r=(1.4, 3.2), alpha=(115, 230)):
        """点苔：聚成小簇，沿脊线与石棱分布，不像噪点。"""
        rng = self.rng
        d = ImageDraw.Draw(self.sheet)
        placed = 0
        guard = 0
        while placed < count and guard < count * 12:
            guard += 1
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if ridge_or_none is not None:
                ry = sample_ridge(ridge_or_none, x)
                if not (-10 < y - ry < 200):
                    continue
            if mask[int(y), int(x)] < 0.5:
                continue
            # 一簇 3~7 点
            k = rng.randint(3, 7)
            for _ in range(k):
                xx = x + rng.gauss(0, 7)
                yy = y + rng.gauss(0, 6)
                rr = rng.uniform(*r)
                a = int(rng.uniform(*alpha))
                d.ellipse([xx - rr, yy - rr * 0.8, xx + rr, yy + rr * 0.8], fill=a)
            placed += 1
        self.flush(0.95, clip=mask)

    def highlight(self, mask_arr, strength=0.9, blur=14.0):
        arr = np.clip(mask_arr, 0, 1)
        if blur:
            arr = np.asarray(Image.fromarray((arr * 255).astype(np.uint8))
                             .filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) / 255.0
        self.commit(arr, -strength)

    def tree(self, x, y, h, trunk_w=12.0, lean=-0.05, density=1.0):
        rng = self.rng
        d = ImageDraw.Draw(self.sheet)
        curve = [(x + lean * h * (s / 13) + math.sin(s / 13 * 2.4) * h * 0.030,
                  y - h * (s / 13)) for s in range(14)]
        for i in range(len(curve) - 1):
            t = i / (len(curve) - 1)
            self.brush([curve[i], curve[i + 1]], width=trunk_w * (1.05 - 0.72 * t),
                       alpha=238, taper=False, jitter=trunk_w * 0.10)
        for dx, dy in ((-2.2, 1.0), (2.0, 0.8)):
            d.line([curve[0][0], curve[0][1], curve[0][0] + trunk_w * dx,
                    curve[0][1] + trunk_w * dy], fill=215, width=max(2, int(trunk_w * 0.45)))

        nb = max(5, int(h / 62))
        for b in range(nb):
            t = 0.24 + 0.72 * b / max(1, nb - 1)
            i = min(len(curve) - 2, int(t * (len(curve) - 1)))
            bx, by = curve[i]
            side = -1 if b % 2 == 0 else 1
            bl = h * rng.uniform(0.17, 0.30) * (0.70 + 0.50 * t)
            ex, ey = bx + side * bl, by - bl * rng.uniform(0.10, 0.40)
            self.brush([(bx, by), (ex, ey)], width=max(2.2, trunk_w * 0.34),
                       alpha=226, taper=True)
            # 针团：枝梢为中心的椭圆云，短线密布
            fan_x, fan_y = bl * 0.34, bl * 0.26
            for _ in range(int(290 * density)):
                px = ex + rng.gauss(0, fan_x)
                py = ey + rng.gauss(0, fan_y) - fan_y * 0.5
                base = math.degrees(math.atan2(py - by, px - bx))
                ang = math.radians(base + rng.uniform(-50, 50))
                ln = rng.uniform(11, 22) * (0.8 + 0.45 * h / 520.0)
                self.brush([(px, py), (px + math.cos(ang) * ln, py + math.sin(ang) * ln)],
                           width=rng.uniform(1.0, 1.5), alpha=rng.uniform(110, 190),
                           taper=False)
        self.flush(0.96)


# ----------------------------------------------------------------------
def sky(sc):
    """天空：极淡的墨韵，避免大片死白。"""
    n = fractal_noise(W, H, sc.rng, 340, 3)
    ys = np.arange(H, dtype=np.float32)[:, None]
    fade = np.clip((1000.0 - ys) / 1000.0, 0, 1) ** 1.2
    sc.commit(np.clip(0.05 * fade * (0.4 + 1.1 * n), 0, 1))


def distant(sc):
    """远山四叠，最远一叠几乎化入天际。"""
    bands = [
        (1180, 0.085, 105, 300, 3),
        (1300, 0.15, 130, 340, 5),
        (1400, 0.20, 100, 250, 9),
        (1480, 0.25, 72, 180, 13),
    ]
    for base, amt, amp, step, seed in bands:
        r2 = random.Random(seed)
        ctrl = []
        for i in range(13):
            ctrl.append((-40 + (W + 80) * i / 12,
                         base - abs(math.sin(i * 1.7 + seed * 0.7)) * amp
                         - r2.uniform(0, 14)))
        ridge = spline(ctrl, 24, rough=3.4, rng=r2)
        top = min(p[1] for p in ridge)
        body = ridge + [(W + 80, H), (-80, H)]
        m = poly_mask(body)
        sc.wash(body, top, base + 200, amt, grain=0.28, blur=4.5)
        sc.flush(0.8, blur=1.0, clip=m)
        sc.brush(ridge, width=1.8, alpha=70 * (amt / 0.15), taper=False, jitter=0.8)
        sc.flush(0.85, clip=m)
        if amt > 0.12:
            sc.moss(ridge, 22, 40, W - 40, top + 10, base + 40, m,
                    r=(1.2, 2.6), alpha=(55, 120))


def main_peak(sc):
    """主峰：多脊错落的不对称山体，主峰偏左，右侧长坡受光。"""
    rng = sc.rng
    ctrl = [
        (140, 1660), (208, 1512), (262, 1350), (296, 1150), (342, 1000),
        (392, 1190), (430, 1420), (462, 1310), (508, 1130), (548, 980),
        (592, 1150), (630, 960), (668, 790),
        (706, 960), (752, 1120), (806, 1020), (852, 1140), (900, 1040),
        (964, 1180), (1040, 1330), (1130, 1470), (1234, 1630),
    ]
    ridge = spline(ctrl, 26, rough=4.0, rng=rng)
    peak_y = min(p[1] for p in ridge)
    bot = 1810
    body = ridge + [(1280, H), (110, H)]
    m = poly_mask(body)
    sc.wash(body, peak_y, bot, 0.44, grain=0.5, blur=2.5)
    sc.flush(0.95, blur=0.6, clip=m)
    sc.ma_field(ridge, peak_y, bot, m, amount=0.60, count=1200, lmin=90, lmax=360)

    shade = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shade).polygon(
        [(620, 800), (800, 1080), (960, 1300), (1100, 1500), (1234, 1630),
         (1060, 1340), (880, 1060), (740, 880)], fill=130)
    shade = np.asarray(shade.filter(ImageFilter.GaussianBlur(34)), dtype=np.float32) / 255.0
    sc.commit(shade * m * 0.40)

    sc.axe(500, 1730, 200, 100, 300, m)
    sc.axe(950, 1710, 180, 95, 220, m)
    sc.moss(ridge, 130, 200, 1260, 780, 1740, m, r=(1.5, 3.4), alpha=(105, 215))
    sc.flush(1.0, clip=m)
    sc.brush(ridge, width=3.2, alpha=232, taper=True, jitter=0.7)
    sc.flush(1.0)

    c2 = [(10, 1670), (95, 1520), (170, 1380), (250, 1510), (325, 1640)]
    r2 = spline(c2, 22, rough=3.0, rng=rng)
    b2 = r2 + [(350, H), (-10, H)]
    m2 = poly_mask(b2)
    sc.wash(b2, min(p[1] for p in r2), 1730, 0.18, grain=0.45)
    sc.flush(0.9, clip=m2)
    sc.brush(r2, width=2.2, alpha=164, taper=True, jitter=0.6)
    sc.flush(1.0, clip=m2)
    sc.moss(r2, 16, 30, 320, 1400, 1610, m2, r=(1.2, 3.0), alpha=(90, 175))

    # 峰前近山（压住主峰山脚）
    c3 = [(430, 1780), (560, 1680), (720, 1720), (880, 1660), (1010, 1730),
          (1120, 1800)]
    r3 = spline(c3, 22, rough=3.0, rng=rng)
    b3 = r3 + [(1140, H), (410, H)]
    m3 = poly_mask(b3)
    sc.wash(b3, min(p[1] for p in r3), 1980, 0.22, grain=0.45)
    sc.flush(0.9, clip=m3)
    sc.axe(760, 1860, 180, 80, 160, m3, alpha=(45, 110))
    sc.brush(r3, width=2.6, alpha=190, taper=True, jitter=0.8)
    sc.flush(0.95, clip=m3)
    return ridge, peak_y


def waterfall(sc, ridge, peak_y):
    """瀑口：先在山体上擦出一道水口，再在口内画数道随收放行笔的水线。"""
    rng = sc.rng
    cx = 880
    segs = [(1020, 1300, 20, 52), (1330, 1610, 16, 40)]

    # 1) 水口留白
    cut = Image.new("L", (W, H), 0)
    cd = ImageDraw.Draw(cut)
    for top_y, bot_y, w0, w1 in segs:
        for s in range(80):
            u = s / 79
            yy = top_y + (bot_y - top_y) * u
            spread = w0 + (w1 - w0) * u
            cd.line([cx - spread, yy, cx + spread, yy], fill=int(255 * (1.0 - 0.35 * u)), width=6)
    sc.highlight(np.asarray(cut, dtype=np.float32) / 255.0, strength=1.05, blur=16)

    # 2) 水口两侧压深，靠明暗而非描边界定岩壁
    wall = Image.new("L", (W, H), 0)
    wd = ImageDraw.Draw(wall)
    for top_y, bot_y, w0, w1 in segs:
        for sign in (-1, 1):
            pts = [(cx + sign * (w0 + (w1 - w0) * s / 10 + 16 + 6 * math.sin(s * 0.6)),
                    top_y + (bot_y - top_y) * s / 10) for s in range(11)]
            for i in range(len(pts) - 1):
                wd.line([pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]],
                        fill=120, width=16)
    wall = np.asarray(wall.filter(ImageFilter.GaussianBlur(9)), dtype=np.float32) / 255.0
    sc.commit(wall * 0.55)
    sc.flush(1.0)

    # 3) 道道水线：起笔重、行笔带波折、收笔淡；越往下越宽
    for top_y, bot_y, w0, w1 in segs:
        for i in range(3):
            off = (-0.45 + 0.45 * i) * w0
            pts = []
            for s in range(14):
                u = s / 13
                yy = top_y + (bot_y - top_y) * u
                xx = cx + off * (1.0 + 0.9 * u) + math.sin(u * 5.0 + i * 2.1) * 3.4 * (0.5 + u)
                pts.append((xx, yy))
            sc.brush(pts, width=1.9, alpha=118, taper=True, jitter=0.5)
        sc.flush(1.0)

    # 4) 崖唇与潭面的横向短笔
    d = ImageDraw.Draw(sc.sheet)
    for top_y, bot_y, w0, w1 in segs:
        for _ in range(22):
            side = rng.choice([-1, 1])
            u = rng.random()
            x = cx + side * (w1 * (0.6 + 0.7 * rng.random()))
            y = top_y + (bot_y - top_y) * u
            ln = rng.uniform(8, 26) * side
            d.line([x, y, x + ln, y + rng.uniform(-5, 5)], fill=int(rng.uniform(60, 130)),
                   width=2)
    for top_y, bot_y, w0, w1 in segs:
        for _ in range(50):
            x = cx + rng.gauss(0, w1 * 1.5 + 22)
            y = bot_y + rng.uniform(-14, 32)
            r = rng.uniform(2, 6.5)
            d.ellipse([x - r, y - r * 0.5, x + r, y + r * 0.5], fill=int(rng.uniform(70, 150)))
    sc.flush(0.9)

    # 5) 瀑顶小水口与潭
    top = Image.new("L", (W, H), 0)
    ImageDraw.Draw(top).polygon([(cx - 12, 980), (cx + 14, 980), (cx + 16, 1030),
                                 (cx - 14, 1030)], fill=235)
    sc.highlight(np.asarray(top, dtype=np.float32) / 255.0, strength=0.9, blur=9)
    pool = Image.new("L", (W, H), 0)
    ImageDraw.Draw(pool).ellipse([cx - 155, 1588, cx + 170, 1676], fill=255)
    sc.highlight(np.asarray(pool, dtype=np.float32) / 255.0, strength=0.42, blur=26)


def mist(sc):
    """云气：窄而透，只在山腰以下提亮。"""
    n = fractal_noise(W, H, sc.rng, 300, 3)
    arr = np.zeros((H, W), dtype=np.float32)
    for cy, th, stre in [(1380, 60, 0.62), (1560, 44, 0.46), (1230, 36, 0.26)]:
        yy = np.arange(H, dtype=np.float32)[:, None]
        g = np.exp(-((yy - cy) ** 2) / (2 * th * th))
        arr = np.maximum(arr, g * stre * (0.40 + 0.90 * n))
    arr = np.asarray(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
                     .filter(ImageFilter.GaussianBlur(12)), dtype=np.float32) / 255.0
    band = np.zeros((H, W), dtype=np.float32)
    band[1200:] = 1.0
    band = Image.fromarray((band * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(80))
    band = np.asarray(band, dtype=np.float32) / 255.0
    sc.highlight(arr * band, strength=0.66, blur=0)


def water_and_bank(sc):
    rng = sc.rng
    ys = np.arange(H, dtype=np.float32)[:, None]
    band = np.clip((ys - 1690) / 340.0, 0, 1) * 0.13
    band = band * (0.55 + 0.9 * fractal_noise(W, H, rng, 150, 2))
    sc.commit(band)

    for i in range(11):
        y = 1755 + i * 26
        alpha = max(14, 52 - i * 3.4)
        pts = [(x, y + math.sin(x / 250 + i * 0.8) * 4.0) for x in range(-20, W + 40, 28)]
        sc.brush(pts, width=1.4, alpha=alpha, taper=False, jitter=1.0)
    # 水口倒影：短促竖墨
    for _ in range(70):
        x = 880 + rng.gauss(0, 70)
        y0 = 1680 + rng.uniform(0, 120)
        sc.brush([(x, y0), (x, y0 + rng.uniform(8, 26))], width=rng.uniform(1.2, 2.4),
                 alpha=rng.uniform(30, 70), taper=False)
    sc.flush(0.75)

    bank = [(0, 1830), (150, 1760), (330, 1800), (470, 1880), (560, 2010), (600, 2200),
            (0, 2200)]
    mb = poly_mask(bank)
    sc.wash(bank, 1760, 2200, 0.34, grain=0.5)
    sc.flush(0.92, clip=mb)
    sc.axe(230, 2010, 180, 120, 180, mb)
    sc.brush(bank[:6], width=3.0, alpha=214, taper=True, jitter=0.8)
    sc.flush(1.0, clip=mb)
    sc.moss(None, 26, 40, 580, 1810, 2140, mb, r=(1.7, 4.0), alpha=(120, 225))

    up = [(80, 1836), (155, 1778), (240, 1822), (282, 1904), (255, 2050), (110, 2090),
          (60, 1990)]
    ru = spline(up[:5], 20, rough=2.6, rng=rng)
    mu = poly_mask(ru + [(300, H), (60, H)])
    sc.commit(mu * 0.22)
    sc.brush(ru, width=2.6, alpha=198, taper=True, jitter=0.8)
    sc.flush(1.0, clip=mu)
    sc.moss(None, 14, 80, 280, 1830, 2050, mu, r=(1.4, 3.4), alpha=(110, 205))

    ctrl = [(1210, 1930), (1305, 1862), (1420, 1896), (1530, 1960), (1600, 2050)]
    r2 = spline(ctrl, 22, rough=2.6, rng=rng)
    mr = poly_mask(r2 + [(1620, H), (1150, H)])
    sc.wash(r2 + [(1620, H), (1150, H)], min(p[1] for p in r2), 2200, 0.40, grain=0.55)
    sc.flush(0.92, clip=mr)
    sc.axe(1400, 2060, 130, 100, 150, mr)
    sc.brush(r2, width=3.4, alpha=228, taper=True, jitter=0.9)
    sc.flush(1.0, clip=mr)
    sc.moss(None, 28, 1210, 1590, 1900, 2150, mr, r=(1.7, 4.0), alpha=(120, 225))

    for _ in range(160):
        x = rng.choice([rng.uniform(-10, 620), rng.uniform(1160, W + 10)])
        y = rng.uniform(1750, 2170)
        ln = rng.uniform(26, 90)
        a = math.radians(rng.uniform(-118, -62))
        sc.brush([(x, y), (x + math.cos(a) * ln * 0.5, y + math.sin(a) * ln)],
                 width=rng.uniform(0.8, 1.6), alpha=rng.uniform(70, 165), taper=True)
    sc.flush(0.85)


def boat(sc, x, y, s=1.0):
    sc.brush([(x - 92 * s, y), (x - 56 * s, y + 20 * s), (x + 56 * s, y + 20 * s),
              (x + 94 * s, y)], width=4.0 * s, alpha=220, taper=True)
    sc.brush([(x - 62 * s, y + 5 * s), (x + 64 * s, y + 5 * s)], width=2.0 * s, alpha=145,
             taper=False)
    sc.brush([(x - 20 * s, y + 2 * s), (x - 20 * s, y - 28 * s), (x + 32 * s, y - 28 * s),
              (x + 32 * s, y + 2 * s)], width=3.0 * s, alpha=208, taper=False)
    sc.brush([(x + 48 * s, y - 78 * s), (x + 78 * s, y + 28 * s)], width=2.2 * s,
             alpha=188, taper=True)
    d = ImageDraw.Draw(sc.sheet)
    d.ellipse([x + 40 * s, y - 44 * s, x + 52 * s, y - 30 * s], fill=208)
    for k in range(1, 4):
        rx, ry = 96 * s + k * 34 * s, 9 * s + k * 4 * s
        d.arc([x - rx, y + 26 * s - ry / 2, x + rx, y + 26 * s + ry / 2],
              0, 180, fill=max(18, 56 - k * 12))
    for i in range(5):
        sc.brush([(x - 80 * s + i * 34 * s, y + 34 * s + i * 9 * s),
                  (x - 46 * s + i * 34 * s, y + 34 * s + i * 9 * s)],
                 width=2.0 * s, alpha=42 - i * 6, taper=False)
    sc.flush(0.96)


def colophon(sc):
    title = font_at(58)
    verse = font_at(32)
    if title is None:
        return
    lay = Image.new("L", (W, H), 0)

    def vtext(text, x, y0, fnt, step):
        yy = y0
        for ch in text:
            tile = Image.new("L", (110, 110), 0)
            ImageDraw.Draw(tile).text((55, 55), ch, fill=255, font=fnt, anchor="mm")
            box = (int(x - 55), int(yy - 55), int(x + 55), int(yy + 55))
            lay.paste(Image.fromarray(np.maximum(np.asarray(lay.crop(box)),
                                                np.asarray(tile))), box[:2])
            yy += step

    vtext("\u5c71\u6cb3\u7ed8", 1415, 300, title, 76)
    if verse is not None:
        vtext("\u8fdc\u5cab\u4e91\u5f00\u89c1\u7fe0\u5fae", 1415, 668, verse, 46)

    seal = Image.new("L", (120, 120), 0)
    sd = ImageDraw.Draw(seal)
    sd.rectangle([5, 5, 115, 115], outline=255, width=7)
    sf = font_at(60)
    if sf is not None:
        sd.text((60, 62), "\u5c71", fill=255, font=sf, anchor="mm")
    box = (1370, 1130, 1490, 1250)
    lay.paste(Image.fromarray(np.maximum(np.asarray(lay.crop(box)), np.asarray(seal))), box[:2])
    sc.commit(np.asarray(lay, dtype=np.float32) / 255.0)


def composite(sc, paper_noise):
    base = np.repeat(PAPER[None, None, :], W, axis=1).repeat(H, axis=0)
    base = base * (0.968 + 0.062 * paper_noise[..., None])

    ink = np.zeros((H, W), dtype=np.float32)
    for arr, w in sc.layers:
        if w >= 0:
            ink += arr * w
        else:
            ink -= arr * (-w) * np.clip(ink, 0, 1)
    ink = np.clip(ink, 0, 0.97)

    out = base * (1.0 - ink[..., None]) + INK[None, None, :] * ink[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def main(seed=SEED, suffix=""):
    rng = random.Random(seed)
    np.random.seed(seed)
    sc = Scene(rng)

    paper = fractal_noise(W, H, rng, 110, 4)
    paper = (paper - paper.min()) / max(1e-6, float(paper.max() - paper.min()))

    sky(sc)
    distant(sc)
    ridge, peak_y = main_peak(sc)
    waterfall(sc, ridge, peak_y)
    mist(sc)
    water_and_bank(sc)
    sc.tree(290, 1975, 500, trunk_w=17, lean=-0.05, density=1.05)
    sc.tree(455, 2075, 320, trunk_w=11, lean=0.06)
    sc.tree(1300, 2030, 300, trunk_w=10, lean=-0.04)
    sc.tree(1440, 2105, 215, trunk_w=7, lean=0.05)
    boat(sc, 880, 1930, 1.0)
    colophon(sc)

    img = composite(sc, paper).filter(ImageFilter.GaussianBlur(0.35))

    margin = 70
    framed = Image.new("RGB", (W + margin * 2, H + margin * 2), (250, 246, 236))
    framed.paste(img, (margin, margin))
    ImageDraw.Draw(framed).rectangle(
        [margin - 12, margin - 12, W + margin + 11, H + margin + 11],
        outline=(152, 142, 124), width=2)

    os.makedirs(OUT_DIR, exist_ok=True)
    png = OUT_PNG if not suffix else OUT_PNG.replace(".png", "%s.png" % suffix)
    jpg = OUT_JPG if not suffix else OUT_JPG.replace(".jpg", "%s.jpg" % suffix)
    framed.save(png)
    framed.convert("RGB").save(jpg, quality=92)
    print("OK: %s (%dx%d)" % (png, framed.width, framed.height))
    return 0


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    if argv and argv[0] == "--seed":
        raise SystemExit(main(int(argv[1]), argv[2] if len(argv) > 2 else "_s%s" % argv[1]))
    raise SystemExit(main())




