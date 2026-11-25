# -*- coding: utf-8 -*-
"""把名家参数渲染成一张可画的中国山水，同时记录逐笔轨迹。"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from . import brush, trajectory


class Group:
    def __init__(self, size):
        self.color = Image.new("RGBA", size, (0, 0, 0, 0))
        self.ink = Image.new("RGBA", size, (0, 0, 0, 0))
        self.dc = ImageDraw.Draw(self.color)
        self.di = ImageDraw.Draw(self.ink)


class MasterRenderer:
    def __init__(self, profile, layout, seed=0, width=1600, detail=1.0, abstraction=0.0,
                 include_trees=False, line_only=False):
        self.p = profile
        self.lay = layout
        self.seed = int(seed)
        self.rng = random.Random(seed + 9173)
        self.npr = np.random.default_rng(seed + 9173)
        rw, rh = layout.get("ratio", (16, 10))
        self.W = int(width)
        self.H = int(round(width * rh / rw))
        self.scale = self.W / 1600.0
        self.detail = detail * (0.88 + 0.10 * (seed % 4))
        self.abstraction = float(max(0.0, min(0.88, abstraction)))
        self.include_trees = bool(include_trees)
        self.clean_lines = True
        self.line_only = bool(line_only)
        self.groups = {k: Group((self.W, self.H)) for k in ("far", "mid", "near")}
        self.plan = []
        self.image = None
        self.tree_templates = []
        self.stroke_library = []
        try:
            import json as _json
            lp = Path(__file__).resolve().parent / "data" / "stroke_library" / (str(profile.get("id")) + ".json")
            if lp.exists():
                self.stroke_library = _json.loads(lp.read_text(encoding="utf-8"))
        except Exception:
            self.stroke_library = []
        try:
            import json as _json
            tp = Path(__file__).resolve().parent / "data" / "tree_model.json"
            if tp.exists():
                self.tree_templates = _json.loads(tp.read_text(encoding="utf-8")).get("templates", [])
        except Exception:
            self.tree_templates = []
        self.stroke_library = []
        try:
            import json as _json
            lp = Path(__file__).resolve().parent / "data" / "stroke_library" / (str(profile.get("id")) + ".json")
            if lp.exists():
                self.stroke_library = _json.loads(lp.read_text(encoding="utf-8"))
        except Exception:
            self.stroke_library = []
        self.peak_points = {}

    def pt(self, nx, ny):
        return (float(nx) * self.W, float(ny) * self.H)

    def ink_rgb(self, depth="mid"):
        c = self.p["palette"]["ink"]
        if depth == "far":
            return tuple(min(255, int(v + 48)) for v in c)
        if depth == "near":
            return tuple(max(0, int(v - 2)) for v in c)
        return tuple(c)

    def color_rgb(self, name):
        return tuple(self.p["palette"].get(name, self.p["palette"]["ink"]))

    def _keep(self, role):
        # 细节密度直接控制碎笔的保留率：0.7 约保留一半，1.3 全保留并配合更多笔数
        if role in ("texture", "moss", "needle", "water", "bark"):
            detail_keep = max(0.30, min(1.0, 0.42 + (self.detail - 0.70) * 1.75))
            if self.rng.random() > detail_keep:
                return False
        if self.abstraction <= 0.001:
            return True
        a = self.abstraction
        if role in ("outline", "tree", "detail"):
            keep = 1.0 - a * 0.15
        elif role == "needle":
            keep = 1.0 - a * 1.0
        elif role == "bark":
            keep = 1.0 - a * 1.4
        elif role == "water":
            keep = 1.0 - a * 1.9
        elif role == "moss":
            keep = 1.0 - a * 2.3
        else:
            keep = 1.0 - a * 1.7
        return self.rng.random() <= max(0.0, keep)

    def _add(self, pts, width, alpha, role, depth="mid", color="ink", dry=None,
             taper=None, pressure=None, smooth=True, record=True, jitter=0.0, speed=None):
        if len(pts) < 2 or not self._keep(role):
            return
        pts = [(float(x), float(y)) for x, y in pts]
        if smooth and len(pts) >= 3:
            pts = brush.catmull_rom(pts, 7)
        pts = brush.resample(pts, max(1.2, 1.7 * self.scale))
        if len(pts) < 2:
            return
        if jitter > 0:
            pts = self._wobble(pts, jitter * self.scale)
        n = len(pts)
        if pressure is None:
            pressure = brush.pressure_envelope(
                n, 0.08, 1.0, taper if taper is not None else self.p["line"]["taper"],
                self.rng, 0.16)
        elif len(pressure) != n:
            pressure = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(pressure)), pressure)
        pressure = np.asarray(pressure, dtype=np.float32)
        base_speed = float(speed if speed is not None else self.p["line"]["speed"])
        motion = self.p.get("motion", {})
        pts, pressure, speeds = trajectory.optimize(
            pts, pressure, base_speed,
            curvature_slow=motion.get("curvature_slow", 0.40),
            press_gain=motion.get("press_gain", 0.07),
            min_speed=motion.get("min_speed", 18))
        n = len(pts)
        base_w = max(1.0, width * self.scale)
        a = int(max(4, min(255, alpha)))
        if color == "ink" and role in ("texture", "moss", "needle"):
            if depth == "far":
                a = int(a * 0.55)
            elif depth == "near":
                a = int(min(255, a * 1.10))
        dry = self.p["line"]["dry"] if dry is None else dry
        group = self.groups[depth]
        target = group.di if color == "ink" else group.dc
        rgb = (24, 27, 23) if self.line_only else (self.ink_rgb(depth) if color == "ink" else self.color_rgb(color))
        if self.line_only:
            a = int(min(255, a * 1.15))
        if self.clean_lines and role in ("ridge", "outline", "texture", "branch", "tree"):
            image = group.ink if color == "ink" else group.color
            brush.draw_ink_stroke(image, pts, base_w, rgb, a, pressure, self.rng,
                                  dry=dry, taper=taper or self.p["line"]["taper"])
        else:
            brush.draw_stroke(target, pts, base_w, rgb, a, pressure, self.rng,
                              dry=dry, taper=taper or self.p["line"]["taper"],
                              bristle=2 if base_w >= 2.2 else 1)
        if record:
            step = max(1, n // 220)
            keep = list(range(0, n, step))
            if keep[-1] != n - 1:
                keep.append(n - 1)
            self.plan.append({
                "points": [[round(pts[i][0], 2), round(pts[i][1], 2)] for i in keep],
                "press": [round(float(pressure[i]), 3) for i in keep],
                "width_px": round(base_w, 2),
                "alpha": a,
                "role": role,
                "layer": depth,
                "color": color,
                "speed": [round(float(speeds[i]), 1) for i in keep],
                "curvature": None,
            })

    def _wobble(self, pts, amp):
        n = len(pts)
        if n < 3 or amp <= 0:
            return pts
        vals = np.array([self.rng.gauss(0, 1) for _ in range(max(4, n // 6))], dtype=np.float32)
        vals = np.convolve(vals, np.ones(3) / 3, mode="same")
        vals = np.interp(np.linspace(0, len(vals) - 1, n), np.arange(len(vals)), vals)
        out = []
        for i, (x, y) in enumerate(pts):
            nx, ny = brush.normal_at(pts, i)
            out.append((x + nx * vals[i] * amp, y + ny * vals[i] * amp))
        return out

    def _wash_dab(self, x, y, rx, ry, alpha, depth="mid", color="ink", record=True):
        group = self.groups[depth]
        rgb = (24, 27, 23) if self.line_only else (self.ink_rgb(depth) if color == "ink" else self.color_rgb(color))
        if self.line_only:
            a = int(min(255, a * 1.15))
        group.dc.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(*rgb, int(alpha)))
        if record and alpha >= 28:
            self.plan.append({
                "points": [[round(x - rx * 0.6, 2), round(y, 2)], [round(x + rx * 0.6, 2), round(y, 2)]],
                "press": [0.28, 0.24], "width_px": round(rx * 0.8, 2), "alpha": int(alpha),
                "role": "texture", "layer": depth, "color": color,
                "speed": [max(18, int(self.p["line"]["speed"] * 0.65))] * 2,
            })

    def _dot(self, x, y, size, alpha, role="moss", depth="mid", color="ink", stretch=1.2):
        if not self._keep(role):
            return
        w = max(1.2, size * self.scale)
        group = self.groups[depth]
        rgb = (24, 27, 23) if self.line_only else (self.ink_rgb(depth) if color == "ink" else self.color_rgb(color))
        if self.line_only:
            a = int(min(255, a * 1.15))
        target = group.di if color == "ink" else group.dc
        r = w * 0.52
        target.ellipse([x - r * stretch, y - r, x + r * stretch, y + r], fill=(*rgb, int(alpha)))
        if role != "wash":
            self.plan.append({
                "points": [[round(x - r * stretch, 2), round(y, 2)], [round(x + r * stretch, 2), round(y, 2)]],
                "press": [0.92, 0.72], "width_px": round(w, 2), "alpha": int(alpha),
                "role": role, "layer": depth, "color": color,
                "speed": [int(self.p["line"]["speed"]), int(self.p["line"]["speed"])],
            })

    def _paint_alpha(self, depth, color, alpha):
        group = self.groups[depth]
        a = np.clip(alpha, 0, 1)
        rgba = np.zeros((self.H, self.W, 4), dtype=np.uint8)
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = np.clip(a * 255, 0, 255).astype(np.uint8)
        group.color = Image.alpha_composite(group.color, Image.fromarray(rgba, "RGBA"))

    def _ink_wash(self, depth, mask, alpha, color=None, edge=True, multiply=None):
        a = brush.mask_array(mask)
        if multiply is not None:
            a = a * multiply
        if edge:
            er = np.asarray(mask.filter(ImageFilter.GaussianBlur(max(1.2, 2.2 * self.scale))), dtype=np.float32) / 255.0
            ed = np.clip(a - er * 0.93, 0, 1)
            a = np.clip(a + ed * 0.30, 0, 1)
        grain = self.npr.normal(1.0, 0.075, a.shape).astype(np.float32)
        # 低频云斑：让墨色有浓淡，不是一整块平涂
        small = self.npr.random((max(6, self.H // 55), max(8, self.W // 55))).astype(np.float32)
        cloud = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((self.W, self.H), Image.BICUBIC), dtype=np.float32) / 255.0
        a = a * (0.58 + 0.72 * cloud)
        self._paint_alpha(depth, color or self.ink_rgb(depth), np.clip(a * grain, 0, 1) * alpha)

    def _mountain_geometry(self, m):
        W, H = self.W, self.H
        cx = m["cx"] * W
        base = m["base"] * H
        h = m["height"] * H * 0.92
        w = m["width"] * W * 1.28
        lean = m.get("lean", 0.0)
        main_u = 0.50 + lean * 0.12
        # 3–5 个真实的山峰，而不是一个钟形包络
        peaks = [(main_u, 1.0, self.rng.uniform(0.22, 0.30))]
        for _ in range(self.rng.randint(1, 3)):
            u = self.rng.uniform(0.08, 0.92)
            if min(abs(u - q[0]) for q in peaks) < 0.18:
                continue
            peaks.append((u, self.rng.uniform(0.40, 0.75), self.rng.uniform(0.14, 0.22)))
        n = 56
        ridge = []
        phase = self.rng.uniform(0, 6.28)
        for i in range(n + 1):
            u = i / n
            v = 0.012
            for pu, ph, pw in peaks:
                t = abs(u - pu) / pw
                if t < 1.0:
                    v = max(v, ph * max(0.0, 1.0 - t * t) ** 0.82)
            # 宽厚山脚：先有一座山的体量，再谈峰
            v = max(v, 0.16 * max(0.0, 1.0 - (2 * u - 1) ** 2) ** 0.75)
            v = max(v, 0.10 * math.exp(-((u - 0.5) / 0.55) ** 2))
            v += 0.030 * math.sin(u * 47 + phase) * (0.25 + v)
            v += self.rng.uniform(-1, 1) * m.get("jag", 0.10) * 0.12 * (0.2 + v)
            v = max(0.0, v)
            x = cx - w / 2 + u * w
            ridge.append((x, base - h * v))
        # 垂直崖口
        for _ in range(self.rng.randint(1, 3)):
            i = self.rng.randint(4, max(5, len(ridge) - 5))
            x, y = ridge[i]
            ridge.insert(i + 1, (x + self.rng.uniform(1, 4) * self.scale,
                                 y + h * self.rng.uniform(0.04, 0.11)))
        bottom = []
        for k in range(7):
            bx = cx - w / 2 + w * k / 6
            by = base + (self.rng.uniform(-0.012, 0.020) * H)
            bottom.append((bx, by))
        poly = [(cx - w / 2, base + 2)] + ridge + bottom[::-1]
        peak = min(ridge, key=lambda q: q[1])
        return poly, ridge, peak

    def _draw_mountain(self, m, depth):
        W, H = self.W, self.H
        poly, ridge, peak = self._mountain_geometry(m)
        mask = brush.soft_polygon((W, H), poly, blur=max(0.6, 1.3 * self.scale), feather=0)
        if not mask.getbbox():
            return
        self.peak_points[id(m)] = peak
        if self.line_only:
            self._ridge_outline(ridge, m, depth)
            if self.p["texture"]["type"] == "pima":
                self._pima_from_ridge(ridge, mask, m, depth, peak, 1.35)
            elif self.p["texture"]["type"] != "none":
                for _ in range(3):
                    self._texture(mask, m, depth, peak)
            return
        base_a = (0.055 if depth == "far" else 0.14 if depth == "mid" else 0.22) * (0.60 + 0.60 * m.get("tone", 0.6))
        # 山脚渐隐：把平直的底边藏进云气里
        fade = np.clip((m["base"] * H - np.mgrid[0:H, 0:W][0]) / (H * 0.16), 0, 1)
        fade = fade ** 1.8
        self._ink_wash(depth, mask, base_a * self.detail, self.ink_rgb(depth), multiply=fade)
        side = np.mgrid[0:H, 0:W][1].astype(np.float32) / max(1, W)
        side = np.clip((side - (m["cx"] - m["width"] * 0.5)) / max(0.08, m["width"]), 0, 1)
        shade = 0.35 + 0.65 * (0.45 + 0.55 * np.sin(side * math.pi))
        a = brush.mask_array(mask) * fade * shade
        self._paint_alpha(depth, self.ink_rgb(depth), a * base_a * 0.70 * fade)
        # 山脊压深：让轮廓附近先有一层重墨，山脚再淡下去
        band_top = [(x, y) for x, y in ridge]
        band_bottom = [(x, y + H * 0.055 + self.rng.uniform(-1, 1) * H * 0.01) for x, y in ridge]
        band = brush.soft_polygon((W, H), band_top + band_bottom[::-1], blur=2.5 * self.scale, feather=0)
        self._paint_alpha(depth, self.ink_rgb(depth), brush.mask_array(band) * (0.09 if depth != "far" else 0.05))
        # 积墨：按画家的墨法层数，错位叠几层淡墨，让山体有厚度而不糊成一块。
        layers = int(self.p["wash"].get("layers", 1))
        for k in range(1, max(1, layers)):
            dx = (k - 1) * 4 * self.scale
            dy = (k - 1) * 2 * self.scale
            m2 = mask.transform(mask.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), resample=Image.BICUBIC)
            am = brush.mask_array(m2) * fade * (0.55 + 0.45 * shade)
            self._paint_alpha(depth, self.ink_rgb(depth), am * base_a * (0.92 if self.p["wash"]["palette"] == "dark" else 0.34 if self.p["wash"]["palette"] == "wet" else 0.14))
        sat = self.p["wash"]["saturation"]
        if sat > 0.03 and m.get("color", 0) > 0.05:
            self._mountain_color(m, mask, peak, depth, sat)
        if depth == "far":
            # 远山只保留淡墨/淡色块，不画线，避免与中近景线条叠在一起
            return
        self._ridge_outline(ridge, m, depth)
        if depth != "near":
            # 中景必须有皴法，只是密度减半；否则画面只剩色块
            old_d = self.p["texture"]["density"]
            self.p["texture"]["density"] = max(0.30, old_d * 0.55)
            if self.p["texture"]["type"] == "pima":
                self._pima_from_ridge(ridge, mask, m, depth, peak, .70)
            else:
                self._texture(mask, m, depth, peak)
            self.p["texture"]["density"] = old_d
            self._mid_texture(mask, m, depth, peak)
            return
        self._texture_from_library(mask, m, depth)
        ttype = self.p["texture"]["type"]
        if ttype in ("pima", "zhedai", "fupi", "fupi_midian"):
            self._facets(ridge, peak, m, depth, mask)
        if ttype in ("pima", "zhedai"):
            self._contours(ridge, m, depth, mask)
            self._spurs(ridge, peak, m, depth, mask)
        if ttype != "none":
            if ttype == "pima":
                self._pima_from_ridge(ridge, mask, m, depth, peak, 1.0)
            else:
                self._texture(mask, m, depth, peak)
            self._moss(ridge, m, depth)
        if self.rng.random() < 0.72:
            self._color_sweeps(mask, m, depth, peak)

    def _mountain_color(self, m, mask, peak, depth, sat):
        W, H = self.W, self.H
        cx, cy = peak
        blue = self.color_rgb("blue")
        green = self.color_rgb("green")
        ochre = self.color_rgb("ochre")
        yy, xx = np.mgrid[0:H, 0:W]
        base_a = brush.mask_array(mask)
        base_fade = np.clip((m["base"] * H - yy) / (H * 0.16), 0, 1) ** 1.2
        base_a = base_a * base_fade
        top_glow = np.exp(-(((xx - cx) / (W * 0.24)) ** 2 + ((yy - cy - H * 0.10) / (H * 0.20)) ** 2) * 1.6)
        mid_glow = np.exp(-(((xx - cx) / (W * 0.32)) ** 2 + ((yy - (cy + H * 0.22)) / (H * 0.18)) ** 2) * 1.2)
        a_blue = np.clip(base_a * top_glow * 1.25 * sat * m.get("color", 0.5), 0, 0.70)
        a_green = np.clip(base_a * mid_glow * 1.10 * sat * m.get("color", 0.5), 0, 0.62)
        self._paint_alpha(depth, blue, a_blue * self.detail)
        self._paint_alpha(depth, green, a_green * self.detail)
        if self.p["wash"]["palette"] == "light_ochre":
            foot = brush.gradient_array(mask, "y", 0.28, 0.01, gamma=1.2) * base_fade
            self._paint_alpha(depth, ochre, foot * 0.34 * sat)
        elif self.p["wash"]["palette"] == "wet":
            foot = brush.gradient_array(mask, "y", 0.22, 0.01, gamma=1.0) * base_fade
            self._paint_alpha(depth, ochre, foot * 0.09 * sat)

    def _texture_from_library(self, mask, m, depth):
        if depth == "far" or not self.stroke_library:
            return
        bbox = mask.getbbox()
        if not bbox:
            return
        x0, y0, x1, y1 = bbox
        entry = self.rng.choice(self.stroke_library)
        strokes = entry.get("strokes", [])
        if not strokes:
            return
        self.rng.shuffle(strokes)
        ml = mask.load()
        n = 0
        for st in strokes:
            if n >= 14:
                break
            pts = [(x0 + px * (x1 - x0), y0 + py * (y1 - y0)) for px, py in st]
            inside = [p for p in pts if 0 <= int(p[0]) < self.W and 0 <= int(p[1]) < self.H and ml[int(p[0]), int(p[1])] > 30]
            if len(inside) < 3:
                continue
            self._add(inside, self.p["texture"]["width"] * self.rng.uniform(0.35, 0.60),
                      self.p["line"]["alpha"] * self.rng.uniform(0.55, 0.78), "texture", depth,
                      dry=0.35, taper=0.3, jitter=0.3)
            n += 1

    def _mid_texture(self, mask, m, depth, peak):
        """中景只给极淡的结构线，主要靠墨色层次，不与近景线条打架。"""
        W, H = self.W, self.H
        ml = mask.load()
        bbox = mask.getbbox()
        if not bbox:
            return
        x0, y0, x1, y1 = bbox
        n = 6 + int(4 * self.detail)
        for _ in range(n):
            x = self.rng.uniform(x0, x1); y = self.rng.uniform(y0, y1)
            if not (0 <= int(x) < W and 0 <= int(y) < H) or ml[int(x), int(y)] < 40:
                continue
            ang = math.atan2(y - peak[1], x - peak[0]) + self.rng.uniform(-0.2, 0.2)
            pts = [(x, y)]
            for k in range(3):
                ang += self.rng.uniform(-0.08, 0.08)
                x += math.cos(ang) * self.p["texture"]["length"] * self.scale * 0.14
                y += math.sin(ang) * self.p["texture"]["length"] * self.scale * 0.14
                if not (0 <= int(x) < W and 0 <= int(y) < H) or ml[int(x), int(y)] < 30:
                    break
                pts.append((x, y))
            if len(pts) >= 2:
                self._add(pts, self.p["texture"]["width"] * 0.35,
                          self.p["line"]["alpha"] * 0.32, "texture", depth,
                          dry=0.5, taper=0.4, jitter=0.4)
        for _ in range(2):
            x = self.rng.uniform(x0, x1); y = self.rng.uniform(y0, y1)
            if ml[int(x), int(y)] > 40:
                self._wash_dab(x, y, (x1 - x0) * .28, (y1 - y0) * .12, 22, depth)
        # 中景补 3 组很淡的披麻，不要密
        for _ in range(3):
            x = self.rng.uniform(x0, x1); y = self.rng.uniform(y0, y1)
            if ml[int(x), int(y)] < 40:
                continue
            ang = math.atan2(y - peak[1], x - peak[0]) + self.rng.uniform(-.15, .15)
            for j in range(4):
                ox = math.cos(ang + math.pi / 2) * (j - 1.5) * 4 * self.scale
                oy = math.sin(ang + math.pi / 2) * (j - 1.5) * 4 * self.scale
                pts = [(x + ox, y + oy)]
                for k in range(3):
                    px = pts[-1][0] + math.cos(ang) * 9 * self.scale
                    py = pts[-1][1] + math.sin(ang) * 9 * self.scale
                    if not (0 <= int(px) < W and 0 <= int(py) < H) or ml[int(px), int(py)] < 30:
                        break
                    pts.append((px, py))
                if len(pts) >= 2:
                    self._add(pts, self.p["texture"]["width"] * .45,
                              self.p["line"]["alpha"] * .42, "texture", depth,
                              dry=.55, taper=.35, jitter=.3)

    def _line_contours(self, mask, ridge, m, depth):
        """纯线稿：沿山脊向内偏移的多层等高线，像参考图的层层石纹。"""
        W, H = self.W, self.H
        ml = mask.load()
        h = m["height"] * H
        for level in range(1, 8):
            off = h * (0.055 + 0.075 * level)
            pts = []
            for k, (x, y) in enumerate(ridge):
                yy = y + off + math.sin(k * 0.6 + level) * 2.5 * self.scale
                xx = x + math.sin(k * 0.45 + level * 1.7) * 3.5 * self.scale
                if not (0 <= int(xx) < W and 0 <= int(yy) < H) or ml[int(xx), int(yy)] < 25:
                    if len(pts) >= 4:
                        self._add(pts, self.p["texture"]["width"] * .55,
                                  235 - level * 12, "texture", depth, dry=.05, taper=.3, jitter=.2)
                    pts = []
                else:
                    pts.append((xx, yy))
            if len(pts) >= 4:
                self._add(pts, self.p["texture"]["width"] * .55,
                          235 - level * 12, "texture", depth, dry=.05, taper=.3, jitter=.2)
        # 少量从山脊向下的短披麻，打破纯等高线
        for _ in range(6):
            x = self.rng.uniform(m["cx"] * W - m["width"] * W * .25, m["cx"] * W + m["width"] * W * .25)
            y = min(yy for _, yy in ridge)
            pts = [(x, y)]
            for k in range(4):
                x += self.rng.uniform(-6, 6) * self.scale
                y += h * .10
                if not (0 <= int(x) < W and 0 <= int(y) < H) or ml[int(x), int(y)] < 25:
                    break
                pts.append((x, y))
            if len(pts) >= 2:
                self._add(pts, self.p["texture"]["width"] * .38, 190, "texture", depth,
                          dry=.1, taper=.3, jitter=.2)

    def _ridge_outline(self, ridge, m, depth):
        i = 0
        n = len(ridge)
        line = self.p["line"]
        while i < n - 2:
            span = self.rng.randint(2, 4)
            seg = ridge[i:i + span + 1]
            if len(seg) >= 2 and self.rng.random() > line["broken"] * 0.65:
                width = line["width"] * (1.25 if depth == "far" else 1.65) * (0.9 + 0.28 * self.rng.random())
                alpha = min(255, line["alpha"] * (0.92 if depth == "far" else 0.58 if depth == "mid" else 1.18) * (0.9 + 0.16 * self.rng.random()))
                self._add(seg, width, alpha, "outline", depth,
                          dry=line["dry"] * (1.3 if depth == "far" else 1.0),
                          jitter=line["wobble"] * 0.45)
                if self.rng.random() < line["double"]:
                    off = [(x, y + self.rng.uniform(2, 5) * self.scale) for x, y in seg]
                    self._add(off, width * 0.45, alpha * 0.65, "outline", depth,
                              dry=line["dry"] * 1.4, jitter=line["wobble"] * 0.35)
            i += span

    def _facets(self, ridge, peak, m, depth, mask):
        """把山体分成几道大的石面，皴法才有地方长。"""
        if depth == "far":
            return
        W, H = self.W, self.H
        ml = mask.load()
        h = m["height"] * H
        count = 2 + int(4 * self.detail)
        for _ in range(max(2, count)):
            idx = self.rng.randint(3, max(4, len(ridge) - 4))
            x, y = ridge[idx]
            pts = [(x, y)]
            drift = (x - peak[0]) / max(1.0, m["width"] * self.W)
            for k in range(5):
                y += h * self.rng.uniform(0.065, 0.11)
                x += drift * h * 0.06 + self.rng.uniform(-1, 1) * 7 * self.scale
                if not (0 <= int(x) < W and 0 <= int(y) < H) or ml[int(x), int(y)] < 30:
                    break
                pts.append((x, y))
            if len(pts) < 3:
                continue
            self._add(pts, self.p["line"]["width"] * self.rng.uniform(0.42, 0.62),
                      self.p["line"]["alpha"] * 0.62, "outline", depth,
                      dry=self.p["line"]["dry"] * 1.1, taper=0.3, jitter=0.7)
            if self.rng.random() < 0.55:
                ox = self.p["line"]["width"] * self.rng.uniform(1.2, 2.2)
                self._add([(p[0] + ox, p[1]) for p in pts],
                          self.p["line"]["width"] * 0.28, self.p["line"]["alpha"] * 0.38,
                          "texture", depth, dry=0.55, taper=0.3, jitter=0.5)

    def _contours(self, ridge, m, depth, mask):
        """山体内部的层理线：让皴法有结构，不是一把乱草。"""
        W, H = self.W, self.H
        ml = mask.load()
        h = m["height"] * H
        levels = [0.07, 0.16, 0.27, 0.40]
        for li, lv in enumerate(levels):
            pts = []
            for x, y in ridge:
                nx = x + self.rng.uniform(-2, 2) * self.scale
                ny = y + h * lv + self.rng.uniform(-1, 1) * h * 0.018
                if 0 <= int(nx) < W and 0 <= int(ny) < H and ml[int(nx), int(ny)] > 30:
                    pts.append((nx, ny))
                else:
                    if len(pts) >= 3:
                        self._contour_segments(pts, li, depth)
                    pts = []
            if len(pts) >= 3:
                self._contour_segments(pts, li, depth)

    def _contour_segments(self, pts, level, depth):
        line = self.p["line"]
        i = 0
        n = len(pts)
        while i < n - 2:
            span = self.rng.randint(3, 7)
            seg = pts[i:i + span]
            if len(seg) >= 2 and self.rng.random() > 0.42:
                self._add(seg, line["width"] * (0.38 + 0.16 * (level % 3)),
                          min(220, line["alpha"] * (0.72 - 0.07 * level)),
                          "texture", depth, dry=line["dry"] * 1.4, taper=0.32, jitter=0.5)
                if self.rng.random() < 0.45:
                    off = [(x, y + self.rng.uniform(2, 5) * self.scale) for x, y in seg]
                    self._add(off, line["width"] * 0.22, line["alpha"] * 0.30,
                              "texture", depth, dry=0.55, taper=0.3, jitter=0.4)
            i += span + self.rng.randint(1, 3)

    def _spurs(self, ridge, peak, m, depth, mask):
        line = self.p["line"]
        density = self.p["texture"]["spur"] * (0.5 + m.get("tone", 0.6))
        count = int(len(ridge) * density * 0.9)
        ml = mask.load()
        for _ in range(max(0, count)):
            idx = self.rng.randint(2, max(3, len(ridge) - 3))
            x, y = ridge[idx]
            h = m["height"] * self.H
            dirx = (x - peak[0]) / max(1.0, m["width"] * self.W) * 0.5
            pts = [(x, y)]
            for k in range(1, 4):
                x += dirx * h * 0.11 + self.rng.uniform(-1, 1) * 5 * self.scale
                y += h * self.rng.uniform(0.045, 0.085)
                if not (0 <= int(x) < self.W and 0 <= int(y) < self.H) or ml[int(x), int(y)] < 40:
                    break
                pts.append((x, y))
            if len(pts) >= 2:
                self._add(pts, line["width"] * 0.52, line["alpha"] * 0.46, "texture", depth,
                          dry=line["dry"] * 1.15, jitter=line["wobble"] * 0.5)

    def _texture(self, mask, m, depth, peak):
        tex = self.p["texture"]
        if tex["type"] == "none" or tex["density"] < 0.04:
            return
        if tex["type"] == "pima":
            self._texture_pima(mask, m, depth, peak, tex)
        elif tex["type"] == "zhedai":
            self._texture_zhedai(mask, m, depth, peak, tex)
        elif tex["type"] in ("fupi", "fupi_midian"):
            self._texture_fupi(mask, m, depth, peak, tex)
        elif tex["type"] == "midian":
            self._texture_midian(mask, m, depth, peak, tex)
        else:
            self._texture_random(mask, m, depth, peak, tex)

    def _pima_from_ridge(self, ridge, mask, m, depth, peak, factor=1.0):
        """披麻的骨架：从山脊落笔，顺坡向内下方走，避免随机竖线像下雨。"""
        if not ridge:
            return
        W, H = self.W, self.H
        ml = mask.load()
        h = m["height"] * H
        tex = self.p["texture"]
        # 脊线按弧长重采样，避免所有笔都挤在尖峰
        rp = brush.resample(ridge, max(8, int(len(ridge) / 10)))
        n = int((12 + 10 * tex["density"]) * max(.35, self.detail) * factor)
        for _ in range(max(5, n)):
            if self.rng.random() < .35:
                x0 = self.rng.uniform(m["cx"] * W - m["width"] * W * .32, m["cx"] * W + m["width"] * W * .32)
                y0 = self.rng.uniform(min(y for _, y in rp) + h * .10, m["base"] * H - h * .18)
            else:
                x0, y0 = rp[self.rng.randrange(2, max(3, len(rp) - 2))]
            side = 1 if x0 > peak[0] else -1
            length = h * self.rng.uniform(.26, .52)
            ex = x0 + side * length * self.rng.uniform(.05, .20)
            ey = y0 + length
            cx = x0 + side * length * self.rng.uniform(.04, .12)
            cy = y0 + length * .48
            pts = []
            for k in range(9):
                t = k / 8
                x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * ex
                y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * ey
                if not (0 <= int(x) < W and 0 <= int(y) < H) or ml[int(x), int(y)] < 22:
                    break
                pts.append((x, y))
            if len(pts) >= 4:
                self._add(pts, tex["width"] * self.rng.uniform(.45, .70),
                          min(255, tex["alpha"] * self.rng.uniform(.85, 1.15)), "texture", depth,
                          dry=.18, taper=.3, jitter=.25)
                # 其中一部分加一根并行短线，像复笔
                if self.rng.random() < .35:
                    off = tex["width"] * self.rng.uniform(1.2, 2.2)
                    p2 = [(x + off, y) for x, y in pts]
                    self._add(p2, tex["width"] * .32, tex["alpha"] * .65, "texture", depth,
                              dry=.4, taper=.35, jitter=.2)

    def _texture_pima(self, mask, m, depth, peak, tex):
        """披麻：成组、顺坡、略带弧度的长线；不要草叶一样的密集短线。"""
        W, H = self.W, self.H
        ml = mask.load()
        x0, y0, x1, y1 = mask.getbbox() or (0, 0, W, H)
        mw, mh = max(1, x1 - x0), max(1, y1 - y0)
        groups = int((5 + 4 * tex["density"]) * (0.75 + 0.35 * self.detail))
        for _ in range(max(4, groups)):
            sx = self.rng.uniform(x0 + mw * .12, x1 - mw * .12)
            sy = self.rng.uniform(y0 + mh * .10, y0 + mh * .46)
            if not (0 <= int(sx) < W and 0 <= int(sy) < H) or ml[int(sx), int(sy)] < 30:
                continue
            side = 1 if sx > peak[0] else -1
            n_line = self.rng.randint(3, 4)
            for li in range(n_line):
                bx = sx + (li - (n_line - 1) / 2) * tex["width"] * self.rng.uniform(2.6, 3.8)
                by = sy + self.rng.uniform(-2, 2) * self.scale
                ln = tex["length"] * self.scale * self.rng.uniform(0.75, 1.15)
                ex = bx + side * ln * self.rng.uniform(0.10, 0.24)
                ey = by + ln
                cx = (bx + ex) / 2 - side * ln * self.rng.uniform(0.06, 0.16)
                cy = (by + ey) / 2
                pts = []
                for k in range(9):
                    t = k / 8
                    x = (1 - t) ** 2 * bx + 2 * (1 - t) * t * cx + t * t * ex
                    y = (1 - t) ** 2 * by + 2 * (1 - t) * t * cy + t * t * ey
                    if not (0 <= int(x) < W and 0 <= int(y) < H) or ml[int(x), int(y)] < 24:
                        break
                    pts.append((x, y))
                if len(pts) >= 4:
                    self._add(pts, tex["width"] * self.rng.uniform(0.55, 0.85),
                              min(255, tex["alpha"] * self.rng.uniform(1.05, 1.35)), "texture", depth,
                              dry=self.p["line"]["dry"] * 0.8, taper=0.35, jitter=0.2)

    def _texture_zhedai(self, mask, m, depth, peak, tex):
        W, H = self.W, self.H
        ml = mask.load()
        x0, y0, x1, y1 = mask.getbbox() or (0, 0, W, H)
        bands = int(9 + 16 * tex["density"])
        for _ in range(max(5, bands)):
            y = self.rng.uniform(y0 + (y1 - y0) * 0.10, y0 + (y1 - y0) * 0.78)
            x = self.rng.uniform(x0, x1)
            steps = self.rng.randint(2, 5)
            pts = []
            for k in range(steps):
                nx = x + (x1 - x0) * self.rng.uniform(0.06, 0.16) * (1 if self.rng.random() < 0.7 else -1)
                ny = y + self.rng.uniform(-4, 4) * self.scale
                if 0 <= int(nx) < W and 0 <= int(ny) < H and ml[int(nx), int(ny)] > 35:
                    pts.extend([(x, y), (nx, y + self.rng.uniform(-2, 2) * self.scale), (nx, ny)])
                    x, y = nx, ny
                else:
                    break
            if len(pts) >= 2:
                self._add(pts, tex["width"] * self.rng.uniform(0.8, 1.3),
                          tex["alpha"] * self.rng.uniform(0.7, 1.1), "texture", depth,
                          dry=0.55, taper=0.25, jitter=0.3)

    def _texture_fupi(self, mask, m, depth, peak, tex):
        W, H = self.W, self.H
        ml = mask.load()
        x0, y0, x1, y1 = mask.getbbox() or (0, 0, W, H)
        n = int((45 + 90 * tex["density"]) * (0.7 + 0.5 * self.detail))
        for _ in range(max(20, n)):
            x = self.rng.uniform(x0, x1)
            y = self.rng.uniform(y0, y1)
            if not (0 <= int(x) < W and 0 <= int(y) < H) or ml[int(x), int(y)] < 50:
                continue
            ang = math.pi / 2 + self.rng.uniform(-0.45, 0.45)
            ln = tex["length"] * self.scale * self.rng.uniform(0.25, 0.55)
            pts = [(x, y)]
            for k in range(2):
                ang += self.rng.uniform(-0.12, 0.12)
                x += math.cos(ang) * ln / 2
                y += math.sin(ang) * ln / 2
                pts.append((x, y))
            self._add(pts, tex["width"] * self.rng.uniform(0.9, 1.5),
                      tex["alpha"] * self.rng.uniform(0.75, 1.2), "texture", depth,
                      dry=0.25, taper=0.2, jitter=0.6)
            if tex["type"] == "fupi_midian" and self.rng.random() < 0.35:
                rx = W * self.rng.uniform(0.010, 0.020)
                self._wash_dab(x + self.rng.uniform(-8, 8), y + self.rng.uniform(2, 10),
                               rx, rx * 0.6, self.rng.uniform(30, 55), depth)

    def _texture_midian(self, mask, m, depth, peak, tex):
        W, H = self.W, self.H
        ml = mask.load()
        x0, y0, x1, y1 = mask.getbbox() or (0, 0, W, H)
        n = int((18 + 30 * tex["density"]) * max(0.45, self.detail ** 1.7))
        for _ in range(max(12, n)):
            x = self.rng.uniform(x0, x1)
            y = self.rng.uniform(y0, y1)
            if not (0 <= int(x) < W and 0 <= int(y) < H) or ml[int(x), int(y)] < 45:
                continue
            # 先铺湿墨团：大而软、低透明，叠成"黑密"而不是撒芝麻
            for _k in range(self.rng.randint(3, 6)):
                rx = W * self.rng.uniform(0.012, 0.026)
                ry = rx * self.rng.uniform(0.45, 0.75)
                self._wash_dab(x + self.rng.uniform(-1.2, 1.2) * rx,
                               y + self.rng.uniform(-0.8, 0.8) * ry,
                               rx, ry, self.rng.uniform(26, 62), depth)
            # 只留少量清晰的米点作骨
            for _k in range(self.rng.randint(0, 2)):
                self._dot(x + self.rng.uniform(-10, 10) * self.scale,
                          y + self.rng.uniform(-8, 8) * self.scale,
                          tex["width"] * self.rng.uniform(0.55, 0.95),
                          self.rng.uniform(90, 150), "moss", depth)
            # 米点之间要有短披麻/折带把山石结构串起来
            if self.rng.random() < 0.55:
                pts = [(x, y)]
                ang = math.atan2(y - peak[1], x - peak[0]) + self.rng.uniform(-0.25, 0.25)
                for k in range(3):
                    ang += self.rng.uniform(-0.12, 0.12)
                    x += math.cos(ang) * tex["length"] * self.scale * 0.22
                    y += math.sin(ang) * tex["length"] * self.scale * 0.22
                    if not (0 <= int(x) < W and 0 <= int(y) < H) or ml[int(x), int(y)] < 35:
                        break
                    pts.append((x, y))
                if len(pts) >= 2:
                    self._add(pts, tex["width"] * self.rng.uniform(0.45, 0.75),
                              tex["alpha"] * self.rng.uniform(0.80, 1.15), "texture", depth,
                              dry=0.45, taper=0.3, jitter=0.5)

    def _texture_random(self, mask, m, depth, peak, tex):
        W, H = self.W, self.H
        ml = mask.load()
        x0, y0, x1, y1 = mask.getbbox() or (0, 0, W, H)
        n = int((50 + 90 * tex["density"]) * (0.7 + 0.5 * self.detail))
        for _ in range(max(15, n)):
            x = self.rng.uniform(x0, x1)
            y = self.rng.uniform(y0, y1)
            if not (0 <= int(x) < W and 0 <= int(y) < H) or ml[int(x), int(y)] < 60:
                continue
            pts = self._texture_path(x, y, peak, m, tex)
            inside = [p for p in pts if 0 <= int(p[0]) < W and 0 <= int(p[1]) < H and ml[int(p[0]), int(p[1])] > 35]
            if len(inside) >= 2:
                self._add(inside, tex["width"] * self.rng.uniform(0.8, 1.3),
                          tex["alpha"] * self.rng.uniform(0.7, 1.15), "texture", depth,
                          dry=self.p["line"]["dry"] * 1.1, taper=0.35, jitter=0.5)

    def _texture_path(self, x, y, peak, m, tex):
        typ = tex["type"]
        h = m["height"] * self.H
        length = tex["length"] * self.scale * self.rng.uniform(0.65, 1.30)
        if typ == "pima":
            dx, dy = x - peak[0], y - peak[1]
            d = math.hypot(dx, dy) or 1.0
            dx, dy = dx / d, dy / d
            ang = math.atan2(dy, dx) + self.rng.uniform(-0.55, 0.55)
        elif typ == "zhedai":
            ang = (math.pi / 2 if self.rng.random() < 0.70 else 0.0) + self.rng.uniform(-0.25, 0.25)
        elif typ in ("fupi", "fupi_midian"):
            ang = math.pi / 2 + self.rng.uniform(-0.35, 0.35)
            length *= 0.55
        elif typ == "yudian":
            ang = math.pi / 2 + self.rng.uniform(-0.18, 0.18)
            length *= 0.36
        else:
            ang = self.rng.uniform(0, math.pi * 2)
        pts = [(x, y)]
        step = length / 4.0
        for k in range(1, 5):
            if typ == "pima":
                ang += self.rng.uniform(-0.16, 0.16) + 0.07 * math.sin(k)
            elif typ == "zhedai":
                ang += self.rng.choice([-0.9, 0.9]) * (0.45 if k % 2 else 0.1)
            elif typ in ("fupi", "fupi_midian"):
                ang += self.rng.uniform(-0.10, 0.10)
            else:
                ang += self.rng.uniform(-0.32, 0.32)
            x += math.cos(ang) * step
            y += math.sin(ang) * step
            pts.append((x, y))
        return pts

    def _moss(self, ridge, m, depth):
        dots = self.p["dots"]
        if dots["density"] < 0.05:
            return
        n_cluster = int(len(ridge) * dots["density"] * 0.26 * (0.55 + 0.5 * self.detail))
        for _ in range(max(1, n_cluster)):
            idx = self.rng.randint(1, max(2, len(ridge) - 2))
            x, y = ridge[idx]
            x += self.rng.uniform(-10, 10) * self.scale
            y += self.rng.uniform(0, 12) * self.scale
            n = self.rng.randint(1, max(2, int(2 + dots["cluster"] * 4)))
            for _k in range(n):
                self._dot(x + self.rng.uniform(-1, 1) * dots["spread"] * 30 * self.scale,
                          y + self.rng.uniform(-0.6, 1.0) * dots["spread"] * 16 * self.scale,
                          dots["size"] * self.rng.uniform(0.62, 1.15),
                          dots["alpha"] * self.rng.uniform(0.55, 0.85), "moss", depth)
        # 浓密点苔只落在山脊与山体内部；不再把墨点撒到天空。

    def _color_sweeps(self, mask, m, depth, peak):
        sat = self.p["wash"]["saturation"]
        if sat < 0.08:
            return
        W, H = self.W, self.H
        ml = mask.load()
        bbox = mask.getbbox()
        if not bbox:
            return
        x0, y0, x1, y1 = bbox
        colors = []
        if self.rng.random() < 0.55:
            colors.append(("blue", 0.20 + 0.20 * sat))
        if self.rng.random() < 0.55:
            colors.append(("green", 0.22 + 0.18 * sat))
        if not colors:
            return
        for color, alpha in colors:
            for _ in range(self.rng.randint(3, 7)):
                y = self.rng.uniform(y0 + (y1 - y0) * 0.10, y0 + (y1 - y0) * 0.70)
                xa = self.rng.uniform(x0, x0 + (x1 - x0) * 0.35)
                xb = xa + (x1 - x0) * self.rng.uniform(0.30, 0.60)
                pts = []
                for k in range(5):
                    t = k / 4
                    xx = xa + (xb - xa) * t
                    yy = y + math.sin(t * math.pi) * self.rng.uniform(-6, 6) * self.scale
                    if 0 <= int(xx) < W and 0 <= int(yy) < H and ml[int(xx), int(yy)] > 70:
                        pts.append((xx, yy))
                if len(pts) >= 2:
                    self._add(pts, self.rng.uniform(10, 18) * self.scale,
                              alpha * 255 * 0.28, "wash", depth, color=color,
                              dry=0.0, taper=0.9, jitter=0.3)
    def _draw_tree_template(self, t, depth):
        if not self.tree_templates:
            return False
        W, H = self.W, self.H
        x, base, h = t["x"] * W, t["base"] * H, t["height"] * H
        kind = t["kind"]
        # 按树型选模板：松取线多，阔叶取墨重，疏取线少
        pools = self.tree_templates
        if kind in ("pine", "mixed"):
            pools = sorted(self.tree_templates, key=lambda z: -len(z["strokes"]))[:8]
        elif kind in ("broadleaf",):
            pools = sorted(self.tree_templates, key=lambda z: -len(z["strokes"]))[4:12] or pools
        tmpl = self.rng.choice(pools)
        strokes = tmpl.get("strokes", [])
        if not strokes:
            return False
        line = self.p["line"]
        tw = line["width"] * (1.0 if kind != "sparse" else .8)
        for st in strokes[:18]:
            pts = []
            for px, py in st:
                if py > 0.94:
                    continue
                pts.append((x + (px - .5) * h * 0.95, base - py * h * 0.98))
            if len(pts) < 2:
                continue
            # 太长的横线多半是坡脚，不当树枝
            if abs(pts[-1][0] - pts[0][0]) > h * .8 and abs(pts[-1][1] - pts[0][1]) < h * .08:
                continue
            self._add(pts, tw * self.rng.uniform(0.55, 1.0),
                      line["alpha"] * self.rng.uniform(0.65, 0.9), "tree", depth,
                      dry=line["dry"] * 0.8, taper=0.35, jitter=0.4)
        # 树冠：直接压墨团，保证不是一根棍
        if kind != "bare":
            from PIL import ImageDraw as _ID
            dr = _ID.Draw(self.groups[depth].ink)
            ink = (*self.ink_rgb(depth), 86)
            for dx, dy, rx, ry in ((-.16, .82, .23, .13), (.12, .86, .25, .14), (0, .94, .19, .10)):
                dr.ellipse([x + (dx - rx) * h, base - (dy + ry) * h,
                            x + (dx + rx) * h, base - (dy - ry) * h], fill=ink)
            for dx, dy in ((-0.16, 0.78), (0.12, 0.84), (0.0, 0.92)):
                self._dot(x + dx * h, base - dy * h, h * 0.16,
                          self.p["dots"]["alpha"] * 0.75, "moss", depth, stretch=1.35)
        # 在较高处补 2–3 个叶团
        highs = [p for st in strokes for p in st if p[1] < .45]
        for px, py in self.rng.sample(highs, min(3, len(highs))) if highs else []:
            self._pad(x + (px - .5) * h * 0.95, base - py * h * 0.98, t, depth, kind, self.p["foliage"]["density"])
        return True

    def _draw_tree(self, t, depth):
        if self._draw_tree_template(t, depth):
            return
        W, H = self.W, self.H
        x, base, h = t["x"] * W, t["base"] * H, t["height"] * H
        lean = t.get("lean", 0.0)
        line = self.p["line"]
        s = self.scale
        kind = t["kind"]
        dense = self.p["foliage"]["density"]
        # 主干：故意歪、有节、上细下粗
        trunk = [
            (x, base + 4 * s),
            (x + lean * h * 0.05 - self.rng.uniform(2, 8) * s, base - h * 0.16),
            (x + lean * h * 0.14 + self.rng.uniform(-9, 9) * s, base - h * 0.34),
            (x + lean * h * 0.24 + self.rng.uniform(-10, 10) * s, base - h * 0.54),
            (x + lean * h * 0.32 + self.rng.uniform(-8, 8) * s, base - h * 0.73),
            (x + lean * h * 0.38 + self.rng.uniform(-6, 6) * s, base - h * 0.90),
        ]
        tw = line["width"] * (2.6 if kind != "sparse" else 2.0)
        self._add(trunk, tw, min(255, line["alpha"] * 1.12), "tree", depth,
                  dry=0.0, taper=0.18, jitter=0.35)
        for side in (-1, 1):
            self._add([(x, base), (x + side * self.rng.uniform(12, 24) * s, base + self.rng.uniform(2, 8) * s)],
                      tw * 0.35, line["alpha"] * 0.62, "tree", depth, dry=0.4, taper=0.6, jitter=0.3)
        # 枝：从主干不同高度向外伸展，弯而不直
        tips = []
        levels = [(0.44, 0.24), (0.62, 0.24), (0.78, 0.18)]
        for li, (frac, rel) in enumerate(levels):
            i = min(len(trunk) - 2, int(frac * (len(trunk) - 1)))
            sx, sy = trunk[i]
            sides = [-1, 1] if self.rng.random() < 0.22 else [self.rng.choice([-1, 1])]
            for side in sides:
                if kind == "sparse" and li % 2 == 1:
                    continue
                if self.rng.random() < 0.18:
                    continue
                length = h * rel * self.rng.uniform(0.78, 1.15) * (0.75 if kind == "sparse" else 1.0)
                ang = math.radians(-8 if side > 0 else -172) + self.rng.uniform(-0.22, 0.22)
                # 主枝
                pts = [(sx, sy)]
                bx, by = sx, sy
                for k in range(5):
                    ang += side * self.rng.uniform(0.02, 0.09)
                    bx += math.cos(ang) * length / 4.2
                    by += math.sin(ang) * length / 4.2
                    pts.append((bx, by))
                self._add(pts, tw * 0.36, line["alpha"] * 0.92, "tree", depth,
                          dry=line["dry"] * 0.95, taper=0.35, jitter=0.8)
                # 侧枝，末端形成叶团
                for k in (2, 3):
                    ax, ay = pts[k]
                    for sgn in (-1, 1):
                        if self.rng.random() < 0.3:
                            continue
                        a2 = ang + sgn * self.rng.uniform(0.25, 0.65)
                        l2 = length * self.rng.uniform(0.22, 0.42)
                        p2 = [(ax, ay)]
                        for _ in range(3):
                            a2 += self.rng.uniform(-0.12, 0.12)
                            ax += math.cos(a2) * l2 / 3.0
                            ay += math.sin(a2) * l2 / 3.0
                            p2.append((ax, ay))
                        self._add(p2, tw * 0.18, line["alpha"] * 0.80, "tree", depth,
                                  dry=line["dry"] * 1.2, taper=0.5, jitter=0.4)
                        tips.append(p2[-1])
                tips.append(pts[-1])
        # 叶团：沿干、沿枝成层分布，形成树冠而不是几颗黑豆
        for li, frac in enumerate((0.52, 0.70, 0.86)):
            i = min(len(trunk) - 2, int(frac * (len(trunk) - 1)))
            px0, py0 = trunk[i]
            side = 1 if li % 2 == 0 else -1
            px = px0 + side * h * self.rng.uniform(0.05, 0.10)
            py = py0 + self.rng.uniform(-0.02, 0.02) * h
            self._pad(px, py, t, depth, kind, dense)
        for tip in tips:
            self._pad(tip[0], tip[1], t, depth, kind, dense)
        for _ in range(self.rng.randint(4, 9)):
            self._dot(x + self.rng.uniform(-20, 20) * s, base + self.rng.uniform(-2, 7) * s,
                      self.p["dots"]["size"] * 0.55, self.p["dots"]["alpha"] * 0.7, "moss", depth)

    def _pad(self, x, y, t, depth, kind, dense):
        s = self.scale
        f = self.p["foliage"]
        green = f.get("green", 0.0) * self.p["wash"]["saturation"]
        rx = (22 + 18 * f["size"]) * s * self.rng.uniform(0.85, 1.20)
        ry = rx * (0.48 if kind == "pine" else 0.68)
        if green > 0.02:
            mask = Image.new("L", (self.W, self.H), 0)
            ImageDraw.Draw(mask).ellipse([x - rx * 1.4, y - ry * 1.4, x + rx * 1.4, y + ry * 1.4],
                                          fill=int(58 * green))
            mask = mask.filter(ImageFilter.GaussianBlur(7 * s))
            self._paint_alpha(depth, self.color_rgb("green"), brush.mask_array(mask) * 0.8)
        # 两三个湿墨团做树冠体块，不再用几十颗小点
        for _ in range(self.rng.randint(2, 3)):
            self._wash_dab(x + self.rng.uniform(-0.25, 0.25) * rx,
                           y + self.rng.uniform(-0.20, 0.20) * ry,
                           rx * self.rng.uniform(0.55, 0.85), ry * self.rng.uniform(0.65, 0.95),
                           self.rng.uniform(58, 86), depth)
        for _ in range(self.rng.randint(2, 3)):
            self._dot(x + self.rng.gauss(0, 0.25) * rx, y + self.rng.gauss(0, 0.25) * ry,
                      self.p["dots"]["size"] * self.rng.uniform(0.28, 0.46),
                      self.p["dots"]["alpha"] * self.rng.uniform(0.55, 0.80), "moss", depth)
        if kind == "pine":
            for _ in range(int(12 + 10 * dense)):
                a = self.rng.uniform(-math.pi * 0.95, -math.pi * 0.05)
                ln = self.rng.uniform(0.55, 1.05) * rx
                p0 = (x + self.rng.uniform(-0.2, 0.2) * rx, y + self.rng.uniform(-0.15, 0.15) * ry)
                p1 = (p0[0] + math.cos(a) * ln, p0[1] + math.sin(a) * ln)
                self._add([p0, p1], 1.35 * s, self.p["line"]["alpha"] * 0.78, "needle", depth,
                          dry=0.1, taper=0.95, jitter=0.05)
        elif kind in ("broadleaf", "sparse", "mixed"):
            for _ in range(int(2 + 3 * dense)):
                px = x + self.rng.uniform(-0.6, 0.6) * rx
                py = y + self.rng.uniform(-0.5, 0.5) * ry
                for a in (-0.55, 0.0, 0.55):
                    self._add([(px, py), (px + math.sin(a) * 8 * s, py + math.cos(a) * 9 * s)],
                              1.5 * s, self.p["line"]["alpha"] * 0.78, "moss", depth,
                              dry=0.15, taper=0.9, jitter=0.1)

    def _draw_pavilion(self, p, depth):
        W, H = self.W, self.H
        x, base, w = p["x"] * W, p["base"] * H, p["width"] * W
        h = w * 1.05
        line = self.p["line"]
        lw = line["width"] * (0.72 if depth == "mid" else 0.95)
        alpha = line["alpha"] * 0.90
        roof_l = (x - w * 0.62, base - h * 0.58)
        roof_m = (x, base - h * 0.92)
        roof_r = (x + w * 0.62, base - h * 0.58)
        self._add([roof_l, (x - w * 0.28, base - h * 0.83), roof_m], lw * 1.15, alpha, "detail", depth,
                  dry=0.18, taper=0.55, jitter=0.3)
        self._add([roof_m, (x + w * 0.28, base - h * 0.83), roof_r], lw * 1.15, alpha, "detail", depth,
                  dry=0.18, taper=0.55, jitter=0.3)
        self._add([(x - w * 0.70, base - h * 0.50), (x, base - h * 0.60), (x + w * 0.70, base - h * 0.50)],
                  lw * 0.85, alpha * 0.9, "detail", depth, dry=0.25, taper=0.5, jitter=0.3)
        for px in (x - w * 0.43, x - w * 0.15, x + w * 0.15, x + w * 0.43):
            self._add([(px, base - h * 0.54), (px, base)], lw * 0.55, alpha * 0.82, "detail", depth,
                      dry=0.3, taper=0.35, jitter=0.2)
        self._add([(x - w * 0.58, base), (x + w * 0.58, base)], lw * 0.9, alpha, "detail", depth,
                  dry=0.2, taper=0.3, jitter=0.2)
        self._add([(x - w * 0.52, base - h * 0.16), (x + w * 0.52, base - h * 0.16)], lw * 0.45,
                  alpha * 0.7, "detail", depth, dry=0.35, taper=0.4, jitter=0.2)
        for _ in range(4):
            bx = x + self.rng.uniform(-w * 0.4, w * 0.4)
            self._add([(bx, base - h * 0.84), (bx + self.rng.uniform(-4, 4), base - h * 0.68)],
                      lw * 0.25, alpha * 0.48, "detail", depth, dry=0.5, taper=0.3, jitter=0.2)

    def _draw_boat(self, b, depth):
        W, H = self.W, self.H
        x, y, w = b["x"] * W, b["y"] * H, b["width"] * W
        line = self.p["line"]
        lw = line["width"] * (0.48 if depth == "far" else 0.72)
        alpha = line["alpha"] * (0.68 if depth == "far" else 0.90)
        hull = [(x - w * 0.55, y - w * 0.10), (x - w * 0.22, y + w * 0.12), (x + w * 0.28, y + w * 0.10),
                (x + w * 0.58, y - w * 0.12)]
        self._add(hull, lw, alpha, "detail", depth, dry=0.15, taper=0.45, jitter=0.2)
        self._add([(x - w * 0.40, y - w * 0.10), (x + w * 0.42, y - w * 0.10)], lw * 0.55,
                  alpha * 0.8, "detail", depth, dry=0.2, taper=0.5, jitter=0.2)
        self._add([(x - w * 0.16, y - w * 0.10), (x - w * 0.10, y - w * 0.42),
                   (x + w * 0.10, y - w * 0.43), (x + w * 0.18, y - w * 0.10)],
                  lw * 0.7, alpha, "detail", depth, dry=0.2, taper=0.4, jitter=0.2)
        for k in range(3):
            bx = x - w * 0.10 + k * w * 0.10
            self._add([(bx, y - w * 0.12), (bx + w * 0.02, y - w * 0.38)], lw * 0.3,
                      alpha * 0.55, "detail", depth, dry=0.3, taper=0.3, jitter=0.1)
        px, py = x + w * 0.26, y - w * 0.12
        self._dot(px, py - w * 0.22, lw * 1.2, alpha, "detail", depth, stretch=0.75)
        self._add([(px, py - w * 0.14), (px + w * 0.03, py + w * 0.02)], lw * 0.5, alpha, "detail", depth,
                  dry=0.2, taper=0.3, jitter=0.1)
        self._add([(x - w * 0.10, y - w * 0.02), (x - w * 0.50, y - w * 0.30)], lw * 0.4,
                  alpha * 0.78, "detail", depth, dry=0.2, taper=0.6, jitter=0.1)

    def _water(self, depth="near"):
        W, H = self.W, self.H
        y0 = self.lay.get("water", 0.70) * H
        y1 = 0.955 * H
        if y1 <= y0:
            return
        density = {"ni_zan": 0.28, "bada_shanren": 0.22, "gong_xian": 0.34}.get(self.p["id"], 1.0)
        n = int((36 * density + 18) * (0.8 + 0.3 * self.detail))
        for i in range(n):
            y = self.rng.uniform(y0 + (y1 - y0) * 0.04, y1)
            if y < y0:
                continue
            x = self.rng.uniform(0.10, 0.90) * W
            ln = self.rng.uniform(0.025, 0.12) * W * (0.7 + 0.6 * (y - y0) / max(1, y1 - y0))
            pts = [(x, y)]
            for k in range(1, 4):
                pts.append((x + ln * k / 3, y + math.sin(k * 1.3 + i) * 1.5 * self.scale))
            a = self.p["line"]["alpha"] * self.rng.uniform(0.10, 0.28)
            self._add(pts, self.p["line"]["width"] * self.rng.uniform(0.18, 0.36), a,
                      "water", depth, dry=0.45, taper=0.85, jitter=0.2)
        for _ in range(3 if self.p["id"] not in ("ni_zan", "bada_shanren") else 1):
            y = self.rng.uniform(y0 + (y1 - y0) * 0.55, y1 * 0.99)
            x = self.rng.uniform(0.15, 0.75) * W
            ln = self.rng.uniform(0.10, 0.22) * W
            self._add([(x, y), (x + ln * 0.5, y - 2 * self.scale), (x + ln, y + 1 * self.scale)],
                      self.p["line"]["width"] * 0.55, self.p["line"]["alpha"] * 0.34,
                      "water", depth, dry=0.35, taper=0.8, jitter=0.3)

    def _waterfall(self, wf):
        W, H = self.W, self.H
        x = wf["x"] * W
        top = wf["top"] * H
        bottom = wf["bottom"] * H
        width = wf["width"] * W
        strength = 0.85 if not wf.get("faint") else wf.get("faint", 0.55)
        pts = [(x - width * 0.55, top), (x - width * 0.48, top + (bottom - top) * 0.5),
               (x - width * 0.55, bottom), (x + width * 0.55, bottom),
               (x + width * 0.50, top + (bottom - top) * 0.5), (x + width * 0.55, top)]
        mask = brush.soft_polygon((W, H), pts, blur=2.2 * self.scale, feather=3)
        self._paint_alpha("mid", (246, 241, 229), brush.mask_array(mask) * strength)
        self._paint_alpha("near", (246, 241, 229), brush.mask_array(mask) * strength * 0.55)
        for side in (-1, 1):
            bx = x + side * width * 0.55
            self._add([(bx, top + (bottom - top) * 0.1), (bx + side * width * 0.6, top + (bottom - top) * 0.35),
                       (bx + side * width * 0.35, top + (bottom - top) * 0.62)],
                      self.p["line"]["width"] * 1.2, self.p["line"]["alpha"] * 0.82, "outline", "mid",
                      dry=0.2, taper=0.5, jitter=0.5)
        for _ in range(int(9 * strength) + 4):
            bx = x + self.rng.uniform(-width * 0.55, width * 0.55)
            self._add([(bx, top + self.rng.uniform(0, (bottom - top) * 0.2)),
                       (bx + self.rng.uniform(-2, 2) * self.scale, bottom - self.rng.uniform(0, (bottom - top) * 0.15))],
                      self.p["line"]["width"] * 0.30, self.p["line"]["alpha"] * 0.35, "water", "mid",
                      dry=0.25, taper=0.85, jitter=0.4)
        mist = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(mist)
        d.ellipse([x - width * 1.7, bottom - H * 0.05, x + width * 1.7, bottom + H * 0.06],
                  fill=(246, 241, 229, int(150 * strength)))
        self.groups["near"].color = Image.alpha_composite(
            self.groups["near"].color, mist.filter(ImageFilter.GaussianBlur(16 * self.scale)))

    def _mist_band(self, img, y, height, strength):
        W, H = self.W, self.H
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        # 不是一条横贯画面的带子，而是一组互相咬合的软云头。
        for k in range(4):
            cx = self.rng.uniform(-0.10, 1.10) * W
            cy = (y + self.rng.uniform(-0.15, 0.45) * height) * H
            rw = self.rng.uniform(0.12, 0.26) * W
            rh = self.rng.uniform(0.40, 0.70) * height * H
            a = int(210 * strength * self.rng.uniform(0.45, 0.95))
            d.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=(246, 241, 229, a))
        layer = layer.filter(ImageFilter.GaussianBlur(max(8, 16 * self.scale)))
        return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")

    def _font(self, size):
        for p in (r"C:\Windows\Fonts\STXINGKA.TTF", r"C:\Windows\Fonts\STKAITI.TTF",
                  r"C:\Windows\Fonts\simkai.ttf", r"C:\Windows\Fonts\msyh.ttc"):
            try:
                return ImageFont.truetype(p, int(size))
            except OSError:
                continue
        return ImageFont.load_default()

    def _text_layer(self, text, size, vertical, spacing=1.06):
        font = self._font(size)
        tmp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        if vertical:
            w = int(size * 1.25)
            h = int(size * spacing * len(text) + size * 0.4)
        else:
            bbox = d.textbbox((0, 0), text, font=font)
            w = int(bbox[2] - bbox[0] + size * 0.5)
            h = int(size * 1.5)
        layer = Image.new("RGBA", (max(2, w), max(2, h)), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        ink = tuple(self.p["palette"]["ink"])
        if vertical:
            for i, ch in enumerate(text):
                bbox = d.textbbox((0, 0), ch, font=font)
                cw = bbox[2] - bbox[0]
                d.text(((w - cw) / 2 - bbox[0], i * size * spacing), ch, font=font, fill=(*ink, 232))
        else:
            bbox = d.textbbox((0, 0), text, font=font)
            d.text((size * 0.2 - bbox[0], (h - (bbox[3] - bbox[1])) / 2 - bbox[1]), text, font=font, fill=(*ink, 232))
        return layer

    def _draw_inscription(self, img):
        W, H = self.W, self.H
        ins = self.p["inscription"]
        size = max(18, int(ins["size"] * H))
        vertical = ins["vertical"]
        layer = self._text_layer(ins["text"], size, vertical)
        sub_size = max(12, int(size * 0.38))
        sub = self._text_layer(ins.get("sub", ""), sub_size, vertical=vertical, spacing=1.02)
        pos = ins["position"]
        if pos == "top_left":
            x, y = int(W * 0.055), int(H * 0.055)
        elif pos == "top_right":
            x = int(W * 0.90 - max(layer.width, sub.width))
            y = int(H * 0.055)
        elif pos == "right":
            x = int(W * 0.88 - max(layer.width, sub.width))
            y = int(H * 0.32)
        else:
            x, y = int(W * 0.055), int(H * 0.24)
        angle = self.rng.uniform(-1.6, 1.6)
        layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
        img = Image.alpha_composite(img.convert("RGBA"), _paste_layer((W, H), layer, x, y)).convert("RGB")
        if vertical:
            sx = x - sub.width - int(size * 0.18)
            sy = y + int(size * 0.2)
        else:
            sx = x + int(size * 0.15)
            sy = y + layer.height + int(size * 0.28)
        if sx < 0:
            sx = x + layer.width + int(size * 0.15)
        sub = sub.rotate(self.rng.uniform(-1.2, 1.2), expand=True, resample=Image.BICUBIC)
        img = Image.alpha_composite(img.convert("RGBA"), _paste_layer((W, H), sub, sx, sy)).convert("RGB")
        seal = brush.seal_image(ins.get("seal", "山河"), max(22, int(size * 0.82)), self.rng,
                                color=(165, 52, 42), font_path=r"C:\Windows\Fonts\simkai.ttf")
        if vertical:
            px = sx - seal.width - int(size * 0.18)
            py = sy + sub.height + int(size * 0.18)
        else:
            px = x + layer.width + int(size * 0.10)
            py = y + layer.height + int(size * 0.10)
        if px < 4 or py + seal.height > H - 4:
            px, py = x + int(size * 0.4), y + layer.height + sub.height + int(size * 0.3)
        img = Image.alpha_composite(img.convert("RGBA"), _paste_layer((W, H), seal, int(px), int(py))).convert("RGB")
        return img

    def render(self):
        if self.line_only:
            self.image = Image.new("RGB", (self.W, self.H), (255, 255, 255))
            # 纯线稿只画主峰，避免中近景轮廓互相穿插
            mains = [m for m in self.lay["mountains"] if m.get("depth") != "far"]
            if not mains:
                mains = self.lay["mountains"]
            main = max(mains, key=lambda m: m["width"] * m["height"] * (1.25 if m.get("depth") == "near" else 1.0))
            self._draw_mountain(main, main.get("depth", "mid"))
            self.image = self._composite_group(main.get("depth", "mid"))
            return self.image
        self.image = brush.paper_base(self.W, self.H, self.rng)
        for m in self.lay["mountains"]:
            if m["depth"] == "far":
                self._draw_mountain(m, "far")
        if self.lay.get("waterfall"):
            self._waterfall(self.lay["waterfall"])
        for t in (self.lay.get("trees", []) if self.include_trees else []):
            if t["depth"] == "far":
                self._draw_tree(t, "far")
        self.image = self._composite_group("far")
        water = float(self.lay.get("water", 0.70))
        if self.lay.get("mountains"):
            self.image = self._mist_band(self.image, max(0.30, water - 0.15), 0.050, 0.32 * self.p["wash"]["mist"])
            self.image = self._mist_band(self.image, max(0.34, water - 0.075), 0.038, 0.22 * self.p["wash"]["mist"])
        for m in self.lay["mountains"]:
            if m["depth"] == "mid":
                self._draw_mountain(m, "mid")
        for t in (self.lay.get("trees", []) if self.include_trees else []):
            if t["depth"] == "mid":
                self._draw_tree(t, "mid")
        if self.lay.get("pavilion") and self.lay["pavilion"].get("depth") == "mid":
            self._draw_pavilion(self.lay["pavilion"], "mid")
        if self.lay.get("boat") and self.lay["boat"].get("depth") == "mid":
            self._draw_boat(self.lay["boat"], "mid")
        self.image = self._composite_group("mid")
        self.image = self._mist_band(self.image, max(0.34, water - 0.030), 0.035, 0.20 * self.p["wash"]["mist"])
        for m in self.lay["mountains"]:
            if m["depth"] == "near":
                self._draw_mountain(m, "near")
        for t in (self.lay.get("trees", []) if self.include_trees else []):
            if t["depth"] == "near":
                self._draw_tree(t, "near")
        if self.lay.get("pavilion") and self.lay["pavilion"].get("depth") == "near":
            self._draw_pavilion(self.lay["pavilion"], "near")
        if self.lay.get("boat") and self.lay["boat"].get("depth") == "near":
            self._draw_boat(self.lay["boat"], "near")
        self._water("near")
        self.image = self._composite_group("near")
        self.image = brush.add_paper_grain(self.image, self.rng, 0.075)
        self.image = self._draw_inscription(self.image)
        from PIL import ImageEnhance
        self.image = ImageEnhance.Contrast(self.image).enhance(1.06)
        self.image = ImageEnhance.Color(self.image).enhance(1.04)
        return self.image

    def _composite_group(self, name):
        g = self.groups[name]
        if self.p["wash"]["palette"] in ("wet", "dark") and name != "far":
            g.color = g.color.filter(ImageFilter.GaussianBlur(max(0.8, 1.4 * self.scale)))
        img = Image.alpha_composite(self.image.convert("RGBA"), g.color).convert("RGB")
        ink_soft = g.ink.filter(ImageFilter.GaussianBlur(max(0.8, 2.0 * self.scale)))
        r, gg, b, a = ink_soft.split()
        a = a.point(lambda v: int(v * 0.34))
        halo = Image.merge("RGBA", (r, gg, b, a))
        img = Image.alpha_composite(img.convert("RGBA"), halo).convert("RGB")
        img = Image.alpha_composite(img.convert("RGBA"), g.ink).convert("RGB")
        return img


def _paste_layer(size, layer, x, y):
    base = Image.new("RGBA", size, (0, 0, 0, 0))
    base.alpha_composite(layer, (int(x), int(y)))
    return base


def render(profile, layout, seed=0, width=1600, detail=1.0, abstraction=0.0, include_trees=False,
           line_only=False):
    r = MasterRenderer(profile, layout, seed=seed, width=width, detail=detail, abstraction=abstraction,
                       include_trees=include_trees, line_only=line_only)
    img = r.render()
    return img, r.plan