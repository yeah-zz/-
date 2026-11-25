"""Independent neutral-ink contour planner; no color planning or hardware calls."""
import cv2
import numpy as np
import shuimo
import vision
from config import OutlineConfig


def _local_widths(points, distance, scale):
    """沿笔迹取参考图里的局部笔画宽度（毫米）。"""
    idx = np.rint(np.asarray(points, float)).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, distance.shape[1] - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, distance.shape[0] - 1)
    return 2.0 * distance[idx[:, 1], idx[:, 0]] * scale


def _smooth(values, window):
    values = np.asarray(values, float)
    if window < 3 or len(values) < window:
        return values
    kernel = np.ones(int(window)) / float(window)
    pad = int(window) // 2
    padded = np.r_[np.full(pad, values[0]), values, np.full(pad, values[-1])]
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def _longest_run(mask):
    """最长的连续 True 段，返回 (lo, hi)。"""
    best = (0, 0)
    start = None
    for i, flag in enumerate(list(np.asarray(mask, bool)) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def _geometry(stroke, widths_mm, scale):
    """切线/法向/半宽剖面 + 两道闸门。返回 None 表示这条笔触"不干净"，不描边。"""
    points = np.asarray(stroke["points"], float)
    if len(points) < 3:
        return None
    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    ok = seg_len > 1e-6
    if int(ok.sum()) < 3:
        return None
    unit = seg[ok] / seg_len[ok][:, None]
    turn = np.degrees(np.arccos(np.clip((unit[:-1] * unit[1:]).sum(axis=1), -1.0, 1.0)))
    if len(turn) and float(np.percentile(turn, 90)) > float(OutlineConfig.EDGE_MAX_TURN_DEG):
        return None
    widths_arr = np.asarray(widths_mm, float)
    ratio = float(np.percentile(widths_arr, 90) / max(1e-6, np.percentile(widths_arr, 10)))
    if ratio > float(OutlineConfig.EDGE_MAX_WIDTH_RATIO):
        return None
    return _raw_geometry(points, widths_arr, cap=float(OutlineConfig.EDGE_MAX_OFFSET_MM))


def _raw_geometry(points, widths_mm, cap):
    """不做闸门的几何（墨块铺毫要用：那里本来就弯、宽度变化也大）。"""
    tangent = np.gradient(points, axis=0)
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    length[length < 1e-9] = 1.0
    tangent = tangent / length
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    half_mm = _smooth(np.asarray(widths_mm, float), 5) / 2.0 - float(OutlineConfig.EDGE_INSET_MM)
    half_mm = np.clip(half_mm, 0.0, cap)
    return points, normal, half_mm


def _make(stroke, points, presses, speeds, path_len_mm, suffix, side, press_scale, source):
    pen_width = float(np.mean([shuimo.press_to_width_mm(v) for v in presses])) if len(presses) else 0.0
    out = dict(stroke, id=stroke["id"] + suffix,
               name=str(stroke.get("name", "")) + suffix,
               points=np.round(points, 3).tolist(),
               press=[round(float(v), 4) for v in presses],
               press_max=round(float(np.max(presses)), 4) if len(presses) else 0.0,
               press_mean=round(float(np.mean(presses)), 4) if len(presses) else 0.0,
               width_mm=round(pen_width, 3), length_mm=round(path_len_mm, 2),
               source=source, edge_side=side, press_scale=press_scale)
    if speeds is not None:
        out["speed"] = [round(float(v), 2) for v in speeds]
    return out


def _emit(stroke, points, normal, half_mm, presses_all, speeds_all, scale,
          offset_mm, suffix, side, press_scale, source, fly_white=0.0, extend=False):
    """在给定偏移处生成一条线。extend=True = 墨块铺毫（尽量铺到墨的边缘）。"""
    if extend:
        mask = half_mm >= (offset_mm - float(OutlineConfig.MASS_FILL_SPACING_MM) * 0.5)
    else:
        mask = (half_mm - offset_mm) >= 0.2
    lo, hi = _longest_run(mask)
    if hi - lo < 3:
        return None
    press = presses_all[lo:hi] * press_scale
    speeds = speeds_all[lo:hi] if speeds_all is not None else None
    pts = points[lo:hi] + normal[lo:hi] * (side * offset_mm / max(1e-9, scale))
    if fly_white > 0:
        rng = np.random.default_rng(abs(hash((stroke["id"], suffix))) % (2 ** 32))
        if rng.random() < fly_white and len(pts) > 8:
            keep = int(len(pts) * float(rng.uniform(0.45, 0.75)))
            start = int(rng.integers(0, max(1, len(pts) - keep)))
            pts = pts[start:start + keep]
            press = press[start:start + keep]
            speeds = speeds[start:start + keep] if speeds is not None else None
    seg_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) * scale
    return _make(stroke, pts, press, speeds, seg_len, suffix, suffix, press_scale, source)


def _sub_strokes(stroke, widths_mm, scale):
    """普通宽笔触：左右两条边线（+ 主干够粗时加内芯线）。"""
    geo = _geometry(stroke, widths_mm, scale)
    if geo is None:
        return []
    points, normal, half_mm = geo
    presses_all = np.asarray(stroke["press"], float)
    speeds_all = np.asarray(stroke.get("speed", []), float)
    speeds_all = speeds_all if len(speeds_all) == len(points) else None
    wide = float(np.percentile(np.asarray(widths_mm, float), 75))
    out = []
    lo, hi = _longest_run(half_mm >= float(OutlineConfig.EDGE_MIN_OFFSET_MM))
    if hi - lo >= 3:
        jitter_mm = float(getattr(OutlineConfig, "EDGE_JITTER_MM", 0.0) or 0.0)
        asym = float(getattr(OutlineConfig, "EDGE_ASYMMETRY", 0.0) or 0.0)
        for side, letter in ((-1.0, "L"), (1.0, "R")):
            base_px = half_mm[lo:hi] / max(1e-9, scale) * (1.0 + side * asym)
            if jitter_mm > 0:
                rng = np.random.default_rng(abs(hash(stroke["id"])) % (2 ** 32))
                jit = _smooth(rng.normal(0.0, 1.0, len(base_px)), 5)
                jit = jit / max(1e-9, float(np.abs(jit).max())) * (jitter_mm / max(1e-9, scale))
                base_px = np.maximum(0.0, base_px + (jit if side < 0 else -jit))
            press = presses_all[lo:hi] * float(OutlineConfig.EDGE_PRESS_SCALE)
            speeds = speeds_all[lo:hi] if speeds_all is not None else None
            pts = points[lo:hi] + normal[lo:hi] * (side * base_px)[:, None]
            seg_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) * scale
            out.append(_make(stroke, pts, press, speeds, seg_len, letter, letter,
                             float(OutlineConfig.EDGE_PRESS_SCALE), "neutral_ink_skeleton_edge"))
    if wide >= float(OutlineConfig.FILL_MIN_WIDTH_MM) and OutlineConfig.INCLUDE_CENTERLINE:
        spacing = float(OutlineConfig.FILL_LINE_SPACING_MM)
        for k in range(1, int(OutlineConfig.FILL_MAX_PER_SIDE) + 1):
            for side, letter in ((-1.0, "%da" % k), (1.0, "%db" % k)):
                item = _emit(stroke, points, normal, half_mm, presses_all, speeds_all, scale,
                             k * spacing, "F%d%s" % (k, letter), side,
                             float(OutlineConfig.FILL_PRESS_SCALE), "neutral_ink_skeleton_fill")
                if item:
                    out.append(item)
    return out


def _mass_lines(pixels_xy, scale, press_mm, spacing_mm, fly_white):
    """把一个墨块用等距平行线填满：沿主轴方向扫，垂直于主轴按笔宽排。

    为什么不用"最长骨架 + 偏移"：那块浓墨的骨架最长段只有 20 多毫米，而墨块本身
    更大、更碎，单靠一条脊铺不满（实测总长反而比原来的乱麻少）。所以改成对整块
    墨的像素做主成分分析，按主轴铺满 —— 这才是"一片墨"。
    """
    pts = np.asarray(pixels_xy, float)
    if len(pts) < 30:
        return []
    centre = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - centre, full_matrices=False)
    axis = vt[0]
    normal = np.asarray([-axis[1], axis[0]])
    t = (pts - centre) @ axis
    n = (pts - centre) @ normal
    spacing_px = spacing_mm / max(1e-9, scale)
    k_max = int(np.floor(max(abs(n.min()), abs(n.max())) / spacing_px))
    rng = np.random.default_rng(12345)
    out = []
    for k in range(-k_max, k_max + 1):
        offset = k * spacing_px
        sel = np.abs(n - offset) <= max(1.0, spacing_px * 0.5)
        if int(sel.sum()) < 5:
            continue
        ts = np.sort(t[sel])
        segs = np.split(ts, np.nonzero(np.diff(ts) > max(2.0, spacing_px * 1.2))[0] + 1)
        seg = max(segs, key=len)
        if len(seg) < 3 or (seg[-1] - seg[0]) * scale < 3.0:
            continue
        if fly_white > 0 and rng.random() < fly_white:          # 飞白：裁掉一段
            keep = float(rng.uniform(0.45, 0.75))
            span = (seg[-1] - seg[0]) * keep
            start = seg[0] + (seg[-1] - seg[0] - span) * float(rng.random())
            seg = np.asarray([start, start + span])
        count = int(np.clip(round((seg[-1] - seg[0]) * scale / 1.5) + 1, 2, 40))
        tt = np.linspace(seg[0], seg[-1], count)
        pxy = centre[None, :] + axis[None, :] * tt[:, None] + normal[None, :] * offset
        out.append((pxy, press_mm))
    return out


def build_plan(work, scale):
    gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
    ink=vision.neutral_ink_mask(work, gray_limit=155).astype(bool)
    neutral=np.full_like(work,255)
    neutral[ink]=np.repeat(gray[ink,None],3,axis=1)
    # Solid graphic ink should stay solid. Textured wash keeps intentional fly-white.
    solid_graphic = bool(ink.any() and np.percentile(gray[ink], 90) <= 100)
    fly_white = 0.0 if solid_graphic else float(OutlineConfig.MASS_FLY_WHITE)
    strokes,stats=shuimo.build_framework_strokes(neutral,scale)
    distance=cv2.distanceTransform(ink.astype(np.uint8),cv2.DIST_L2,3)

    # ---- 宽墨块识别：局部宽度 >= 阈值的连通区 ----
    mass_id = np.zeros_like(distance, dtype=np.int32)
    if OutlineConfig.MASS_FILL:
        mass_mask = ((2.0 * distance * scale) >= float(OutlineConfig.MASS_MIN_WIDTH_MM)).astype(np.uint8)
        if mass_mask.any():
            _n, mass_id = cv2.connectedComponents(mass_mask, 8)

    info = []
    for i, s in enumerate(strokes):
        widths = _local_widths(s['points'], distance, scale)
        pts = np.rint(np.asarray(s['points'], float)).astype(int)
        pts[:, 0] = np.clip(pts[:, 0], 0, mass_id.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, mass_id.shape[0] - 1)
        on = mass_id[pts[:, 1], pts[:, 0]]
        ratio = float(np.mean(on > 0)) if len(on) else 0.0
        dominant = int(np.bincount(on).argmax()) if len(on) else 0
        info.append(dict(stroke=s, widths=widths, ratio=ratio, mass=dominant,
                         wide=float(np.percentile(widths, 75)) if len(widths) else 0.0))

    # 每个墨块只留最长的 N 条骨架当"脊"（原始笔触还没有 id，用下标当键）
    spines = set()
    if OutlineConfig.MASS_FILL:
        by_mass = {}
        for i, it in enumerate(info):
            if it['mass'] > 0 and it['ratio'] >= float(OutlineConfig.MASS_DROP_RATIO):
                by_mass.setdefault(it['mass'], []).append(i)
        for mid, idxs in by_mass.items():
            idxs.sort(key=lambda i: -float(info[i]['stroke'].get('length_mm', 0.0)))
            for i in idxs[:int(OutlineConfig.MASS_KEEP_LONGEST)]:
                spines.add(i)

    # 每个墨块：取【墨块像素外扩一圈 ∩ 墨迹】当填充区域，用整块墨的像素做扫描线填充
    mass_pixels = {}
    if OutlineConfig.MASS_FILL:
        wanted = sorted({info[i]['mass'] for i in spines if info[i]['mass'] > 0})
        grow = int(round(float(OutlineConfig.MASS_MIN_WIDTH_MM) * 0.5 / max(1e-9, scale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        for mid in wanted:
            m = (mass_id == mid).astype(np.uint8)
            if grow > 0:
                m = cv2.dilate(m, kernel)
            m = (m > 0) & (ink > 0)
            ys, xs = np.nonzero(m)
            if len(xs) >= 30:
                mass_pixels[mid] = np.column_stack([xs, ys]).astype(float)

    clean = []
    for i, it in enumerate(info):
        s, widths, wide = it['stroke'], it['widths'], it['wide']
        base = dict(s, id='outline-%04d' % i, source='neutral_ink_skeleton', image_width_mm=round(wide, 2))
        in_mass = (OutlineConfig.MASS_FILL and it['mass'] > 0
                   and it['ratio'] >= float(OutlineConfig.MASS_DROP_RATIO))
        if in_mass:
            if i not in spines:
                continue                                 # 墨块里的碎线：交给铺毫填充
            # 脊这一条：先用整块墨的扫描线把整片铺满，再保留它自己当中线
            mid = it['mass']
            pixels = mass_pixels.get(mid)
            if pixels is None or len(pixels) < 30:
                clean.append(base)
                continue
            presses = list(np.asarray(s['press'], float))
            press_mm = float(np.median(presses)) * float(OutlineConfig.MASS_FILL_PRESS_SCALE)
            # ★ 2026-09-26：间距不能是死数。铺毫线的笔压是"脊线中位笔压 × 比例"，
            #   一旦把 PRESS_MIN/MAX 收窄，中位笔压跟着降，线宽就变细；
            #   此时若还按固定的 1.5mm 排线，整片墨块会漏出白条。
            #   所以取 min(配置值, 0.6×该笔压的实测线宽)：
            #   宽线时不至于排得过密（省时间），细线时自动收紧保证盖满。
            spacing_mm = float(OutlineConfig.MASS_FILL_SPACING_MM)
            line_width_mm = float(shuimo.press_to_width_mm(press_mm))
            spacing_mm = min(spacing_mm, max(0.4, 0.6 * line_width_mm))
            lines = _mass_lines(pixels, scale, press_mm, spacing_mm, fly_white)
            parts = [base]
            for j, (pxy, pr) in enumerate(lines):
                speed = float(np.mean(np.asarray(s.get('speed', [28.0]), float)))
                # 铺毫也做提按：笔身保持覆盖，首尾渐轻，避免等宽排线的机器感。
                phase = np.linspace(0., 1., len(pxy))
                taper = np.interp(phase, [0., .08, .18, .78, .93, 1.],
                                  [.68, .9, 1., 1., .88, .58])
                varied_press = np.round(pr * taper, 4)
                parts.append(dict(base, id='outline-%04dM%02d' % (i, j),
                                  name='铺毫%02d' % j,
                                  points=np.round(pxy, 3).tolist(),
                                  press=varied_press.tolist(),
                                  press_max=round(float(varied_press.max()), 4),
                                  press_mean=round(float(varied_press.mean()), 4),
                                  width_mm=round(float(np.mean([
                                      shuimo.press_to_width_mm(v) for v in varied_press])), 3),
                                  length_mm=round(float(np.linalg.norm(np.diff(pxy, axis=0), axis=1).sum()) * scale, 2),
                                  speed=[round(speed, 2)] * len(pxy),
                                  source='neutral_ink_skeleton_massfill'))
            clean.extend(parts)
            continue
        parts = [base]
        if (OutlineConfig.EDGE_LINES and wide >= float(OutlineConfig.EDGE_MIN_WIDTH_MM)
                and float(s.get('length_mm', 0.0)) >= float(OutlineConfig.EDGE_MIN_LEN_MM)):
            subs = _sub_strokes(base, widths, scale)
            if subs:
                parts = ([base] if OutlineConfig.INCLUDE_CENTERLINE else []) + subs
        clean.extend(parts)

    return dict(format='outline-design-v1',ready_for_execution=False,image_size=[work.shape[1],work.shape[0]],
        mm_per_px=scale,strokes=clean,stroke_count=len(clean),
        ink_rendering='solid' if solid_graphic else 'textured',
        execution_profile='existing_ink_profile_unchanged',requires=['Matching paper registration and reviewed ink execution job'])


def draw(canvas,plan):
    for s in plan['strokes']:
        cv2.polylines(canvas,[np.rint(s['points']).astype(np.int32)],False,(35,35,35),
            max(1,int(round(s['width_mm']/plan['mm_per_px']))),cv2.LINE_AA)
    return canvas
