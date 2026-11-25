"""Two-color plum: neutral branches, then five pressed red dots per flower.

按花心和五个瓣向分块；默认每瓣只压一个短点。原来的铺色笔迹仍可通过
``petal_style='strokes'`` 生成，便于对照。小花苞按连通域处理。
"""
import math

import cv2
import numpy as np

import vision
import outline_pass
import watercolor
import shuimo
from geometry import CanvasMap

# ---- 花瓣参数 ----
# 默认按用户要求：每瓣先深压、向花心短拖，一瓣一笔。旧铺色参数留作对照。
# 2.00 mm 实画五瓣仍偏小；下一档试 2.50 mm，只先画单花小样。
PETAL_PRESS_MM = .90          # 旧铺色模式的单笔峰值下压
PETAL_DOT_PRESS_MM = 2.50     # 五点梅花：比已实画的 2.00 mm 再深 0.50 mm
PETAL_DOT_RADIUS_MM = 5.5     # 五点围绕花心的半径；毛笔点痕直径约 3~4 mm
PETAL_DOT_LENGTH_MM = 3.0     # 厾笔后朝花心短拖约 3 mm，形成一片小花瓣
PETAL_STYLE = 'dots'           # 默认每朵五笔；'strokes' 保留旧铺色计划供对照
# 从瓣的外缘向花心行笔：外侧铺满，中部留住宽度，靠花心提笔。
PETAL_PRESS_PROFILE = np.array([.68, .96, 1., 1., .94, .78, .48, .18, 0.])
PETAL_LINE_SPACING_MM = 1.55  # 相邻笔痕略重叠，避免瓣面变成几道分离的弧线
PETAL_MAX_LINES = 4           # 宽瓣最多四笔；笔数增加会增加蘸色与总时长
PETAL_INSET_MM = .25          # 只轻微内缩；之前与整图腐蚀叠加，削掉了过多瓣面
PETAL_INNER_SCALE = 1.0       # 靠瓣中轴的笔迹满压力
PETAL_OUTER_SCALE = .82       # 靠瓣边缘的笔迹略淡，但仍与中笔相接
PETAL_SAMPLE_MM = 1.35        # 多取几个点以保住瓣的曲线和逐点笔压形状
PETAL_MIN_POINTS = 2
PETAL_MAX_POINTS = 10
PETAL_MIN_AREA_MM2 = .5       # 小于此面积的红块当蕊点，只点一下
PETAL_MIN_LEN_MM = 1.8        # 中线短于此长度也只点一下
DAB_LENGTH_MM = .5            # "点一下"的极短笔长（执行器要求 >= 2 个点）
FLOWER_CLUSTER_MM = 1.8       # 红块膨胀这么多毫米后仍连通的算同一朵花（决定画花顺序）


def _midline(blob, scale):
    """主轴 + 逐切片均值 -> 中线；返回 (中线点, 切片步长px)。"""
    ys, xs = np.nonzero(blob)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    centre = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - centre, full_matrices=False)
    axis = vt[0]
    t = (pts - centre) @ axis
    lo, hi = float(t.min()), float(t.max())
    if hi - lo <= 0:
        return None
    length_mm = (hi - lo) * scale
    n = int(np.clip(round(length_mm / PETAL_SAMPLE_MM), PETAL_MIN_POINTS, PETAL_MAX_POINTS))
    step = (hi - lo) / (n - 1)
    mid = []
    for k in range(n):
        tk = lo + step * k
        band = np.abs(t - tk) <= max(1.0, step * .6)
        if band.any():
            mid.append(pts[band].mean(axis=0))
    if len(mid) < PETAL_MIN_POINTS:
        return None
    return np.asarray(mid, np.float32), step


def _half_width(point, distance):
    y = int(np.clip(round(float(point[1])), 0, distance.shape[0] - 1))
    x = int(np.clip(round(float(point[0])), 0, distance.shape[1] - 1))
    return float(distance[y, x])


def _petal_lines(blob, distance, scale, flower_centre=None):
    """一瓣 -> [(点, 压力数组)]：沿中线、按宽度铺开的若干条方向性短笔。

    先把花瓣向内缩 PETAL_INSET_MM，让相邻花瓣之间留出白缝；
    靠花心的那条用满压力，越靠外的越淡（近似"心浓边淡"）。
    """
    inset_px = max(0, int(round(PETAL_INSET_MM / scale)))
    core = blob
    if inset_px > 0:
        eroded = cv2.erode(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset_px + 1,) * 2))
        if eroded.sum() > 0:
            core = eroded
    midline = _midline(core, scale)
    if midline is None:
        return []
    mid, _step = midline
    half = np.asarray([_half_width(p, distance) for p in mid], np.float32)
    count = int(np.clip(math.ceil(float(half.max()) * 2.0 * scale / PETAL_LINE_SPACING_MM),
                        1, PETAL_MAX_LINES))
    direction = mid[-1] - mid[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        return []
    direction = direction / norm
    normal = np.asarray([-direction[1], direction[0]], np.float32)
    profile = np.interp(np.linspace(0., 1., len(mid)),
                        np.linspace(0., 1., len(PETAL_PRESS_PROFILE)), PETAL_PRESS_PROFILE)
    offset_px = PETAL_LINE_SPACING_MM / scale
    lines = []
    for j in range(count):
        shift = (j - (count - 1) / 2.0) * offset_px
        pts = []
        for i, point in enumerate(mid):
            room = max(0.0, half[i] - 0.4 / scale)      # 不要把笔迹铺到花瓣外
            pts.append(point + normal * float(np.clip(shift, -room, room)))
        pts = np.asarray(pts, np.float32)
        # 从外瓣向花心画：较重的笔痕落在外瓣宽处，收笔在花心附近。
        # 对孤立小块（没有可用花心）仍按两端局部宽度判定方向。
        if flower_centre is not None:
            centre = np.asarray(flower_centre, np.float32)
            reverse = np.linalg.norm(pts[-1] - centre) > np.linalg.norm(pts[0] - centre)
        else:
            reverse = _half_width(pts[-1], distance) > _half_width(pts[0], distance)
        if reverse:
            pts = pts[::-1]
        # 越靠外的笔迹越淡；count==1 时就是贴中线那条（满压力）
        edge = 1.0 if count >= 3 and j in (0, count - 1) else 0.0
        scale_j = PETAL_INNER_SCALE + (PETAL_OUTER_SCALE - PETAL_INNER_SCALE) * edge
        lines.append((pts, profile * PETAL_PRESS_MM * scale_j))
    return lines


def _dab(centre, scale):
    """花苞 / 蕊点：极短的一小笔（执行器要求至少两个点）。"""
    step = DAB_LENGTH_MM / scale
    pts = np.asarray([[centre[0], centre[1]], [centre[0] + step, centre[1]]], np.float32)
    return pts, np.asarray([PETAL_PRESS_MM * .6, 0.], np.float32)


def _dot(centre, flower_centre, scale, press=None, length_mm=None):
    """先深压，再朝花心短拖并逐渐提笔；一瓣一条连续轨迹。"""
    press = PETAL_DOT_PRESS_MM if press is None else float(press)
    length_mm = PETAL_DOT_LENGTH_MM if length_mm is None else float(length_mm)
    direction = np.asarray(flower_centre, np.float32) - np.asarray(centre, np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:                 # 花苞没有明确花心，沿短水平线厾笔
        direction = np.asarray([1., 0.], np.float32)
    else:
        direction /= norm
    start = np.asarray(centre, np.float32) - direction * (.4 / scale)
    middle = start + direction * (length_mm * .55 / scale)
    tip = start + direction * (length_mm / scale)
    return np.asarray([start, middle, tip], np.float32), np.asarray(
        [press, press * .9, press * .32], np.float32)


def _flower_centre(yellow, red, flower_mask):
    """用参考图中花蕊汇聚处定位花心；无花蕊时退回红色质心。"""
    ys, xs = np.nonzero(flower_mask)
    fallback = (float(xs.mean()), float(ys.mean()))
    if yellow is None:
        return fallback
    x0, x1 = max(0, int(xs.min())), min(red.shape[1], int(xs.max()) + 1)
    y0, y1 = max(0, int(ys.min())), min(red.shape[0], int(ys.max()) + 1)
    crop = yellow[y0:y1, x0:x1].astype(np.float32)
    if int(crop.sum()) < 10:
        return fallback
    density = cv2.GaussianBlur(crop, (0, 0), 2.0)
    _min, peak, _min_xy, (x, y) = cv2.minMaxLoc(density)
    return (float(x + x0), float(y + y0)) if peak >= .25 else fallback


def _radial_petals(flower_mask, centre, scale):
    """沿五个外瓣的主方向分割整朵花，避免相连花瓣被当成一瓣。"""
    ys, xs = np.nonzero(flower_mask)
    dx, dy = xs - centre[0], ys - centre[1]
    radius = np.hypot(dx, dy)
    angle = np.mod(np.degrees(np.arctan2(dy, dx)), 360.)
    far = radius >= 3.2 / scale
    if int(far.sum()) < 50:
        return [flower_mask]
    bins = np.floor(angle[far]).astype(np.int32)
    weights = np.minimum(radius[far], 12.0 / scale)
    hist = np.bincount(bins, weights=weights, minlength=360).astype(np.float32)
    smooth = cv2.GaussianBlur(np.r_[hist[-30:], hist, hist[:30]][None, :],
                              (0, 0), 7.0).ravel()[30:390]
    peaks = []
    for candidate in np.argsort(smooth)[::-1]:
        if all(min(abs(int(candidate) - p), 360 - abs(int(candidate) - p)) > 40
               for p in peaks):
            peaks.append(int(candidate))
        if len(peaks) == 5:
            break
    if len(peaks) != 5:
        return [flower_mask]
    delta = np.abs(angle[:, None] - np.asarray(peaks)[None, :])
    owner = np.argmin(np.minimum(delta, 360. - delta), axis=1)
    petals = []
    for k in range(5):
        blob = np.zeros_like(flower_mask)
        blob[ys[owner == k], xs[owner == k]] = 1
        if int(blob.sum()) >= 8:
            petals.append(blob)
    return petals


def _open_flower_cutoff(areas_mm2):
    """用本图花/花苞面积的自然断层定门槛，不按某张图写死尺寸。"""
    areas = np.sort(np.asarray([a for a in areas_mm2 if a >= 1.], float))
    if len(areas) < 3:
        return 20.0
    ratios = areas[1:] / areas[:-1]
    candidates = [(float(ratios[i]), i + 1) for i in range(1, len(areas) - 1)
                  if i + 1 >= max(2, int(len(areas) * .2))]
    if candidates:
        ratio, split = max(candidates)
        if ratio >= 2.2:
            return float(math.sqrt(areas[split - 1] * areas[split]))
    return 20.0


def petal_plan(red, scale, yellow=None, style='dots', open_area_mm2=None,
               return_cutoff=False):
    """红色掩膜 -> 五点梅花；style='strokes' 保留旧铺色法供对照。"""
    if style not in ('dots', 'strokes'):
        raise ValueError('Unsupported petal style: %s' % style)
    distance = cv2.distanceTransform(red, cv2.DIST_L2, 3)
    cluster_px = max(1, int(round(FLOWER_CLUSTER_MM / scale)))
    grown = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cluster_px + 1,) * 2))
    _n_flowers, flower_label, _stats, _centroids = cv2.connectedComponentsWithStats(grown, 8)
    clusters = []
    for flower in range(1, _n_flowers):
        flower_mask = ((flower_label == flower) & (red > 0)).astype(np.uint8)
        area = int(flower_mask.sum())
        if area == 0:
            continue
        clusters.append((flower, flower_mask, area))
    cutoff = (_open_flower_cutoff([area * scale * scale for _, _, area in clusters])
              if open_area_mm2 is None else float(open_area_mm2))
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('open_area_mm2 must be positive and finite')
    grouped = []
    for flower, flower_mask, area in clusters:
        fcx, fcy = _flower_centre(yellow, red, flower_mask)
        if area * scale * scale >= cutoff:
            blobs = _radial_petals(flower_mask, (fcx, fcy), scale)
        else:
            count, labels = cv2.connectedComponents(flower_mask, 8)
            blobs = [(labels == k).astype(np.uint8) for k in range(1, count)]
        for blob in blobs:
            area_mm2 = float(blob.sum()) * scale * scale
            if area_mm2 < PETAL_MIN_AREA_MM2 * .15:
                continue
            ys, xs = np.nonzero(blob)
            cx, cy = float(xs.mean()), float(ys.mean())
            lines = []
            if style == 'dots':
                direction = np.asarray([cx - fcx, cy - fcy], np.float32)
                norm = float(np.linalg.norm(direction))
                if len(blobs) == 5 and norm > 1e-6:
                    unit = direction / norm
                    # 小花要把落笔点放到瓣的外半部；按固定 3 mm 向内拖会穿过花心，
                    # 使五瓣在预览和纸上都并成一团。
                    radial_mm = ((xs - fcx) * unit[0] + (ys - fcy) * unit[1]) * scale
                    outer_mm = float(np.percentile(radial_mm, 85))
                    radius_mm = min(PETAL_DOT_RADIUS_MM, max(2.8, outer_mm - .7))
                    length_mm = min(PETAL_DOT_LENGTH_MM, max(.6, radius_mm - 2.3))
                    centre = np.asarray([fcx, fcy], np.float32) + unit * (radius_mm / scale)
                    lines = [_dot(centre, (fcx, fcy), scale, length_mm=length_mm)]
                else:
                    # 花苞是一枚小点，不沿枝条拖出 3 mm 的红色长条。
                    centre = np.asarray([cx, cy], np.float32)
                    lines = [_dot(centre, centre, scale,
                                  press=min(PETAL_DOT_PRESS_MM * .55, 1.35),
                                  length_mm=.8)]
            elif area_mm2 >= PETAL_MIN_AREA_MM2:
                for pts, press in _petal_lines(blob, distance, scale, (fcx, fcy)):
                    length_mm = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) * scale
                    if length_mm >= PETAL_MIN_LEN_MM:
                        lines.append((pts, press))
            if not lines:
                lines = [_dab((cx, cy), scale)]
            grouped.append((flower, math.atan2(cy - fcy, cx - fcx), lines))
    strokes = []
    for flower, _angle, lines in sorted(grouped, key=lambda item: (item[0], item[1])):
        for pts, press in lines:
            strokes.append(dict(id='petal-%03d' % len(strokes), color='red',
                                points=np.round(pts, 3).tolist(), region_id=int(flower),
                                length_mm=float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) * scale,
                                press=[round(float(v), 4) for v in press],
                                # 预览渲染要求这两个数组长度 = 点数（按点压力给示意宽度）
                                target_width_mm=[round(shuimo.press_to_width_mm(float(p)), 2) for p in press],
                                target_opacity=[.8] * len(press),
                                brush_width_mm_estimate=max(
                                    shuimo.press_to_width_mm(float(p)) for p in press)))
    return (strokes, cutoff) if return_cutoff else strokes


def build_plan(reference, petal_style=None, open_area_mm2=None):
    petal_style = PETAL_STYLE if petal_style is None else petal_style
    work, info = vision.prepare_image(vision.load_image(str(reference)), work_size=1000)
    # Fit the ENTIRE reference well inside the experiment's 20 mm paper margin.
    base = CanvasMap(info['prepared_size'], info['image_corners'])
    pad = int(np.ceil(25 / base.mm_per_px))
    h, w = work.shape[:2]
    small = cv2.resize(work, (w - 2 * pad, h - 2 * pad), interpolation=cv2.INTER_AREA)
    work = np.full_like(work, 255)
    work[pad:h - pad, pad:w - pad] = small
    geometry = dict(image_size=[w, h], image_corners=base.image_corners, mm_per_px=base.mm_per_px)
    # 深红阴影的亮度常低于旧 HSV 门槛；按红通道优势保留完整瓣面。
    b, g, r = cv2.split(work)
    red_raw = ((r > 55) & (r > g.astype(np.float32) * 1.27)
           & (r > b.astype(np.float32) * 1.27)
           & ((r.astype(np.int16) - g.astype(np.int16)) > 25)
           & ((r.astype(np.int16) - b.astype(np.int16)) > 25)).astype(np.uint8)
    # Shared neutral-ink extraction keeps colored petals out of the black branch plan.
    ink = outline_pass.build_plan(work, base.mm_per_px)
    ink.update(geometry)
    radius = max(1, int(round(.2 / base.mm_per_px)))
    red = cv2.erode(red_raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    yellow = (((hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 40)
               & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 100))).astype(np.uint8)
    strokes, cutoff = petal_plan(red, base.mm_per_px, yellow, style=petal_style,
                                 open_area_mm2=open_area_mm2, return_cutoff=True)
    red_plan = dict(format='sparse-color-design-v1', ready_for_execution=False, **geometry,
                    petal_style=petal_style, open_area_mm2=cutoff,
                    open_area_mode='auto' if open_area_mm2 is None else 'manual',
                    strokes=strokes, stroke_count=len(strokes), palette_bgr=watercolor.PALETTE)
    return dict(format='plum-experiment-v1', ready_for_execution=False, **geometry,
                stages=['outline', 'color'], modules={'outline': ink, 'color': red_plan},
                note=('Black branches then five pressed red dots per open flower; '
                      'yellow stamens omitted; measured cups required')), work
