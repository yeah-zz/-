"""Source-constrained ink centerlines. Pure offline image processing, BGR input.

This estimates visible line geometry, not original stroke order or brush force.
No model downloads, robot imports, or execution-ready task files.
"""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from original_d_tracer import LandscapeBrushGenerator as Trace


@dataclass(frozen=True)
class InkConfig:
    # All spatial parameters are relative to a 1110 px long image edge.
    strong_contrast: float = 22.
    weak_contrast: float = 11.
    max_value: float = 212.
    min_component_length: float = 10.
    simplify_error: float = .55
    max_gap: float = 4.
    gap_support: float = .9


def _length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _graph(skeleton):
    ys, xs = np.where(skeleton > 0)
    pixels = set(zip(xs.tolist(), ys.tolist()))
    graph = {}
    for x, y in sorted(pixels, key=lambda p: (p[1], p[0])):
        links = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = x + dx, y + dy
                if not (dx or dy) or q not in pixels:
                    continue
                # Do not turn an ordinary right-angle corner into a triangle.
                if dx and dy and ((x + dx, y) in pixels or (x, y + dy) in pixels):
                    continue
                links.append(q)
        graph[x, y] = links
    return graph


def _trace(graph):
    visited, paths = set(), []

    def edge(a, b):
        return tuple(sorted((a, b)))

    def walk(a, b):
        path = [a]
        visited.add(edge(a, b))
        while True:
            path.append(b)
            if len(graph[b]) != 2 or b == path[0]:
                break
            choices = [q for q in graph[b] if q != a and edge(b, q) not in visited]
            if not choices:
                break
            a, b = b, choices[0]
            visited.add(edge(a, b))
        return np.asarray(path, np.float32)

    nodes = [p for p in graph if len(graph[p]) != 2]
    # Second pass preserves closed loops, such as a boat cabin or enclosed rock.
    for p in nodes + list(graph):
        for q in graph[p]:
            if edge(p, q) not in visited:
                paths.append(walk(p, q))
    return paths


def _direction(points, at_end, reach):
    p = points[::-1] if at_end else points
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    k = min(len(p) - 1, max(1, int(np.searchsorted(arc, reach))))
    v = p[k] - p[0]
    return v / max(float(np.linalg.norm(v)), 1e-8)


def _samples(a, b):
    n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) * 2)) + 1)
    return np.rint(np.linspace(a, b, n)).astype(int)


def _join(paths, evidence, ratio, cfg):
    """Join shared nodes or short gaps only with source support and tangency."""
    joined_count = 0
    for _ in range(16):
        grid, candidates = {}, []
        cell = max(1., cfg.max_gap * ratio)
        endpoints = []
        for i, p in enumerate(paths):
            if np.array_equal(p[0], p[-1]):
                continue
            for end in (0, 1):
                q = p[-1] if end else p[0]
                key = tuple(np.floor(q / cell).astype(int))
                endpoints.append((i, end, q, key))
                grid.setdefault(key, []).append((i, end, q))
        for i, ei, a, key in endpoints:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for j, ej, b in grid.get((key[0] + dx, key[1] + dy), []):
                        if i >= j:
                            continue
                        d = float(np.linalg.norm(b - a))
                        if d > cell:
                            continue
                        # Outward vector of first segment, inward of second.
                        u = -_direction(paths[i], ei == 1, max(4., 7 * ratio))
                        v = _direction(paths[j], ej == 1, max(4., 7 * ratio))
                        alignment = float(u @ v)
                        if alignment < (.78 if d < .01 else .92):
                            continue
                        if d > .01:
                            gap = (b - a) / d
                            if min(float(u @ gap), float(v @ gap)) < .88:
                                continue
                            q = _samples(a, b)
                            if np.mean(evidence[q[:, 1], q[:, 0]] > 0) < cfg.gap_support:
                                continue
                        candidates.append((alignment - .02 * d, i, ei, j, ej))
        used, merged = set(), []
        for _, i, ei, j, ej in sorted(candidates, reverse=True):
            if i in used or j in used:
                continue
            a = paths[i] if ei else paths[i][::-1]
            b = paths[j][::-1] if ej else paths[j]
            skip = 1 if np.array_equal(a[-1], b[0]) else 0
            merged.append(np.concatenate((a, b[skip:])))
            used.update((i, j))
            joined_count += 1
        if not used:
            break
        paths = [p for i, p in enumerate(paths) if i not in used] + merged
    return paths, joined_count


def _paper_normalize(image):
    value = image.max(axis=2)
    chroma = image.max(axis=2).astype(float) - image.min(axis=2)
    eligible = (value > 175) & (value < 253) & (chroma < 40)
    if np.count_nonzero(eligible) < 100:
        return image.copy()
    bright = eligible & (value >= np.percentile(value[eligible], 85))
    paper = np.median(image[bright], axis=0)
    return np.clip(image.astype(float) * 255 / np.maximum(paper, 1), 0, 255).astype(np.uint8)


def extract(image, config=None, pad=None):
    """Return geometry + diagnostic maps, with source pixel coordinates intact.

    `pad`, when supplied, is (top, bottom, left, right). Geometry is not scaled
    into robot coordinates. Widths are optical source estimates in pixels.
    """
    cfg = config or InkConfig()
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected a nonempty uint8 BGR image')
    if min(image.shape[:2]) < 5:
        raise ValueError('Image must be at least 5 x 5 pixels')
    h, w = image.shape[:2]
    ratio = max(h, w) / 1110.
    normalized = _paper_normalize(image)
    smooth = cv2.bilateralFilter(normalized, 5, 22, max(1., ratio * 2))
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    value = smooth.max(axis=2)
    # V retains the darkness distinction between blue ink and blue wash;
    # unlike the old mask, saturation alone never rejects colored ink.
    signal = np.rint(.55 * gray + .45 * value).astype(np.uint8)
    responses = []
    for diameter, minimum in ((5, 3), (9, 5), (15, 9)):
        # At low resolution an actual 3 px pen line can occupy 5 raster
        # pixels. Do not shrink every detector below that physical pixel width.
        size = max(minimum, int(round(diameter * ratio)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        responses.append(cv2.morphologyEx(signal, cv2.MORPH_BLACKHAT, kernel))
    contrast = np.maximum.reduce(responses).astype(np.float32)
    neutral_dark = (value < 85)
    strong = ((contrast >= cfg.strong_contrast) & (value < cfg.max_value) & (gray < 185)) | neutral_dark
    weak = ((contrast >= cfg.weak_contrast) & (value < min(235, cfg.max_value + 18)) & (gray < 210)) | neutral_dark
    if pad is not None:
        top, bottom, left, right = map(int, pad)
        valid = np.zeros((h, w), bool)
        valid[top:h-bottom if bottom else h, left:w-right if right else w] = True
        strong &= valid
        weak &= valid
    # Bounded hysteresis: retain weak pixels near strong ink, without allowing
    # a single seed to flood an entire textured watercolor region.
    mask = strong.astype(np.uint8)
    for _ in range(max(1, int(round(2 * ratio)))):
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8)) & weak.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(count, np.uint8)
    dots = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        extent = max(stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT])
        if area >= max(4, 5 * ratio * ratio) and extent >= max(3, cfg.min_component_length * ratio):
            keep[label] = 1
        elif max(5, 7 * ratio * ratio) <= area <= max(15, 60 * ratio * ratio):
            yy, xx = np.where(labels == label)
            # Preserve compact, strong isolated marks separately, not as long
            # paths or executable taps. Weak specks are still discarded.
            if np.median(value[yy, xx]) < 100 and np.mean(strong[yy, xx]) > .7:
                dots.append(dict(center=[float(xx.mean()), float(yy.mean())], area_px=int(area)))
    mask = keep[labels]
    width_map = 2 * cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Padding the thinning input handles real strokes touching image borders.
    skeleton = Trace._thin_binary(np.pad(mask * 255, 1))[1:-1, 1:-1]
    graph = _graph(skeleton)
    raw_paths = _trace(graph)
    removed_spurs = 0
    # Prune only tiny terminal spurs attached to a junction, never isolated
    # short strokes, main branches, or closed loops.
    for p in raw_paths:
        a, b = tuple(p[0].astype(int)), tuple(p[-1].astype(int))
        terminal = len(graph[a]) == 1 and len(graph[b]) >= 3
        reverse = len(graph[b]) == 1 and len(graph[a]) >= 3
        if not (terminal or reverse):
            continue
        q = a if terminal else b
        limit = min(max(2., 3 * ratio), max(1.5, width_map[q[1], q[0]] * .55))
        if _length(p) < limit:
            drop = p[:-1] if terminal else p[1:]
            for x, y in drop.astype(int):
                skeleton[y, x] = 0
            removed_spurs += 1
    raw_paths = _trace(_graph(skeleton))
    paths, joins = _join(raw_paths, weak, ratio, cfg)
    result = []
    min_length = max(2.5, cfg.min_component_length * ratio)
    for raw in paths:
        length = _length(raw)
        if length < min_length:
            continue
        closed = bool(np.array_equal(raw[0], raw[-1]))
        # RDP has a bounded source-polyline deviation; no free-moving CLIP
        # coordinates or unconstrained spline smoothing.
        p = cv2.approxPolyDP(raw.reshape(-1, 1, 2), max(.6, cfg.simplify_error * ratio), False)[:, 0, :]
        ij = np.rint(p).astype(int)
        result.append(dict(points=p.astype(float).tolist(),
                           width_px=width_map[ij[:, 1], ij[:, 0]].astype(float).tolist(),
                           closed=closed, length_px=_length(p),
                           mean_contrast=float(np.mean(contrast[raw[:, 1].astype(int), raw[:, 0].astype(int)]))))
    result.sort(key=lambda p: (-p['length_px'], p['points'][0][1], p['points'][0][0]))
    raster = np.zeros((h, w), np.uint8)
    for p in result:
        cv2.polylines(raster, [np.rint(p['points']).astype(np.int32)], False, 255, 1)
    stats_out = dict(algorithm='source-constrained-ink-v2', config=asdict(cfg),
                     input_size=[w, h], coordinate_units='source_pixels', paths=len(result),
                     total_length_px=sum(p['length_px'] for p in result),
                     compact_marks=len(dots), removed_terminal_spurs=removed_spurs,
                     source_supported_joins=joins, raw_graph_segments=len(raw_paths),
                     ready_for_execution=False, brush_width_calibrated=False)
    return dict(paths=result, dots=dots, stats=stats_out,
                mask=mask * 255, skeleton=skeleton, raster=raster,
                contrast=contrast, weak_mask=weak.astype(np.uint8) * 255)


def build_strokes(work, scale, pad=None, source_image=None):
    """Adapt visible geometry to the planner's existing stroke schema.

    Pressure uses the previous conservative heuristic, not a newly calibrated
    brush model. Keep every fitted vertex when densifying (including corners).
    Compact dot candidates remain diagnostic until they have a dot policy.
    `work` is already the shared prepared image; do not remap source_image.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid mm-per-pixel scale')
    from config import ShuimoConfig
    from ink_control import refresh_stats
    from shuimo import ink_palette_index, press_to_width_mm

    extracted = extract(work, pad=pad)
    strokes = []
    low = float(ShuimoConfig.PRESS_MIN_MM)
    high = float(ShuimoConfig.PRESS_MAX_MM)
    if not 0 <= low < high:
        raise ValueError('Invalid configured pressure limits')
    for index, path in enumerate(extracted['paths']):
        p = np.asarray(path['points'], float)
        widths = np.asarray(path['width_px'], float)
        if path['length_px'] * scale < 1.:
            continue
        dense, dw = [], []
        for k, (a, b) in enumerate(zip(p, p[1:])):
            count = max(1, int(math.ceil(float(np.linalg.norm(b-a)) * scale / .35)))
            t = np.arange(count) / count
            dense.extend(a[None, :] + t[:, None] * (b-a)[None, :])
            dw.extend(widths[k] + t * (widths[k+1] - widths[k]))
        points = np.vstack((np.asarray(dense), p[-1]))
        dw = np.r_[dw, widths[-1]]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))] * scale
        presses = np.interp(dw * scale, [.25, .55, 1., 1.6], [.06, .1, .2, .3])
        if not path['closed']:
            ramp = np.minimum(np.clip(arc/.6, 0, 1), np.clip((arc[-1]-arc)/.8, 0, 1))
            presses *= .55 + .45 * ramp
        presses = np.clip(presses, low, min(high, max(low, .3)))
        stroke = dict(points=points.tolist(), press=presses.tolist(), speed=[18.] * len(points),
                      source_width_mm=(dw * scale).tolist(), source_closed=path['closed'],
                      layer=0 if np.mean(points[:, 1]) < work.shape[0]*.28 else 1,
                      pal_index=ink_palette_index(), tail_mm=0., lift_mm=4.,
                      role='source_center', name='原图约束中心线 v2',
                      source_path_id=index, design_origin='source-constrained-ink-v2')
        strokes.append(refresh_stats(stroke, scale))
    stats = dict(extracted['stats'])
    stats.update(profile='mountain', detail='color-aware ink and source-supported centerlines',
                 source_paths=len(extracted['paths']), strokes=len(strokes),
                 layers={k: sum(s['layer'] == k for s in strokes) for k in range(3)},
                 length_mm=sum(s['length_mm'] for s in strokes),
                 mean_width_mm=float(np.mean([s['width_mm'] for s in strokes])) if strokes else 0.,
                 max_width_mm=max((press_to_width_mm(s['press_max']) for s in strokes), default=0.),
                 mean_press_mm=float(np.mean([s['press_mean'] for s in strokes])) if strokes else 0.,
                 max_press_mm=max((s['press_max'] for s in strokes), default=0.),
                 pressure_mapping='previous conservative source-width heuristic; NOT calibrated',
                 pressure_caps_mm={str(k): min(high, max(low, .3)) for k in range(3)},
                 brush_width_calibrated_for_this_job=False, broad_rails=0)
    return strokes, stats
