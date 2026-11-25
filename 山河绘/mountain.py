"""Conservative mountain line-art profile. All distances are paper millimetres.

Upper ink-run centerlines retain the silhouette of each connected mountain; interior
lines must run downhill and leave room for the estimated brush footprint.
This is a landscape heuristic, not a general semantic image recognizer.
No calibration or global drawing settings are changed.
"""
import cv2
import numpy as np

from config import ShuimoConfig
import shuimo
import vision


def length(path):
    return float(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1).sum())


def runs(points, keep):
    start = None
    for i, yes in enumerate(list(keep) + [False]):
        if yes and start is None:
            start = i
        elif not yes and start is not None:
            if i - start >= 2:
                yield points[start:i]
            start = None


def resample(path, scale):
    # Retain corners explicitly: uniform sampling alone cuts across peaks.
    p = cv2.approxPolyDP(np.asarray(path, np.float32).reshape(-1, 1, 2),
                         0.16 / scale, False).reshape(-1, 2)
    out = [p[0]]
    for a, b in zip(p, p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) * scale / 1.0)))
        out.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.asarray(out, float)


def join_junctions(paths):
    """Continue only through the very same skeleton node, never across white.

    Pair the most nearly opposite outgoing tangents; do not arbitrarily take
    a DFS branch at a crossroad. Rebuild local endpoints after each sweep.
    """
    paths = {i: list(p) for i, p in enumerate(paths)}
    while True:
        endpoints = {}
        for i, p in paths.items():
            for end in (0, -1):
                endpoints.setdefault(tuple(p[end]), []).append((i, end))
        changed = False
        for node, ends in sorted(endpoints.items()):
            best = None
            for n, (i, a) in enumerate(ends):
                if i not in paths or tuple(paths[i][a]) != node: continue
                for j, b in ends[n+1:]:
                    if i == j or j not in paths or tuple(paths[j][b]) != node: continue
                    pa = paths[i] if a == 0 else paths[i][::-1]
                    pb = paths[j] if b == 0 else paths[j][::-1]
                    va = np.asarray(pa[min(3, len(pa)-1)]) - node
                    vb = np.asarray(pb[min(3, len(pb)-1)]) - node
                    norm = np.linalg.norm(va) * np.linalg.norm(vb)
                    score = -float(np.dot(va, vb)) / norm if norm else -1
                    if score > .80 and (best is None or score > best[0]):
                        best = score, i, j, pa, pb
            if best:
                _, i, j, pa, pb = best
                paths[i] = pa[::-1] + pb[1:]
                del paths[j]
                changed = True
        if not changed: return list(paths.values())


def upper_ink_centerline(region):
    """One point per column through the upper ink run of a component.

    A capped run keeps attached vertical hatching from pulling the ridge
    downwards. The returned path remains inside source ink at each column.
    """
    h, w = region.shape
    top = np.argmax(region, axis=0)
    columns = np.arange(w)
    continuing = region[top, columns].copy()
    depth = np.ones(w, dtype=np.int32)
    for offset in range(1, 11):
        continuing &= region[np.minimum(top + offset, h - 1), columns] & (top + offset < h)
        depth += continuing
    return np.column_stack((columns, top + (depth - 1) * .5)).astype(float)


def contour_depth(path, scale, top_y, bottom_y):
    """A composition band, inferred from ridge prominence and page position."""
    rise = float(np.ptp(path[:, 1]) * scale)
    position = (float(path[:, 1].min()) - top_y) / max(bottom_y - top_y, 1.)
    if rise < 6 or (position < .30 and rise < 12):
        return "far"
    if rise >= 25 or position > .68:
        return "near"
    return "middle"


def build_strokes(work, scale, pad=None, detail="standard"):
    if detail == "structured":
        from mountain_structure import restructure
        base, stats = build_strokes(work, scale, pad, "enriched")
        return restructure(base, stats, work, scale, pad)
    if detail not in ("standard", "enriched"):
        raise ValueError("Unknown mountain detail: " + str(detail))
    enriched = detail == "enriched"
    interior_min = 5.5 if enriched else 6.0
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    # Preserve fine lines at >= 1600 px; no closing across white gaps.
    ink = vision.neutral_ink_mask(work, gray_limit=210) * 255
    if pad:
        t, b, l, r = pad
        if t: ink[:t] = 0
        if b: ink[-b:] = 0
        if l: ink[:, :l] = 0
        if r: ink[:, -r:] = 0
    skeleton = vision._thin(ink)
    count, labels, boxes, _ = cv2.connectedComponentsWithStats(ink, 8)
    outlines = []
    for label in range(1, count):
        x, y, w, h, area = boxes[label]
        if w * scale < 10 or area < 12:
            continue
        region = labels[y:y+h, x:x+w] == label
        path = upper_ink_centerline(region) + (x, y)
        # Discontinuous upper envelopes (e.g. overlapping mountain faces)
        # are split rather than connected by invented vertical lines.
        cuts = np.flatnonzero(np.abs(np.diff(path[:, 1])) * scale > 1.3) + 1
        for part in np.split(path, cuts):
            if len(part) >= 2 and length(part) * scale >= 8:
                outlines.append(part)

    accepted = []
    occupied = np.zeros_like(ink)
    minimum = float(ShuimoConfig.PRESS_MIN_MM)
    maximum = float(ShuimoConfig.PRESS_MAX_MM)
    # Physical pressure ceilings, not normalization to the darkest source pixel.
    # Relative levels are design targets, not measured brush widths.
    contour_boost = {"far": .08, "middle": .16, "near": .24}
    detail_boost = {"far": .0, "middle": .04, "near": .06}

    def add(path, layer, contour=False, depth_band="middle"):
        p = resample(path, scale)
        if not contour and p[-1, 1] < p[0, 1]:
            p = p[::-1]  # Interior strokes always travel downhill.
        d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] * scale
        if d[-1] < (8 if layer == 0 else interior_min):
            return
        # Release along the last 2 mm of the intended path, never beyond it.
        envelope = np.minimum(np.clip(d / 1.2, 0, 1), np.clip((d[-1]-d) / 2., 0, 1))
        boost = contour_boost[depth_band] if contour else (detail_boost[depth_band] if layer == 1 else 0.)
        cap = min(maximum, minimum + boost)
        presses = minimum + (cap - minimum) * envelope
        presses *= np.clip((d[-1] - d) / 1.2, 0, 1)
        widths = [shuimo.press_to_width_mm(v) for v in presses]
        stroke = dict(points=p.tolist(), press=presses.tolist(),
                      speed=[float(ShuimoConfig.SPEED_BASE)] * len(p),
                      pass_=1, pal_index=shuimo.ink_palette_index(), layer=layer,
                      name=["主轮廓", "次级山脊", "下行皴线"][layer],
                      brush=ShuimoConfig.DEFAULT_BRUSH, lift_mm=ShuimoConfig.LIFT_MM,
                      tail_mm=0.0, length_mm=float(d[-1]),
                      role="silhouette" if contour else "downhill",
                      depth_band=depth_band,
                      width_mm=float(np.mean(widths)),
                      press_mean=float(presses.mean()), press_max=float(presses.max()))
        stroke["pass"] = stroke.pop("pass_")
        accepted.append(stroke)
        # Reserve an explicit white gap outside the estimated contact width.
        # cv2's line thickness is a diameter. Reserve the estimated brush
        # contact plus a modest white margin; the former margin removed many
        # distinct source ridges immediately beside a dark silhouette.
        reserve_mm = max(widths) + ((.9 if contour else 1.1) if enriched else (1.0 if contour else 1.3))
        cv2.polylines(occupied, [np.rint(p).astype(np.int32)], False, 255,
                      max(1, int(np.ceil(reserve_mm / scale))))

    top_y = min((float(p[:, 1].min()) for p in outlines), default=0.)
    bottom_y = max((float(p[:, 1].min()) for p in outlines), default=top_y)
    contour_sources = []
    for path in sorted(outlines, key=lambda p: (float(p[:, 1].min()), float(p[0, 0]))):
        # Low, shallow distant ridges carry less pressure than steep peaks.
        band = contour_depth(path, scale, top_y, bottom_y)
        before = len(accepted)
        add(path, 1 if band == "far" else 0, contour=True, depth_band=band)
        if len(accepted) > before:
            contour_sources.append((np.asarray(accepted[-1]["points"]), band))

    outline_distance = cv2.distanceTransform(255 - occupied, cv2.DIST_L2, 5) * scale
    texture_cells = {}

    # Skeleton segments outside the silhouette's reserved footprint.
    residual = skeleton.copy()
    residual[occupied > 0] = 0
    candidates = join_junctions(vision._trace_paths(residual, epsilon=.12 / scale))
    candidates.sort(key=lambda p: (-length(p), p[0][1], p[0][0]))
    rejected = 0
    for path in candidates:
        # Densify before testing spacing, so long simplified segments cannot
        # jump across an already accepted stroke unnoticed.
        p = resample(path, scale)
        if p[-1, 1] < p[0, 1]: p = p[::-1]
        delta = np.diff(p, axis=0)
        total = length(p) * scale
        drop = (p[-1, 1] - p[0, 1]) * scale
        downhill = float(np.maximum(delta[:, 1], 0).sum())
        uphill = float(np.maximum(-delta[:, 1], 0).sum())
        if total < interior_min or drop < total * .48 or uphill > .12 * max(downhill, 1):
            rejected += 1
            continue
        ij = np.rint(p).astype(int)
        clear = occupied[ij[:, 1], ij[:, 0]] == 0
        pieces = list(runs(p, clear))
        if not pieces:
            rejected += 1
        for piece in pieces:
            size = length(piece) * scale
            if size >= interior_min:
                start = np.rint(piece[0]).astype(int)
                cell = tuple(np.floor(piece[0] * scale / 16.).astype(int))
                # Keep strokes rooted near a ridge, with a local detail budget.
                # This suppresses the loose flecks deep inside dense source art.
                if outline_distance[start[1], start[0]] > (12. if enriched else 9.) or texture_cells.get(cell, 0) >= (5 if enriched else 3):
                    rejected += 1
                    continue
                band = min(contour_sources, key=lambda item: np.min(np.sum(
                    (item[0] - piece[0]) ** 2, axis=1)))[1] if contour_sources else "middle"
                add(piece, 1 if size >= 10 else 2, depth_band=band)
                texture_cells[cell] = texture_cells.get(cell, 0) + 1

    # Layer order is stable; within a layer nearest start reduces pen-up travel
    # without reversing the designed downhill direction or moving details first.
    ordered = []
    current = np.array([0., 0.])
    for layer in range(3):
        remaining = [s for s in accepted if s["layer"] == layer]
        while remaining:
            i = min(range(len(remaining)), key=lambda j: float(np.linalg.norm(
                np.array(remaining[j]["points"][0]) - current)))
            s = remaining.pop(i)
            ordered.append(s)
            current = np.array(s["points"][-1])
    stats = dict(profile="mountain", detail=detail, strokes=len(ordered), paths=len(candidates),
                 length_mm=sum(s["length_mm"] for s in ordered),
                 layers={k: sum(s["layer"] == k for s in ordered) for k in range(3)},
                 ink_ratio=float(np.mean(ink > 0)), ink_px=int(np.count_nonzero(ink)),
                 skeleton_px=int(np.count_nonzero(skeleton)), otsu_gray=210,
                 white_ratio=float(np.mean(gray > 245)), pad_excluded=bool(pad),
                 candidates_rejected=rejected, mm_per_px=scale,
                 mean_width_mm=float(np.mean([s["width_mm"] for s in ordered])) if ordered else 0.,
                 max_width_mm=max((s["width_mm"] for s in ordered), default=0.),
                 mean_press_mm=float(np.mean([s["press_mean"] for s in ordered])) if ordered else 0.,
                 max_press_mm=max((s["press_max"] for s in ordered), default=0.))
    return ordered, stats
