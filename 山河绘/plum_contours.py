"""Image-derived double-hook branches, petal contours and small bud deposits.

Offline extension of vision/shuimo; never changes calibration or legacy plans.
"""
from pathlib import Path
import hashlib

import cv2
import numpy as np

import imgio
import shuimo
import vision
from config import ShuimoConfig
from geometry import CanvasMap

INK_PRESS_MM = .12
COLOR_CONTOUR_PRESS_MM = .12
BUD_PRESS_MM = .40
BUD_MAX_AREA_MM2 = 65.
BRANCH_DOUBLE_WIDTH_MM = 1.8
SAMPLE_MM = .50
MAX_POINTS = 80


def masks(work):
    b, g, r = cv2.split(work.astype(np.float32))
    # Channel differences retain the very pale pink that HSV S>100 discarded.
    red = ((r-g > 3) & (r-b > 4) & (r > 135)).astype(np.uint8)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.linalg.norm(lab[:, :, 1:]-128., axis=2)
    ink = (((gray < 215) & (chroma < 16)) | (gray < 65)).astype(np.uint8)
    return ink, red


def _runs(points, flags, closed=False):
    points, flags = np.asarray(points, float), np.asarray(flags, bool)
    if not flags.any():
        return []
    if closed and flags.all():
        return [np.vstack([points, points[0]])]
    if closed:
        start = int(np.flatnonzero(~flags)[0])
        points, flags = np.roll(points, -start, axis=0), np.roll(flags, -start)
    indices = np.flatnonzero(flags)
    return [points[g] for g in np.split(indices, np.flatnonzero(np.diff(indices)>1)+1)
            if len(g) >= 2]


def _sample(points, scale):
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    pts = cv2.approxPolyDP(pts.reshape(-1, 1, 2), .08/scale, False).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    d = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))] * scale
    keep = np.r_[True, np.diff(d)>1e-7]
    pts, d = pts[keep], d[keep]
    if len(pts)<2:
        return pts
    t = np.linspace(0., d[-1], max(2, int(np.ceil(d[-1]/SAMPLE_MM))+1))
    return np.column_stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])])


def _append(strokes, points, scale, kind, press, region=0, color='ink', min_mm=.6):
    pts = _sample(points, scale)
    if len(pts)<2 or np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()*scale < min_mm:
        return
    for start in range(0, len(pts)-1, MAX_POINTS-1):
        part = pts[start:start+MAX_POINTS]
        profile = np.full(len(part), press, float)
        if kind == 'bud_dab':
            profile = np.linspace(press, 0., len(part))
        elif kind in ('ink_detail', 'ink_dot'):
            profile[-1] *= .65
        widths = [shuimo.press_to_width_mm(v) for v in profile]
        strokes.append(dict(id='%s-%04d' % (kind, len(strokes)), color=color,
            kind=kind, source='source_image_'+kind, region_id=int(region),
            points=np.round(part, 3).tolist(), press=np.round(profile, 4).tolist(),
            speed=[20.] * len(part), length_mm=round(float(np.linalg.norm(
                np.diff(part, axis=0), axis=1).sum())*scale, 3),
            press_max=round(float(profile.max()), 4), press_mean=round(float(profile.mean()), 4),
            width_mm=round(float(np.mean(widths)), 3), target_width_mm=widths,
            target_opacity=[.8]*len(part), brush_width_mm_estimate=float(np.mean(widths))))


def ink_plan(ink, scale, flower_mask=None):
    skeleton = vision._thin(ink*255)
    if not np.any(skeleton):
        return []
    flower_mask = np.zeros_like(ink) if flower_mask is None else flower_mask
    source_distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    # Close fine dry-brush cracks for branch silhouette only. Fine stamens below
    # still use the untouched source skeleton and are never merged by this step.
    radius = max(1, round(.45/scale))
    envelope = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2))
    contours, _ = cv2.findContours(envelope, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(envelope, contours, -1, 1, cv2.FILLED)
    distance = cv2.distanceTransform(envelope, cv2.DIST_L2, 5)
    medial = vision._thin(envelope*255)
    # Associate boundary pixels with their nearest medial-axis source width.
    _, labels = cv2.distanceTransformWithLabels((medial==0).astype(np.uint8),
        cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    widths = np.zeros(int(labels.max())+1, np.float32)
    widths[labels[medial>0]] = 2*distance[medial>0]*scale
    nearest_width = widths[labels]
    inset = max(1, int(round(shuimo.press_to_width_mm(INK_PRESS_MM)/2/scale)))
    inner = cv2.erode(envelope, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*inset+1,)*2))
    contours, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    strokes = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        pts = contour[:, 0, :]
        flags = ((nearest_width[pts[:, 1], pts[:, 0]] >= BRANCH_DOUBLE_WIDTH_MM)
                 & (flower_mask[pts[:, 1], pts[:, 0]]==0))
        for run in _runs(pts, flags, closed=True):
            _append(strokes, run, scale, 'branch_contour', INK_PRESS_MM)
    # Existing thinning/path tracing retains fine twigs and radial stamens.
    paths = shuimo._join_collinear_branches(vision._trace_paths(skeleton, epsilon=.35))
    for path in paths:
        pts = _sample(path, scale)
        if len(pts)<2:
            continue
        xy = np.rint(pts).astype(int)
        flags = nearest_width[xy[:, 1], xy[:, 0]] <= BRANCH_DOUBLE_WIDTH_MM+.15
        in_flower = flower_mask[xy[:, 1], xy[:, 0]]>0
        flags[in_flower] = 2*source_distance[xy[in_flower, 1], xy[in_flower, 0]]*scale <= 1.95
        for run in _runs(pts, flags):
            _append(strokes, run, scale, 'ink_detail', INK_PRESS_MM, min_mm=.55)
    # Isolated anther dots are smaller than the legacy 2 mm stroke cutoff.
    n, labels, stats, centers = cv2.connectedComponentsWithStats(ink, 8)
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]*scale*scale
        if .07 <= area <= 1.5:
            center = centers[k]
            _append(strokes, [center, center+[.18/scale, 0]], scale,
                    'ink_dot', INK_PRESS_MM, min_mm=.1)
    return strokes


def _petals(blob, scale, ink=None):
    # Source silhouette supplies all outer curves. Five radial lobe separators
    # are inferred where overlapping pale petals do not have explicit boundaries.
    radius = max(1, round(.35/scale))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2))
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(blob, contours, -1, 1, cv2.FILLED)
    ys, xs = np.nonzero(blob)
    center = np.asarray([xs.mean(), ys.mean()])
    yy, xx = np.indices(blob.shape)
    if ink is not None:
        density = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), .7/scale)
        near = ((xx-center[0])**2+(yy-center[1])**2) <= (3.5/scale)**2
        iy, ix = np.unravel_index(np.argmax(density*near), blob.shape)
        if density[iy, ix] > 0:
            center = np.asarray([ix, iy], float)
    angle = (np.degrees(np.arctan2(ys-center[1], xs-center[0]))+360)%360
    radial = np.zeros(360)
    np.maximum.at(radial, np.floor(angle).astype(int), np.hypot(xs-center[0], ys-center[1]))
    # Circular smoothing is used only to locate tips/valleys, never to replace
    # the actual source contour with an idealized flower shape.
    radial = np.convolve(np.r_[radial[-7:], radial, radial[:7]], np.ones(15)/15, 'valid')
    best = None
    for phase in range(72):
        peaks = []
        for k in range(5):
            candidates = (phase+k*72+np.arange(-22, 23))%360
            peaks.append(int(candidates[np.argmax(radial[candidates])]))
        score = sum(radial[p] for p in peaks)
        if best is None or score>best[0]:
            best = (score, sorted(peaks))
    peaks = best[1]
    boundaries = []
    for k in range(5):
        lo, hi = peaks[k], peaks[(k+1)%5]+(360 if k==4 else 0)
        indices = np.arange(lo+8, hi-7)
        boundaries.append(int(indices[np.argmin(radial[indices%360])]))
    all_angles = (np.degrees(np.arctan2(yy-center[1], xx-center[0]))+360)%360
    unwrapped = (all_angles-boundaries[0])%360
    limits = sorted((b-boundaries[0])%360 for b in boundaries)+[360]
    petals = []
    for lo, hi in zip(limits[:-1], limits[1:]):
        heart_gap = (xx-center[0])**2+(yy-center[1])**2 >= (.45/scale)**2
        petal = ((unwrapped>=lo)&(unwrapped<hi)&(blob>0)&heart_gap).astype(np.uint8)
        petals.append(petal)
    return petals, center


def color_plan(red, scale, ink=None):
    n, labels, stats, centers = cv2.connectedComponentsWithStats(red, 8)
    strokes, regions = [], []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]*scale*scale
        if area < 2.:
            continue
        blob = (labels==k).astype(np.uint8)
        small = area < BUD_MAX_AREA_MM2
        regions.append(dict(id=k, kind='bud' if small else 'flower', area_mm2=round(area, 2),
                            center=centers[k].tolist()))
        if small:
            # Two tiny deposits follow the bud's own major axis; no large petal fill.
            ys, xs = np.nonzero(blob)
            coords = np.column_stack([xs, ys]).astype(float)
            _, _, vt = np.linalg.svd(coords-centers[k], full_matrices=False)
            axis = vt[0]
            for offset in (-.55, .55):
                center = centers[k]+axis*(offset/scale)
                _append(strokes, [center-axis*(.35/scale), center+axis*(.35/scale)],
                        scale, 'bud_dab', BUD_PRESS_MM, k, 'red', min_mm=.1)
        else:
            petals, flower_center = _petals(blob, scale, ink)
            for petal in petals:
                # Mild closing removes pixel pinholes without losing pale outer lobes.
                petal = cv2.morphologyEx(petal, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                contours, _ = cv2.findContours(petal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    if cv2.contourArea(contour)*scale*scale < 5.:
                        continue
                    pts = contour[:, 0, :]
                    # Open each petal's root before the dense black stamens, so
                    # the second color pass does not paint over the flower heart.
                    flags = np.linalg.norm(pts-flower_center, axis=1)*scale >= 2.
                    for run in _runs(pts, flags, closed=True):
                        _append(strokes, run, scale, 'petal_contour',
                                COLOR_CONTOUR_PRESS_MM, k, 'red')
    return strokes, regions


def build_plan(reference, margin_mm=24., work_size=1400):
    work, info = vision.prepare_image(vision.load_image(str(reference)), work_size=work_size)
    if info.get('used_corners'):
        raise ValueError('Contour fit needs automatic image corners; custom registration requires a separate fit')
    mapper = CanvasMap(info['prepared_size'], info['image_corners'])
    h, w = work.shape[:2]
    pad = int(np.ceil(margin_mm/mapper.mm_per_px))
    gain = min((w-2*pad)/w, (h-2*pad)/h)
    if gain <= 0:
        raise ValueError('Canvas too small for the requested margin')
    small = cv2.resize(work, (round(w*gain), round(h*gain)), interpolation=cv2.INTER_AREA)
    work = np.full_like(work, 255)
    y, x = (h-small.shape[0])//2, (w-small.shape[1])//2
    work[y:y+small.shape[0], x:x+small.shape[1]] = small
    ink, red = masks(work)
    geometry = dict(image_size=[w, h], image_corners=mapper.image_corners, mm_per_px=mapper.mm_per_px)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    flower_ids = [k for k in range(1, n)
                  if stats[k, cv2.CC_STAT_AREA]*mapper.mm_per_px**2 >= BUD_MAX_AREA_MM2]
    flowers = np.isin(labels, flower_ids).astype(np.uint8)
    radius = max(1, round(1./mapper.mm_per_px))
    flowers = cv2.dilate(flowers, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2))
    black = ink_plan(ink, mapper.mm_per_px, flowers)
    colored, regions = color_plan(red, mapper.mm_per_px, ink)
    modules = {}
    for name, fmt, strokes in [('outline', 'outline-design-v1', black),
                               ('color', 'sparse-color-design-v1', colored)]:
        modules[name] = dict(format=fmt, ready_for_execution=False, **geometry,
            strokes=strokes, stroke_count=len(strokes), drawing_style='contours_and_small_dabs',
            mass_fill=False, calibrated_preview=True)
    modules['color']['regions'] = regions
    source = Path(reference).resolve()
    return dict(format='plum-experiment-v1', ready_for_execution=False, **geometry,
        reference_image=str(source), reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        stages=['outline', 'color'], modules=modules, algorithm='vision_shuimo_source_contours_v1',
        petal_separator_model='Five radial lobes inferred from source tips and valleys; outer curves from source mask',
        fit=dict(margin_mm=margin_mm, uniform_gain=gain),
        note='Pure branch contours, source fine lines/stamens, red petal outlines, small bud dabs; no fill'), work


def render(plan, work, calibrated=True, stage=None):
    canvas = np.full_like(work, 255)
    for key in ([stage] if stage else plan['stages']):
        for stroke in plan['modules'][key]['strokes']:
            pts = np.rint(stroke['points']).astype(np.int32)
            color = (35, 35, 35) if key=='outline' else (135, 135, 225)
            for i in range(len(pts)-1):
                width = ((shuimo.press_to_width_mm(stroke['press'][i])+
                          shuimo.press_to_width_mm(stroke['press'][i+1]))/2
                         /plan['mm_per_px']) if calibrated else 1.
                cv2.line(canvas, tuple(pts[i]), tuple(pts[i+1]), color,
                         max(1, round(width)), cv2.LINE_AA)
    return canvas


def save_image(path, image):
    if not imgio.imwrite(str(path), image):
        raise OSError('Cannot save '+str(path))
