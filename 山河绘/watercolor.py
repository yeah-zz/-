"""Reference-driven, OFFLINE watercolor design in image coordinates.

No hardware imports, robot coordinates, pressure commands or invented pigment
recipes. Color opacity is an optical target requiring physical swatch calibration.
"""
from pathlib import Path
import math
import cv2
import numpy as np
import color_pass
import imgio
import vision
from geometry import CanvasMap

COLOR_ORDER = ('blue', 'yellow', 'red')
RECIPE_LEVELS = {'wash': .16, 'body': .36, 'accent': .70}
PALETTE = {'red': (30, 48, 245), 'yellow': (38, 186, 248), 'blue': (160, 105, 48), 'green': (72, 112, 65)}


def analyze(work, mm_per_px):
    """Separate chromatic paint from neutral ink and the white mist/background."""
    labels, _ = color_pass.classify_colors(work, min_chroma=2.5)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.linalg.norm(lab[:, :, 1:]-128., axis=2)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    ink = ((gray < 155) & (chroma < 15)) | (gray < 65)
    labels[ink] = -1
    strength = np.zeros(labels.shape, np.float32)
    delta = 255.-work.astype(np.float32)
    for i, name in enumerate(color_pass.COLORS):
        basis = 255.-np.asarray(PALETTE[name], np.float32)
        alpha = np.clip((delta @ basis)/(basis @ basis), 0., .95)
        strength[labels == i] = alpha[labels == i]
    # Small-scale pigment texture should not fragment a coherent mountain face.
    sigma = max(.7, .45/mm_per_px)
    fields = {}
    for i, name in enumerate(color_pass.COLORS):
        evidence = ((labels == i) & (strength >= .025)).astype(np.uint8)
        evidence = cv2.morphologyEx(evidence, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*max(2, int(.8/mm_per_px))+1,)*2))
        values = cv2.GaussianBlur(strength*(labels == i), (0, 0), sigma)
        fields[name] = (evidence.astype(bool), values)
    return ink, fields


def flow_field(work, mm_per_px):
    """Structure-tensor tangents follow local rock/paint features, oriented downhill.

Where evidence is weak, fall back to a downward slope, not horizontal hatching.
"""
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    gray = cv2.GaussianBlur(gray, (0, 0), max(1., .35/mm_per_px))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sigma = max(1., 1.7/mm_per_px)
    smooth = lambda a: cv2.GaussianBlur(a, (0, 0), sigma)
    xx, xy, yy = smooth(gx*gx), smooth(gx*gy), smooth(gy*gy)
    angle = .5*np.arctan2(2*xy, xx-yy)+np.pi/2
    vx, vy = np.cos(angle), np.sin(angle)
    sign = np.where(vy < 0, -1., 1.)
    vx, vy = vx*sign, vy*sign
    coherence = np.sqrt((xx-yy)**2+4*xy**2)/(xx+yy+1e-8)
    weight = .85*np.clip(coherence, 0., 1.)
    vx = weight*vx+(1-weight)*.18
    vy = np.maximum(weight*vy+(1-weight), .22)
    norm = np.sqrt(vx*vx+vy*vy)
    return np.stack([vx/norm, vy/norm], axis=2)


def line_samples(a, b, step=.45):
    count = max(1, int(math.ceil(np.linalg.norm(b-a)/step)))
    return a+(b-a)*np.linspace(0, 1, count+1)[:, None]


def inside_segment(mask, a, b):
    q = np.rint(line_samples(a, b)).astype(int)
    h, w = mask.shape
    if (q < 0).any() or (q[:, 0] >= w).any() or (q[:, 1] >= h).any():
        return False
    return bool(mask[q[:, 1], q[:, 0]].all())


def trace(seed, valid, field, step_px, max_length_px):
    def half(sign):
        path = [np.asarray(seed, float)]
        previous = None
        for _ in range(int(max_length_px/(2*step_px))):
            x, y = np.rint(path[-1]).astype(int)
            direction = field[y, x]*sign
            if previous is not None:
                direction = .7*previous+.3*direction
            direction /= max(np.linalg.norm(direction), 1e-8)
            nxt = path[-1]+direction*step_px
            if not inside_segment(valid, path[-1], nxt): break
            path.append(nxt); previous = direction
        return path
    backward, forward = half(-1), half(1)
    return np.asarray(backward[:0:-1]+forward)


def stroke_profile(points, alpha_field, width_mm):
    q = np.rint(points).astype(int)
    target = np.clip(alpha_field[q[:, 1], q[:, 0]], .035, .95)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    u = arc/max(arc[-1], 1e-8)
    # A long full-width body with short tapered ends avoids bead-like strokes.
    taper = np.minimum(np.minimum(.55+u/.15*.45, .25+(1-u)/.20*.75), 1.)
    # Requested footprint only; no conversion to pressure without measured lookup.
    widths = width_mm*taper*(.65+.35*np.sqrt(target/max(float(target.max()), .035)))
    mean = float(np.mean(target))
    recipe = 'wash' if mean < .22 else ('body' if mean < .48 else 'accent')
    return target.tolist(), widths.tolist(), recipe


def plan_paths(work, mm_per_px, brush_width_mm=3., ink_gap_mm=.6):
    if not math.isfinite(mm_per_px) or mm_per_px <= 0: raise ValueError('Invalid image scale')
    if not math.isfinite(brush_width_mm) or not .3 <= brush_width_mm <= 10:
        raise ValueError('Design brush width must be in [0.3, 10] mm')
    if not math.isfinite(ink_gap_mm) or not 0 <= ink_gap_mm <= 10:
        raise ValueError('Design ink gap must be in [0, 10] mm')
    ink, fields = analyze(work, mm_per_px)
    flow = flow_field(work, mm_per_px)
    gap = int(math.ceil(ink_gap_mm/mm_per_px))
    h, w = ink.shape
    strokes = []
    for color in COLOR_ORDER:
        region, strength = fields[color]
        occupied = np.zeros((h, w), np.uint8)
        field = flow if color != 'red' else np.broadcast_to(np.array([1., 0.]), (h, w, 2))
        # Broad wash first, then narrower strokes where ridges constrain the brush.
        # These are requested widths, not assumed executable pressure settings.
        widths_to_try = sorted({brush_width_mm, min(brush_width_mm, max(.45, .52*brush_width_mm)),
                               min(brush_width_mm, max(.3, .24*brush_width_mm))}, reverse=True)
        for width in widths_to_try:
            radius = int(math.ceil(width/(2*mm_per_px)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)
            ink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(radius+gap)+1,)*2)
            blocked = cv2.dilate(ink.astype(np.uint8), ink_kernel).astype(bool)
            centers = cv2.erode(region.astype(np.uint8), kernel,
                borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) & ~blocked
            pitch = max(2, int(round(.42*width/mm_per_px)))
            seeds = [(x, y) for y in range(pitch//2, h, pitch)
                     for x in range(pitch//2+(pitch//2 if (y//pitch)%2 else 0), w, pitch)
                     if centers[y, x]]
            seeds.sort(key=lambda p: (-float(strength[p[1], p[0]]), p[1], p[0]))
            for x, y in seeds:
                if occupied[y, x]: continue
                valid = centers & (occupied == 0)
                points = trace((x, y), valid, field, max(.75, .45/mm_per_px), 35./mm_per_px)
                if len(points) < 3: continue
                length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()*mm_per_px)
                if length < max(1.5, width): continue
                alpha, widths, recipe = stroke_profile(points, strength, width)
                if color == 'red': widths = [width]*len(points)
                strokes.append(dict(color=color, recipe=recipe, layer=list(RECIPE_LEVELS).index(recipe),
                    points=points.tolist(), target_opacity=alpha, target_width_mm=widths,
                    brush_width_mm_estimate=width, length_mm=length,
                    path_style='flowing_slope' if color != 'red' else 'sun_fill',
                    pressure_mm=None, pigment_recipe_calibrated=False))
                cv2.polylines(occupied, [np.rint(points).astype(np.int32)], False, 255,
                              max(1, int(round(.60*width/mm_per_px))))
    strokes.sort(key=lambda s: (COLOR_ORDER.index(s['color']), s['layer'], s['points'][0][1]))
    for i, s in enumerate(strokes): s['id'] = 'watercolor-%04d'%i
    return strokes, ink, fields


def schedule(strokes):
    steps = []
    previous_color = None
    for color in COLOR_ORDER:
        if not any(s['color'] == color for s in strokes): continue
        if previous_color is not None:
            steps.append(dict(kind='manual_wash_wipe_and_verify_clean', from_color=previous_color, to_color=color))
        for layer, recipe in enumerate(RECIPE_LEVELS):
            ids = [s['id'] for s in strokes if s['color'] == color and s['recipe'] == recipe]
            if not ids: continue
            if layer > 0:
                steps.append(dict(kind='observe_previous_layer_dry', color=color, automatic_seconds=None))
            steps.extend([dict(kind='manual_load_calibrated_tone', color=color, recipe=recipe,
                               concentration=None, water_ratio=None, calibrated=False),
                          dict(kind='paint_design_layer', color=color, recipe=recipe, stroke_ids=ids)])
        previous_color = color
    return steps


def render_image(work, plan, include_ink=True):
    """Return (preview_bgr_uint8, path_view_bgr_uint8) without writing any file.

Preview uses generated paths only; original chromatic pixels are never pasted.
Coverage is a soft footprint envelope, NOT simulated pigment/water diffusion.
"""
    h, w = work.shape[:2]
    canvas = np.full((h, w, 3), 255., np.float32)
    path_view = np.full((h, w, 3), 255, np.uint8)
    scale = plan['mm_per_px']
    palette = plan.get('palette_bgr', PALETTE)
    colors = (*COLOR_ORDER, 'green') if any(s['color']=='green' for s in plan['strokes']) else COLOR_ORDER
    for color in colors:
        coverage = np.zeros((h, w), np.float32)
        for s in plan['strokes']:
            if s['color'] != color: continue
            points = np.asarray(s['points'])
            # Local envelopes are max-composited to avoid false dark overlap stripes.
            pad = int(math.ceil(s['brush_width_mm_estimate']/scale))+3
            lo = np.maximum(np.floor(points.min(0)).astype(int)-pad, 0)
            hi = np.minimum(np.ceil(points.max(0)).astype(int)+pad+1, (w, h))
            local = np.zeros((hi[1]-lo[1], hi[0]-lo[0]), np.float32)
            for j, (a, b) in enumerate(zip(points, points[1:])):
                width = (s['target_width_mm'][j]+s['target_width_mm'][j+1])/2/scale
                alpha = (s['target_opacity'][j]+s['target_opacity'][j+1])/2
                cv2.line(local, tuple(np.rint(a-lo).astype(int)), tuple(np.rint(b-lo).astype(int)),
                         float(alpha), max(1, int(round(width))), cv2.LINE_AA)
            local = cv2.GaussianBlur(local, (0, 0), max(.45, .10*s['brush_width_mm_estimate']/scale))
            roi = coverage[lo[1]:hi[1], lo[0]:hi[0]]
            np.maximum(roi, local, out=roi)
            cv2.polylines(path_view, [np.rint(points).astype(np.int32)], False, tuple(palette[color]), 1, cv2.LINE_AA)
        bgr = np.asarray(palette[color], np.float32)
        canvas = canvas*(1-coverage[:, :, None])+bgr*coverage[:, :, None]
    if include_ink:
        ink, _ = analyze(work, scale)
        canvas[ink] = work[ink]
        path_view[ink] = work[ink]
    return np.clip(canvas, 0, 255).astype(np.uint8), path_view


def render(work, plan, preview_path, paths_path=None, include_ink=True):
    """Write the preview (and optionally the path-only view) to disk."""
    preview, path_view = render_image(work, plan, include_ink)
    if not imgio.imwrite(str(preview_path), preview):
        raise OSError('Cannot write watercolor preview')
    if paths_path and not imgio.imwrite(str(paths_path), path_view):
        raise OSError('Cannot write watercolor path preview')


def build_plan(reference_path, brush_width_mm=3., ink_gap_mm=.6, work_size=1000):
    source = vision.load_image(str(reference_path))
    work, info = vision.prepare_image(source, work_size=work_size)
    mapper = CanvasMap(info['prepared_size'], info['image_corners'])
    strokes, ink, fields = plan_paths(work, mapper.mm_per_px, brush_width_mm, ink_gap_mm)
    if not strokes: raise ValueError('No usable colored regions in reference')
    plan = dict(format='watercolor-reference-design-v1', ready_for_execution=False,
        composition='new_reference_composition', existing_paper_registration=None,
        reference_image=str(Path(reference_path).resolve()), reference_sha256=color_pass._digest(reference_path),
        image_size=list(info['prepared_size']), mm_per_px=mapper.mm_per_px,
        image_corners=mapper.image_corners, brush_width_mm_estimate=brush_width_mm,
        ink_gap_mm=ink_gap_mm, preview_model='Generated variable-width path envelopes; NOT fluid simulation',
        palette_bgr=PALETTE,
        requires=['New paper or explicitly registered matching ink composition',
                  'Measured brush-width versus pressure table for the selected brush',
                  'On-paper light/body/accent pigment swatches and retained-color distance',
                  'Verified manual wash/load steps or separately calibrated pigment stations'],
        tone_kind='Optical image targets; no inferred concentration, water ratio, or pressure',
        strokes=strokes, sequence=schedule(strokes))
    return plan, work
