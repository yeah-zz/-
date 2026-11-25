"""Offline, image-coordinate planning for several painting subjects.

This module never imports a robot executor. Every result is a design proposal,
not a calibrated motion job. New subjects implement ``Planner(analysis)`` and
register their name in PLANNERS.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Callable

import cv2
import numpy as np

import imgio
import vision
from geometry import CanvasMap


@dataclass(frozen=True)
class Analysis:
    """Common image evidence passed to each subject planner (pixel coordinates)."""
    work: np.ndarray                 # uint8 BGR, after vision.prepare_image
    mm_per_px: float                 # approximate paper scale, for design widths only
    pad: tuple                       # top, bottom, left, right in prepared pixels
    image_corners: list
    ink: np.ndarray                  # uint8 0/1 neutral ink mask
    colors: dict                     # hue name -> uint8 0/1 mask


def analyze(reference, work_size=1000):
    """Shared preprocessing, neutral ink extraction, and coarse hue separation."""
    work, info = vision.prepare_image(vision.load_image(str(reference)), work_size=work_size)
    mapper = CanvasMap(info['prepared_size'], info['image_corners'])
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    colored = (saturation >= 45) & (value >= 35)
    ink = vision.neutral_ink_mask(work, gray_limit=210)
    colored &= ink == 0
    # OpenCV hue is [0, 179]. These masks are evidence, not pigment recipes.
    bands = {
        'red': (hue < 11) | (hue >= 170),
        'orange': (hue >= 11) & (hue < 20),
        'yellow': (hue >= 20) & (hue < 38),
        'green': (hue >= 38) & (hue < 85),
        'blue': (hue >= 85) & (hue < 135),
        'purple': (hue >= 135) & (hue < 170),
    }
    colors = {name: (colored & band).astype(np.uint8) for name, band in bands.items()}
    return Analysis(work, mapper.mm_per_px, tuple(info['pad']), mapper.image_corners,
                    ink, colors)


def _module(strokes, algorithm, **extra):
    return dict(format='subject-strokes-v1', algorithm=algorithm,
                ready_for_execution=False, strokes=strokes,
                stroke_count=len(strokes), **extra)


def _outline(a):
    import outline_pass
    plan = outline_pass.build_plan(a.work, a.mm_per_px)
    plan['algorithm'] = 'outline_pass.build_plan'
    return plan


def _generic(a):
    import sparse_color
    strokes, regions = sparse_color.plan_paths(a.work, a.mm_per_px)
    return {'color': _module(strokes, 'sparse_color.plan_paths', regions=regions),
            'outline': _outline(a)}


def _plum(a):
    import plum_blossom
    red = a.colors['red'] | a.colors['purple']
    strokes = plum_blossom.petal_plan(red, a.mm_per_px, a.colors['yellow'])
    return {'outline': _outline(a),
            'petals': _module(strokes, 'plum_blossom.petal_plan',
                              color='red', petal_style='dots')}


def _mountain(a):
    import mountain
    strokes, stats = mountain.build_strokes(a.work, a.mm_per_px, a.pad)
    return {'mountain_ink': _module(strokes, 'mountain.build_strokes', stats=stats)}


def _component_axis(mask, scale, color, minimum_area_mm2=3.0):
    """Trial lotus strokes: major axis of each isolated colored component."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    strokes = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] * scale * scale < minimum_area_mm2:
            continue
        yy, xx = np.nonzero(labels == label)
        pixels = np.column_stack((xx, yy)).astype(float)
        center = pixels.mean(axis=0)
        _, _, basis = np.linalg.svd(pixels - center, full_matrices=False)
        axis = basis[0]
        projection = (pixels - center) @ axis
        lo, hi = np.percentile(projection, [12, 88])
        if (hi - lo) * scale < 1.0:
            continue
        points = np.stack([center + axis * t for t in (lo, (lo + hi) / 2, hi)])
        press = [.16, .32, 0.] if color == 'pink' else [.12, .22, 0.]
        strokes.append(dict(id=f'{color}-{len(strokes):03d}', color=color,
                            points=np.round(points, 3).tolist(), press=press,
                            length_mm=round(float((hi - lo) * scale), 3),
                            source='component_major_axis_trial'))
    return strokes


def _lotus(a):
    # No user lotus reference or measured brush marks exist. This is a small,
    # replaceable geometry baseline for testing the subject interface only.
    pink = a.colors['red'] | a.colors['purple']
    petals = _component_axis(pink, a.mm_per_px, 'pink')
    green = _component_axis(a.colors['green'], a.mm_per_px, 'green')
    return {'petals': _module(petals, 'lotus.component_axis_trial', color='pink'),
            'leaves_stems': _module(green, 'lotus.component_axis_trial', color='green')}


PLANNERS: dict[str, Callable[[Analysis], dict]] = {
    'generic': _generic, 'plum': _plum, 'mountain': _mountain, 'lotus': _lotus}


def build_plan(reference, subject, work_size=1000):
    """Return (design JSON object, prepared BGR image); never produce motion."""
    if subject not in PLANNERS:
        raise ValueError(f'Unknown subject {subject!r}; choose one of {", ".join(PLANNERS)}')
    reference = Path(reference)
    if not reference.is_file():
        raise FileNotFoundError(reference)
    a = analyze(reference, work_size)
    modules = PLANNERS[subject](a)
    h, w = a.work.shape[:2]
    status = 'experimental_no_lotus_reference' if subject == 'lotus' else 'offline_design'
    plan = dict(format='multi-painting-offline-v1', ready_for_execution=False,
                subject=subject, status=status, reference_image=str(reference.resolve()),
                reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                coordinate_space='prepared_image_pixels',
                image_size=[w, h], image_corners=a.image_corners,
                mm_per_px=a.mm_per_px, stages=list(modules), modules=modules,
                analysis=dict(ink_pixels=int(a.ink.sum()),
                              color_pixels={k: int(v.sum()) for k, v in a.colors.items()},
                              pad=list(a.pad)),
                requires=['Reference review', 'Paper registration',
                          'Measured brush footprint, pigment, and drying behavior'])
    return plan, a.work


def preview(work, plan):
    """Draw only proposed image-space paths, never the original filled artwork."""
    canvas = np.full_like(work, 255)
    palette = {'red': (45, 45, 190), 'pink': (130, 110, 220),
               'green': (80, 130, 55), 'blue': (190, 110, 50),
               'yellow': (60, 190, 210)}
    for module in plan['modules'].values():
        for stroke in module['strokes']:
            pts = np.asarray(stroke['points'], dtype=float)
            if len(pts) < 2:
                continue
            color = palette.get(stroke.get('color'), (40, 40, 40))
            width = max(1, int(round(float(stroke.get('width_mm', 0.6)) /
                                     plan['mm_per_px'])))
            cv2.polylines(canvas, [np.rint(pts).astype(np.int32)], False,
                          color, min(width, 12), cv2.LINE_AA)
    return canvas


def save_outputs(plan, work, json_path, preview_path):
    """Write a reviewable design and preview, refusing existing output files."""
    json_path, preview_path = Path(json_path), Path(preview_path)
    if json_path.resolve() == preview_path.resolve() or json_path.exists() or preview_path.exists():
        raise FileExistsError('Output path already exists or paths are identical')
    json_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False)
    picture = preview(work, plan)
    if not imgio.imwrite(str(preview_path), picture):
        raise OSError(f'Cannot write preview: {preview_path}')
    json_path.write_text(data + '\n', encoding='utf-8')
