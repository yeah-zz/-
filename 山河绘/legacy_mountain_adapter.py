"""Use the supplied legacy ink tracer with the current calibrated job pipeline.

The old image-to-skeleton and path ordering code is vendored unchanged in
legacy_landscape_painting.py. This adapter keeps its relative source-width and
local-density signals while using current paper coordinates and pressure caps.
It never imports a robot driver.
"""

import cv2
import numpy as np
from dataclasses import asdict, dataclass

from config import ShuimoConfig
from ink_control import refresh_stats
from legacy_landscape_painting import LandscapeBrushGenerator
from shuimo import ink_palette_index, press_to_width_mm


class LegacyTraceConfig:
    LANDSCAPE_SCALE = 0.12
    LANDSCAPE_JOIN_GAP_MM = 0.0
    LANDSCAPE_DENSE_RADIUS_MM = 3.0
    LANDSCAPE_DENSE_PRESS_REDUCTION = 0.55
    LANDSCAPE_DENSE_SPEED_BOOST = 0.45
    LANDSCAPE_LIGHT_INK_GRAY = 238
    LANDSCAPE_LIGHT_INK_CLOSE_PX = 1
    LANDSCAPE_DIRECTIONAL_JOIN_GAP_MM = 4.0
    LANDSCAPE_DIRECTIONAL_JOIN_ANGLE_DEG = 38.0
    LANDSCAPE_DIRECTIONAL_JOIN_MIN_LENGTH_MM = 1.5


@dataclass(frozen=True)
class BrushExpression:
    """Relative expression controls; absolute ceiling stays in site config.

    Defaults preserve the existing adapter. These are design parameters, not
    measured force or a promise of physical line width.
    """
    pressure_floor_mm: float = .04
    layer_cap_factors: tuple = (1., .72, .48)
    width_gamma: float = 1.
    density_reduction: float = .55
    belly_variation: float = 0.
    belly_period_mm: float = 18.
    entry_mm: float = 1.8
    exit_mm: float = 2.5
    min_length_mm: float = 2.
    support_distance_mm: float = .75
    directional_join_gap_mm: float = 4.

    def validate(self):
        numeric = [v for k, v in asdict(self).items() if k != 'layer_cap_factors']
        if not np.isfinite(numeric).all() or not np.isfinite(self.layer_cap_factors).all():
            raise ValueError('Expression parameters must be finite')
        if len(self.layer_cap_factors) != 3 or not all(0 < v <= 1 for v in self.layer_cap_factors):
            raise ValueError('Three layer caps must be in (0, 1]')
        if not 0 <= self.pressure_floor_mm < min(self.layer_cap_factors) * ShuimoConfig.PRESS_MAX_MM:
            raise ValueError('Pressure floor must be below every layer cap')
        if not .25 <= self.width_gamma <= 4 or not 0 <= self.density_reduction <= 1:
            raise ValueError('Invalid width gamma or density reduction')
        if not 0 <= self.belly_variation <= .3 or self.belly_period_mm <= 0:
            raise ValueError('Invalid belly variation')
        if min(self.entry_mm, self.exit_mm, self.min_length_mm) <= 0:
            raise ValueError('Taper and minimum length must be positive')
        if not 0 <= self.support_distance_mm <= 1 or not 0 <= self.directional_join_gap_mm <= 4:
            raise ValueError('Gap/support settings exceed the reviewed design range')
        return self


def _dense_samples(raw, max_step_px):
    """Interpolate coordinates and old width/density signals together."""
    out = [raw[0]]
    for a, b in zip(raw, raw[1:]):
        distance = float(np.linalg.norm(b[:2] - a[:2]))
        for i in range(1, max(1, int(np.ceil(distance / max_step_px))) + 1):
            n = max(1, int(np.ceil(distance / max_step_px)))
            out.append(a + (b - a) * i / n)
    return np.asarray(out, float)


def _supported_runs(samples, support):
    start = None
    for i, good in enumerate(list(support) + [False]):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start >= 2:
                yield samples[start:i]
            start = None


def build_strokes(work, scale, pad=None, expression=None, trace_size=600):
    """Return current-format strokes from the old line-art extractor."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid paper scale")
    top, bottom, left, right = (pad or (0, 0, 0, 0))
    h, w = work.shape[:2]
    crop = work[top:h-bottom if bottom else h, left:w-right if right else w]
    if min(crop.shape[:2]) < 3:
        raise ValueError("Image padding leaves no source area")
    ch, cw = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if not 300 <= trace_size <= 1800:
        raise ValueError("Trace size must be between 300 and 1800")
    if max(ch, cw) > trace_size:
        ratio = trace_size / max(ch, cw)
        small = cv2.resize(gray, (int(cw * ratio), int(ch * ratio)),
                           interpolation=cv2.INTER_LINEAR)
    else:
        small = gray
    expr = (expression or BrushExpression(
        pressure_floor_mm=float(ShuimoConfig.PRESS_MIN_MM),
        width_gamma=.75, min_length_mm=1., directional_join_gap_mm=0.
    )).validate()
    trace_config = LegacyTraceConfig()
    trace_config.LANDSCAPE_DIRECTIONAL_JOIN_GAP_MM = expr.directional_join_gap_mm
    source = LandscapeBrushGenerator(trace_config)
    traced = source._extract_continuous_ink_strokes(small)

    press_min = expr.pressure_floor_mm
    press_max = float(ShuimoConfig.PRESS_MAX_MM)
    caps = [press_max * factor for factor in expr.layer_cap_factors]
    if not 0 <= press_min < caps[2]:
        raise ValueError("legacy-source needs a lighter pressure floor for fine detail")

    # Reject unsupported stretches created by the old gap join. A path may be
    # split, but no new line is drawn through the middle of a white area.
    white = (gray >= 245).astype(np.uint8)
    source_distance_mm = cv2.distanceTransform(white, cv2.DIST_L2, 5) * scale
    max_supported_distance_mm = expr.support_distance_mm
    strokes = []
    split_gaps = 0
    omitted_short = 0
    for old_index, old in enumerate(traced):
        raw = np.asarray(old["trajectory"], float)
        if raw.shape[0] < 3 or raw.shape[1] < 4:
            continue
        xy = np.column_stack((left + raw[:, 0] * cw / 1024,
                              top + raw[:, 1] * ch / 1024))
        # Capture widths at the original skeleton samples, before integer-1024
        # coordinate quantization. Resampling widths off the quantized XY would
        # create spurious zero-width gaps beside very thin source lines.
        source_width = np.asarray(old['source_width_1024'],float) * cw / 1024 * scale
        samples = _dense_samples(np.column_stack((xy, raw[:, 2:4], source_width)), .5 / scale)
        ij = np.rint(samples[:, :2] - [left, top]).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, cw - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, ch - 1)
        support = source_distance_mm[ij[:, 1], ij[:, 0]] <= max_supported_distance_mm
        pieces = list(_supported_runs(samples, support))
        split_gaps += max(0, len(pieces) - 1)
        for piece_index, piece in enumerate(pieces):
            points = piece[:, :2]
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))] * scale
            total = float(arc[-1])
            if total < expr.min_length_mm:
                omitted_short += 1
                continue
            layer = min(2, int(old["layer"]))
            width_strength = np.clip(piece[:, 2], 0, 1) ** expr.width_gamma
            density_strength = np.clip(piece[:, 3], 0, 1)
            press = press_min + (caps[layer] - press_min) * width_strength
            press = press_min + (press - press_min) * (1 - expr.density_reduction * density_strength)
            # A broad, deterministic rise/fall, without shifting the centreline.
            press *= 1 - expr.belly_variation * (.5 + .5 * np.cos(2 * np.pi * arc / expr.belly_period_mm))
            taper = np.minimum(np.clip(arc / expr.entry_mm, 0, 1),
                               np.clip((total - arc) / expr.exit_mm, 0, 1))
            press = np.clip(press * taper, 0, press_max)
            speed = ShuimoConfig.SPEED_BASE * (1.05 - .25 * press / press_max)
            speed *= 1 + .45 * density_strength
            speed *= .5 + .5 * taper
            stroke = dict(
                points=points.tolist(), press=press.tolist(),
                speed=np.clip(speed, 5, 100).tolist(),
                pass_=1, pal_index=ink_palette_index(), layer=layer,
                name=("旧版长轮廓", "旧版结构线", "旧版短皴")[layer],
                role=("legacy_long", "legacy_structure", "legacy_texture")[layer],
                design_origin="legacy-source-width-and-density",
                source_stroke_id=old_index, source_piece=piece_index,
                tail_mm=0.0, lift_mm=4.0,
            )
            stroke['source_width_mm'] = piece[:, 4].tolist()
            stroke['source_width_kind'] = 'source-distance-transform-design-target-not-calibration'
            strokes.append(refresh_stats(stroke, scale))

    summary = dict(
        profile="mountain", detail="legacy-source", strokes=len(strokes),
        source_paths=len(traced), split_white_gaps=split_gaps,
        omitted_under_2mm=omitted_short, source_support_mm=max_supported_distance_mm,
        omitted_short=omitted_short, min_length_mm=expr.min_length_mm,
        expression=asdict(expr),
        pressure_mapping="legacy relative source width and density; current bounded layer caps",
        pressure_caps_mm={str(i): round(c, 3) for i, c in enumerate(caps)},
        brush_width_calibrated_for_this_job=False,
        length_mm=sum(s["length_mm"] for s in strokes),
        layers={k: sum(s["layer"] == k for s in strokes) for k in range(3)},
        mean_width_mm=float(np.mean([s["width_mm"] for s in strokes])) if strokes else 0.,
        max_width_mm=max((press_to_width_mm(s["press_max"]) for s in strokes), default=0.),
        mean_press_mm=float(np.mean([s["press_mean"] for s in strokes])) if strokes else 0.,
        max_press_mm=max((s["press_max"] for s in strokes), default=0.),
    )
    return strokes, summary
