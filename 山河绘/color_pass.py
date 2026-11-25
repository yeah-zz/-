"""Offline second-pass color planning for an existing balanced mountain ink job.

The output is a design plan, deliberately without robot waypoints. Pigment
stations, brush footprint and color strength have not been measured on paper.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from config import CanvasConfig
import imgio
import job
import vision


COLORS = {
    "red": (42, 53, 214),     # BGR preview swatches, not measured pigments
    "yellow": (52, 194, 225),
    "blue": (181, 115, 50),
}
TONE_LIMITS = (72.0, 86.0)  # design-image L*, not a pigment recipe


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _lab_hues():
    swatches = np.array([[[bgr for bgr in COLORS.values()]]], dtype=np.uint8).reshape(1, 3, 3)
    lab = cv2.cvtColor(swatches, cv2.COLOR_BGR2LAB)[0, :, 1:].astype(np.float32) - 128.0
    return lab / np.maximum(np.linalg.norm(lab, axis=1, keepdims=True), 1e-6)


def classify_colors(work_bgr, min_chroma=14.0):
    """Classify visibly colored pixels by hue; dark neutral ink stays unpainted."""
    lab = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[:, :, 1:] - 128.0
    chroma = np.linalg.norm(ab, axis=2)
    direction = ab / np.maximum(chroma[:, :, None], 1e-6)
    affinity = direction @ _lab_hues().T
    labels = np.argmax(affinity, axis=2).astype(np.int8)
    labels[(chroma < min_chroma) | (np.max(affinity, axis=2) < .72)] = -1
    return labels, lab[:, :, 0] * (100.0 / 255.0)


def ink_exclusion_mask(ink_work, base_strokes, gap_px):
    """Exclude source ink and planned ink paths with a requested white gap."""
    mask = (cv2.cvtColor(ink_work, cv2.COLOR_BGR2GRAY) < 215).astype(np.uint8) * 255
    for stroke in base_strokes:
        points = np.rint(np.asarray(stroke["points"])).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(mask, [points], False, 255, 3)
    dilation = max(1, int(gap_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2)
    return cv2.dilate(mask, kernel) > 0


def _runs(row):
    edges = np.diff(np.r_[False, row, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _tone(lightness):
    if lightness >= TONE_LIMITS[1]:
        return "light"
    if lightness >= TONE_LIMITS[0]:
        return "medium"
    return "strong"


def plan_strokes(labels, lightness, blocked, mm_per_px, brush_width_mm=3.0,
                 min_area_mm2=30.0, min_length_mm=8.0):
    """Fill safe color regions with separated horizontal brush-center paths."""
    if not math.isfinite(mm_per_px) or mm_per_px <= 0:
        raise ValueError("Invalid canvas scale")
    if not math.isfinite(brush_width_mm) or not 0 < brush_width_mm <= 20:
        raise ValueError("Estimated brush width must be in (0, 20] mm")
    radius_px = max(1, int(math.ceil(brush_width_mm / (2 * mm_per_px))))
    step_px = max(1, int(round(.68 * brush_width_mm / mm_per_px)))
    min_run_px = max(2, int(math.ceil(min_length_mm / mm_per_px)))
    strokes = []
    for color_index, name in enumerate(COLORS):
        mask = ((labels == color_index) & ~blocked).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] * mm_per_px**2 < min_area_mm2:
                continue
            x, y, width, height = stats[component, :4]
            region = (components[y:y+height, x:x+width] == component).astype(np.uint8)
            # Erosion keeps the whole estimated brush disk inside its color patch.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1,) * 2)
            centers = cv2.erode(region, kernel, borderType=cv2.BORDER_CONSTANT,
                                borderValue=0) > 0
            for local_y in range(step_px // 2, height, step_px):
                for start, end in _runs(centers[local_y]):
                    if end - start < min_run_px:
                        continue
                    yy = y + local_y
                    xx0, xx1 = x + start, x + end - 1
                    sample = lightness[yy, xx0:xx1+1]
                    tone = _tone(float(np.median(sample)))
                    strokes.append({"color": name, "target_tone": tone,
                                    "points": [[float(xx0), float(yy)],
                                               [float(xx1), float(yy)]],
                                    "brush_width_mm_estimate": brush_width_mm,
                                    "length_mm": round((xx1-xx0)*mm_per_px, 3)})
    return strokes


def render_preview(ink_work, strokes, mm_per_px, out_path):
    canvas = np.full_like(ink_work, 255)
    for stroke in strokes:
        color = np.asarray(COLORS[stroke["color"]], np.float32)
        alpha = {"light": .35, "medium": .62, "strong": .90}[stroke["target_tone"]]
        bgr = tuple(int(round(v)) for v in (255 * (1-alpha) + color * alpha))
        points = [tuple(map(int, p)) for p in stroke["points"]]
        thickness = max(1, int(round(stroke["brush_width_mm_estimate"] / mm_per_px)))
        for p0, p1 in zip(points, points[1:]):
            cv2.line(canvas, p0, p1, bgr, thickness, cv2.LINE_AA)
    ink = cv2.cvtColor(ink_work, cv2.COLOR_BGR2GRAY) < 215
    canvas[ink] = ink_work[ink]
    if not imgio.imwrite(str(out_path), canvas):
        raise OSError("Could not save preview: " + str(out_path))


def manual_color_sequence(strokes):
    """Planning order; each color change requires an observed clean-brush check."""
    sequence = []
    for color in ("yellow", "red", "blue"):
        ids = [i for i, stroke in enumerate(strokes) if stroke["color"] == color]
        if not ids:
            continue
        if sequence:
            sequence.append({"kind": "manual_wash_and_wipe"})
        sequence.extend(({"kind": "verify_brush_clean"},
                         {"kind": "load_pigment", "color": color,
                          "station_calibrated": False},
                         {"kind": "manual_test_swatch", "color": color},
                         {"kind": "paint_design_strokes", "color": color,
                          "stroke_ids": ids}))
    return sequence


def build_plan(ink_job_path, color_image_path, brush_width_mm=3.0, gap_mm=1.0,
               ink_image_override=None):
    _, meta = job.load_job(str(ink_job_path))
    if meta.get("ink_policy") != "balanced" or meta.get("profile") != "mountain":
        raise ValueError("Color pass requires a balanced mountain ink job")
    base = meta.get("ink_execution", {}).get("base_strokes")
    if not base:
        raise ValueError("Ink job has no base strokes")
    if meta.get("canvas_corners") != [list(p) for p in CanvasConfig.ROBOT_CORNERS]:
        raise ValueError("Canvas calibration differs from ink job")
    ink_image_path = ink_image_override or meta.get("image")
    if ink_image_path and not os.path.isfile(ink_image_path):
        nearby = Path(ink_job_path).with_name(Path(ink_image_path).name)
        if nearby.is_file() and not ink_image_override:
            ink_image_path = str(nearby)
    if not ink_image_path or not os.path.isfile(ink_image_path):
        raise ValueError("Ink source image is unavailable; restore the original ink image")
    ink_image = vision.load_image(ink_image_path)
    color_image = vision.load_image(str(color_image_path))
    if ink_image.shape[:2] != color_image.shape[:2]:
        raise ValueError("Color design must have exactly the same pixel dimensions as the ink source")
    ink_work, ink_info = vision.prepare_image(ink_image, work_size=1800)
    color_work, color_info = vision.prepare_image(color_image, work_size=1800)
    if (list(ink_info["prepared_size"]) != meta["image_size"] or
            ink_info["pad"] != color_info["pad"]):
        raise ValueError("Color design does not align with the ink job work image")
    mm_per_px = float(meta["ink_execution"]["mm_per_px"])
    if not math.isfinite(gap_mm) or not 0 <= gap_mm <= 10:
        raise ValueError("Ink gap must be in [0, 10] mm")
    labels, lightness = classify_colors(color_work)
    gap_px = math.ceil(gap_mm / mm_per_px)
    blocked = ink_exclusion_mask(ink_work, base, gap_px)
    strokes = plan_strokes(labels, lightness, blocked, mm_per_px, brush_width_mm)
    if not strokes:
        raise ValueError("No safe color strokes found; try a simpler design or smaller brush")
    plan = {"format": "color-pass-design-v1", "ready_for_execution": False,
            "reason": "Pigment stations, brush width and tone recipes require on-paper calibration",
            "ink_job": os.path.abspath(str(ink_job_path)),
            "ink_job_sha256": _digest(ink_job_path),
            "ink_image": os.path.abspath(str(ink_image_path)),
            "ink_image_sha256": _digest(ink_image_path),
            "color_image": os.path.abspath(str(color_image_path)),
            "color_image_sha256": _digest(color_image_path),
            "image_size": meta["image_size"], "canvas_corners": meta["canvas_corners"],
            "mm_per_px": mm_per_px, "brush_width_mm_estimate": brush_width_mm,
            "ink_gap_mm": gap_mm, "tone_kind": "design-image Lab lightness target only",
            "strokes": strokes, "sequence": manual_color_sequence(strokes)}
    return plan, ink_work


def save_plan(path, plan):
    output = Path(path)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing plan: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
