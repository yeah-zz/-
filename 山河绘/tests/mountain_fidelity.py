"""Offline mountain extraction review; run from the project root.

python tests/mountain_fidelity.py
Writes comparison overlays and JSON to artifacts/mountain/fidelity_review/.
The source mask is a proxy for line location, not an annotated ground truth.
"""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mountain
import vision
from config import ShuimoConfig
from geometry import CanvasMap


OUT = Path("artifacts/mountain/fidelity_review")
SOURCES = ("desired_mountain_reference", "mountain_baimiao_v3")


def review(source, detail, save=True):
    work, info = vision.prepare_image(
        vision.load_image(f"artifacts/reference/{source}.png"), work_size=1800)
    mapper = CanvasMap(info["prepared_size"], info["image_corners"])
    with patch("socket.socket", side_effect=AssertionError("Offline only")):
        strokes, stats = mountain.build_strokes(work, mapper.mm_per_px,
                                                info["pad"], detail=detail)
    source_ink = (vision.neutral_ink_mask(work, gray_limit=210) * 255)
    t, b, l, r = info["pad"]
    if t: source_ink[:t] = 0
    if b: source_ink[-b:] = 0
    if l: source_ink[:, :l] = 0
    if r: source_ink[:, -r:] = 0
    source_center = vision._thin(source_ink)
    path_mask = np.zeros_like(source_ink)
    for s in strokes:
        p = np.rint(s["points"]).astype(np.int32)
        cv2.polylines(path_mask, [p], False, 255, 1, cv2.LINE_8)
    tolerance = max(2, round(0.55 / mapper.mm_per_px))
    radius = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * tolerance + 1, 2 * tolerance + 1))
    near_path = cv2.dilate(path_mask, radius) > 0
    near_source = cv2.dilate(source_ink, radius) > 0
    center = source_center > 0
    path = path_mask > 0
    missed = center & ~near_path
    unsupported = path & ~near_source
    distance_to_ink = cv2.distanceTransform(255 - source_ink, cv2.DIST_L2, 5)
    path_distance = distance_to_ink[path]
    result = {
        "source": source, "detail": detail, "tolerance_px": tolerance,
        "strokes": len(strokes), "length_mm": round(stats["length_mm"], 1),
        "source_center_px": int(center.sum()), "path_px": int(path.sum()),
        "coverage": round(float((center & near_path).sum() / max(center.sum(), 1)), 4),
        "source_support": round(float((path & near_source).sum() / max(path.sum(), 1)), 4),
        "exact_ink_support": round(float((path & (source_ink > 0)).sum() / max(path.sum(), 1)), 4),
        "path_ink_distance_p95_px": round(float(np.percentile(path_distance, 95)), 2) if path_distance.size else 0,
        "white_gap_px_over_3": int((path_distance > 3).sum()),
        "unsupported_px": int(unsupported.sum()),
        "shorter_than_8mm": sum(s["length_mm"] < 8 for s in strokes),
        "layers": stats["layers"],
        "bands": {band: {
            role: {
                "count": len(group),
                "peak_press_mm": round(max((s["press_max"] for s in group), default=0), 3),
                "mean_estimated_width_mm": round(float(np.mean([s["width_mm"] for s in group])), 3) if group else 0,
            }
            for role, group in (("outline", [s for s in strokes if s["depth_band"] == band and s["role"] == "silhouette"]),
                                ("interior", [s for s in strokes if s["depth_band"] == band and s["role"] != "silhouette"]))}
            for band in ("far", "middle", "near")},
    }
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        canvas = cv2.cvtColor(cv2.addWeighted(
            work, .55, np.full_like(work, 255), .45, 0), cv2.COLOR_BGR2RGB)
        canvas[missed] = (240, 50, 190)  # missed source center: magenta
        canvas[path] = (25, 150, 55)    # supported output: green
        canvas[path & (distance_to_ink > 1)] = (235, 45, 35)  # path over white: red
        cv2.imwrite(str(OUT / f"{source}_{detail}_overlay.png"),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        layered = np.full_like(work, 255)
        # BGR: silhouette is darker than interior; distant hills are paler.
        color = {
            "near": ((45, 45, 50), (130, 125, 115)),
            "middle": ((95, 95, 100), (170, 160, 145)),
            "far": ((160, 160, 165), (205, 195, 175)),
        }
        for s in strokes:
            contour = s["role"] == "silhouette"
            p = np.rint(s["points"]).astype(np.int32)
            width = max(1, round(s["width_mm"] / mapper.mm_per_px))
            cv2.polylines(layered, [p], False,
                          color[s["depth_band"]][0 if contour else 1],
                          width, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"{source}_{detail}_layers.png"), layered)
        header = np.full((52, work.shape[1] * 2, 3), 245, np.uint8)
        cv2.putText(header, "SOURCE", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, .8, (40, 40, 40), 2)
        cv2.putText(header, "EXTRACTED: darker/thicker outer contours; paler distant hills",
                    (work.shape[1] + 16, 34), cv2.FONT_HERSHEY_SIMPLEX, .7, (40, 40, 40), 2)
        cv2.imwrite(str(OUT / f"{source}_{detail}_comparison.png"),
                    np.vstack((header, np.hstack((work, layered)))))
    return result


if __name__ == "__main__":
    ShuimoConfig.PRESS_MIN_MM = .15
    results = [review(s, d) for s in SOURCES
               for d in ("standard", "enriched", "structured")]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(result)
