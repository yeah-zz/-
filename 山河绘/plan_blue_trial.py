"""Pick a few pale-blue accent paths beside v6 ink ridges, without robot motion.

This is a provisional placement study for an existing sheet. It does not set Z,
assume real brush width, or prove that the paper remains registered.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

import color_pass
from config import CanvasConfig
from geometry import CanvasMap
import job
import vision


def _safe_runs(valid):
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def make_trial(job_path, ink_image_path, brush_width_mm=1.0, gap_mm=2.0, count=1,
               min_separation_mm=45.0):
    _, meta = job.load_job(str(job_path))
    if meta.get("profile") != "mountain" or meta.get("ink_policy") != "balanced":
        raise ValueError("Expected a balanced mountain ink job")
    if meta.get("canvas_corners") != [list(p) for p in CanvasConfig.ROBOT_CORNERS]:
        raise ValueError("Current canvas calibration differs from the saved drawing")
    if not math.isfinite(brush_width_mm) or not 0 < brush_width_mm <= 10:
        raise ValueError("Invalid estimated brush width")
    if not math.isfinite(gap_mm) or not 0 <= gap_mm <= 10:
        raise ValueError("Invalid ink gap")
    if not math.isfinite(min_separation_mm) or not 0 <= min_separation_mm <= 100:
        raise ValueError("Invalid candidate separation")
    source = vision.load_image(str(ink_image_path))
    work, info = vision.prepare_image(source, work_size=1800)
    if list(info["prepared_size"]) != meta["image_size"]:
        raise ValueError("Ink image does not match saved task geometry")
    scale = float(meta["ink_execution"]["mm_per_px"])
    mapper = CanvasMap(info["prepared_size"], info["image_corners"])
    ink = (cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) < 215).astype(np.uint8)*255
    base = meta["ink_execution"]["base_strokes"]
    for stroke in base:
        points = np.rint(stroke["points"]).astype(np.int32)
        if len(points) > 1:
            cv2.polylines(ink, [points], False, 255, 3)
    clearance = cv2.distanceTransform(255-ink, cv2.DIST_L2, 5)*scale
    h, w = ink.shape
    required_mm = brush_width_mm/2 + gap_mm
    candidates = []
    for source_stroke in base:
        if source_stroke.get("role") != "face_ridge" or source_stroke["length_mm"] < 10:
            continue
        path = np.asarray(source_stroke["points"], float)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]*scale
        middle = (arc >= .15*arc[-1]) & (arc <= .82*arc[-1])
        path = path[middle]
        if len(path) < 3:
            continue
        tangent = np.gradient(path, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        for side in (-1, 1):
          for offset_mm in (2.0, 3.0, 4.0, 5.0, 6.0):
            points = path + side*(offset_mm/scale)*normal
            ij = np.rint(points).astype(int)
            in_bounds = ((ij[:, 0] >= 0) & (ij[:, 0] < w) &
                         (ij[:, 1] >= 0) & (ij[:, 1] < h))
            safe = np.zeros(len(points), bool)
            safe[in_bounds] = clearance[ij[in_bounds, 1], ij[in_bounds, 0]] >= required_mm
            for a, b in _safe_runs(safe):
                segment = points[a:b]
                if len(segment) < 3:
                    continue
                length_mm = np.linalg.norm(np.diff(segment, axis=0), axis=1).sum()*scale
                if length_mm < 6:
                    continue
                # Short accents are easier to check and less likely to flood an ink ridge.
                if length_mm > 9:
                    subarc = np.r_[0., np.cumsum(np.linalg.norm(
                        np.diff(segment, axis=0), axis=1))]*scale
                    segment = segment[:max(3, int(np.searchsorted(subarc, 9, side="right")))]
                    length_mm = np.linalg.norm(np.diff(segment, axis=0), axis=1).sum()*scale
                center = mapper.to_robot([segment[len(segment)//2]])[0]
                edge_mm = cv2.pointPolygonTest(
                    np.asarray(CanvasConfig.ROBOT_CORNERS, np.float32),
                    (float(center[0]), float(center[1])), True)
                if edge_mm < 20:
                    continue
                candidates.append({"color": "blue", "target_tone": "light",
                                   "points": segment.tolist(),
                                   "brush_width_mm_estimate": brush_width_mm,
                                   "length_mm": round(float(length_mm), 3),
                                   "source_group": source_stroke["mountain_group"],
                                   "source_depth": source_stroke["depth_band"],
                                   "offset_from_source_mm": offset_mm,
                                   "min_ink_clearance_mm": round(float(np.min(
                                       clearance[ij[a:b, 1], ij[a:b, 0]])), 3),
                                   "canvas_edge_clearance_mm": round(float(edge_mm), 3),
                                   "center_robot_xy_mm": center.tolist()})
    candidates.sort(key=lambda s: (s["source_depth"] != "far",
                                   -s["canvas_edge_clearance_mm"],
                                   -s["length_mm"], s["offset_from_source_mm"],
                                   s["source_group"]))
    selected = []
    for candidate in candidates:
        if any(candidate["source_group"] == s["source_group"] for s in selected):
            continue
        center = np.asarray(candidate["center_robot_xy_mm"])
        if any(np.linalg.norm(center-np.asarray(s["center_robot_xy_mm"])) < min_separation_mm
               for s in selected):
            continue
        selected.append(candidate)
        if len(selected) >= count:
            break
    if not selected:
        raise ValueError("No separated blue trial strokes with the requested clearance")
    plan = {"format": "blue-trial-design-v1", "ready_for_execution": False,
            "reason": "Need brush width/pressure measurement and paper registration check",
            "ink_job": str(Path(job_path).resolve()),
            "ink_job_sha256": color_pass._digest(job_path),
            "ink_image": str(Path(ink_image_path).resolve()),
            "ink_image_sha256": color_pass._digest(ink_image_path),
            "image_size": meta["image_size"], "canvas_corners": meta["canvas_corners"],
            "paper_z_snapshot": meta["paper_z"], "mm_per_px": scale,
            "brush_width_mm_estimate": brush_width_mm, "ink_gap_mm": gap_mm,
            "strokes": selected,
            "sequence": color_pass.manual_color_sequence(selected)}
    return plan, work


def main():
    parser = argparse.ArgumentParser(description="Offline blue accent study; never connects to robot")
    parser.add_argument("--job", default="artifacts/mountain/mountain_v6_job.json")
    parser.add_argument("--ink-image", default="artifacts/reference/desired_mountain_reference.png")
    parser.add_argument("--brush-width", type=float, default=1.0)
    parser.add_argument("--gap", type=float, default=2.0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--min-separation", type=float, default=45.0)
    parser.add_argument("--job-out", default="artifacts/blue/mountain_v6_blue_trial_plan.json")
    parser.add_argument("--preview-out", default="artifacts/blue/mountain_v6_blue_trial_preview.png")
    args = parser.parse_args()
    output = Path(args.job_out)
    preview = Path(args.preview_out)
    if not 1 <= args.count <= 30 or output.exists() or preview.exists() or output.resolve() == preview.resolve():
        parser.error("Count must be 1..30 and output paths must be distinct and unused")
    plan, work = make_trial(args.job, args.ink_image, args.brush_width, args.gap,
                            args.count, args.min_separation)
    color_pass.render_preview(work, plan["strokes"], plan["mm_per_px"], preview)
    color_pass.save_plan(output, plan)
    print("Offline blue trial: %d strokes, no robot connection" % len(plan["strokes"]))
    print("Plan:", output.resolve())
    print("Preview:", preview.resolve())


if __name__ == "__main__":
    main()
