"""Offline sparse blue accents spread over the full mountain drawing."""

# --- path bootstrap: this script lives in scripts/, put the project root on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
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


def safe_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def make_full_plan(job_path, ink_image_path, brush_width_mm=1.0,
                   ink_gap_mm=5.0, max_strokes=18):
    _, meta = job.load_job(str(job_path))
    if meta.get("profile") != "mountain" or meta.get("ink_policy") != "balanced":
        raise ValueError("Expected reviewed balanced mountain job")
    if meta.get("canvas_corners") != [list(p) for p in CanvasConfig.ROBOT_CORNERS]:
        raise ValueError("Canvas calibration changed")
    if not math.isfinite(brush_width_mm) or not 0 < brush_width_mm <= 1:
        raise ValueError("Full blue pass requires measured brush width <=1 mm")
    if not math.isfinite(ink_gap_mm) or ink_gap_mm < 5:
        raise ValueError("Full pass requires >=5 mm nominal ink gap")
    source = vision.load_image(str(ink_image_path))
    work, info = vision.prepare_image(source, work_size=1800)
    if list(info["prepared_size"]) != meta["image_size"]:
        raise ValueError("Ink source dimensions differ from saved job")
    scale = float(meta["ink_execution"]["mm_per_px"])
    mapper = CanvasMap(info["prepared_size"], info["image_corners"])
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    ink = (gray < 215).astype(np.uint8) * 255
    base = meta["ink_execution"]["base_strokes"]
    for stroke in base:
        points = np.rint(stroke["points"]).astype(np.int32)
        if len(points) > 1:
            cv2.polylines(ink, [points], False, 255, 3)
    clearance = cv2.distanceTransform(255-ink, cv2.DIST_L2, 5) * scale
    h, w = ink.shape
    polygon = np.asarray(CanvasConfig.ROBOT_CORNERS, np.float32)
    candidates = []
    for source_stroke in base:
        if source_stroke.get("role") not in ("silhouette", "face_ridge"):
            continue
        if source_stroke["length_mm"] < 14:
            continue
        path = np.asarray(source_stroke["points"], float)
        if len(path) < 4:
            continue
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))] * scale
        path = path[(arc >= .1*arc[-1]) & (arc <= .9*arc[-1])]
        if len(path) < 3:
            continue
        tangent = np.gradient(path, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal[normal[:, 1] < 0] *= -1  # Blue under the ridgeline in image space.
        for offset_mm in (8., 12., 16.):
            points = path + normal * (offset_mm/scale)
            ij = np.rint(points).astype(int)
            in_bounds = ((ij[:, 0] >= 0) & (ij[:, 0] < w) &
                         (ij[:, 1] >= 0) & (ij[:, 1] < h))
            safe = np.zeros(len(points), bool)
            safe[in_bounds] = (clearance[ij[in_bounds, 1], ij[in_bounds, 0]] >=
                               ink_gap_mm + brush_width_mm/2)
            for a, b in safe_runs(safe):
                segment = points[a:b]
                if len(segment) < 3:
                    continue
                seg_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(segment, axis=0), axis=1))] * scale
                if seg_arc[-1] < 7:
                    continue
                # Multiple independent short accents can arise from one ridge.
                for start_mm in np.arange(0, seg_arc[-1]-6.5, 12.):
                    take = (seg_arc >= start_mm) & (seg_arc <= start_mm+10.)
                    part = segment[take]
                    if len(part) < 3:
                        continue
                    length_mm = float(np.linalg.norm(np.diff(part, axis=0), axis=1).sum()*scale)
                    if not 6 <= length_mm <= 11:
                        continue
                    robot_xy = mapper.to_robot(part)
                    edge = min(cv2.pointPolygonTest(polygon, tuple(map(float, p)), True)
                               for p in robot_xy)
                    if edge < 25:
                        continue
                    pix = np.rint(part).astype(int)
                    min_ink = float(np.min(clearance[pix[:, 1], pix[:, 0]]))
                    center = robot_xy[len(robot_xy)//2]
                    candidates.append(dict(color="blue", target_tone="light",
                                           points=part.tolist(), length_mm=round(length_mm, 3),
                                           brush_width_mm_estimate=brush_width_mm,
                                           min_ink_clearance_mm=round(min_ink, 3),
                                           canvas_edge_clearance_mm=round(edge, 3),
                                           center_robot_xy_mm=center.tolist(),
                                           source_group=source_stroke.get("mountain_group"),
                                           source_depth=source_stroke.get("depth_band"),
                                           source_role=source_stroke.get("role")))
    # Select one accent per broad visual cell, evenly distributed across the sheet.
    cells = {}
    for c in candidates:
        center_px = np.asarray(c["points"])[len(c["points"])//2]
        key = (min(5, int(center_px[0] * 6 / w)),
               min(3, int(center_px[1] * 4 / h)))
        score = c["min_ink_clearance_mm"] + .1*c["canvas_edge_clearance_mm"]
        if key not in cells or score > cells[key][0]:
            cells[key] = (score, c)
    selected = []
    for _, c in sorted(cells.values(), key=lambda row: (-row[0], row[1]["source_group"])):
        center = np.asarray(c["center_robot_xy_mm"])
        if any(np.linalg.norm(center-np.asarray(old["center_robot_xy_mm"])) < 25
               for old in selected):
            continue
        selected.append(c)
        if len(selected) >= max_strokes:
            break
    selected.sort(key=lambda c: (c["source_depth"] != "far",
                                 c["source_depth"] == "near",
                                 c["center_robot_xy_mm"][1],
                                 c["center_robot_xy_mm"][0]))
    if len(selected) < 6:
        raise ValueError("Could not find enough safe accents for a full blue pass")
    plan = dict(format="blue-full-design-v1", ready_for_execution=False,
                reason="Requires paper registration and supervised manual pigment reloads",
                ink_job=str(Path(job_path).resolve()), ink_job_sha256=color_pass._digest(job_path),
                ink_image=str(Path(ink_image_path).resolve()),
                ink_image_sha256=color_pass._digest(ink_image_path),
                image_size=meta["image_size"], canvas_corners=meta["canvas_corners"],
                paper_z_snapshot=meta["paper_z"], mm_per_px=scale,
                brush_width_mm_estimate=brush_width_mm, ink_gap_mm=ink_gap_mm,
                strokes=selected,
                sequence=color_pass.manual_color_sequence(selected))
    return plan, work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default="artifacts/mountain/mountain_v6_job.json")
    parser.add_argument("--ink-image", default="artifacts/reference/desired_mountain_reference.png")
    parser.add_argument("--job-out", default="artifacts/blue/mountain_v6_blue_full_plan.json")
    parser.add_argument("--preview-out", default="artifacts/blue/mountain_v6_blue_full_preview.png")
    args = parser.parse_args()
    if Path(args.job_out).exists() or Path(args.preview_out).exists():
        parser.error("Output files already exist")
    plan, work = make_full_plan(args.job, args.ink_image)
    color_pass.render_preview(work, plan["strokes"], plan["mm_per_px"], args.preview_out)
    color_pass.save_plan(args.job_out, plan)
    print("Offline full blue plan:", len(plan["strokes"]), "strokes; no robot connection")


if __name__ == "__main__":
    main()
