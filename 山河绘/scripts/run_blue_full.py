"""Paint a sparse blue layer in two-stroke batches with manual pigment loading.

Default is read-only. Each executed batch must start with a loaded brush parked
high outside the paper; it returns there after both strokes. No automatic dip.
"""

# --- path bootstrap: this script lives in scripts/, put the project root on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

import color_pass
from config import ArmConfig, CanvasConfig, ShuimoConfig, z_for_lift, z_for_press
from geometry import CanvasMap, check_waypoints
from robot import E6Robot, RobotError
from run_mountain_v4_robot import connect_readonly, no_errors
from run_mountain_balanced import check_orientation
from prepare_robot import close_to, move_to
from park_color_loading import target_pose as loading_pose


PRESS_MM = 0.30
HOVER_MM = 15.0
BATCH_SIZE = 2


def load_targets(plan_path, batch):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("format") != "blue-full-design-v1":
        raise ValueError("Requires full blue design plan")
    if plan.get("ink_job_sha256") != color_pass._digest(plan["ink_job"]):
        raise ValueError("Saved ink job changed")
    if plan.get("ink_image_sha256") != color_pass._digest(plan["ink_image"]):
        raise ValueError("Saved ink source changed")
    if plan.get("canvas_corners") != [list(p) for p in CanvasConfig.ROBOT_CORNERS]:
        raise ValueError("Canvas calibration changed")
    if plan.get("paper_z_snapshot") != ArmConfig.PAPER_Z:
        raise ValueError("Paper Z changed")
    if not ShuimoConfig.PRESS_MIN_MM <= PRESS_MM <= ShuimoConfig.PRESS_MAX_MM:
        raise ValueError("Pressure outside calibrated fine-brush range")
    strokes = plan.get("strokes", [])
    if not strokes or batch < 0 or batch * BATCH_SIZE >= len(strokes):
        raise ValueError("Batch index outside plan")
    mapper = CanvasMap(plan["image_size"])
    polygon = np.asarray(CanvasConfig.ROBOT_CORNERS, np.float32)
    entries = []
    for i in range(batch*BATCH_SIZE, min((batch+1)*BATCH_SIZE, len(strokes))):
        stroke = strokes[i]
        if (stroke.get("color") != "blue" or stroke.get("brush_width_mm_estimate", 99) > 1
                or stroke.get("min_ink_clearance_mm", 0) < 6
                or stroke.get("canvas_edge_clearance_mm", 0) < 25):
            raise ValueError("Unsafe stroke metadata at index %d" % i)
        px = np.asarray(stroke["points"], float)
        if px.ndim != 2 or px.shape[1] != 2 or len(px) < 3 or not np.isfinite(px).all():
            raise ValueError("Invalid stroke points at index %d" % i)
        xy = mapper.to_robot(px)
        for p in xy:
            if cv2.pointPolygonTest(polygon, tuple(map(float, p)), True) < 25:
                raise ValueError("Stroke enters paper-edge margin")
        arc = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        if not 5 <= arc <= 12:
            raise ValueError("Blue accent length outside review limit")
        start = xy[0]
        end = xy[-1]
        high = (float(start[0]), float(start[1]), z_for_lift(75), *ArmConfig.RPY)
        hover = (float(start[0]), float(start[1]), z_for_lift(HOVER_MM), *ArmConfig.RPY)
        contact = [(float(p[0]), float(p[1]), z_for_press(PRESS_MM), *ArmConfig.RPY)
                   for p in xy]
        lift = (float(end[0]), float(end[1]), z_for_lift(HOVER_MM), *ArmConfig.RPY)
        raised = (float(end[0]), float(end[1]), z_for_lift(75), *ArmConfig.RPY)
        sequence = [("transfer-high", high), ("hover", hover), ("touch", contact[0])]
        sequence.extend(("draw-%d" % j, p) for j, p in enumerate(contact[1:], 1))
        sequence.extend((("lift", lift), ("raise", raised)))
        entries.append((i, sequence, arc))
    points = [loading_pose()] + [pose for _, seq, _ in entries for _, pose in seq]
    ok, problems = check_waypoints([dict(x=p[0], y=p[1], z=p[2]) for p in points])
    if not ok:
        raise ValueError("Blue batch exceeds configured reach: " + str(problems))
    return entries


def already_done(plan_path):
    digest = color_pass._digest(plan_path)
    finished = set()
    for path in Path("robot_runs").glob("blue_full_*.json"):
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if old.get("plan_sha256") == digest:
            finished.update(old.get("completed_strokes", []))
    return finished


def run(plan_path, batch, execute=False, confirm_loaded=False):
    if execute:
        raise ValueError("Legacy sparse runner disabled after safety review; use tested run_blue_wash.py with hover registration")
    if execute and not confirm_loaded:
        raise ValueError("--execute requires --confirm-loaded")
    entries = load_targets(plan_path, batch)
    indices = [i for i, _, _ in entries]
    if execute and set(indices) & already_done(plan_path):
        raise ValueError("This batch contains an already painted stroke")
    robot = E6Robot()
    record = dict(action="paint" if execute else "check", batch=batch,
                  plan=str(Path(plan_path).resolve()),
                  plan_sha256=color_pass._digest(plan_path),
                  stroke_indices=indices, completed_strokes=[], success=False)
    try:
        connect_readonly(robot)
        mode, alarms, pose = robot.get_mode(), robot.get_error_id(), robot.get_pose()
        record.update(initial_mode=mode, initial_alarms=alarms, initial_pose=pose)
        if mode != 5 or not no_errors(alarms):
            raise RobotError("Requires idle enabled robot without alarms")
        check_orientation(pose)
        if not close_to(pose, loading_pose()):
            raise RobotError("Brush must start at the reviewed high outside-paper loading point")
        print("BLUE FULL batch", batch, "strokes", indices)
        for i, seq, length in entries:
            print("Stroke %d: %.2f mm | high %s | contact points %d" %
                  (i, length, seq[0][1][:3], sum(n.startswith("draw") for n, _ in seq)+1))
        if not execute:
            record["success"] = True
            print("READ-ONLY: no motion sent")
            return 0
        if not close_to(robot.get_pose(), pose) or robot.get_mode() != 5 or not no_errors(robot.get_error_id()):
            raise RobotError("Robot changed since planning; no motion sent")
        response = robot.send("SpeedFactor(10)")
        if not robot._ok(response):
            raise RobotError("Cannot set blue-pass speed: " + response)
        for i, seq, _ in entries:
            for name, target in seq:
                print("MOVING: stroke %d %s" % (i, name), flush=True)
                move_to(robot, target)
            record["completed_strokes"].append(i)
        print("MOVING: return to outside-paper high loading point", flush=True)
        move_to(robot, loading_pose())
        final = robot.get_pose()
        if not close_to(final, loading_pose()) or robot.get_mode() != 5 or not no_errors(robot.get_error_id()):
            raise RobotError("Final outside-paper parking not confirmed")
        record.update(success=True, final_pose=final)
        print("BATCH COMPLETE: parked high outside paper; manual reload before next batch")
        return 0
    except Exception as exc:
        record["error"] = str(exc)
        print("STOPPED:", exc, flush=True)
        return 4
    finally:
        robot.close(disable=False)
        folder = Path("robot_runs")
        folder.mkdir(exist_ok=True)
        path = folder / ("blue_full_%s_%s.json" %
                         (datetime.now().strftime("%Y%m%d_%H%M%S_%f"), uuid4().hex[:8]))
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Blue pass record:", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="artifacts/blue/mountain_v6_blue_full_plan.json")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-loaded", action="store_true")
    args = parser.parse_args()
    return run(args.plan, args.batch, args.execute, args.confirm_loaded)


if __name__ == "__main__":
    raise SystemExit(main())
