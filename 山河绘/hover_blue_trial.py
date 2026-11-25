"""Position the brush above the first blue accent for visual registration.

Default is read-only. --execute-hover only moves with the brush raised; this
module has no paint/contact action and never clears alarms or enables the arm.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from config import ArmConfig, CanvasConfig, z_for_lift
from geometry import CanvasMap, check_waypoints
from robot import E6Robot, RobotError
from run_mountain_v4_robot import connect_readonly, no_errors
from run_mountain_balanced import check_orientation
from prepare_robot import close_to, move_to
import color_pass


HIGH_LIFT_MM = 75.0
HOVER_LIFT_MM = 15.0
EDGE_MARGIN_MM = 20.0


def load_hover_target(plan_path):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan.get("format") != "blue-trial-design-v1" or len(plan.get("strokes", [])) != 1:
        raise ValueError("Requires exactly one reviewed blue trial stroke")
    if plan.get("ink_job_sha256") != color_pass._digest(plan["ink_job"]):
        raise ValueError("Saved ink job has changed")
    if plan.get("ink_image_sha256") != color_pass._digest(plan["ink_image"]):
        raise ValueError("Saved ink image has changed")
    if plan.get("paper_z_snapshot") != ArmConfig.PAPER_Z:
        raise ValueError("Paper Z has changed")
    if plan.get("canvas_corners") != [list(p) for p in CanvasConfig.ROBOT_CORNERS]:
        raise ValueError("Canvas calibration has changed")
    stroke = plan["strokes"][0]
    if stroke.get("color") != "blue" or stroke.get("brush_width_mm_estimate", 99) > 1:
        raise ValueError("Blue trial width requirement not met")
    if stroke.get("min_ink_clearance_mm", 0) < 2.5:
        raise ValueError("Insufficient planned ink clearance")
    points = np.asarray(stroke["points"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Invalid blue accent points")
    mapper = CanvasMap(plan["image_size"])
    target_xy = mapper.to_robot_point(points[0])
    distance = cv2.pointPolygonTest(np.asarray(CanvasConfig.ROBOT_CORNERS, np.float32),
                                    target_xy, True)
    if distance < EDGE_MARGIN_MM:
        raise ValueError("Hover target is too close to the canvas edge")
    return target_xy


def plan_hover(pose, target_xy):
    if pose is None or len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise RobotError("Current pose cannot be read")
    check_orientation(pose)
    if ArmConfig.PAPER_Z - pose[2] < 60:
        raise RobotError("Start must be at least 60 mm above the saved paper reference")
    high_z = z_for_lift(HIGH_LIFT_MM)
    if abs(pose[2] - high_z) > .25:
        raise RobotError("Start must be at the 75 mm prepared height")
    high = (float(target_xy[0]), float(target_xy[1]), high_z, *ArmConfig.RPY)
    hover = (float(target_xy[0]), float(target_xy[1]),
             z_for_lift(HOVER_LIFT_MM), *ArmConfig.RPY)
    bounds = [dict(x=p[0], y=p[1], z=p[2]) for p in (pose, high, hover)]
    ok, problems = check_waypoints(bounds)
    if not ok:
        raise RobotError("Hover move outside configured reach: " + str(problems))
    return (high, hover)


def run(plan_path, execute=False, confirm_clearance=False):
    if execute and not confirm_clearance:
        raise ValueError("--execute-hover requires --confirm-clearance")
    target_xy = load_hover_target(plan_path)
    robot = E6Robot()
    record = dict(plan=str(Path(plan_path).resolve()), action="hover" if execute else "check",
                  success=False, steps=[])
    try:
        connect_readonly(robot)
        mode, alarms, pose = robot.get_mode(), robot.get_error_id(), robot.get_pose()
        record.update(initial_mode=mode, initial_alarms=alarms, initial_pose=pose)
        if mode != 5 or not no_errors(alarms):
            raise RobotError("Requires idle enabled robot without alarms")
        high, hover = plan_hover(pose, target_xy)
        print("BLUE TRIAL: pen-up hover only")
        print("Current:", pose)
        print("High transfer:", high)
        print("Hover 15 mm above saved paper Z:", hover)
        if not execute:
            record["success"] = True
            print("READ-ONLY: no motion sent")
            return 0
        if not close_to(robot.get_pose(), pose) or robot.get_mode() != 5 or not no_errors(robot.get_error_id()):
            raise RobotError("Robot changed since planning; no motion sent")
        response = robot.send("SpeedFactor(10)")
        if not robot._ok(response):
            raise RobotError("Cannot set hover speed: " + response)
        for name, target in (("transfer-high", high), ("hover-above-paper", hover)):
            print("MOVING:", name, flush=True)
            move_to(robot, target)
            record["steps"].append(dict(name=name, target=target, arrived=True))
        final = robot.get_pose()
        if not close_to(final, hover) or robot.get_mode() != 5 or not no_errors(robot.get_error_id()):
            raise RobotError("Hover arrival or idle state not confirmed")
        record.update(success=True, final_pose=final)
        print("HOVER READY: brush remains raised; no paint stroke sent")
        return 0
    except Exception as exc:
        record["error"] = str(exc)
        print("STOPPED:", exc, flush=True)
        return 4
    finally:
        robot.close(disable=False)
        out = Path("robot_runs")
        out.mkdir(exist_ok=True)
        path = out / ("blue_hover_%s_%s.json" %
                      (datetime.now().strftime("%Y%m%d_%H%M%S_%f"), uuid4().hex[:8]))
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Hover record:", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="artifacts/blue/mountain_v6_blue_trial_plan.json")
    parser.add_argument("--execute-hover", action="store_true")
    parser.add_argument("--confirm-clearance", action="store_true")
    args = parser.parse_args()
    return run(args.plan, args.execute_hover, args.confirm_clearance)


if __name__ == "__main__":
    raise SystemExit(main())
