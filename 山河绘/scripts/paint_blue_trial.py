"""One short blue contact trial on the previously registered paper.

Default is read-only. Execution starts only from hover_blue_trial.py's verified
15 mm hover, paints the first segment (about 3.7 mm), then lifts vertically.
There is no automatic dipping, washing, alarm clearing, or enabling.
"""

# --- path bootstrap: this script lives in scripts/, put the project root on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from config import ArmConfig, CanvasConfig, ShuimoConfig, z_for_lift, z_for_press
from geometry import CanvasMap, check_waypoints
from robot import E6Robot, RobotError
from run_mountain_v4_robot import connect_readonly, no_errors
from run_mountain_balanced import check_orientation
from prepare_robot import close_to, move_to
from hover_blue_trial import HOVER_LIFT_MM, load_hover_target
import color_pass


TRIAL_PRESS_MM = 0.30  # Existing calibrated fine preset: nominal 0.9 mm line.
EDGE_MARGIN_MM = 20.0


def plan_trial(plan_path):
    start_xy = load_hover_target(plan_path)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    stroke = plan["strokes"][0]
    points_px = np.asarray(stroke["points"], float)
    if len(points_px) < 3:
        raise ValueError("Expected at least three candidate points")
    # The last point approaches the conservative 20 mm paper-edge margin.
    # The first segment is enough to verify this first real color mark.
    xy = CanvasMap(plan["image_size"]).to_robot(points_px[:2])
    polygon = np.asarray(CanvasConfig.ROBOT_CORNERS, np.float32)
    for value in xy:
        if cv2.pointPolygonTest(polygon, tuple(map(float, value)), True) < EDGE_MARGIN_MM:
            raise ValueError("Blue trial enters paper-edge margin")
    if not ShuimoConfig.PRESS_MIN_MM <= TRIAL_PRESS_MM <= ShuimoConfig.PRESS_MAX_MM:
        raise ValueError("Fine brush pressure is outside configured range")
    hover_start = (*map(float, start_xy), z_for_lift(HOVER_LIFT_MM), *ArmConfig.RPY)
    contact_start = (float(xy[0, 0]), float(xy[0, 1]),
                     z_for_press(TRIAL_PRESS_MM), *ArmConfig.RPY)
    contact_end = (float(xy[1, 0]), float(xy[1, 1]),
                   z_for_press(TRIAL_PRESS_MM), *ArmConfig.RPY)
    lift_end = (float(xy[1, 0]), float(xy[1, 1]),
                z_for_lift(HOVER_LIFT_MM), *ArmConfig.RPY)
    targets = [contact_start, contact_end, lift_end]
    ok, problems = check_waypoints([dict(x=p[0], y=p[1], z=p[2])
                                    for p in [hover_start] + targets])
    if not ok:
        raise ValueError("Blue trial outside configured reach: " + str(problems))
    length = float(np.linalg.norm(xy[1]-xy[0]))
    return hover_start, targets, length


def run(plan_path, execute=False, confirm_loaded=False):
    if execute and not confirm_loaded:
        raise ValueError("--execute requires --confirm-loaded: blue paint applied to brush")
    plan_digest = color_pass._digest(plan_path)
    if execute:
        for old_path in Path("robot_runs").glob("blue_paint_*.json"):
            try:
                old = json.loads(old_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (old.get("action") == "paint" and old.get("success") is True
                    and (old.get("plan_sha256") == plan_digest
                         or old.get("plan") == str(Path(plan_path).resolve()))):
                raise ValueError("This blue trial was already painted; refusing a duplicate stroke")
    hover_start, targets, length = plan_trial(plan_path)
    robot = E6Robot()
    record = dict(plan=str(Path(plan_path).resolve()), plan_sha256=plan_digest,
                  action="paint" if execute else "check",
                  nominal_press_mm=TRIAL_PRESS_MM, nominal_length_mm=length,
                  success=False, steps=[])
    try:
        connect_readonly(robot)
        mode, alarms, pose = robot.get_mode(), robot.get_error_id(), robot.get_pose()
        record.update(initial_mode=mode, initial_alarms=alarms, initial_pose=pose)
        if mode != 5 or not no_errors(alarms):
            raise RobotError("Requires idle enabled robot without alarms")
        check_orientation(pose)
        if not close_to(pose, hover_start):
            raise RobotError("Brush is not at the reviewed 15 mm blue-trial hover")
        print("BLUE TRIAL: %.2f mm, nominal press %.2f mm" % (length, TRIAL_PRESS_MM))
        for name, target in zip(("touch", "short-blue-stroke", "lift"), targets):
            print("PLAN:", name, target)
        if not execute:
            record["success"] = True
            print("READ-ONLY: no motion sent")
            return 0
        if not close_to(robot.get_pose(), pose) or robot.get_mode() != 5 or not no_errors(robot.get_error_id()):
            raise RobotError("Robot changed since planning; no motion sent")
        response = robot.send("SpeedFactor(10)")
        if not robot._ok(response):
            raise RobotError("Cannot set trial speed: " + response)
        for name, target in zip(("touch", "short-blue-stroke", "lift"), targets):
            print("MOVING:", name, flush=True)
            move_to(robot, target)
            record["steps"].append(dict(name=name, target=target, arrived=True))
        final = robot.get_pose()
        if not close_to(final, targets[-1]) or robot.get_mode() != 5 or not no_errors(robot.get_error_id()):
            raise RobotError("Final raised pose or idle state not confirmed")
        record.update(success=True, final_pose=final)
        print("TRIAL COMPLETE: brush lifted; inspect color and alignment before any more painting")
        return 0
    except Exception as exc:
        record["error"] = str(exc)
        print("STOPPED:", exc, flush=True)
        return 4
    finally:
        robot.close(disable=False)
        out = Path("robot_runs")
        out.mkdir(exist_ok=True)
        path = out / ("blue_paint_%s_%s.json" %
                      (datetime.now().strftime("%Y%m%d_%H%M%S_%f"), uuid4().hex[:8]))
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Paint record:", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="artifacts/blue/mountain_v6_blue_trial_plan.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-loaded", action="store_true")
    args = parser.parse_args()
    return run(args.plan, args.execute, args.confirm_loaded)


if __name__ == "__main__":
    raise SystemExit(main())
