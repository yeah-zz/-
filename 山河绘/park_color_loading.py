"""Park the brush high over the existing, calibrated outside-paper ink station.

The station is used only as a high parking point for manual color loading.
This program never descends to ink, clears alarms, or enables the robot.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from config import ArmConfig, CanvasConfig, PALETTE, station_approach_z
from geometry import check_waypoints
from robot import E6Robot, RobotError
from run_mountain_v4_robot import connect_readonly, no_errors
from run_mountain_balanced import check_orientation
from prepare_robot import close_to, move_to


def target_pose():
    station = PALETTE[0]
    if station.get("name") != "墨黑" or "dip" not in station:
        raise ValueError("Expected calibrated ink station")
    x, y = map(float, station["dip"])
    if cv2.pointPolygonTest(np.asarray(CanvasConfig.ROBOT_CORNERS, np.float32),
                            (x, y), True) > -20:
        raise ValueError("Loading station is not at least 20 mm outside the paper")
    return (x, y, station_approach_z(), *ArmConfig.RPY)


def run(execute=False, confirm_clearance=False):
    if execute and not confirm_clearance:
        raise ValueError("--execute requires --confirm-clearance")
    target = target_pose()
    robot = E6Robot()
    record = dict(action="park-outside-paper" if execute else "check", success=False,
                  target=target)
    try:
        connect_readonly(robot)
        mode, alarms, pose = robot.get_mode(), robot.get_error_id(), robot.get_pose()
        record.update(initial_mode=mode, initial_alarms=alarms, initial_pose=pose)
        if mode != 5 or not no_errors(alarms):
            raise RobotError("Requires idle enabled robot without alarms")
        check_orientation(pose)
        if abs(pose[2] - station_approach_z()) > .25:
            raise RobotError("Brush must already be at the 75 mm high transfer level")
        ok, problems = check_waypoints([dict(x=p[0], y=p[1], z=p[2])
                                        for p in (pose, target)])
        if not ok:
            raise RobotError("Outside-paper transfer exceeds configured reach: " + str(problems))
        print("Current:", pose)
        print("Outside-paper high parking target:", target)
        if not execute:
            record["success"] = True
            print("READ-ONLY: no motion sent")
            return 0
        if not close_to(robot.get_pose(), pose) or robot.get_mode() != 5 or not no_errors(robot.get_error_id()):
            raise RobotError("Robot changed since planning; no motion sent")
        response = robot.send("SpeedFactor(10)")
        if not robot._ok(response):
            raise RobotError("Cannot set parking speed: " + response)
        print("MOVING: high transfer to outside-paper station", flush=True)
        move_to(robot, target)
        final = robot.get_pose()
        if not close_to(final, target) or robot.get_mode() != 5 or not no_errors(robot.get_error_id()):
            raise RobotError("High parking arrival not confirmed")
        record.update(success=True, final_pose=final)
        print("PARKED: high outside paper; no station descent or dip sent")
        return 0
    except Exception as exc:
        record["error"] = str(exc)
        print("STOPPED:", exc, flush=True)
        return 4
    finally:
        robot.close(disable=False)
        folder = Path("robot_runs")
        folder.mkdir(exist_ok=True)
        path = folder / ("color_park_%s_%s.json" %
                         (datetime.now().strftime("%Y%m%d_%H%M%S_%f"), uuid4().hex[:8]))
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Parking record:", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-clearance", action="store_true")
    args = parser.parse_args()
    return run(args.execute, args.confirm_clearance)


if __name__ == "__main__":
    raise SystemExit(main())
