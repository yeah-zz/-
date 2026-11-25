"""Recover drawing orientation from a limited, operator-cleared start region.

Default: query and print the plan only. --execute --confirm-clearance opts in
to physical motion. This is NOT a collision planner or a general home command.
The tool origin can differ from the brush tip; clearance requires site checks.
"""
import argparse
from datetime import datetime
import json
import math
import time
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
from config import ArmConfig, CanvasConfig, station_approach_z
from geometry import check_waypoints
from robot import E6Robot, RobotError
from run_mountain_v4_robot import connect_readonly, no_errors
from run_mountain_balanced import load_reviewed, check_orientation

MIN_START_HEIGHT_MM = 40.
MIN_ALIGNED_LIFT_HEIGHT_MM = 4.
CANVAS_MARGIN_MM = 20.
MAX_ROTATION_PER_AXIS_DEG = 20.


def plan_recovery(pose):
    if pose is None or len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise RobotError("No valid pose; cannot plan recovery")
    p = tuple(float(v) for v in pose)
    delta = [abs(a-b) for a,b in zip(p[3:],ArmConfig.RPY)]
    needs_rotation = max(delta) > .15  # Same tolerance as drawing preflight.
    sign = 1 if ArmConfig.Z_INCREASES_DOWN else -1
    height = sign * (ArmConfig.PAPER_Z-p[2])
    # A calibrated, already raised drawing pose only needs a vertical lift.
    # Never permit orientation correction in the near-paper branch.
    required_height = MIN_START_HEIGHT_MM if needs_rotation else MIN_ALIGNED_LIFT_HEIGHT_MM
    if height < required_height:
        raise RobotError("Start is too low (reference height %.1f mm, requires >= %.1f mm for %s). "
                         "Manually move to clear space above the paper first." %
                         (height, required_height, "orientation recovery" if needs_rotation else "vertical lift"))
    polygon = np.asarray(CanvasConfig.ROBOT_CORNERS, np.float32)
    edge_distance = cv2.pointPolygonTest(polygon, (p[0],p[1]), True)
    if edge_distance < CANVAS_MARGIN_MM:
        raise RobotError("Start XY must be inside the paper region, >=20 mm from its edges. "
                         "No automatic recovery near the ink bowl or outside this region.")
    if max(delta) > MAX_ROTATION_PER_AXIS_DEG:
        raise RobotError("Orientation change exceeds 20 deg per axis; use supervised manual recovery")
    # Raise vertically with the CURRENT orientation. Never use move_linear()
    # for this stage: that method always inserts the calibrated target RPY.
    high_z = min(p[2], station_approach_z()) if sign == 1 else max(p[2], station_approach_z())
    high = (p[0],p[1],high_z,*p[3:])
    aligned = (p[0],p[1],high_z,*ArmConfig.RPY)
    moves = []
    if abs(high_z-p[2]) > .01: moves.append(("raise-with-current-orientation", high))
    if needs_rotation: moves.append(("restore-drawing-orientation", aligned))
    bounds = [dict(x=v[0],y=v[1],z=v[2]) for v in [p, high, aligned]]
    ok, problems = check_waypoints(bounds)
    if not ok: raise RobotError("Recovery outside configured bounds: " + str(problems))
    return moves


def read_idle_pose(robot, record=None):
    mode, alarms, pose = robot.get_mode(), robot.get_error_id(), robot.get_pose()
    if record is not None:
        record.update(robot_mode=mode, alarms=alarms, observed_pose=pose)
    print("RobotMode:", mode, "| Alarms:", alarms)
    print("User %d / Tool %d | Current XYZ/RxRyRz: %s" %
          (ArmConfig.USER_INDEX, ArmConfig.TOOL_INDEX, pose))
    replies = (str(alarms),)
    if any("control mode is not tcp" in value.lower() or "not tcp" in value.lower()
           for value in replies):
        raise RobotError("当前是软件在线控制模式。请在 DobotStudio 切回 TCP 模式，断开其 29999 连接后重试；未发送运动。")
    try:
        clear = no_errors(alarms)
    except (ValueError, TypeError, IndexError, AttributeError):
        clear = False
    if not clear:
        raise RobotError("报警存在或无法可靠读取。请在控制器查看报警原因并人工排除；不自动清报警或使能。")
    if mode != 5:
        if mode == -1:
            raise RobotError('RobotMode=-1：未能可靠读取控制器模式。请检查TCP/IP控制模式、网络连接及控制端占用；未自动复位或使能。')
        raise RobotError("RobotMode=%s，需已使能且空闲(5)。请在现场确认停机/急停/碰撞/暂停或占用原因；不自动复位。" % mode)
    if pose is None or len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise RobotError("Cannot read valid pose")
    return tuple(pose)


def close_to(actual, target):
    return (actual is not None and len(actual) == 6 and all(math.isfinite(v) for v in actual)
            and all(abs(a-b) <= (.25 if i < 3 else .15)
                    for i,(a,b) in enumerate(zip(actual,target))))


def wait_arrival(robot, target, timeout=60.):
    from motion_diagnostics import attached
    journal = attached(robot)
    if journal:
        journal.record('arrival_wait', target=list(target), timeout_s=timeout)
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        mode = robot.get_mode()
        # Keep the alarm reply even when the mode itself rejects continuation.
        alarms = robot.get_error_id()
        if mode not in (5,7,8):
            error = RobotError("Recovery interrupted; mode=%s; alarms=%s. No further moves sent." % (mode, alarms))
            if journal: journal.fault(robot, error, mode, alarms)
            raise error
        try:
            clear = no_errors(alarms)
        except (ValueError, TypeError, IndexError, AttributeError):
            clear = False
        if not clear:
            error = RobotError("Recovery alarm or unreadable alarm reply: " + str(alarms))
            if journal: journal.fault(robot, error, mode, alarms)
            raise error
        if mode == 5:
            pose = robot.get_pose()
            if close_to(pose, target):
                if journal: journal.arrival(target, pose)
                return
        time.sleep(journal.poll_interval if journal else .15)
    error = RobotError("Arrival not confirmed within %g s; no further moves sent" % timeout)
    if journal: journal.fault(robot, error)
    raise error


def move_to(robot, target):
    command = ("MovL(pose={%.4f,%.4f,%.4f,%.4f,%.4f,%.4f},user=%d,tool=%d,v=10,cp=0)" %
               (*target, ArmConfig.USER_INDEX, ArmConfig.TOOL_INDEX))
    response = robot.send(command)
    if not robot._ok(response): raise RobotError("Recovery move rejected: " + response)
    wait_arrival(robot, target)


def prepare(path, execute=False, confirm_clearance=False, log_dir="robot_runs"):
    if execute and not confirm_clearance:
        raise ValueError("Execution requires --confirm-clearance: unchanged calibration/mounting, "
                         "brush suspended, lift and rotation area clear, operator present")
    # Match the existing paper, tool, RPY and ink settings to the reviewed job.
    robot = E6Robot()
    record = dict(job=str(path), action="prepare-motion" if execute else "read-only-diagnosis",
                  started_at=datetime.now().astimezone().isoformat(), success=False, steps=[])
    try:
        load_reviewed(path)
        connect_readonly(robot)
        initial = read_idle_pose(robot, record)
        record["initial_pose"] = initial
        moves = plan_recovery(initial)
        record["plan"] = [dict(name=name, target=target) for name, target in moves]
        for name, target in moves: print("PLAN:", name, target)
        if not execute:
            print("READ-ONLY: 未发送运动或设置指令。计划无法检测笔尖摆动和现场障碍。")
            record["success"] = True
            return 0
        # Re-query immediately before enabling any planned movement.
        if not close_to(read_idle_pose(robot), initial):
            raise RobotError("Robot moved since planning; restart recovery check")
        if moves:
            response = robot.send("SpeedFactor(10)")
            if not robot._ok(response): raise RobotError("Cannot set recovery speed: " + response)
        previous = initial
        for name, target in moves:
            if not close_to(read_idle_pose(robot), previous):
                raise RobotError("Unexpected pose change; no further moves sent")
            print("MOVING:", name, flush=True)
            move_to(robot, target)
            record["steps"].append(dict(name=name, target=target, arrived=True))
            previous = target
        final = read_idle_pose(robot)
        check_orientation(final)
        if not close_to(final, previous):
            raise RobotError("Final pose differs from planned target; no drawing started")
        record["final_pose"] = final
        record["success"] = True
        print("READY: drawing orientation restored; brush remains raised. No drawing started.")
        print("Final pose:", final)
        return 0
    except Exception as exc:
        record["error"] = str(exc)
        print("STOPPED:", exc, flush=True)
        return 4
    finally:
        robot.close(disable=False)
        try:
            folder = Path(log_dir)
            folder.mkdir(parents=True, exist_ok=True)
            record_path = folder / ("prepare_%s_%s.json" %
                                    (datetime.now().strftime("%Y%m%d_%H%M%S_%f"), uuid4().hex[:8]))
            record["finished_at"] = datetime.now().astimezone().isoformat()
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print("诊断/复位记录:", record_path)
        except OSError as exc:
            print("无法写入诊断/复位记录:", exc)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default="artifacts/mountain/mountain_v6_job.json")
    parser.add_argument("--execute", action="store_true", help="Physically raise and restore orientation")
    parser.add_argument("--confirm-clearance", action="store_true",
                        help="Confirm calibration/mounting unchanged, brush suspended, path clear and operator present")
    parser.add_argument("--log-dir", default="robot_runs", help="Directory for unique diagnostic/recovery records")
    args = parser.parse_args(argv)
    if args.execute and not args.confirm_clearance:
        parser.error("--execute requires --confirm-clearance; first run without --execute to review")
    return prepare(args.job, args.execute, args.confirm_clearance, args.log_dir)


if __name__ == "__main__": raise SystemExit(main())
