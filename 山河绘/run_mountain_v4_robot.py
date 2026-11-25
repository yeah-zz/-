"""Execute the reviewed v4 JSON with explicit high-level station transfers.

Offline by default. --execute requires already confirmed paper/site readiness.
Never clears errors or re-enables the robot automatically.
"""
import argparse
import json
import math
import socket
import time
from pathlib import Path

from config import ArmConfig, CanvasConfig, ShuimoConfig, station_approach_z, transit_z, validate
from geometry import check_waypoints
from job import load_job, save_job, _wp
from robot import E6Robot, RobotError


def safe_transfers(waypoints):
    out = []
    high = station_approach_z()
    for wp in waypoints:
        if out and wp["tag"] == "dip-in":
            prev = out[-1]
            out.append(_wp(prev["x"], prev["y"], high, True, 40, tag="station-raise"))
        if out and out[-1]["tag"] == "dip-out" and wp["tag"].startswith("hover:"):
            out.append(_wp(wp["x"], wp["y"], high, True, 40, tag="station-return-high"))
        out.append(dict(wp))
    return out


def no_errors(reply):
    if not reply.startswith("0,"): return False
    data = json.loads(reply.split("{", 1)[1].split("}", 1)[0] or "[]")
    def clear(value):
        return all(clear(v) for v in value) if isinstance(value, list) else value == 0
    return clear(data)


def connect_readonly(robot):
    robot.sock = socket.create_connection((robot.ip, robot.port), timeout=5)
    robot.sock.settimeout(.6)
    try:
        banner = robot.sock.recv(256)
        if banner: raise RobotError("Unexpected connection banner: %r" % banner)
    except socket.timeout:
        pass
    robot.sock.settimeout(float(ArmConfig.REPLY_TIMEOUT))
    robot.connected = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    original, meta = load_job("artifacts/mountain/mountain_v4_job.json")
    if validate(): raise RuntimeError(validate())
    if meta.get("profile") != "mountain" or meta["strokes"] != 70:
        raise RuntimeError("Expected the reviewed 70-stroke mountain job")
    if meta["paper_z"] != ArmConfig.PAPER_Z or meta["canvas_corners"] != [list(p) for p in CanvasConfig.ROBOT_CORNERS]:
        raise RuntimeError("Calibration differs from the reviewed job")
    if any(w["tag"].startswith("tail:") for w in original):
        raise RuntimeError("Unexpected extending tail")
    waypoints = safe_transfers(original)
    ok, problems = check_waypoints(waypoints)
    if not ok: raise RuntimeError(problems)
    execution_meta = dict(meta, station_transfer="vertical-high-horizontal-vertical",
                          rest_after_strokes=[20, 40, 60], rest_seconds=20)
    save_job("artifacts/mountain/mountain_v4_safe_execution_job.json", waypoints, execution_meta)
    print("Validated: %d strokes / %d points / %d dips" %
          (meta["strokes"], len(waypoints), meta["dips"]), flush=True)
    if not args.execute: return

    robot = E6Robot()
    completed = 0
    sent = 0
    started = time.time()
    result = {"completed_strokes": 0, "success": False}
    log = Path("artifacts/mountain/mountain_v4_execution_log.json")
    try:
        connect_readonly(robot)
        if robot.get_mode() != 5 or not no_errors(robot.get_error_id()):
            raise RobotError("Robot must already be enabled, idle and free of alarms")
        pose = robot.get_pose()
        if pose is None or any(abs(a-b) > .15 for a, b in zip(pose[3:], ArmConfig.RPY)):
            raise RobotError("Pose orientation does not match calibration")
        startup = [_wp(pose[0], pose[1], station_approach_z(), True, 40, tag="initial-raise")]
        if not check_waypoints(startup)[0]: raise RobotError("Initial raise outside bounds")
        for command in ("SpeedFactor(%d)" % ArmConfig.SPEED_FACTOR,
                        "User(%d)" % ArmConfig.USER_INDEX, "Tool(%d)" % ArmConfig.TOOL_INDEX):
            if not robot._ok(robot.send(command)): raise RobotError("Setup rejected: " + command)
        robot.move_safe(pose[0], pose[1], station_approach_z(), 40)
        pending = 0
        for index, wp in enumerate(waypoints):
            response = robot.move_linear(wp["x"], wp["y"], wp["z"], wp["velocity"],
                                         0 if wp["pen_up"] else wp.get("cp", 20))
            if not robot._ok(response):
                raise RobotError("Point %d rejected: %s; %s" % (index, response, robot.get_error_id()))
            sent += 1
            pending += 1
            # Flush at the end of each stroke, every station step, pauses and
            # bounded batches. Dip dwell starts only after arrival is confirmed.
            boundary = wp["pen_up"] or wp["pause"] > 0 or pending >= 180
            if boundary:
                if not robot.wait_idle(120): raise RobotError("Motion did not finish")
                pending = 0
            if wp["pause"]: time.sleep(wp["pause"])
            if wp["tag"].startswith("lift:"):
                completed += 1
                print("Completed %d/70 strokes | %d/%d points | %.1f min" %
                      (completed, sent, len(waypoints), (time.time()-started)/60), flush=True)
                result.update(completed_strokes=completed, points_sent=sent,
                              elapsed_seconds=round(time.time()-started, 1))
                log.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                if completed in (20, 40, 60):
                    robot.move_safe(wp["x"], wp["y"], transit_z(), 40)
                    print("High pause: 20 seconds", flush=True)
                    time.sleep(20)
                    robot.move_safe(wp["x"], wp["y"], wp["z"], 40)
            time.sleep(ArmConfig.MOVE_DELAY)
        if not robot.wait_idle(120): raise RobotError("Final motion did not finish")
        last = waypoints[-1]
        robot.move_safe(last["x"], last["y"], transit_z(), 40)
        result.update(robot_mode=robot.get_mode(), alarms=robot.get_error_id(), pose=robot.get_pose())
        result["success"] = result["robot_mode"] == 5 and no_errors(result["alarms"])
        if not result["success"]: raise RobotError("Final status not clear")
        print("COMPLETE: 70/70, mode=5, no alarms, pen lifted 60 mm", flush=True)
    except Exception as exc:
        result["error"] = str(exc)
        print("STOPPED: %s" % exc, flush=True)
        raise
    finally:
        result.update(completed_strokes=completed, points_sent=sent,
                      elapsed_seconds=round(time.time()-started, 1))
        log.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        robot.close(disable=False)


if __name__ == "__main__": main()
