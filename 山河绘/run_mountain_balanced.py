import numpy as np
"""Reviewed mountain execution with an actual-clock ink feedback loop.

Default CLI only validates. Call with --execute after site/paper confirmation.
Only the ink model is feedback-controlled: there is no optical ink sensor.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from datetime import datetime
from uuid import uuid4
from painting_session import PauseRequested, PAUSED

from config import ArmConfig, CanvasConfig, ShuimoConfig, PALETTE, station_approach_z, transit_z, validate
from geometry import CanvasMap, check_waypoints
from ink_control import InkPolicy, InkState, cost, duration, condition_fresh, dip_step
import job
from robot import E6Robot, RobotError
from run_mountain_v4_robot import connect_readonly, no_errors


def check_orientation(pose):
    """Describe a failed preflight without moving or relaxing its tolerance."""
    if pose is None or len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise RobotError("Cannot read a valid 6-value pose in User %d / Tool %d; no motion sent" %
                         (ArmConfig.USER_INDEX, ArmConfig.TOOL_INDEX))
    delta = [a-b for a, b in zip(pose[3:], ArmConfig.RPY)]
    if any(abs(d) > .15 for d in delta):
        rows = ["Orientation differs from calibration; no motion sent.",
                "User %d / Tool %d | tolerance 0.1500 deg per axis" %
                (ArmConfig.USER_INDEX, ArmConfig.TOOL_INDEX),
                "Axis    Current(deg)   Required(deg)   Difference(deg)"]
        for name, current, target, diff in zip(("Rx", "Ry", "Rz"), pose[3:], ArmConfig.RPY, delta):
            rows.append("%-4s %13.4f %15.4f %+17.4f" % (name, current, target, diff))
        rows.append("Restore the calibrated orientation with clear space around the pen before retrying.")
        raise RobotError("\n".join(rows))


def check_robot_readonly(path):
    """Validate the selected job and query state only; do not create run logs."""
    strokes, _, _, _ = load_reviewed(path)
    robot = E6Robot()
    try:
        connect_readonly(robot)
        mode, alarms, pose = robot.get_mode(), robot.get_error_id(), robot.get_pose()
        print("READ-ONLY | %d strokes | User %d / Tool %d" %
              (len(strokes), ArmConfig.USER_INDEX, ArmConfig.TOOL_INDEX))
        print("RobotMode:", mode, "| Alarms:", alarms)
        print("Current XYZ/RxRyRz:", pose)
        print("Required RxRyRz:", ArmConfig.RPY)
        if mode != 5 or not no_errors(alarms):
            raise RobotError("Requires idle enabled robot without alarms; no state changed")
        check_orientation(pose)
        print("PASS: job, idle state and orientation checks passed. No motion sent.")
        return 0
    except Exception as exc:
        print("CHECK FAILED:", exc, flush=True)
        return 4
    finally:
        robot.close(disable=False)


def reviewed_ink_palette(strokes):
    """Use the same single ink station as planning, including legacy strokes.

    Some reviewed hand-added branches omit pal_index. Their stage is already
    black ink; no color is inferred from geometry. Reject an explicit different
    or malformed station before any motion callback or robot connection.
    """
    from shuimo import ink_palette_index
    index = ink_palette_index()
    for stroke in strokes:
        value = stroke.get('pal_index', index)
        if type(value) is not int or value != index:
            raise ValueError('Reviewed black-ink stroke has invalid palette index: %r (%s)'
                             % (value, stroke.get('id', 'unnamed')))
    return index


def run_feedback(strokes, scale, policy, perform_dip, perform_stroke, perform_rest,
                 clock=time.monotonic, on_progress=None, before_stroke=None):
    """Injectable clock and motion callbacks allow real delay/drying regression tests."""
    palette_index = reviewed_ink_palette(strokes)
    state = InkState(policy)
    events = []
    dips = 0
    current = None
    for i, base in enumerate(strokes):
        if before_stroke and before_stroke(i):
            state = InkState(policy)
            current = None
        if policy.rest_every and i and i % policy.rest_every == 0:
            perform_rest(policy.rest_s)
        travel = 1. if current is None else (((base["points"][0][0]-current[0])**2 +
                  (base["points"][0][1]-current[1])**2)**.5 * scale / policy.paper_travel_mm_s + 1.)
        # Whole-stroke wall time includes travel/lift overhead; it is not drawing speed.
        # Freshness is a refill heuristic, not a reason to abort a reviewed trajectory.
        draw_estimate = duration(base, policy)
        reasons = state.reasons(clock(), cost(base, scale), draw_estimate + travel)
        if reasons:
            step = dip_step(palette_index, state.dip_at is None, reasons, policy)
            perform_dip(step)
            state.load(clock())
            dips += 1
        s = condition_fresh(base, state, scale)
        row = dict(stroke=i+1, dip_cycle=dips, refill_reason=reasons,
                   age_before_transfer_s=round(clock()-state.dip_at, 3),
                   used_before_eq_mm=round(state.used, 3), demand_eq_mm=round(cost(s, scale), 3))
        started = clock()
        measured = perform_stroke(s)
        row.update(actual_transfer_and_stroke_s=round(clock()-started, 3),
                   actual_end_age_s=round(clock()-state.dip_at, 3))
        if measured and measured.get("draw_seconds", 0) > 0:
            draw_seconds = measured["draw_seconds"]
            row["actual_draw_s"] = round(draw_seconds, 3)
        state.consume(s, scale, clock())
        events.append(row)
        current = s["points"][-1]
        if on_progress: on_progress(events, dips)
    return events, dips


def normalized(value):
    return json.loads(json.dumps(value))


def drawing_calibration(value):
    """Cup loading settings are runtime inputs, not drawing registration."""
    value = normalized(value)
    for entry in value['palette']:
        for key in ('dip', 'surface_z', 'dip_mm', 'dip_time_s'):
            entry.pop(key, None)
    return value


def load_reviewed(path):
    if isinstance(path, dict):
        import copy
        data = copy.deepcopy(path)
        planned, meta = data.get("waypoints", []), data.get("meta", {})
    else:
        planned, meta = job.load_job(path)
    if meta.get("ink_policy") != "balanced" or meta.get("profile") != "mountain":
        raise ValueError("Requires reviewed balanced mountain job")
    problems = validate(need_stations=True)
    if problems: raise ValueError(problems)
    snap = meta["ink_execution"]
    air_speed = meta.get('air_transfer_velocity_percent', 40)
    if type(air_speed) is not int or not 1 <= air_speed <= 80:
        raise ValueError('Ink air transfer velocity must be an integer in [1, 80]')
    expected = dict(rpy=list(ArmConfig.RPY), user=ArmConfig.USER_INDEX, tool=ArmConfig.TOOL_INDEX,
                    speed_factor=ArmConfig.SPEED_FACTOR, palette=PALETTE)
    if (drawing_calibration(snap["calibration"]) != drawing_calibration(expected) or meta["paper_z"] != ArmConfig.PAPER_Z or
            meta["canvas_corners"] != normalized(CanvasConfig.ROBOT_CORNERS)):
        raise ValueError("Calibration/settings changed; regenerate and review")
    mapper = CanvasMap(meta["image_size"])
    if abs(mapper.mm_per_px-snap["mm_per_px"]) > 1e-9:
        raise ValueError("Canvas scale changed")
    # Scheduled rest follows the current runtime setting, including older jobs.
    runtime_policy = dict(snap["policy"])
    runtime_policy["rest_every"] = InkPolicy().rest_every
    runtime_policy["rest_s"] = InkPolicy().rest_s
    policy = InkPolicy(**runtime_policy)
    if runtime_policy["rest_every"] != snap["policy"].get("rest_every"):
        print("固定周期休息使用当前设置：每 %d 笔（0 表示关闭）。" % policy.rest_every, flush=True)
    snap["policy"] = runtime_policy
    strokes = snap["base_strokes"]
    if not strokes or len(strokes) != meta["strokes"]:
        raise ValueError("Stroke manifest mismatch")
    reviewed_ink_palette(strokes)
    for s in strokes:
        if s.get("tail_mm") != 0 or len(s["points"]) != len(s["press"]):
            raise ValueError("Invalid stroke data")
        if cost(s, mapper.mm_per_px) + policy.reserve_eq_mm > policy.capacity_eq_mm + 1e-6:
            raise ValueError("Unsplit over-budget stroke")
        if duration(s, policy) + policy.return_allowance_s > policy.max_age_s:
            raise ValueError("Overlong time budget")
        if any(not 0 <= p <= ShuimoConfig.PRESS_MAX_MM for p in s["press"]):
            raise ValueError("Pressure outside calibrated limits")
        if not check_waypoints(job.stroke_waypoints(mapper, s))[0]:
            raise ValueError("Stroke outside reach")
    # Validate today's cup even if only its rim changed (rim is not in PALETTE).
    from pigment_station import PigmentStation, read_measurements
    measurements = read_measurements()
    if (measurements['user'] != ArmConfig.USER_INDEX or measurements['tool'] != ArmConfig.TOOL_INDEX
            or measurements['rpy'] != list(ArmConfig.RPY)):
        raise ValueError('Cup coordinate frame/orientation changed; recalibrate')
    PigmentStation(**measurements['black']).validate()
    if snap['calibration'] != normalized(expected):
        # Rebuild loading/transfer waypoints in memory, retaining the exact base
        # strokes and ink policy. Never overwrite the reviewed display artifact.
        from ink_control import build_plan
        from shuimo import ink_palette_index
        from run_mountain_v4_robot import safe_transfers
        steps, _ = build_plan(strokes, ink_palette_index(), mapper.mm_per_px, policy)
        planned, _ = job.steps_to_waypoints(steps, mapper)
        planned = safe_transfers(planned)
        meta['design_calibration'] = normalized(snap['calibration'])
        snap['calibration'] = normalized(expected)
        print('使用当前墨盘配置，已重新校验蘸墨轨迹；原画稿 JSON 保持不变。')
    if not check_waypoints(planned)[0]: raise ValueError("Plan outside reach")
    return strokes, mapper, policy, meta


class Motion:
    def __init__(self, robot, mapper, pose, air_transfer_velocity_percent=40):
        if type(air_transfer_velocity_percent) is not int or not 1 <= air_transfer_velocity_percent <= 80:
            raise ValueError('Ink air transfer velocity must be an integer in [1, 80]')
        self.air_transfer_velocity_percent = air_transfer_velocity_percent
        self.robot, self.mapper = robot, mapper
        self.current = dict(x=pose[0], y=pose[1], z=pose[2])
        self.waypoints = []
        self.at_station = False
        from pigment_station import PigmentStation, read_measurements
        values = read_measurements()
        self.ink_station = PigmentStation(**values['black']).validate()
        self.completed_dips = 0
        self.completed_drawing_length_mm = 0.0

    def send(self, waypoints, *, continuous=False):
        if not check_waypoints(waypoints)[0]: raise RobotError("Generated move outside reach")
        from queued_motion import send_batch
        from prepare_robot import wait_arrival
        batch = []
        def flush():
            if not batch: return
            motions=[]
            for i, wp in enumerate(batch):
                # Preserve every vertical/horizontal transition exactly.
                cp = wp.get('cp',20) if (not wp['pen_up'] and i+1<len(batch)
                     and not batch[i+1]['pen_up'] and str(wp.get('tag','')).startswith('draw:')) else 0
                motions.append(((wp['x'],wp['y'],wp['z'],*ArmConfig.RPY),
                                int(max(1,min(100,round(wp['velocity'])))),cp))
            # Preserve submitted intent even if interrupted before arrival.
            self.waypoints.extend(dict(wp) for wp in batch)
            options = {'continuous': True} if continuous else {}
            send_batch(self.robot,motions,lambda r,p: wait_arrival(r,p,timeout=120), metadata=batch, **options)
            self.current = batch[-1]
            batch.clear()
        for wp in waypoints:
            batch.append(wp)
            if wp['pause']:
                flush()
                time.sleep(wp['pause'])
        flush()

    def raise_at_current(self, z, pause=0, tag="station-raise"):
        self.send([job._wp(self.current["x"], self.current["y"], z, True,
                          self.air_transfer_velocity_percent, pause=pause, tag=tag)])

    def dip(self, step):
        offset = self.ink_station.level_offset(self.completed_dips, self.completed_drawing_length_mm)
        points = job.dip_waypoints(step["pal_index"], self.ink_station.dwell_s, self.ink_station.drip_s)
        # Only the high approach to the cup is accelerated. Cup descent and
        # withdrawal retain their existing speeds and vertical clearances.
        for point in points:
            if point['tag'] == 'dip-in':
                point['velocity'] = max(point['velocity'], self.air_transfer_velocity_percent)
        if self.ink_station.level_compensation_enabled:
            from config import station_dip_z
            for point in points:
                if point['tag'] == 'dip':
                    point['z'] = station_dip_z(point['z'], offset)
                    point['level_offset_mm'] = offset
            print('黑墨液面下降补偿 %.3f mm（累计行笔 %.1f mm，第 %d 次蘸墨）' % (offset,self.completed_drawing_length_mm,self.completed_dips+1),flush=True)
        if not check_waypoints(points)[0]: raise RobotError('Compensated dip outside reach')
        self.raise_at_current(station_approach_z())
        self.send(points)
        self.completed_dips += 1
        self.at_station = True

    def stroke(self, stroke):
        points = job.stroke_waypoints(self.mapper, stroke)
        if self.at_station:
            # First hover after a cup return is vertical descent, not XY travel.
            points[0]['velocity'] = job.TRANSIT_SPEED
            self.send([job._wp(points[0]["x"], points[0]["y"], station_approach_z(),
                               True, self.air_transfer_velocity_percent, tag="station-return-high")])
        started = time.monotonic()
        self.send(points, continuous=True)
        self.at_station = False
        xy = self.mapper.to_robot(stroke['points'])
        self.completed_drawing_length_mm += float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        return {"draw_seconds": time.monotonic()-started}

    def rest(self, seconds):
        self.raise_at_current(transit_z(), pause=seconds, tag="rest")


def format_progress(completed, total, dips=0):
    ratio = completed / total if total else 0.
    filled = min(24, max(0, int(ratio * 24)))
    return "[%s%s] 已完成 %d/%d 笔（%.1f%%）| 蘸墨 %d 次" % (
        "#" * filled, "-" * (24-filled), completed, total, ratio*100, dips)


def execute_reviewed_job(path, *, start_index=0, control=None, trial_acceleration_percent=None):
    # Explicit terminal trials only; normal Web/CLI calls inherit the controller.
    if trial_acceleration_percent is not None and (
            type(trial_acceleration_percent) is not int or not 1 <= trial_acceleration_percent <= 100):
        raise ValueError('Trial acceleration must be an integer from 1 to 100')
    strokes, mapper, policy, meta = load_reviewed(path)
    if type(start_index) is not int or not 0 <= start_index <= len(strokes):
        raise ValueError("Invalid stroke continuation index")
    total_strokes = len(strokes)
    strokes = strokes[start_index:]
    path = Path(path)
    # Every attempt has its own files. A failed retry must not replace a
    # successful drawing record or its actual trajectory.
    run_dir = path.with_name(path.stem + "_runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
    log_path = run_dir / (run_id + "_execution_log.json")
    actual_path = run_dir / (run_id + "_actual.json")
    print("Execution record:", log_path, flush=True)
    robot = E6Robot()
    from motion_diagnostics import MotionJournal
    diagnostic_path = run_dir / (run_id + '_diagnostic.jsonl')
    journal = MotionJournal(diagnostic_path)
    robot._diagnostic = journal
    result = dict(success=False, completed_strokes=start_index, total_strokes=total_strokes,
                  dips=0, ink_cycles=[], interrupted=False, active_action=None,
                  job_path=str(path.resolve()))
    result.update(diagnostic_path=str(diagnostic_path.resolve()),
                  started_at=datetime.now().astimezone().isoformat(),
                  actual_file_meaning='planned/submitted intent; not measured trajectory')
    started = time.monotonic()
    motion = None

    def persist():
        if getattr(getattr(control, 'session', None), 'save_progress', True) is False:
            return  # Current-run bookkeeping stays in RAM; final diagnostic remains.
        result['elapsed_seconds'] = round(time.monotonic()-started, 1)
        log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def perform(kind, callback, *args):
        result['active_action'] = dict(kind=kind,
            stroke_number=result['completed_strokes']+1 if kind == 'stroke' else None)
        journal.context = dict(action=kind, stroke_number=result['completed_strokes']+1,
                               stroke_id=args[0].get('id') if kind == 'stroke' else None,
                               dip_number=result['dips']+1 if kind == 'dip' else result['dips'])
        journal.record('action_start')
        persist()
        outcome = callback(*args)
        journal.record('action_complete')
        if kind == 'dip':
            result['dips'] += 1
        # Stroke completion is committed by progress() after feedback bookkeeping.
        if kind != 'stroke':
            result['active_action'] = None
            persist()
        return outcome

    persist()
    if control: control.progress(start_index)
    else: print(format_progress(start_index, total_strokes), flush=True)
    print("Ctrl+C 会中断程序，不是可靠的机械臂暂停；不能自动断点续画。", flush=True)
    try:
        connect_readonly(robot)
        if robot.get_mode() != 5 or not no_errors(robot.get_error_id()):
            raise RobotError("Requires idle enabled robot without alarms; no auto recovery")
        pose = robot.get_pose()
        result["initial_pose"] = pose
        result["required_rpy"] = list(ArmConfig.RPY)
        check_orientation(pose)
        from experiment_start import validate_start
        initial_high = validate_start(pose, require_park=False)
        for command in ("SpeedFactor(%d)" % ArmConfig.SPEED_FACTOR,
                        "User(%d)" % ArmConfig.USER_INDEX, "Tool(%d)" % ArmConfig.TOOL_INDEX):
            if not robot._ok(robot.send(command)) or not robot.wait_idle(120):
                raise RobotError("Setup failed")
        acceleration = 'inherited controller setting; not read back'
        if trial_acceleration_percent is not None:
            from motion_settings import apply_acceleration
            acceleration = dict(joint=trial_acceleration_percent, linear=trial_acceleration_percent)
            apply_acceleration(robot, acceleration)
        journal.record('motion_settings', global_speed_factor=ArmConfig.SPEED_FACTOR,
                       acceleration=acceleration)
        motion = Motion(robot, mapper, pose, meta.get('air_transfer_velocity_percent', 40))
        if abs(pose[2] - initial_high[2]) > .01:
            perform('initial-lift', lambda: motion.raise_at_current(initial_high[2], tag="initial-vertical-lift"))

        def progress(events, dips):
            result.update(completed_strokes=start_index+len(events), dips=dips,
                          ink_cycles=events, active_action=None)
            persist()
            if control: control.progress(start_index+len(events))
            else: print(format_progress(start_index+len(events), total_strokes, dips), flush=True)

        from park_color_loading import target_pose
        from prepare_robot import close_to, read_idle_pose
        from run_painting_experiment import calibration_snapshot
        initial_calibration = calibration_snapshot()

        def park():
            perform('pause-raise', lambda: motion.raise_at_current(station_approach_z()))
            target = target_pose()
            perform('pause-park', lambda: motion.send([
                job._wp(*target[:3], True, 40, tag='pause-park')]))
            motion.at_station = True

        def verify():
            if calibration_snapshot() != initial_calibration or not close_to(read_idle_pose(robot), target_pose()):
                raise RobotError('暂停期间姿态或标定改变，不能直接继续。')

        def boundary(_=None):
            return control.boundary(park, verify) if control else False

        run_feedback(strokes, mapper.mm_per_px, policy,
                     lambda step: perform('dip', motion.dip, step),
                     lambda stroke: perform('stroke', motion.stroke, stroke),
                     lambda seconds: perform('rest', motion.rest, seconds),
                     on_progress=progress, before_stroke=boundary)
        boundary()
        if control:
            park()
        else:
            perform('final-lift', lambda: motion.raise_at_current(transit_z(), tag="final-lift"))
        result.update(robot_mode=robot.get_mode(), alarms=robot.get_error_id(), pose=robot.get_pose())
        result["success"] = result["robot_mode"] == 5 and no_errors(result["alarms"])
        if not result["success"]: raise RobotError("Final status not clear")
        return 0
    except PauseRequested:
        result.update(paused=True, active_action=None)
        return PAUSED
    except KeyboardInterrupt:
        journal.record('execution_interrupted', stop_confirmed=False,
                       last_confirmed=journal.last_confirmed)
        result.update(interrupted=True, error="KeyboardInterrupt",
                      interruption_note="Controller stop not sent or confirmed; active action may be partial. No automatic resume.")
        print("\n已中断程序：" + format_progress(result['completed_strokes'], len(strokes), result['dips']), flush=True)
        print("机械臂停止状态未确认；如仍在运动，请用控制器停止/急停。", flush=True)
        print("记录已保留，当前动作可能未完成。重新运行会从第一笔开始，不要在原纸上直接重跑。", flush=True)
        print("如放弃旧画换新纸，可运行 .\\.venv\\Scripts\\python.exe abandon_old_task.py 处理旧上色记录和锁。", flush=True)
        return 130
    except Exception as exc:
        result["error"] = str(exc)
        try:
            journal.fault(robot, exc)
        except Exception as snapshot_error:
            journal.record('fault_snapshot_error', error=str(snapshot_error))
        print("STOPPED:", exc, flush=True)
        return 4
    finally:
        result['finished_at'] = datetime.now().astimezone().isoformat()
        result['last_confirmed_arrival'] = journal.last_confirmed
        result["elapsed_seconds"] = round(time.monotonic()-started, 1)
        try:
            log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if motion:
                job.save_job(str(actual_path), motion.waypoints, dict(meta, actual_run=result))
        finally:
            robot.close(disable=False)
            journal.close()


def main(argv=None):
    from launcher import setup_console
    setup_console()
    parser = argparse.ArgumentParser(description="读取黑线JSON，确认后绘制。")
    parser.add_argument("--job", required=True)
    parser.add_argument("--colors-job", help="黑线完成后使用的完整上色计划；默认找同名配套计划")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    action.add_argument("--check", action="store_true", help="只离线校验，不连接机械臂")
    action.add_argument("--check-robot", action="store_true", help="只读取机械臂状态，不运动")
    args = parser.parse_args(argv)
    try:
        if args.check_robot:
            return check_robot_readonly(args.job)
        strokes, mapper, policy, meta = load_reviewed(args.job)
        print("任务：%s；黑线%d笔。" % (args.job, len(strokes)))
        if args.check:
            print("离线校验通过，没有连接机械臂。")
            return 0
        if input("确认纸张、笔和墨盘已准备好，输入 RUN 开始；其他输入退出：").strip().upper() != "RUN":
            return 0
        result = execute_reviewed_job(args.job)
        if result != 0:
            return result
        task = Path(args.job)
        color_path = Path(args.colors_job) if args.colors_job else (
            task.with_name(task.stem[:-5] + '_计划.json')
            if task.stem.endswith('_黑线任务') else None)
        if color_path is not None and color_path.exists():
            from run_color_from_job import load_plan
            from color_selection import paint_selected_colors
            print('黑线完成，已抬笔。等待墨干和手动换色。')
            return paint_selected_colors(load_plan(color_path, task))
        if args.colors_job:
            raise ValueError('指定的上色计划不存在：' + str(color_path))
        print('黑线完成；未找到同名配套上色计划，本次结束。')
        return 0
    except (ValueError, OSError, EOFError, KeyboardInterrupt) as exc:
        print("已停止：", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
