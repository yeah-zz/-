"""一条命令回到绘画起点位：纸外高位停车点。

绘画实验（run_plum_experiment.py / run_color_experiment.py）要求的起始位是
“纸外墨盘正上方的高位停车点”：

    (park_x, park_y, station_approach_z(), *ArmConfig.RPY)

本脚本只把两个**已经评审过的动作基元**按安全顺序串起来，不新增能力：

    A. 保持当前姿态，在原 XY 竖直抬到站点高位（不横移）
    B. 到位后原地整组下发 RX/RY/RZ（不分别拧三个角）
    C. 在该高度水平移到纸外停车点
    D. 若抬得比接近高度更高，再在停车点正上方竖直下降到接近高度

默认**只读**：只打印计划，不发送任何指令。真运动需要同时给
`--execute --confirm-clearance`，并在提示处输入 GO。
不使能、不清报警、不改 User/Tool、不下蘸、不开始绘画。

沿用（而不是放宽）现有门限：
  * prepare_robot：需要转正时要求“纸内且离四边 >= 20 mm、离纸面 >= 40 mm”，
    已对齐时只要求 >= 4 mm；每轴姿态差 <= 20°；从不下降。
  * 纸外站点分支照 restore_drawing_pose.py：已在接近高位、距停车点 <= 10 mm、
    每轴 <= 3°。
  * geometry.check_waypoints 复核计划里的每一个点是否在可达范围内。

未确认到位的运动会在 robot_runs/blue_wash_motion.lock 留下运动锁（与
run_blue_wash / run_color_experiment / restore_drawing_pose 共用同一把锁），
后续脚本不会越过它继续动。
"""
import argparse
import math
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from config import ArmConfig, CanvasConfig, station_approach_z
from geometry import check_waypoints
from park_color_loading import target_pose as park_target
from prepare_robot import (CANVAS_MARGIN_MM, MAX_ROTATION_PER_AXIS_DEG,
                           MIN_ALIGNED_LIFT_HEIGHT_MM, MIN_START_HEIGHT_MM,
                           close_to, read_idle_pose, wait_arrival)
from project_paths import ROBOT_LOG_DIR
from robot import E6Robot, RobotError
from run_blue_wash import persist
from run_mountain_balanced import check_orientation
from run_mountain_v4_robot import connect_readonly

ORIENT_TOL_DEG = .15        # 与绘画前检查同一容差
RESTORE_MAX_DEG = 3.        # restore_drawing_pose.py 的站点分支上限
RESTORE_PARK_TOL_MM = 10.

# ---------------------------------------------------------------- 速度
# 有效速度 ≈ 全局倍率 × 指令速度（Dobot 的 SpeedFactor 是全局缩放，MovL 的 v 是单条指令速度）。
# 原来的 10 / 10（照抄 prepare_robot）实在太慢：200 mm 的高位横移要一分多钟。
# 现在默认 20 / 20，约为原来的 4 倍；想再快就用参数临时加，例如：
#     uv run python go_to_start.py --execute --confirm-clearance --velocity 40 --speed-factor 30
# ⚠ 越快留给"现场发现障碍并急停"的时间越短。这段运动是纸面上方 60~75 mm 的高位移动，
#   前提始终是：路径已清空、有人看护、手放在急停上。
GLOBAL_SPEED_FACTOR = 20    # 运动前下发的 SpeedFactor（%）
MOVE_VELOCITY = 20          # 每条 MovL 的指令速度 v（%）

SITE_CHECK = (
    '现场确认（--confirm-clearance 的含义）：\n'
    '  1) 笔/纸/装夹/标定没有变，毛笔悬空；\n'
    '  2) 原 XY 的竖直上升路径、高位整组转正的摆动范围、高位横移路径上\n'
    '     没有杯沿、夹具、线缆或人员；\n'
    '  3) 有人现场看护，随时可急停。\n'
    '  这不是碰撞检测，只是软件门限。')


def height_above_paper(z):
    """工具坐标 Z 换算成“离纸面多少毫米”（正数=在纸面上方）。"""
    sign = 1 if ArmConfig.Z_INCREASES_DOWN else -1
    return sign * (ArmConfig.PAPER_Z - float(z))


def edge_distance(xy):
    """正=纸内离最近纸边距离，负=纸外距离。"""
    polygon = np.asarray(CanvasConfig.ROBOT_CORNERS, np.float32)
    return float(cv2.pointPolygonTest(polygon, (float(xy[0]), float(xy[1])), True))


def inward_hint(pose):
    """给出“只点动 X/Y 最小多少毫米”才能进入纸内 >=20 mm 包络的提示。"""
    best = None
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        for step in range(1, 61):
            if edge_distance((pose[0] + dx * step, pose[1] + dy * step)) >= CANVAS_MARGIN_MM:
                if best is None or step < best[0]:
                    best = (step, dx, dy)
                break
    if best is None:
        return None
    step, dx, dy = best
    return ('请在示教器里只点动 %s 轴 %s %.0f mm 以上（不要点姿态角），然后重跑本脚本'
            % ('X' if dx else 'Y', '加' if (dx + dy) > 0 else '减', step))


def plan(pose):
    """返回 [(步骤名, 目标位姿)]；任何一条不满足现有门限就直接拒绝。"""
    if pose is None or len(pose) != 6 or not np.isfinite(np.asarray(pose, dtype=float)).all():
        raise RobotError('无法读取有效的 6 值起始位姿')
    pose = tuple(float(v) for v in pose)
    worst = max(abs(a - b) for a, b in zip(pose[3:], ArmConfig.RPY))
    if worst > MAX_ROTATION_PER_AXIS_DEG:
        raise RobotError(
            '姿态偏差 %.2f° 超过 %.0f°/轴：自动恢复脚本都会拒绝。不要改 ArmConfig.RPY 迁就，'
            '按 ROBOT_STARTUP.md 用示教器人工处理或重新标定。' % (worst, MAX_ROTATION_PER_AXIS_DEG))
    needs_rotation = worst > ORIENT_TOL_DEG
    park = park_target()
    height = height_above_paper(pose[2])
    required = MIN_START_HEIGHT_MM if needs_rotation else MIN_ALIGNED_LIFT_HEIGHT_MM
    if height < required:
        raise RobotError(
            '起点太低（离纸面 %.1f mm，%s需要 >= %.1f mm）。先用示教器把笔竖直抬到有充分间隙的'
            '高位（只点 X/Y/Z 或升降，不要点姿态），再重跑本脚本。'
            % (height, '转正时' if needs_rotation else '抬笔时', required))
    if needs_rotation:
        edge = edge_distance(pose[:2])
        park_distance = math.hypot(pose[0] - park[0], pose[1] - park[1])
        inside = edge >= CANVAS_MARGIN_MM
        station = (-edge >= CANVAS_MARGIN_MM and worst <= RESTORE_MAX_DEG
                   and park_distance <= RESTORE_PARK_TOL_MM)
        if not (inside or station):
            hint = inward_hint(pose)
            raise RobotError(
                '需要整组转正 %.2f°，但当前 XY 不在允许自动转正的区域内：'
                '离纸边 %+.1f mm（纸内需 >= %.0f mm；纸外站点分支需 >= %.0f mm 且距停车点 <= %.0f mm、'
                '每轴 <= %.0f°；当前距停车点 %.1f mm）。'
                % (worst, edge, CANVAS_MARGIN_MM, CANVAS_MARGIN_MM, RESTORE_PARK_TOL_MM,
                   RESTORE_MAX_DEG, park_distance)
                + ('\n' + hint if hint else ''))
    approach = station_approach_z()
    high_z = (min(pose[2], approach) if ArmConfig.Z_INCREASES_DOWN
              else max(pose[2], approach))
    moves = []
    if abs(high_z - pose[2]) > .01:
        moves.append(('raise-with-current-orientation', (pose[0], pose[1], high_z, *pose[3:])))
    if needs_rotation:
        moves.append(('restore-drawing-orientation', (pose[0], pose[1], high_z, *ArmConfig.RPY)))
    last = moves[-1][1] if moves else pose
    if math.hypot(last[0] - park[0], last[1] - park[1]) > .25:
        moves.append(('transfer-to-outside-paper-park', (park[0], park[1], last[2], *ArmConfig.RPY)))
        last = moves[-1][1]
    if abs(last[2] - approach) > .01:
        moves.append(('descend-to-approach-level', (park[0], park[1], approach, *ArmConfig.RPY)))
        last = moves[-1][1]
    ok, problems = check_waypoints([dict(x=p[0], y=p[1], z=p[2])
                                    for p in (pose,) + tuple(t for _, t in moves)])
    if not ok:
        raise RobotError('计划超出配置的可达范围: ' + '; '.join(problems))
    check_orientation(last)      # 终点必须正好是标定姿态
    return moves


def move_to(robot, target, velocity=MOVE_VELOCITY):
    """MovL 到位并确认：指令速度可调（prepare_robot.move_to 固定 v=10，这里放开）。"""
    command = ("MovL(pose={%.4f,%.4f,%.4f,%.4f,%.4f,%.4f},user=%d,tool=%d,v=%d,cp=0)"
               % (*target, ArmConfig.USER_INDEX, ArmConfig.TOOL_INDEX, int(velocity)))
    response = robot.send(command)
    if not robot._ok(response):
        raise RobotError('运动被拒: ' + response)
    wait_arrival(robot, target)


def go(execute=False, confirm_clearance=False, log_dir=ROBOT_LOG_DIR, ask=input,
       speed_factor=GLOBAL_SPEED_FACTOR, velocity=MOVE_VELOCITY):
    if execute and not confirm_clearance:
        raise ValueError('--execute 需要 --confirm-clearance：标定/装夹未变、笔悬空、路径清空、有人看护')
    folder = Path(log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ('go_to_start_%s_%s.json'
                     % (datetime.now().strftime('%Y%m%d_%H%M%S_%f'), uuid4().hex[:8]))
    record = dict(action='go-to-start-motion' if execute else 'check', success=False,
                  steps=[], target=None, global_speed_factor=int(speed_factor),
                  movl_velocity_percent=int(velocity),
                  started_at=datetime.now().astimezone().isoformat())
    robot = E6Robot()
    lock = None
    lock_path = folder / 'blue_wash_motion.lock'
    try:
        if execute:
            # 与其它运动脚本共用同一把锁：同一时刻只允许一个运动脚本。
            lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock, str(path.resolve()).encode('utf8'))
            os.fsync(lock)
        connect_readonly(robot)
        initial = read_idle_pose(robot, record)
        record['initial_pose'] = list(initial)
        park = park_target()
        record['target'] = list(park)
        moves = plan(initial)
        record['plan'] = [dict(name=name, target=list(target)) for name, target in moves]
        print('目标起始位（纸外高位停车点）:', tuple(round(v, 4) for v in park))
        print('速度：SpeedFactor(%d)%% + 指令 v=%d%% → 有效速度约为两者相乘（改为 --speed-factor N --velocity N）'
              % (speed_factor, velocity))
        for name, target in moves:
            print('PLAN: %-32s %s' % (name, tuple(round(v, 4) for v in target)))
        if not moves:
            print('已经就在纸外高位停车点：无需运动，未发送任何指令。')
            record['success'] = True
            return 0
        if not execute:
            print('READ-ONLY: 未发送运动或设置指令。加 --execute --confirm-clearance 才真的移动。')
            record['success'] = True
            return 0
        print(SITE_CHECK)
        if ask('输入 GO 执行以上 %d 步；其他输入取消: ' % len(moves)).strip() != 'GO':
            print('已取消，未发送运动。')
            return 0
        if not close_to(read_idle_pose(robot), initial):
            raise RobotError('确认期间位姿已改变，未发送运动；请重新检查')
        if not robot._ok(robot.send('SpeedFactor(%d)' % int(speed_factor))):
            raise RobotError('速度设置失败，未发送运动')
        previous = initial
        for name, target in moves:
            if not close_to(read_idle_pose(robot), previous):
                raise RobotError('执行 %s 前实测位姿与计划不符，未继续发送运动' % name)
            record['pending_move'] = dict(name=name, target=list(target))
            persist(path, record)
            print('MOVING: %s' % name, flush=True)
            move_to(robot, target, velocity)
            previous = target
            record['steps'].append(dict(name=name, target=list(target), arrived=True))
            record.pop('pending_move', None)
            persist(path, record)
        final = read_idle_pose(robot)
        if not close_to(final, previous):
            raise RobotError('最终位姿与计划目标不符，未开始绘画')
        check_orientation(final)
        record.update(success=True, final_pose=list(final))
        print('READY: 已到纸外高位停车点 = 绘画实验要求的起始位。未蘸墨、未下降、未开始绘画。')
        print('现在可以直接重新运行 run_plum_experiment.py，并在提示处输入 RESUME。')
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        record['error'] = str(exc)
        print('STOPPED:', exc, flush=True)
        return 4
    finally:
        record['finished_at'] = datetime.now().astimezone().isoformat()
        try:
            persist(path, record)
            if lock is not None:
                os.close(lock)
                if record.get('pending_move'):
                    print('运动未确认，保留运动锁 %s；请核查现场和本次日志，勿直接删除。' % lock_path)
                else:
                    lock_path.unlink()
        finally:
            robot.close(disable=False)
        print('记录:', path, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--execute', action='store_true', help='真的发送运动')
    parser.add_argument('--confirm-clearance', action='store_true',
                        help='确认标定/装夹未变、笔悬空、路径清空、有人现场看护')
    parser.add_argument('--speed-factor', type=int, default=GLOBAL_SPEED_FACTOR,
                        help='运动前下发的全局速度倍率 %%（默认 %(default)s；越大越快）')
    parser.add_argument('--velocity', type=int, default=MOVE_VELOCITY,
                        help='每条 MovL 的指令速度 v %%（默认 %(default)s；越大越快）')
    parser.add_argument('--log-dir', default=str(ROBOT_LOG_DIR))
    args = parser.parse_args(argv)
    if args.execute and not args.confirm_clearance:
        parser.error('--execute 需要同时给 --confirm-clearance')
    for name, value in (('--speed-factor', args.speed_factor), ('--velocity', args.velocity)):
        if not 1 <= value <= 100:
            parser.error('%s 必须在 1~100 之间（当前 %s）' % (name, value))
    return go(args.execute, args.confirm_clearance, args.log_dir,
              speed_factor=args.speed_factor, velocity=args.velocity)


if __name__ == '__main__':
    raise SystemExit(main())
