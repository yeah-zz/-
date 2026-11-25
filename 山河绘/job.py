# -*- coding: utf-8 -*-
"""
job.py - 把"步骤列表"翻译成机械臂轨迹：坐标、Z、站点动作、预览、执行
====================================================================

【waypoint 结构】
  {"x", "y", "z", "pause", "velocity", "pen_up", "tag"}
  pen_up=True  抬笔点（移动用）
  pen_up=False 落笔点（行笔用，会启用平滑过渡 cp）

【Z 的三条来源】
  1. 行笔：z_for_press(笔档下压量)
  2. 抬笔：z_for_lift(LIFT_MM)
  3. 站点：station_dip_z(站台面高度, 插入深度)（水盂/调色盘/擦笔布各有自己的面高）

【执行策略】
  逐点下发 MovL，按 pen_up 边界分批等待控制器缓存清空（原项目实测约 260 点会满），
  落笔线段启用 cp 平滑，驻笔用主机端 sleep。
"""

import json
import math
import time

import cv2
import numpy as np

from config import (
    ArmConfig,
    JobConfig,
    PALETTE,
    ShuimoConfig,
    StationConfig,
    brush_of,
    station_dip_z,
    transit_z,
    z_for_lift,
    z_for_press,
)
import imgio

TRANSIT_SPEED = 60          # 站点间空跑速度
TAG_SKIP = "station"


def _wp(x, y, z, pen_up, velocity, pause=0.0, tag="", cp=None):
    """
    cp = 该点的平滑过渡比例（0~100）。None = 用 JobConfig.CONTINUOUS_PATH_RATIO。
    只有水墨路线会显式带上更大的 cp（行笔要流畅，不能每点停一下）。
    """
    waypoint = {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "pause": float(pause),
        "velocity": float(velocity),
        "pen_up": bool(pen_up),
        "tag": tag,
    }
    if cp is not None:
        waypoint["cp"] = int(cp)
    return waypoint


# =====================================================================
# 1. 站点动作
# =====================================================================
def wash_waypoints():
    wash = StationConfig.WASH
    x = float(wash["x"])
    y = float(wash["y"])
    depth_z = station_dip_z(wash["z_surface"], wash["dip_mm"])
    surface_z = float(wash["z_surface"])
    swing = max(0.0, float(wash["stir_dx_mm"]))
    pause = max(0.0, float(wash["pause_s"]))
    times = max(1, int(wash["stir_times"]))

    points = [_wp(x, y, transit_z(), True, TRANSIT_SPEED, tag="wash-in")]
    for _ in range(times):
        points.append(_wp(x, y, depth_z, True, JobConfig.DOWN_SPEED, tag="wash-down"))
        points.append(_wp(x + swing, y, depth_z, True, JobConfig.DOWN_SPEED, tag="wash-stir"))
        points.append(_wp(x - swing, y, depth_z, True, JobConfig.DOWN_SPEED, tag="wash-stir"))
        points.append(_wp(x, y, surface_z, True, JobConfig.DOWN_SPEED,
                          pause=pause, tag="wash-up"))
    points.append(_wp(x, y, transit_z(), True, TRANSIT_SPEED, tag="wash-out"))
    return points


def wipe_waypoints():
    wipe = StationConfig.WIPE
    x0 = float(wipe["x0"])
    y0 = float(wipe["y0"])
    x1 = float(wipe["x1"])
    y1 = float(wipe["y1"])
    press_z = station_dip_z(wipe["z_surface"], wipe["press_mm"])
    speed = float(wipe.get("speed", 30))
    passes = max(1, int(wipe["passes"]))

    points = [_wp(x0, y0, transit_z(), True, TRANSIT_SPEED, tag="wipe-in")]
    for i in range(passes):
        if i % 2 == 0:
            points.append(_wp(x0, y0, press_z, True, JobConfig.DOWN_SPEED, tag="wipe-down"))
            points.append(_wp(x1, y1, press_z, True, speed, tag="wipe"))
        else:
            points.append(_wp(x1, y1, press_z, True, speed, tag="wipe"))
            points.append(_wp(x0, y0, press_z, True, speed, tag="wipe"))
    points.append(_wp(x0 if passes % 2 == 0 else x1,
                      y0 if passes % 2 == 0 else y1,
                      transit_z(), True, TRANSIT_SPEED, tag="wipe-out"))
    return points


def dip_waypoints(pal_index, dwell_s=None, drip_s=None):
    """蘸色：下到该色格颜料表面以下 dip_mm，停留 dip_time_s（每色可单独配置）"""
    from config import (dip_depth_of, dip_position, dip_time_of,
                        palette_surface_z, station_approach_z)
    x, y = dip_position(pal_index)
    dip_z = station_dip_z(palette_surface_z(pal_index), dip_depth_of(pal_index))
    # ★ 用【站点接近高度】而不是站点间高度：墨盘可能是高碗，碗沿远高于纸面，
    #   用站点间高度横移过去会把碗撞翻。见 StationConfig.APPROACH_LIFT_MM。
    approach_z = station_approach_z()
    # 蘸完在碗上方停一下，让笔毛上多余的墨滴回去 —— 不停就走，
    # 湿墨会在空跑时被甩出去，纸上出现一圈飞溅黑点。
    drip = max(0.0, float(getattr(StationConfig, "DIP_DRIP_PAUSE_S", 0.0)))
    dwell = dip_time_of(pal_index) if dwell_s is None else float(dwell_s)
    if drip_s is not None:
        drip = float(drip_s)
    if not all(math.isfinite(v) and 0 <= v <= 10 for v in (dwell, drip)):
        raise ValueError("Invalid dip/drain duration")
    points = [_wp(x, y, approach_z, True, TRANSIT_SPEED, tag="dip-in")]
    points.append(_wp(x, y, dip_z, True, JobConfig.DOWN_SPEED,
                      pause=dwell, tag="dip"))
    points.append(_wp(x, y, approach_z, True, TRANSIT_SPEED,
                      pause=drip, tag="dip-out"))
    return points


def test_waypoints(pal_index, press_mm=None, speed=None):
    """
    试色/试墨区画一小条。press_mm / speed 给定时覆盖笔档默认值 ——
    水墨路线（极细狼毫）的下压量和色块路线的笔档完全不同，不能混用。
    """
    strip = StationConfig.TEST_STRIP
    if not strip.get("enabled", True):
        return []
    brush = brush_of(PALETTE[pal_index])
    press_z = z_for_press(float(press_mm) if press_mm is not None
                          else brush["press_mm"])
    test_speed = float(speed) if speed is not None else brush["speed"]
    x0 = float(strip["x0"])
    y0 = float(strip["y0"])
    x1 = float(strip["x1"])
    y1 = float(strip["y1"])
    points = [_wp(x0, y0, transit_z(), True, TRANSIT_SPEED, tag="test-in")]
    points.append(_wp(x0, y0, press_z, True, JobConfig.DOWN_SPEED, tag="test-down"))
    points.append(_wp(x1, y1, press_z, False, test_speed, tag="test"))
    points.append(_wp(x1, y1, transit_z(), True, TRANSIT_SPEED, tag="test-out"))
    return points


# =====================================================================
# 2. 笔触 -> 轨迹
# =====================================================================
def stroke_waypoints(mapper, stroke):
    """
    一条笔触：抬笔移入 -> 垂直落笔 -> 行笔 -> 垂直抬笔

    两条路径，互不影响：
      - **逐点笔压**（水墨路线）：stroke 带 "press" 列表，和 points 等长，单位 mm。
        这是"一笔之内有粗细浓淡"的唯一入口 —— 色块路线是一条笔触一个固定下压量，
        水墨路线是每个轨迹点各有自己的下压量和速度。
      - **固定笔压**（色块路线）：按笔档的 press_mm，整条一个值。行为与改动前一致。
    """
    name = stroke.get("name", "")
    robot_points = mapper.to_robot(stroke["points"])
    count = int(robot_points.shape[0])
    if count < 2:
        return []

    x0 = float(robot_points[0][0])
    y0 = float(robot_points[0][1])
    x1 = float(robot_points[-1][0])
    y1 = float(robot_points[-1][1])
    # 水墨抬笔比色块低（4mm vs 8mm）：减少空跑，保持笔势连贯
    lift_z = z_for_lift(float(stroke.get("lift_mm", ArmConfig.LIFT_MM)))

    presses = stroke.get("press")
    if presses and len(presses) == count:
        # ---------------- 逐点笔压（水墨框架）----------------
        speeds = stroke.get("speed")
        if not speeds or len(speeds) != count:
            speeds = [float(JobConfig.DOWN_SPEED)] * count
        # 机械安全上限：行笔中 Z 绝不允许超过它（狼毫压过头会劈叉/伤纸）
        top = float(ShuimoConfig.PRESS_MAX_MM)
        # 水墨行笔的平滑过渡比例：比色块路线大得多，否则每个轨迹点都"停一下
        # 再走"，走出来是一段段直线（客户反馈的"直来直去"）。见 config.py 注释。
        shuimo_cp = int(getattr(ShuimoConfig, "CONTINUOUS_PATH_RATIO",
                                JobConfig.CONTINUOUS_PATH_RATIO))
        points = [_wp(x0, y0, lift_z, True, TRANSIT_SPEED, tag="hover:" + name)]
        horizontal_speed = stroke.get('horizontal_velocity_percent')
        if horizontal_speed is not None:
            if type(horizontal_speed) is not int or not 1 <= horizontal_speed <= 80:
                raise ValueError('Ink horizontal velocity must be an integer in [1, 80]')
            points[0]['velocity'] = horizontal_speed
        points.append(_wp(x0, y0, z_for_press(float(presses[0])), False,
                          JobConfig.DOWN_SPEED, tag="down:" + name))
        for i in range(1, count):
            # 下限 0 = 笔尖刚触纸，绝不抬离纸面（否则墨线断成几截）
            press = min(max(float(presses[i]), 0.0), top)
            points.append(_wp(float(robot_points[i][0]),
                              float(robot_points[i][1]),
                              z_for_press(press), False,
                              float(speeds[i]), tag="draw:" + name,
                              cp=shuimo_cp))

        # ---------------- 收笔出锋 ----------------
        # 不在收笔点原地抬笔（那会把墨堆成一个疙瘩），而是沿笔势方向再走
        # 一小段、同时把笔压收到最轻，最后的抬笔动作发生在"锋"的末端。
        # 见 config.py 的 ShuimoConfig.TAIL_MM 注释。
        end_x, end_y = x1, y1
        tail_mm = float(stroke.get("tail_mm", getattr(ShuimoConfig, "TAIL_MM", 0.0)))
        # ★ 点状/串珠笔触（DAB_LENGTH_MM>0）【必须关掉出锋】：
        #   4mm 的小段加 4mm 的出锋 = 整条都是尾巴，笔压永远到不了满值，
        #   画出来是一条细线而不是一个墨点，"串珠"效果就没了。
        #   同理，任何比 2 倍出锋还短的笔触都不加。
        stroke_len_mm = 0.0
        for _i in range(1, count):
            stroke_len_mm += float(np.hypot(
                float(robot_points[_i][0]) - float(robot_points[_i - 1][0]),
                float(robot_points[_i][1]) - float(robot_points[_i - 1][1])))
        if float(getattr(ShuimoConfig, "DAB_LENGTH_MM", 0.0) or 0.0) > 1e-6:
            tail_mm = 0.0
        elif stroke_len_mm < tail_mm * 2.0:
            tail_mm = 0.0
        if tail_mm > 1e-6:
            dx = float(robot_points[-1][0]) - float(robot_points[-2][0])
            dy = float(robot_points[-1][1]) - float(robot_points[-2][1])
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                ux, uy = dx / norm, dy / norm
                n_tail = max(1, int(getattr(ShuimoConfig, "TAIL_POINTS", 4)))
                keep = float(np.clip(
                    getattr(ShuimoConfig, "TAIL_PRESS_RATIO", 0.0), 0.0, 1.0))
                boost = max(0.1, float(getattr(ShuimoConfig,
                                               "TAIL_SPEED_BOOST", 1.0)))
                tail_speed = min(100.0, float(speeds[-1]) * boost)
                last_press = min(max(float(presses[-1]), 0.0), top)
                for k in range(1, n_tail + 1):
                    f = k / float(n_tail)
                    press = last_press * (1.0 - f * (1.0 - keep))
                    points.append(_wp(
                        float(robot_points[-1][0]) + ux * tail_mm * f,
                        float(robot_points[-1][1]) + uy * tail_mm * f,
                        z_for_press(press), False, tail_speed,
                        tag="tail:" + name, cp=shuimo_cp))
                end_x = float(robot_points[-1][0]) + ux * tail_mm
                end_y = float(robot_points[-1][1]) + uy * tail_mm

        points.append(_wp(end_x, end_y, lift_z, True, TRANSIT_SPEED,
                          tag="lift:" + name))
        for tag, key in (('down:', 'touchdown_velocity_percent'), ('lift:', 'lift_velocity_percent')):
            value = stroke.get(key)
            if value is not None:
                if type(value) is not int or not 1 <= value <= 60:
                    raise ValueError('Stroke touchdown/lift speed must be an integer in [1, 60]')
                for point in points:
                    if point['tag'].startswith(tag): point['velocity'] = value
        return points

    # ---------------- 固定笔压（色块路线）----------------
    from config import BRUSH_PRESETS
    # 用笔触自带的档位（vision 会按色块粗细自动降档），缺失时回退到颜色默认档
    brush = BRUSH_PRESETS.get(stroke.get("brush")) or \
        brush_of(PALETTE[stroke["pal_index"]])
    press_z = z_for_press(brush["press_mm"])
    speed = float(brush["speed"])

    points = [_wp(x0, y0, lift_z, True, TRANSIT_SPEED, tag="hover:" + name)]
    points.append(_wp(x0, y0, press_z, False, JobConfig.DOWN_SPEED, tag="down:" + name))
    for i in range(1, count):
        points.append(_wp(float(robot_points[i][0]), float(robot_points[i][1]),
                          press_z, False, speed, tag="draw:" + name))
    points.append(_wp(x1, y1, lift_z, True, TRANSIT_SPEED, tag="lift:" + name))
    return points


def steps_to_waypoints(steps, mapper):
    """步骤列表 -> (waypoints, stats)"""
    waypoints = []
    stats = {"strokes": 0, "travel_mm": 0.0}
    previous = None
    for step in steps:
        kind = step["kind"]
        if kind == "wash":
            chunk = wash_waypoints()
        elif kind == "wipe":
            chunk = wipe_waypoints()
        elif kind == "dip":
            chunk = dip_waypoints(step["pal_index"], step.get("dwell_s"), step.get("drip_s"))
        elif kind == "rest":
            if previous is None:
                raise ValueError("Cannot rest before a known paper position")
            chunk = [_wp(previous["x"], previous["y"], transit_z(), True, TRANSIT_SPEED,
                         pause=float(step["seconds"]), tag="rest")]
        elif kind == "test":
            chunk = test_waypoints(step["pal_index"],
                                   step.get("press_mm"), step.get("speed"))
        elif kind == "stroke":
            chunk = stroke_waypoints(mapper, step["stroke"])
            stats["strokes"] += 1
        else:
            continue

        for wp in chunk:
            if previous is not None:
                stats["travel_mm"] += math.hypot(wp["x"] - previous["x"],
                                                 wp["y"] - previous["y"])
            previous = wp
        waypoints.extend(chunk)
    return waypoints, stats


# =====================================================================
# 3. 预览图
# =====================================================================
def render_preview(source_bgr, strokes, mm_per_px, out_path):
    """
    左侧：处理后的输入图；右侧：按笔宽与调色板模拟出来的"成画预览"。
    这张图就是你按下 --execute 后纸上大致会得到的结果。
    """
    h, w = source_bgr.shape[:2]
    sim = np.full((h, w, 3), 255, dtype=np.uint8)

    for stroke in strokes:
        rgb = PALETTE[stroke["pal_index"]]["rgb"]
        color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        thickness = int(max(1, round(float(stroke["width_mm"]) / max(1e-6, mm_per_px))))
        pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in stroke["points"]],
                       dtype=np.int32)
        if pts.shape[0] < 2:
            continue
        cv2.polylines(sim, [pts], False, color, thickness, cv2.LINE_AA)

    left = source_bgr.copy()
    bar = np.full((24, w, 3), 245, dtype=np.uint8)
    cv2.putText(bar, "SOURCE", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(bar, "SIMULATION (what the robot will paint)", (w // 2 + 8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    left = cv2.vconcat([bar, left])
    right = cv2.vconcat([bar, sim])
    combined = cv2.hconcat([left, np.full((left.shape[0], 4, 3), 200, dtype=np.uint8), right])
    if not imgio.imwrite(out_path, combined):
        raise IOError("预览图写出失败: %s" % out_path)
    return out_path


def render_station_map(mapper, out_path, size=520):
    """
    俯视图：画布 + 水盂/擦笔/调色盘色格/试色区的位置关系。
    上机前用这张图核对站点坐标有没有量错。
    """
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    all_points = []

    def collect(x, y):
        all_points.append((float(x), float(y)))

    for x, y in mapper.robot_corners:
        collect(x, y)
    wash = StationConfig.WASH
    collect(wash["x"], wash["y"])
    wipe = StationConfig.WIPE
    collect(wipe["x0"], wipe["y0"])
    collect(wipe["x1"], wipe["y1"])
    strip = StationConfig.TEST_STRIP
    if strip.get("enabled", True):
        collect(strip["x0"], strip["y0"])
        collect(strip["x1"], strip["y1"])
    from config import dip_position
    for i in range(len(PALETTE)):
        x, y = dip_position(i)
        collect(x, y)

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_x = max(1.0, x_max - x_min)
    span_y = max(1.0, y_max - y_min)
    span = max(span_x, span_y)
    margin = 40

    def to_px(x, y):
        px = margin + int(round((x - x_min) / span * (size - 2 * margin)))
        py = size - margin - int(round((y - y_min) / span * (size - 2 * margin)))
        return (px, py)

    corners_px = [to_px(x, y) for x, y in mapper.robot_corners]
    cv2.polylines(canvas, [np.array(corners_px, dtype=np.int32)], True,
                  (180, 120, 60), 2, cv2.LINE_AA)
    cv2.putText(canvas, "CANVAS", (corners_px[0][0] + 4, corners_px[0][1] + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 120, 60), 1, cv2.LINE_AA)

    wx, wy = to_px(wash["x"], wash["y"])
    cv2.circle(canvas, (wx, wy), 14, (200, 120, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "WASH", (wx - 18, wy - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 120, 40), 1, cv2.LINE_AA)

    w0 = to_px(wipe["x0"], wipe["y0"])
    w1 = to_px(wipe["x1"], wipe["y1"])
    cv2.line(canvas, w0, w1, (60, 160, 60), 3, cv2.LINE_AA)
    cv2.putText(canvas, "WIPE", (w1[0] + 4, w1[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (60, 160, 60), 1, cv2.LINE_AA)

    if strip.get("enabled", True):
        s0 = to_px(strip["x0"], strip["y0"])
        s1 = to_px(strip["x1"], strip["y1"])
        cv2.line(canvas, s0, s1, (150, 60, 150), 2, cv2.LINE_AA)
        cv2.putText(canvas, "TEST", (s1[0] + 4, s1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (150, 60, 150), 1, cv2.LINE_AA)

    for i, entry in enumerate(PALETTE):
        x, y = dip_position(i)
        px, py = to_px(x, y)
        rgb = entry["rgb"]
        cv2.circle(canvas, (px, py), 9, (int(rgb[2]), int(rgb[1]), int(rgb[0])), -1,
                   cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 9, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(canvas, "%d" % i, (px - 4, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "station map (top view, mm)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    if not imgio.imwrite(out_path, canvas):
        raise IOError("站点图写出失败: %s" % out_path)
    return out_path


# =====================================================================
# 4. 任务文件
# =====================================================================
def save_job(path, waypoints, meta):
    data = {"format": "arm-painting job", "version": 1, "meta": meta,
            "waypoints": waypoints}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    return path


def load_job(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("waypoints", []), data.get("meta", {})


# =====================================================================
# 5. 执行（逐点 MovL + 分批等待）
# =====================================================================
def execute_waypoints(robot, waypoints, verbose=True):
    """
    逐点下发并按批次等待。返回实际下发点数。
    任何一条指令返回非 0 错误码都会抛 RuntimeError（避免继续乱画）。
    """
    batch = max(1, int(JobConfig.MOTION_BATCH_SIZE))
    cp_ratio = int(JobConfig.CONTINUOUS_PATH_RATIO)
    pending = 0
    count = 0
    total = len(waypoints)
    for index, wp in enumerate(waypoints):
        velocity = int(max(5, min(100, round(float(wp.get("velocity", 30))))))
        # 落笔点用平滑过渡，抬笔点必须 cp=0（否则会切角、落点不准）。
        # 每个点可以自带 cp（水墨路线给的值更大，让行笔连绵不断）。
        continuous = int(wp.get("cp", cp_ratio)) if wp.get("pen_up") is False else 0
        response = robot.move_linear(wp["x"], wp["y"], wp["z"], velocity, continuous)
        if not response or not response.startswith("0,"):
            error_id = robot.get_error_id()
            raise RuntimeError(
                "第 %d/%d 个轨迹点执行失败: (%.1f, %.1f, %.1f) 响应=%r 控制器错误=%r"
                % (index, total, wp["x"], wp["y"], wp["z"], response, error_id))
        count += 1
        pending += 1
        if float(wp.get("pause", 0.0)) > 0:
            time.sleep(min(5.0, float(wp["pause"])))
        if pending >= batch and (wp.get("pen_up") is True or pending >= batch * 2):
            if not robot.wait_idle(JobConfig.MOTION_TIMEOUT):
                raise RuntimeError("第 %d 个点后等待运动完成超时" % count)
            pending = 0
            if verbose:
                print("   已下发 %d/%d 点..." % (count, total))
        time.sleep(float(ArmConfig.MOVE_DELAY))
    if pending:
        if not robot.wait_idle(JobConfig.MOTION_TIMEOUT):
            raise RuntimeError("任务结束时等待运动完成超时")
    return count
