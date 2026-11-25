# -*- coding: utf-8 -*-

"""

main.py - 机械臂绘画 命令行入口

====================================================================



【常用命令】

  python main.py --check                     配置自检（不连机械臂）

  python main.py --probe                     连机械臂空跑画布四角+中心（验证坐标）

  python main.py --calibrate-z               交互标定纸面高度 PAPER_Z

  python main.py --test-strokes              连机械臂画 4 条标定线（测笔宽）

  python main.py --image lanhua.jpg          只预览：出预览图 + 任务 JSON

  python main.py --image lanhua.jpg --mono --limit 30 --execute

                                             单色试画 30 笔（第一次上纸用这条）

  python main.py --image lanhua.jpg --execute

                                             完整上色绘制



【默认是安全的】

  不带 --execute 时绝不会连接机械臂，只生成 paint_preview.png / paint_job.json。

"""



import argparse

import os

import sys

import time



import numpy as np



from config import (

    ArmConfig,

    BRUSH_PRESETS,

    CanvasConfig,

    JobConfig,

    PALETTE,

    ReachConfig,

    ShuimoConfig,

    StationConfig,

    VisionConfig,

    describe,

    dip_position,

    palette_surface_z,

    station_approach_z,

    transit_z,

    validate,

    z_for_lift,

    z_for_press,

)

from geometry import CanvasMap, check_waypoints

import job as jobmod

import scheduler

import shuimo

import vision

from project_paths import OUTPUT_DIR





# =====================================================================

# 辅助

# =====================================================================

def apply_mono(strokes, name):

    """把所有笔触改成同一个颜色，并跳过洗笔/蘸色动作（第一上机测试用）"""

    index = None

    for i, entry in enumerate(PALETTE):

        if entry["name"] == name:

            index = i

            break

    if index is None:

        print("[!] 调色板里没有 %s，改用第一个颜色 %s" % (name, PALETTE[0]["name"]))

        index = 0

    for stroke in strokes:

        stroke["pal_index"] = index

        stroke["layer"] = 0

        stroke["name"] = PALETTE[index]["name"]

    JobConfig.SKIP_PIGMENT_ACTIONS = True

    print("[单色模式] 全部用 %s 绘制，并跳过洗笔/蘸色/试色动作" % PALETTE[index]["name"])

    return index





def apply_limit(steps, count):

    """只保留前 count 条笔触（含它前面的准备动作）"""

    out = []

    kept = 0

    for step in steps:

        if step["kind"] == "stroke":

            if kept >= count:

                break

            kept += 1

        out.append(step)

    print("[限量] 只绘制前 %d 条笔触" % kept)

    return out





def apply_line_art_preset():

    """

    水墨线稿预设：兰花、竹叶这类"细墨线 + 大量留白"的图。

    默认参数是按"大色块插画"调的，直接用会把细线抹掉：

      - 开运算核(0.8mm≈9px)会把 4px 宽的墨线整条腐蚀掉 -> 关闭开运算

      - 最小色块面积/最小笔长按大色块设置 -> 线条碎片会被全部丢弃

      - 扫掠行距放宽，一根细线只留一条笔触，避免同一片叶子画好几遍

    """

    VisionConfig.MORPH_OPEN_MM = 0.0

    VisionConfig.MORPH_CLOSE_MM = 1.6

    VisionConfig.MIN_REGION_AREA_MM2 = 6.0

    VisionConfig.MIN_STROKE_LEN_MM = 4.0

    VisionConfig.SPACING_FACTOR = 0.85

    VisionConfig.STROKE_BOW_MM = 0.8

    print("[预设] 线稿模式：关闭开运算，最小色块 6mm2，最小笔长 4mm，行距 0.85")





def print_summary(waypoints, plan_stats, job_stats, mapper):

    print("-" * 70)

    print("[轨迹] 共 %d 个轨迹点，空跑距离 %.0f mm" % (len(waypoints), job_stats["travel_mm"]))

    print("[轨迹] 笔触 %d 条，总笔长 %.0f mm" % (plan_stats["strokes"], plan_stats["length_mm"]))

    print("[轨迹] %s" % mapper.summary())

    print("-" * 70)





# =====================================================================

# 各子命令

# =====================================================================

# =====================================================================

# 引导式标定（--teach）

# =====================================================================

TEACH_STEPS = [

    ("canvas_tl", "画布左上角正上方（笔尖对准；建议从纸边内缩 10~15mm）"),

    ("canvas_tr", "画布右上角"),

    ("canvas_br", "画布右下角"),

    ("canvas_bl", "画布左下角"),

    ("paper_center", "纸张中心：笔尖刚好轻触纸面（用来确定 PAPER_Z）"),

    ("wash", "清水盂中心：笔尖刚好接触水面"),

    ("wipe_a", "擦笔布一端：笔尖刚好接触布面"),

    ("wipe_b", "擦笔布另一端"),

    ("strip_a", "试色区起点：笔尖刚好接触纸面"),

    ("strip_b", "试色区终点"),

]





def render_teach_snippet(recorded):

    """把记录到的坐标渲染成可以直接粘贴到 config.py 末尾的代码"""

    lines = ["", "# ===== 现场标定结果（由 --teach 生成，直接粘贴到 config.py 末尾）====="]

    keys = ("canvas_tl", "canvas_tr", "canvas_br", "canvas_bl")

    if all(key in recorded for key in keys):

        lines.append("CanvasConfig.ROBOT_CORNERS = [")

        for key in keys:

            pose = recorded[key]

            lines.append("    (%.1f, %.1f)," % (pose[0], pose[1]))

        lines.append("]")

    if "paper_center" in recorded:

        lines.append("ArmConfig.PAPER_Z = %.1f" % recorded["paper_center"][2])

    if "wash" in recorded:

        pose = recorded["wash"]

        lines.append('StationConfig.WASH["x"] = %.1f' % pose[0])

        lines.append('StationConfig.WASH["y"] = %.1f' % pose[1])

        lines.append('StationConfig.WASH["z_surface"] = %.1f' % pose[2])

    if "wipe_a" in recorded and "wipe_b" in recorded:

        first = recorded["wipe_a"]

        second = recorded["wipe_b"]

        lines.append('StationConfig.WIPE["x0"] = %.1f' % first[0])

        lines.append('StationConfig.WIPE["y0"] = %.1f' % first[1])

        lines.append('StationConfig.WIPE["x1"] = %.1f' % second[0])

        lines.append('StationConfig.WIPE["y1"] = %.1f' % second[1])

        lines.append('StationConfig.WIPE["z_surface"] = %.1f' % first[2])

    if "strip_a" in recorded and "strip_b" in recorded:

        first = recorded["strip_a"]

        second = recorded["strip_b"]

        lines.append('StationConfig.TEST_STRIP["x0"] = %.1f' % first[0])

        lines.append('StationConfig.TEST_STRIP["y0"] = %.1f' % first[1])

        lines.append('StationConfig.TEST_STRIP["x1"] = %.1f' % second[0])

        lines.append('StationConfig.TEST_STRIP["y1"] = %.1f' % second[1])

    for index, entry in enumerate(PALETTE):

        key = "dip_%d" % index

        if key in recorded:

            pose = recorded[key]

            lines.append('PALETTE[%d]["dip"] = (%.1f, %.1f)          # %s'

                         % (index, pose[0], pose[1], entry["name"]))

            lines.append('PALETTE[%d]["surface_z"] = %.1f' % (index, pose[2]))

    if recorded:

        lines.append("StationConfig.STATIONS_ARE_PLACEHOLDER = False")

    lines.append("")

    lines.append("# 还没做的两件事：")

    lines.append("# 1) 量一下水盂/调色盘边缘高出纸面多少，把 ArmConfig.TRANSIT_LIFT_MM 设得更大，否则会撞杯沿")

    lines.append("# 2) uv run python main.py --test-strokes  量实际笔宽，填回 BRUSH_PRESETS")

    return "\n".join(lines) + "\n"





def cmd_teach(args):

    """

    引导式标定：手动把笔尖移到每个目标位置，程序自动读坐标并生成可粘贴的配置。

    前提：机械臂已连接，毛笔已装好，USER_INDEX / TOOL_INDEX 已在 config.py 里设对

    （GetPose 返回的是工具坐标系下的笔尖位置）。

    """

    from robot import E6Robot

    robot = E6Robot()

    ok, message = robot.connect()

    print("[机器人] %s" % message)

    if not ok:

        return 3



    steps = list(TEACH_STEPS)

    for index, entry in enumerate(PALETTE):

        steps.append(("dip_%d" % index,

                      "第 %d 格「%s」中心：笔尖刚好接触颜料表面" % (index + 1, entry["name"])))



    recorded = {}

    try:

        print("")

        print("=" * 66)

        print(" 引导标定：用 DobotStudio Pro 手动移动笔尖，到位停稳后按回车记录")

        print(" 共 %d 个位置。中途 Ctrl+C 可中止，已记录的会保留。" % len(steps))

        print("=" * 66)

        for key, label in steps:

            input("\n>>> %s\n    [%s] 到位后按回车：" % (label, key))

            pose = robot.get_pose()

            if pose is None:

                print("    [!] 读取位姿失败。可在 DobotStudio Pro 里看当前坐标手工输入。")

                raw = input("    手工输入 x,y,z（逗号分隔，直接回车跳过）：").strip()

                if not raw:

                    continue

                try:

                    parts = [float(v) for v in raw.replace("，", ",").split(",")]

                    pose = (parts[0], parts[1], parts[2], 0.0, 0.0, 0.0)

                except (ValueError, IndexError):

                    print("    输入无效，跳过这一项。")

                    continue

            recorded[key] = pose

            print("    已记录 x=%.1f  y=%.1f  z=%.1f" % (pose[0], pose[1], pose[2]))

    except KeyboardInterrupt:

        print("\n[中止] 保留已记录内容。")

    finally:

        robot.close(disable=False)



    snippet = render_teach_snippet(recorded)

    print(snippet)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taught_config.txt")

    try:

        with open(path, "w", encoding="utf-8") as handle:

            handle.write(snippet)

        print("[输出] 已写入 %s：把里面的内容粘贴到 config.py 末尾即可。" % path)

    except OSError as exc:

        print("[!] 写入 %s 失败: %s（上面的内容直接复制也行）" % (path, exc))

    return 0





def cmd_check():

    describe()

    if StationConfig.STATIONS_ARE_PLACEHOLDER:

        print("[提示] 站点坐标还是占位值：只能 --check / --probe / 预览，")

        print("       测量完 StationConfig 后把 STATIONS_ARE_PLACEHOLDER 改成 False。")

    return 0





def cmd_probe(args):

    """在安全高度空跑画布四角与中心，验证画布坐标是否量对"""

    corners = [tuple(p) for p in CanvasConfig.ROBOT_CORNERS]

    centre = (sum(p[0] for p in corners) / 4.0, sum(p[1] for p in corners) / 4.0)

    targets = [(x, y) for (x, y) in corners] + [centre]

    names = ["左上", "右上", "右下", "左下", "中心"]



    probe = [{"x": x, "y": y, "z": transit_z()} for (x, y) in targets]

    ok, problems = check_waypoints(probe)

    if not ok:

        print("[!] 画布坐标超出可达范围，先修 config.py：")

        for p in problems:

            print("   ", p)

        return 2



    from robot import E6Robot

    robot = E6Robot()

    ok, message = robot.connect()

    print("[机器人] %s" % message)

    if not ok:

        return 3

    try:

        for name, (x, y) in zip(names, targets):

            print("[空跑] -> %s (%.1f, %.1f) 高度 %.1f" % (name, x, y, transit_z()))

            robot.move_safe(x, y, transit_z(), 40)

        print("[空跑] 完成。请确认机械臂停的位置是否正好在纸的四角上方。")

    except Exception as exc:

        print("[!] 空跑中断: %s" % exc)

        return 4

    finally:

        robot.close(disable=False)

    return 0





def cmd_calibrate_z(args):

    """

    交互标定纸面高度：从安全高度慢慢下降，由你判断笔尖是否刚好触纸。



    【为什么重写了】原版有个坑，现场表现是"按 d 之后机械臂先上升再下降，

    永远到不了纸面"：



        while True:

            move_safe(cx, cy, current)            # 降到目标

            answer = input()

            move_safe(cx, cy, z_for_lift(15))     # ★ 每次都抬回 15mm 上方

            current += 0.5                        # 而目标只变 0.5mm



    每按一次 d 要上升 15mm、再下降 15mm，而净下降只有 0.5mm ——

    如果起点离纸面 20mm，就要按 42 次，看起来就是一直在原地上下蹦。



    现在改成：

      - 接近一次，之后**不在调整之间抬笔**（纯 Z 移动，不会划伤纸面）

      - 步长可调：2mm 粗调 / 0.5mm / 0.1mm 精调

      - 可以直接输入目标高度跳过去

    """

    corners = [tuple(p) for p in CanvasConfig.ROBOT_CORNERS]

    cx = sum(p[0] for p in corners) / 4.0

    cy = sum(p[1] for p in corners) / 4.0



    from robot import E6Robot

    robot = E6Robot()

    ok, message = robot.connect()

    print("[机器人] %s" % message)

    if not ok:

        return 3



    # 下降方向：Z_INCREASES_DOWN=True 时 Z 变大 = 往下压

    down = 1.0 if ArmConfig.Z_INCREASES_DOWN else -1.0

    current = float(ArmConfig.PAPER_Z)

    # ★ 起步高度必须是【安全的】：换了笔之后，新笔尖和旧笔尖可能差几十毫米。

    #   如果照旧从 PAPER_Z 附近落下去，长笔会直接把笔尖压弯在纸上。

    #   所以统一从"旧纸面上方 50mm"起步 —— 比任何毛笔的长度变化都大。

    lift = -50.0 if ArmConfig.Z_INCREASES_DOWN else 50.0

    safe_start = current + lift

    step = 2.0

    try:

        robot.move_safe(cx, cy, safe_start, 40)

        print("")

        print("=" * 62)

        print(" 触纸标定：纸面中心 (%.1f, %.1f)。手放在急停上！" % (cx, cy))

        print(" 旧纸面高度 %.2f" % current)

        print(" 从它上方 50mm（%.2f）起步，你一步步降下来" % safe_start)

        print(" 嫌慢就直接输入数字跳过去，例如  -80  -70  -62")

        print("=" * 62)

        current = safe_start

        robot.move_safe(cx, cy, current, 20)

        while True:

            print("")

            print("  当前高度 %.2f     步长 %.1f mm     (%s)"

                  % (current, step,

                     "Z 变大 = 下降" if down > 0 else "Z 变小 = 下降"))

            print("  ------------------------------------------------------")

            print("   Enter 或 d  下降一个步长          u  上升一个步长")

            print("   1 / 2 / 3   步长改为 2 / 0.5 / 0.1 mm")

            print("   数字        直接设为该高度（例：-70 或 -68.5）")

            print("   y           笔尖刚好碰到纸面 -> 结束")

            print("   q           放弃")

            answer = input("  > ").strip().lower()



            if answer == "q":

                print("  已放弃，未改动任何配置。")

                return 0

            if answer == "y":

                break

            if answer in ("1", "2", "3"):

                step = {"1": 2.0, "2": 0.5, "3": 0.1}[answer]

                print("  步长已改为 %.1f mm" % step)

                continue

            if answer in ("", "d"):

                target = current + step * down

            elif answer == "u":

                target = current - step * down

            else:

                try:

                    target = float(answer)

                except ValueError:

                    print("  [!] 无法识别，请输入 Enter/d/u/y/数字/q")

                    continue



            reply = robot.move_linear(cx, cy, target, 15, 0)

            if not robot._ok(reply):

                print("  [!] 移动被拒: %r（高度保持在 %.2f）" % (reply, current))

                continue

            if not robot.wait_idle(JobConfig.MOTION_TIMEOUT):

                print("  [!] 等待到位失败 mode=%d，停止标定" % robot.get_mode())

                break

            current = target



        print("")

        print("=" * 62)

        print(" 标定完成：PAPER_Z = %.2f" % current)

        print(" 请把 config.py 里 ArmConfig.PAPER_Z 改成这个值")

        print("=" * 62)

        # 标定完把笔抬起来，别让它一直压在纸上

        robot.move_safe(cx, cy, z_for_lift(10.0), 40)

        print("[机器人] 已抬笔 10mm")

    except Exception as exc:

        print("[!] 标定中断: %s" % exc)

        return 4

    finally:

        robot.close(disable=False)

    return 0





def cmd_calibrate_ink(args):

    """

    交互标定【墨面高度】：笔停在墨盘正上方，一步步下降，由你判断笔尖是否刚好接触液面。



    和 --calibrate-z 的区别：

        --calibrate-z   标的是【纸张平面】，XY 在画布中心

        --calibrate-ink 标的是【墨盘液面】，XY 在调色板那一格



    什么时候要重跑：加了墨、加了水、换了容器、挪了盘子 —— 液面高度都会变。

    液面变了而 surface_z 没跟着改，轻则蘸不到墨（画出来是空的），

    重则笔尖插太深顶到盘底把笔毛压弯。

    """

    if not PALETTE:

        print("[!] PALETTE 为空，先在 config.py 里配一格墨")

        return 2



    index = shuimo.ink_palette_index()

    entry = PALETTE[index]

    tx, ty = dip_position(index)

    old_z = palette_surface_z(index)



    from robot import E6Robot

    robot = E6Robot()

    ok, message = robot.connect()

    print("[机器人] %s" % message)

    if not ok:

        return 3



    # "往上"的方向：Z 增大 = 下压时，往上就是减小 Z

    up = -1.0 if ArmConfig.Z_INCREASES_DOWN else 1.0

    # ★ 起步高度必须是【安全的】：

    #   换笔之后新笔尖和旧笔尖可能差几十毫米，旧液面上方 15mm 未必还在

    #   新笔尖的上方 —— 长笔一上来就直接插进墨里，标定就废了。

    #   统一从"旧液面上方 40mm"起步，比任何毛笔的长度变化都大。

    start = old_z + 40.0 * up

    current = start

    step = 1.0



    try:

        print("")

        print("=" * 64)

        print(" 墨面标定：墨盘中心 (%.2f, %.2f)，格子「%s」" % (tx, ty, entry["name"]))

        print(" 旧液面 %.4f  ->  从它上方 15mm（%.4f）开始下降" % (old_z, start))

        print(" 手放在急停上！")

        print("=" * 64)

        # ★★ 三步式接近 —— 绝不能一步斜着飞过去！

        #   MovL 是直线插补，一步式会"一边横移一边下降"，

        #   如果容器沿比起点高，半路就把墨盘撞翻了。

        approach_z = station_approach_z()

        cur = robot.get_pose()

        if cur is None:

            print("[!] 读不到当前位姿，无法安全接近")

            return 4

        print("  ① 先在原地抬到接近高度 %.3f（纸面上方 %.0fmm）"

              % (approach_z, float(StationConfig.APPROACH_LIFT_MM or 0.0)))

        robot.move_safe(cur[0], cur[1], approach_z, 30)

        print("  ② 水平移动到墨盘正上方")

        robot.move_safe(tx, ty, approach_z, 25)

        print("  ③ 垂直下降到起始高度 %.3f" % start)

        robot.move_safe(tx, ty, start, 15)

        while True:

            print("")

            print("  当前高度 %.3f     步长 %.1f mm     (%s)"

                  % (current, step, "Z 变大 = 下降" if up < 0 else "Z 变小 = 下降"))

            print("  相对旧液面: %+.2f mm" % (current - old_z))

            print("  ------------------------------------------------------")

            print("   Enter 或 d  下降一个步长          u  上升一个步长")

            print("   1 / 2 / 3   步长改为 1 / 0.2 / 0.05 mm")

            print("   数字        直接设为该高度")

            print("   y           笔尖刚好碰到液面 -> 结束")

            print("   q           放弃")

            answer = input("  > ").strip().lower()



            if answer == "q":

                print("  已放弃，未改动。")

                return 0

            if answer == "y":

                break

            if answer in ("1", "2", "3"):

                step = {"1": 1.0, "2": 0.2, "3": 0.05}[answer]

                print("  步长已改为 %.2f mm" % step)

                continue

            if answer in ("", "d"):

                target = current - step * up

            elif answer == "u":

                target = current + step * up

            else:

                try:

                    target = float(answer)

                except ValueError:

                    print("  [!] 无法识别，请输入 Enter/d/u/y/数字/q")

                    continue



            reply = robot.move_linear(tx, ty, target, 10, 0)

            if not robot._ok(reply):

                print("  [!] 移动被拒: %r（高度保持在 %.3f）" % (reply, current))

                continue

            if not robot.wait_idle(JobConfig.MOTION_TIMEOUT):

                print("  [!] 等待到位失败 mode=%d，停止标定" % robot.get_mode())

                break

            current = target



        print("")

        print("=" * 64)

        print(" 标定完成：墨面 surface_z = %.4f" % current)

        print(" 变化量：%+.2f mm（正数 = 液面下降了）" % (current - old_z))

        print("")

        print(" 请把 config.py 里 PALETTE[%d][\"surface_z\"] 改成 %.4f" % (index, current))

        print("=" * 64)

        # 标定完抬起来，别让笔尖泡在墨里

        robot.move_safe(tx, ty, approach_z, 30)

        print("[机器人] 已抬笔回到接近高度 %.3f" % approach_z)

    except Exception as exc:

        print("[!] 标定中断: %s" % exc)

        return 4

    finally:

        robot.close(disable=False)

    return 0





class _IdentityMapper:

    """把已是机器人坐标的点原样返回 —— 供笔宽标定使用"""



    @staticmethod

    def to_robot(points):

        # 必须是 ndarray：job.stroke_waypoints 要用 .shape

        return np.asarray(points, dtype=np.float64).reshape(-1, 2)





def cmd_brush_test(args):

    """

    笔宽标定：画 6 条【笔压递增】的横线，你拿尺子量宽度 -> 填 ShuimoConfig.BRUSHES。



    ★ 走的是真实的 job.stroke_waypoints（恒等映射，点本身已经是机器人坐标），

      所以包含起笔/收笔斜坡和收笔出锋，和正式绘制完全一致。

    ★ 每条线的【中间段】是恒定笔压，量中间那一段的宽度最准。

    ★ 位置可以自己指定（找一块干净的纸）：

        --bt-x0 40 --bt-x1 170 --bt-y0 60 --bt-step 20

    """

    from robot import E6Robot



    x0 = float(getattr(args, "bt_x0", 40.0) or 40.0)

    x1 = float(getattr(args, "bt_x1", 170.0) or 170.0)

    y0 = float(getattr(args, "bt_y0", 60.0) or 60.0)

    dstep = float(getattr(args, "bt_step", 20.0) or 20.0)

    count = 6

    lo = float(ShuimoConfig.PRESS_MIN_MM)

    hi = float(ShuimoConfig.PRESS_MAX_MM)

    if hi <= lo:

        print("[!] PRESS_MAX_MM 必须大于 PRESS_MIN_MM")

        return 2

    custom = str(getattr(args, "bt_presses", "") or "").strip()

    if custom:

        # 指定笔压列表（mm）：用来把实际工作点也量进去，例如

        #   --bt-presses 0.22,0.28,0.5,0.7,0.95    （0.22/0.28 = 花瓣常用档）

        try:

            presses = [float(v) for v in custom.replace("，", ",").split(",") if v.strip()]

        except ValueError:

            print("[!] --bt-presses 只能是逗号分隔的数字，例如 0.22,0.28,0.5,0.7,0.95")

            return 2

        if not presses or any(p < 0 or p > 5 for p in presses):

            print("[!] --bt-presses 每项必须在 0~5 mm 之间")

            return 2

        count = len(presses)

        lo, hi = min(presses), max(presses)

    else:

        presses = [lo + (hi - lo) * i / float(count - 1) for i in range(count)]

    npts = 12

    pal = shuimo.ink_palette_index()



    print("=" * 76)

    print(" 笔宽标定：%d 条横线，笔压 %.2f -> %.2f mm 递增" % (count, lo, hi))

    print("=" * 76)

    print("  线段 X: %.1f -> %.1f   长 %.0fmm" % (x0, x1, x1 - x0))

    print("  行距 %.0fmm，Y 从 %.1f 到 %.1f" % (dstep, y0, y0 + dstep * (count - 1)))

    print("  纸面 %.2f   抬笔 %.2f" % (ArmConfig.PAPER_Z, z_for_lift(ArmConfig.LIFT_MM)))

    print("")

    print("  %-4s %-10s %-12s %-9s %s" % ("#", "笔压mm", "插入Z", "Y", "轨迹点"))

    plan = []

    for i, press in enumerate(presses):

        y = y0 + dstep * i

        pts = [(x0 + (x1 - x0) * k / float(npts - 1), y) for k in range(npts)]

        stroke = {

            "points": pts,

            "press": [press] * npts,

            "speed": [18.0] * npts,

            "pal_index": pal,

            "name": "笔宽%d" % (i + 1),

            "lift_mm": float(ArmConfig.LIFT_MM),

        }

        wps = jobmod.stroke_waypoints(_IdentityMapper(), stroke)

        plan.append((i + 1, press, y, wps))

        print("  %-4d %-10.2f %-12.3f %-9.1f %d"

              % (i + 1, press, z_for_press(press), y, len(wps)))



    total = sum(len(w[3]) for w in plan)

    print("")

    print("  合计 %d 个轨迹点" % total)

    print("")

    print("=" * 76)

    print(" 量法：每条线【中间那一段】的宽度（两端有起笔/收笔斜坡，不要量）")

    print(" 量完把 %d 个宽度发我，我填进 ShuimoConfig.BRUSHES" % count)

    print("=" * 76)

    print("")



    if not args.execute:

        print("以上只是预览，没有连接机械臂。确认位置没问题后加 --execute 上纸。")

        return 0

    if not args.yes:

        try:

            answer = input("要把这 6 条线画到纸上吗？(y/N) ").strip().lower()

        except EOFError:

            print("[!] 非交互环境，需加 --yes")

            return 2

        if answer != "y":

            print("已取消。")

            return 0



    robot = E6Robot()

    ok, message = robot.connect()

    print("[机器人] %s" % message)

    if not ok:

        return 3

    try:

        def do_dip():

            """蘸一次墨。★ 每条线前都要蘸 —— 一笔墨只够画约 150mm，

            而这 6 条线加起来 780mm，不补蘸的话后面几条是干的、量出来全是错的。"""

            for w in jobmod.dip_waypoints(pal):

                reply = robot.move_linear(w["x"], w["y"], w["z"], w["velocity"], 0)

                if not robot._ok(reply):

                    print("    x 蘸墨指令被拒: %r" % reply)

                    return False

                if not robot.wait_idle(JobConfig.MOTION_TIMEOUT):

                    print("    x 蘸墨等待到位失败 mode=%d" % robot.get_mode())

                    robot.clear_error()

                    return False

                if float(w.get("pause", 0.0)) > 0:

                    time.sleep(min(10.0, float(w["pause"])))

            return True



        for idx, press, y, wps in plan:

            print(">>> 第 %d 条  蘸墨..." % idx)

            if not do_dip():

                break

            print("    落笔 Y=%.1f  笔压 %.2fmm  插入 Z=%.3f"

                  % (y, press, z_for_press(press)))

            reply = robot.move_safe(x0, y, z_for_lift(ArmConfig.LIFT_MM), 40)

            if not robot._ok(reply):

                print("    x 移动到起点失败: %r" % reply)

                break

            failed = False

            for w in wps:

                reply = robot.move_linear(w["x"], w["y"], w["z"], w["velocity"], 0)

                if not robot._ok(reply):

                    print("    x 指令被拒: %r" % reply)

                    failed = True

                    break

                if not robot.wait_idle(JobConfig.MOTION_TIMEOUT):

                    print("    x 等待到位失败 mode=%d %s"

                          % (robot.get_mode(), robot.get_error_id()))

                    robot.clear_error()

                    failed = True

                    break

            if failed:

                break

            print("    OK")

        print("")

        print("=" * 76)

        print(" 6 条线画完了。请量每条线【中间段】的宽度，发给我。")

        print("=" * 76)

    except Exception as exc:

        print("[!] 标定中断: %s" % exc)

    finally:

        try:

            robot.move_safe(x0, y0, z_for_lift(30.0), 40)

        except Exception:

            pass

        robot.close(disable=False)

    return 0





def cmd_test_strokes(args):

    """

    在试色区画标定线，用来量实际笔宽。



    --mode shuimo 时用 ShuimoConfig.BRUSHES（极细狼毫的 4 档），

    否则用 BRUSH_PRESETS（色块路线的宽笔 4 档）—— 两套笔完全不同，不能混标。

    """

    strip = StationConfig.TEST_STRIP

    if getattr(args, "mode", "color") == "shuimo":

        table = ShuimoConfig.BRUSHES

        order = [key for key in ("tip", "fine", "mid", "broad") if key in table]

        label = "ShuimoConfig.BRUSHES"

        print("[标定] 水墨模式：极细狼毫 %d 档" % len(order))

        print("[标定] 注意：必须**带着墨**量，不能干笔量 —— 狼毫笔痕宽度和含墨量强相关。")

    else:

        table = BRUSH_PRESETS

        order = [key for key in ("fine", "mid", "wide", "flat") if key in table]

        label = "BRUSH_PRESETS"

    if not order:

        print("[!] %s 里没有可用的笔档" % label)

        return 2

    if StationConfig.STATIONS_ARE_PLACEHOLDER:

        print("[!] 警告：站点坐标仍是占位值，试色区位置可能不对，先用 --probe 确认。")



    from robot import E6Robot

    robot = E6Robot()

    ok, message = robot.connect()

    print("[机器人] %s" % message)

    if not ok:

        return 3



    x0 = float(strip["x0"])

    y_base = float(strip["y0"])

    length = 45.0

    row_gap = 15.0

    try:

        for i, key in enumerate(order):

            preset = table[key]

            y = y_base + i * row_gap

            x1 = x0 + length

            z = z_for_press(preset["press_mm"])

            print("[标定] %s: press=%.2fmm 期望宽度 %.1fmm 速度 %d"

                  % (key, preset["press_mm"], preset["width_mm"], preset["speed"]))

            robot.move_safe(x0, y, transit_z(), 40)

            robot.move_safe(x0, y, z, 15)

            robot.move_linear(x1, y, z, preset["speed"], 0)

            robot.wait_idle(JobConfig.MOTION_TIMEOUT)

            robot.move_safe(x1, y, transit_z(), 40)

        print("")

        print("已画 %d 条线（沿 y 方向间隔 %.0fmm）。请用尺量出各自宽度，" % (len(order), row_gap))

        print("然后把 width_mm 填回 config.py 的 %s。" % label)

    except Exception as exc:

        print("[!] 标定中断: %s" % exc)

        return 4

    finally:

        robot.close(disable=False)

    return 0





def cmd_paint(args):

    """主流程：图像 -> 色块 -> 笔触 -> 排程 -> 轨迹 -> 预览（可选执行）"""

    skip_pigment = bool(args.mono or args.no_pigment)

    # 显式双向赋值：SKIP_PIGMENT_ACTIONS 是模块级开关，而 launcher 会在同一个

    # 进程里反复调用 main()。只设 True 不设 False，上一次的 --no-pigment 就会

    # 泄漏到下一次"完整绘制"，让机器人不去蘸色（静默出错）。

    JobConfig.SKIP_PIGMENT_ACTIONS = skip_pigment

    problems = validate(need_stations=not skip_pigment)

    if problems:

        print("[!] 配置问题:")

        for item in problems:

            print("   ", item)

        if args.execute:

            print("[!] 存在配置问题，拒绝执行。先修好配置（或加 --mono 跳过站点动作）。")

            return 2



    if args.line_art:

        apply_line_art_preset()



    try:

        image = vision.load_image(args.image)

    except ValueError as exc:

        print("[!] %s" % exc)

        return 2



    work, info = vision.prepare_image(image)

    mapper = CanvasMap(info["prepared_size"], info["image_corners"])

    print("[图像] 工作图 %d x %d（留白填充=%s）"

          % (info["prepared_size"][0], info["prepared_size"][1], info["padded"]))

    print("[图像] %s" % mapper.summary())



    strokes, stats = vision.build_strokes(work, mapper.mm_per_px)

    vision.print_stats(stats)

    if not strokes:

        print("[!] 没有生成任何笔触：可能是纸色阈值把整张图判成留白了，"

              "调低 VisionConfig.PAPER_L_MIN 或提高 PAPER_TOLERANCE。")

        return 1



    if args.mono:

        apply_mono(strokes, args.mono)



    steps, plan_stats = scheduler.build_plan(strokes)

    if args.limit:

        steps = apply_limit(steps, int(args.limit))

    scheduler.print_plan(plan_stats)



    waypoints, job_stats = jobmod.steps_to_waypoints(steps, mapper)

    if not waypoints:

        print("[!] 没有生成轨迹点")

        return 1

    print_summary(waypoints, plan_stats, job_stats, mapper)



    reach_ok, reach_problems = check_waypoints(waypoints)

    if not reach_ok:

        print("[!] 轨迹超出可达范围（可改 CanvasConfig.ROBOT_CORNERS 或 ReachConfig）:")

        for item in reach_problems:

            print("   ", item)

        if args.execute:

            print("[!] 拒绝执行。")

            return 2



    if not args.no_preview:

        preview_path = args.preview_out or os.path.join(str(OUTPUT_DIR), JobConfig.PREVIEW_PNG)

        jobmod.render_preview(work, strokes, mapper.mm_per_px, preview_path)

        print("[输出] 预览图: %s" % preview_path)

        if args.diagnostics:

            station_path = os.path.join(os.path.dirname(preview_path), "paint_stations.png")

            jobmod.render_station_map(mapper, station_path)

            print("[输出] 站点图: %s" % station_path)

        if args.dump_labels:

            label_path = os.path.join(os.path.dirname(preview_path), JobConfig.LABELS_PNG)

            vision.render_label_map(stats["labels"], label_path)

            print("[输出] 分区图: %s" % label_path)



    job_path = args.job_out or os.path.join(str(OUTPUT_DIR), JobConfig.JOB_JSON)

    meta = {

        "image": os.path.abspath(args.image),

        "image_size": list(info["prepared_size"]),

        "mm_per_px": round(float(mapper.mm_per_px), 5),

        "canvas_corners": [list(p) for p in CanvasConfig.ROBOT_CORNERS],

        "paper_z": ArmConfig.PAPER_Z,

        "strokes": plan_stats["strokes"],

        "length_mm": round(plan_stats["length_mm"], 1),

        "colors": [{"name": g["name"], "layer": g["layer"], "strokes": g["strokes"]}

                   for g in plan_stats["groups"]],

        "mono": args.mono,

    }

    jobmod.save_job(job_path, waypoints, meta)

    print("[输出] 任务文件: %s（%d 个轨迹点）" % (job_path, len(waypoints)))



    if not args.execute:

        print("")

        print("以上只是预览，没有连接机械臂。确认预览图没问题后加 --execute 上纸。")

        return 0



    if not reach_ok:

        return 2



    if not args.yes:

        print("")

        print("即将上纸绘制：%d 个轨迹点，%d 条笔触。请确认纸已固定、笔已装好、急停在手边。"

              % (len(waypoints), plan_stats["strokes"]))

        try:

            answer = input("确认开始？(输入 yes 继续): ").strip().lower()

        except EOFError:

            print("(非交互环境，视为取消；要直接执行请加 --yes)")

            return 0

        if answer != "yes":

            print("已取消。")

            return 0



    from robot import E6Robot

    robot = E6Robot()

    ok, message = robot.connect()

    print("[机器人] %s" % message)

    if not ok:

        return 3

    try:

        first = waypoints[0]

        robot.move_safe(first["x"], first["y"], transit_z(), 40)

        started = time.time()

        count = jobmod.execute_waypoints(robot, waypoints)

        last = waypoints[-1]

        robot.move_safe(last["x"], last["y"], transit_z(), 40)

        print("[完成] 共下发 %d 个轨迹点，用时 %.1f 分钟"

              % (count, (time.time() - started) / 60.0))

    except Exception as exc:

        print("[!] 绘制中断: %s" % exc)

        print("    机械臂保持当前位置。建议手动抬笔/复位后检查纸面。")

        return 4

    finally:

        # 非交互环境（脚本/计划任务）下 input() 会抛 EOFError，

        # 默认按"保持使能"处理，不能让它在绘制完之后崩掉。

        try:

            answer = input("是否断使能机械臂？(y/n，默认 n): ").strip().lower()

        except EOFError:

            answer = "n"

            print("(非交互环境，默认保持使能)")

        robot.close(disable=(answer == "y"))

    return 0





# =====================================================================

# 水墨框架（Pass 1）

# =====================================================================

def cmd_shuimo(args):

    # launcher invokes main repeatedly in one process: CLI overrides are local.

    original_press = ShuimoConfig.PRESS_MIN_MM

    original_skip = JobConfig.SKIP_PIGMENT_ACTIONS

    try:

        return _cmd_shuimo(args)

    finally:

        ShuimoConfig.PRESS_MIN_MM = original_press

        JobConfig.SKIP_PIGMENT_ACTIONS = original_skip





def _cmd_shuimo(args):

    """

    水墨框架流程：图像 -> 墨线骨架 -> 带逐点笔压的长笔触 -> 排程 -> 轨迹。



    和 --mode color（色块铺色）完全独立：

      - 不做 KMeans 分区，只提墨线

      - 每条笔触每个点各有自己的下压量和速度（一笔之内有粗细浓淡）

      - 不用换色循环，只按笔程补蘸

    """

    if args.press_min is not None:

        press_min = float(args.press_min)

        press_max = float(ShuimoConfig.PRESS_MAX_MM)

        if not np.isfinite(press_min) or press_min < 0.0 or press_min > press_max:

            print("[!] --press-min 必须在 0.0 到 %.2f mm 之间。" % press_max)

            return 2

        ShuimoConfig.PRESS_MIN_MM = press_min

        print("[水墨] 本次最轻笔压覆盖为 %.2f mm（配置文件默认值未修改）" % press_min)



    skip_pigment = bool(args.no_pigment)

    # 同 cmd_paint：必须双向赋值，否则 --no-pigment 会漏到下一次完整绘制

    JobConfig.SKIP_PIGMENT_ACTIONS = skip_pigment

    problems = validate(need_stations=not skip_pigment)

    if problems:

        print("[!] 配置问题:")

        for item in problems:

            print("   ", item)

        if args.execute:

            print("[!] 存在配置问题，拒绝执行。先修好配置（或加 --no-pigment 跳过蘸墨）。")

            return 2



    try:

        image = vision.load_image(args.image)

    except ValueError as exc:

        print("[!] %s" % exc)

        return 2



    profile = getattr(args, "shuimo_profile", "classic")

    balanced = getattr(args, "ink_policy", "legacy") == "balanced"

    if balanced and profile != "mountain":

        print("[!] balanced 蘸墨策略目前只支持 mountain 线稿。")

        return 2

    if balanced and skip_pigment:

        print("[!] balanced 需要启用蘸墨，不能与 --no-pigment 合用。")

        return 2

    work, info = vision.prepare_image(image, work_size=1800 if profile == "mountain" else None)

    mapper = CanvasMap(info["prepared_size"], info["image_corners"])

    print("[图像] 工作图 %d x %d（留白填充=%s）"

          % (info["prepared_size"][0], info["prepared_size"][1], info["padded"]))

    print("[图像] %s" % mapper.summary())



    if profile == "mountain":
        detail = getattr(args, "mountain_detail", "original-d")
        if detail == "original-d":
            from original_d_adapter import build_strokes
            strokes, stats = build_strokes(work, mapper.mm_per_px, pad=info.get("pad"), source_image=image)
        elif detail == "legacy-source":
            from legacy_mountain_adapter import build_strokes
            strokes, stats = build_strokes(work, mapper.mm_per_px, pad=info.get("pad"))
        else:
            import mountain
            strokes, stats = mountain.build_strokes(work, mapper.mm_per_px, pad=info.get("pad"),
                                                   detail=detail)
    else:

        strokes, stats = shuimo.build_framework_strokes(

            work, mapper.mm_per_px, pad=info.get("pad"))

    if balanced:

        import ink_control

        stats["semantic_strokes"] = len(strokes)

        strokes = ink_control.split_long_strokes(strokes, mapper.mm_per_px, ink_control.InkPolicy())

        strokes = ink_control.order_for_ink(strokes, mapper.mm_per_px, ink_control.InkPolicy())

        stats["ink_split_count"] = len(strokes) - stats["semantic_strokes"]

        stats["strokes"] = len(strokes)

        stats["length_mm"] = sum(s["length_mm"] for s in strokes)

        stats["layers"] = {k: sum(s["layer"] == k for s in strokes) for k in range(3)}

    shuimo.print_stats(stats)

    if not strokes:

        print("[!] 没有生成任何墨线笔触，见上面的诊断。")

        return 1



    start = int(args.start or 0)

    if start < 0:

        print("[!] --start 不能为负数。")

        return 2

    if start or args.limit:

        end = start + int(args.limit) if args.limit else len(strokes)

        strokes = strokes[start:end]

        if not strokes:

            print("[!] 分段范围内没有笔触：start=%d limit=%d"

                  % (start, int(args.limit or 0)))

            return 2

        stats["length_mm"] = sum(s["length_mm"] for s in strokes)

        stats["strokes"] = len(strokes)

        stats["layers"] = {k: sum(s.get("layer") == k for s in strokes) for k in range(3)}

        stats["mean_width_mm"] = float(np.mean([s["width_mm"] for s in strokes]))

        stats["mean_press_mm"] = float(np.mean([s["press_mean"] for s in strokes]))

        stats["max_press_mm"] = max(s["press_max"] for s in strokes)

        print("[分段] 绘制原任务第 %d~%d 条笔触，共 %d 条（总长 %.0f mm）"

              % (start + 1, start + len(strokes), len(strokes),

                 stats["length_mm"]))



    pal_index = shuimo.ink_palette_index()

    print("[水墨] 用色：%s（调色板第 %d 格）" % (PALETTE[pal_index]["name"], pal_index))



    base_strokes = strokes

    if balanced:

        steps, plan_stats = ink_control.build_plan(strokes, pal_index, mapper.mm_per_px)

        strokes = [step["stroke"] for step in steps if step["kind"] == "stroke"]

        stats["mean_width_mm"] = float(np.mean([s["width_mm"] for s in strokes]))

        stats["mean_press_mm"] = float(np.mean([s["press_mean"] for s in strokes]))

        stats["max_press_mm"] = max(s["press_max"] for s in strokes)

        policy_values = plan_stats['ink_policy']
        print("[蘸墨] 长度×估计笔宽 + 墨龄 + 停顿联合判断；初蘸%.1fs / 补蘸%.1fs / 沥墨%.1fs" %
              (policy_values['initial_dip_s'], policy_values['refresh_dip_s'], policy_values['drip_s']))
        print("[蘸墨] 以下为未实测模型预测；运行时会按实际时钟重新判断补蘸。")

    else:

        steps, plan_stats = scheduler.build_framework_plan(strokes, pal_index)

    scheduler.print_framework_plan(plan_stats)



    waypoints, job_stats = jobmod.steps_to_waypoints(steps, mapper)

    if balanced:

        from run_mountain_v4_robot import safe_transfers

        waypoints = safe_transfers(waypoints)

    if not waypoints:

        print("[!] 没有生成轨迹点")

        return 1

    print_summary(waypoints, plan_stats, job_stats, mapper)



    reach_ok, reach_problems = check_waypoints(waypoints)

    if not reach_ok:

        print("[!] 轨迹超出可达范围（可改 CanvasConfig.ROBOT_CORNERS 或 ReachConfig）:")

        for item in reach_problems:

            print("   ", item)

        if args.execute:

            print("[!] 拒绝执行。")

            return 2



    base_dir = str(OUTPUT_DIR)

    if not args.no_preview:

        preview_path = args.preview_out or os.path.join(

            base_dir, ShuimoConfig.PREVIEW_PNG)

        shuimo.render_framework_preview(work, strokes, mapper.mm_per_px, preview_path,

                                         mapper=mapper, waypoints=waypoints,

                                         diagnostics=args.diagnostics)

        print("[输出] 预览图: %s" % preview_path)



    job_path = args.job_out or os.path.join(base_dir, ShuimoConfig.JOB_JSON)

    meta = {

        "mode": "shuimo-framework",

        "profile": profile,

        "quality_stats": stats,

        "preview_width_calibrated": False,

        "ink_policy": "balanced" if balanced else "legacy",

        "actual_contact_press_range_mm": [

            min(p for s in strokes for p in s["press"]),

            max(p for s in strokes for p in s["press"])],

        "pass": 1,

        "image": os.path.abspath(args.image),

        "image_size": list(info["prepared_size"]),

        "mm_per_px": round(float(mapper.mm_per_px), 5),

        "canvas_corners": [list(p) for p in CanvasConfig.ROBOT_CORNERS],

        "paper_z": ArmConfig.PAPER_Z,

        "strokes": plan_stats["strokes"],

        "length_mm": round(plan_stats["length_mm"], 1),

        "dips": plan_stats["dips"],

        "ink": PALETTE[pal_index]["name"],

        "press_range_mm": [float(ShuimoConfig.PRESS_MIN_MM),

                           float(ShuimoConfig.PRESS_MAX_MM)],

        "mean_width_mm": round(stats.get("mean_width_mm", 0.0), 3),

    }

    if balanced:

        meta["ink_plan"] = plan_stats

        meta["ink_execution"] = {

            "base_strokes": base_strokes, "policy": plan_stats["ink_policy"],

            "mm_per_px": mapper.mm_per_px,

            "calibration": {"rpy": list(ArmConfig.RPY), "user": ArmConfig.USER_INDEX,

                            "tool": ArmConfig.TOOL_INDEX, "speed_factor": ArmConfig.SPEED_FACTOR,

                            "palette": PALETTE},

        }

    jobmod.save_job(job_path, waypoints, meta)

    print("[输出] 任务文件: %s（%d 个轨迹点）" % (job_path, len(waypoints)))



    if not args.execute:

        print("")

        print("以上只是预览，没有连接机械臂。确认预览图没问题后加 --execute 上纸。")

        return 0



    if not args.yes:

        print("")

        print("即将上纸勾墨线：%d 个轨迹点，%d 条笔触，蘸墨 %d 次。"

              % (len(waypoints), plan_stats["strokes"], plan_stats["dips"]))

        print("请确认纸已固定、狼毫已装好、墨已调好、急停在手边。")

        try:

            answer = input("确认开始？(输入 yes 继续): ").strip().lower()

        except EOFError:

            print("(非交互环境，视为取消；要直接执行请加 --yes)")

            return 0

        if answer != "yes":

            print("已取消。")

            return 0



    if balanced:

        from run_mountain_balanced import execute_reviewed_job

        return execute_reviewed_job(job_path)



    from robot import E6Robot

    robot = E6Robot()

    ok, message = robot.connect()

    print("[机器人] %s" % message)

    if not ok:

        return 3

    try:

        first = waypoints[0]

        robot.move_safe(first["x"], first["y"], transit_z(), 40)

        started = time.time()

        count = jobmod.execute_waypoints(robot, waypoints)

        last = waypoints[-1]

        robot.move_safe(last["x"], last["y"], transit_z(), 40)

        print("[完成] 框架共下发 %d 个轨迹点，用时 %.1f 分钟"

              % (count, (time.time() - started) / 60.0))

        print("[下一步] 框架已干，可以准备上色（Pass 2）。")

    except Exception as exc:

        print("[!] 绘制中断: %s" % exc)

        print("    机械臂保持当前位置。建议手动抬笔/复位后检查纸面。")

        return 4

    finally:

        # 非交互环境（脚本/计划任务）下 input() 会抛 EOFError，

        # 默认按"保持使能"处理，不能让它在绘制完之后崩掉。

        try:

            answer = input("是否断使能机械臂？(y/n，默认 n): ").strip().lower()

        except EOFError:

            answer = "n"

            print("(非交互环境，默认保持使能)")

        robot.close(disable=(answer == "y"))

    return 0





# =====================================================================

# 入口

# =====================================================================

def build_parser():

    parser = argparse.ArgumentParser(

        description="机械臂绘画：图像 -> 笔触 -> E6 轨迹（色块铺色 / 水墨框架）")

    parser.add_argument("--mode", choices=("color", "shuimo", "color-pass", "watercolor", "painting"), default="color",

                        help="color=原有色块铺色；shuimo=水墨框架；color-pass=对齐墨线的离线上色；watercolor=参考图式山水淡彩离线设计")

    parser.add_argument("--check", action="store_true", help="配置自检（不连机械臂）")

    parser.add_argument("--stages", default="color,outline", help="painting 功能顺序：color、outline、color,outline、outline,color")

    parser.add_argument("--strokes-per-region", type=int, default=2, help="稀疏铺色每个色块的笔数上限（正整数，默认2）；通用入口按面积铺色")

    parser.add_argument("--sparse-brush-width", type=float, default=1., help="稀疏上色估计笔宽 mm")

    parser.add_argument("--color-press", type=float, default=.22, help="上色候选下压量 mm，需试纸标定")

    parser.add_argument("--color-draw-speed", type=int, default=15, help="上色候选行笔 v 百分比")

    parser.add_argument("--existing-ink", action="store_true", help="上色设计需避让图中已有墨线")

    parser.add_argument("--probe", action="store_true", help="空跑画布四角与中心")

    parser.add_argument("--calibrate-z", action="store_true", help="交互标定纸面高度")

    parser.add_argument("--calibrate-ink", action="store_true",

                        help="交互标定墨面高度（加了墨/水之后重跑）")

    parser.add_argument("--brush-test", action="store_true",

                        help="笔宽标定：画 6 条笔压递增的横线，量宽度填 BRUSHES")

    parser.add_argument("--bt-x0", type=float, default=40.0, help="笔宽标定线段起点 X")

    parser.add_argument("--bt-x1", type=float, default=170.0, help="笔宽标定线段终点 X")

    parser.add_argument("--bt-y0", type=float, default=60.0, help="笔宽标定第一条线的 Y")

    parser.add_argument("--bt-step", type=float, default=20.0, help="笔宽标定行距 mm")

    parser.add_argument("--bt-presses", default="",

                        help="笔宽标定用指定笔压（逗号分隔 mm），例如 0.22,0.28,0.5,0.7,0.95")

    parser.add_argument("--teach", action="store_true",

                        help="引导式标定：逐点记录画布/水盂/擦笔/调色盘坐标并生成配置")

    parser.add_argument("--test-strokes", action="store_true",

                        help="画笔宽标定线（--mode shuimo 时用 ShuimoConfig.BRUSHES）")

    parser.add_argument("--image", help="输入图片路径")

    parser.add_argument("--ink-job", help="color-pass 使用的已生成 balanced 山体墨线任务")

    parser.add_argument("--ink-image", help="color-pass 使用的原始墨线图；任务中的旧路径无效时指定")

    parser.add_argument("--color-brush-width", type=float, default=3.0,

                        help="color-pass 估计笔宽 mm；实画前必须量笔宽")

    parser.add_argument("--color-ink-gap", type=float, default=1.0,

                        help="color-pass 与墨线之间的保留空隙 mm")

    parser.add_argument("--watercolor-ink-gap", type=float, default=.6,

                        help="watercolor 新构图设计的墨线留白 mm；不是当前纸张的实机安全间隙")

    parser.add_argument("--execute", action="store_true", help="真正连接机械臂执行")

    parser.add_argument("--mono", nargs="?", const="墨黑", default=None,

                        help="单色模式（可跟颜色名），并跳过洗笔/蘸色")

    parser.add_argument("--line-art", action="store_true",

                        help="水墨线稿预设（兰花/竹叶这类细线+大量留白）")

    parser.add_argument("--no-pigment", action="store_true", help="跳过洗笔/蘸色/试色")

    parser.add_argument("--limit", type=int, default=0, help="只绘制前 N 条笔触")

    parser.add_argument("--start", type=int, default=0,

                        help="水墨分段绘制：跳过前 N 条笔触后继续")

    parser.add_argument("--press-min", type=float, default=None,

                        help="水墨本次运行的最轻下压量 mm（不修改配置文件）")

    parser.add_argument("--shuimo-profile", choices=("classic", "mountain"), default="classic",

                        help="山水专用分层、留白筛选；classic 保持通用提取")

    parser.add_argument("--mountain-detail", choices=("standard", "enriched", "structured", "legacy-source", "original-d"), default="original-d",
                        help="original-d 使用 D 原版取线（默认）；enriched 增加源线；structured 组织山面；legacy-source 用旧版源图骨架、墨宽和密度生成更多细皴")
    parser.add_argument("--ink-policy", choices=("legacy", "balanced"), default="legacy",

                        help="balanced 试验策略：短蘸、时间/耗墨前瞻、停顿后刷新")

    parser.add_argument("--no-preview", action="store_true", help="不生成预览图")

    parser.add_argument("--diagnostics", action="store_true",

                        help="额外另存诊断图：水墨的分面板图、色块的站点图、"

                             "分模块/梅花的单模块文件、淡彩的纯路径图（默认都不出）")

    parser.add_argument("--dump-labels", action="store_true", help="输出彩色分区图")

    parser.add_argument("--job-out", help="任务 JSON 输出路径")

    parser.add_argument("--preview-out", help="预览图输出路径")

    parser.add_argument("--yes", action="store_true", help="跳过上纸前的确认")

    return parser





def main(argv=None):

    parser = build_parser()

    args = parser.parse_args(argv)



    if args.mode in ("watercolor", "painting") and any((args.execute, args.probe, args.calibrate_z,

            args.calibrate_ink, args.brush_test, args.teach, args.test_strokes)):

        print("[!] watercolor / painting 只做离线设计，不能与机器人运动/标定参数组合。")

        return 2



    if args.check:

        return cmd_check()

    if args.probe:

        return cmd_probe(args)

    if args.calibrate_z:

        return cmd_calibrate_z(args)

    if args.calibrate_ink:

        return cmd_calibrate_ink(args)

    if args.brush_test:

        return cmd_brush_test(args)

    if args.teach:

        return cmd_teach(args)

    if args.test_strokes:

        return cmd_test_strokes(args)

    if not args.image:

        parser.print_help()

        print("")

        print("提示：先用 python main.py --check 自检，再用 --image 出预览图。")

        return 0

    if args.mode == "painting":

        if args.ink_job:

            print("[!] painting 使用统一参考图坐标，不能套用旧墨线任务。")

            return 2

        from painting_workflow import build_plan, save_outputs

        from color_profile import ColorProfile

        try:

            plan, work = build_plan(args.image, args.stages, args.sparse_brush_width,

                args.watercolor_ink_gap, args.strokes_per_region,

                ColorProfile(press_mm=args.color_press, draw_velocity_percent=args.color_draw_speed),

                args.existing_ink)

            outputs = save_outputs(plan, work,

                                   args.job_out or str(OUTPUT_DIR / "painting_plan.json"),

                                   None if args.no_preview else args.preview_out or str(OUTPUT_DIR / "painting_preview.png"),

                                   split_stages=args.diagnostics)

        except (ValueError, OSError, KeyError) as exc:

            print("[!] 分模块规划失败:", exc)

            return 2

        print("[功能顺序]", " → ".join(plan['stages']))

        for stage, module in plan['modules'].items():

            print("[%s] %d 笔" % (stage, module['stroke_count']))

        for output in outputs: print("[输出]", output.resolve())

        print("[说明] 每块最多两笔；轻压/速度是待试纸标定的上色参数，未启动机械臂。")

        return 0

    if args.mode == "watercolor":

        if args.execute or args.ink_job:

            print("[!] watercolor 是新构图离线设计；不能直接套用已有纸张/墨线任务或执行机器人。")

            return 2

        from pathlib import Path

        import watercolor

        from color_pass import save_plan

        plan_path = Path(args.job_out or str(OUTPUT_DIR / "watercolor_plan.json"))

        preview_path = Path(args.preview_out or str(OUTPUT_DIR / "watercolor_preview.png"))

        paths_path = preview_path.with_name(preview_path.stem + "_paths.png") if args.diagnostics else None

        outputs = [plan_path] + ([] if args.no_preview else ([preview_path] + ([paths_path] if paths_path else [])))

        if len({p.resolve() for p in outputs}) != len(outputs) or any(p.exists() for p in outputs):

            print("[!] 输出路径重复或文件已存在；请指定新的计划/预览路径。")

            return 2

        try:

            plan, work = watercolor.build_plan(args.image, args.color_brush_width, args.watercolor_ink_gap)

            if not args.no_preview:

                preview_path.parent.mkdir(parents=True, exist_ok=True)

                watercolor.render(work, plan, preview_path, paths_path)

            save_plan(plan_path, plan)

        except (ValueError, OSError, KeyError) as exc:

            print("[!] 水彩规划失败:", exc)

            return 2

        print("[水彩设计] %d笔，%.1f mm；蓝/黄/红，淡染→主体→强调。" %

              (len(plan['strokes']), sum(s['length_mm'] for s in plan['strokes'])))

        print("[计划]", plan_path.resolve())

        if not args.no_preview:

            print("[色面预览]", preview_path.resolve())

            if paths_path: print("[路径预览]", paths_path.resolve())

        print("[限制] 浓淡为图像目标；需实测笔宽/配色，不生成压笔或机器人指令。")

        return 0

    if args.mode == "color-pass":

        if args.execute:

            print("[!] 彩色第二遍目前只有离线设计计划；颜料工位和洗笔未验证，拒绝 --execute。")

            return 2

        if not args.ink_job:

            print("[!] color-pass 需要 --ink-job 指定已生成的墨线任务。")

            return 2

        from color_pass import build_plan, render_preview, save_plan

        from pathlib import Path

        plan_path = Path(args.job_out or str(OUTPUT_DIR / "color_pass_plan.json"))

        preview_path = Path(args.preview_out or str(OUTPUT_DIR / "color_pass_preview.png"))

        if not args.no_preview and plan_path.resolve() == preview_path.resolve():

            print("[!] 设计计划和预览图必须使用不同路径。")

            return 2

        if plan_path.exists() or (not args.no_preview and preview_path.exists()):

            print("[!] 输出文件已存在；请用新的 --job-out / --preview-out 路径。")

            return 2

        try:

            plan, ink_work = build_plan(args.ink_job, args.image,

                                        args.color_brush_width, args.color_ink_gap,

                                        args.ink_image)

            if not args.no_preview:

                preview_path.parent.mkdir(parents=True, exist_ok=True)

                render_preview(ink_work, plan["strokes"], plan["mm_per_px"], preview_path)

            save_plan(plan_path, plan)

        except (ValueError, OSError, KeyError, FileExistsError) as exc:

            print("[!] 彩色第二遍规划失败:", exc)

            return 2

        counts = {name: sum(s["color"] == name for s in plan["strokes"])

                  for name in ("red", "yellow", "blue")}

        print("[彩色第二遍] 离线笔触 %d，红/黄/蓝 %d/%d/%d。" %

              (len(plan["strokes"]), counts["red"], counts["yellow"], counts["blue"]))

        print("[输出] 设计计划:", plan_path.resolve())

        if not args.no_preview:

            print("[输出] 叠色预览:", preview_path.resolve())

        print("[限制] 计划没有机器人轨迹；需先标定颜料、笔宽和洗笔效果。")

        return 0

    if args.mode == "shuimo":

        return cmd_shuimo(args)

    return cmd_paint(args)





if __name__ == "__main__":

    sys.exit(main())

