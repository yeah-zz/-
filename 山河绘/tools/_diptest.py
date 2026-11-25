# -*- coding: utf-8 -*-
"""
蘸墨动作单独验证（跑完即删）—— 不画任何东西
====================================================================
直接调用项目自己的 job.dip_waypoints(0)，走真实代码路径：
    dip-in   抬到站点间高度，横移到墨盘正上方
    dip      垂直下降到"墨面 + dip_mm"，停留 dip_time_s
    dip-out  垂直抬回站点间高度

★ 调色板已精简成只有一格墨（index 0）。
★ 机械臂会动，但只在墨盘那一小片区域上下动，不会碰纸。
"""

# --- path bootstrap: this script now lives in 绘画/tools/, put 绘画/ on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import time

import job as jobmod
from config import (PALETTE, ArmConfig, ReachConfig,
                    dip_position, palette_surface_z,
                    dip_depth_of, dip_time_of, transit_z)
from robot import E6Robot

IDX = 0
idx = IDX
x, y = dip_position(idx)
surface = palette_surface_z(idx)
depth = dip_depth_of(idx)
dwell = dip_time_of(idx)

print("=" * 70)
print(" 蘸墨动作验证")
print("=" * 70)
print("  目的地   墨盘中心 (%.4f, %.4f)" % (x, y))
print("  墨面高度 %.4f" % surface)
print("  插入深度 %.1f mm  ->  目标 Z = %.4f" % (depth, surface + depth))
print("  停留     %.1f 秒" % dwell)
print("  接近高度 transit_z = %.2f   (比墨面高 %.2f mm)"
      % (transit_z(), surface - transit_z()))
print("  可达范围 ReachConfig Z = [%.1f, %.1f]" % (ReachConfig.Z_MIN, ReachConfig.Z_MAX))
for nm, z in (("transit", transit_z()), ("插入点", surface + depth)):
    ok = ReachConfig.Z_MIN <= z <= ReachConfig.Z_MAX
    print("     %-8s %8.2f  -> %s" % (nm, z, "OK" if ok else "x 超范围，会被拒绝！"))
print("")

robot = E6Robot()
ok, message = robot.connect()
print("[连接] %s" % message)
if not ok:
    raise SystemExit(3)

start = robot.get_pose()
print("[当前] %s" % (start,))
print("")

waypoints = jobmod.dip_waypoints(idx)
print("蘸墨轨迹 %d 个点（由 job.dip_waypoints 生成）:" % len(waypoints))
for w in waypoints:
    print("   %-9s (%9.3f, %9.3f, %9.3f)  v=%2.0f  pause=%.1f"
          % (w["tag"], w["x"], w["y"], w["z"], w["velocity"], w["pause"]))
print("")

try:
    for i, w in enumerate(waypoints):
        print(">>> [%d/%d] %s  ->  (%.2f, %.2f, %.2f)  v=%.0f"
              % (i + 1, len(waypoints), w["tag"], w["x"], w["y"], w["z"], w["velocity"]))
        reply = robot.move_linear(w["x"], w["y"], w["z"], w["velocity"], 0)
        if not robot._ok(reply):
            print("    x 指令被拒: %s" % reply)
            break
        if not robot.wait_idle(60):
            print("    x 等待到位失败 mode=%d %s"
                  % (robot.get_mode(), robot.get_error_id()))
            robot.clear_error()
            break
        got = robot.get_pose()
        if got:
            print("    OK 实际 (%.3f, %.3f, %.3f)" % (got[0], got[1], got[2]))
        if float(w.get("pause", 0.0)) > 0:
            print("    停留 %.1f 秒（让笔毛吸墨）..." % w["pause"])
            time.sleep(min(10.0, float(w["pause"])))
    print("")
    print("=" * 70)
    print(" 蘸墨完成。请确认：")
    print("   ① 机械臂是不是跑到了墨盘正上方？")
    print("   ② 笔尖是不是插进墨里了（插深 2.5mm）？")
    print("   ③ 笔毛上有没有沾到墨？")
    print("=" * 70)
finally:
    try:
        if start:
            robot.move_safe(start[0], start[1], transit_z(), 40)
            print("[机器人] 已回到起始位置上方")
    except Exception as exc:
        print("[!] 回位失败: %s" % exc)
    robot.close(disable=False)
