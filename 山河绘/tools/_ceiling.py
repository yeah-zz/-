# -*- coding: utf-8 -*-
"""
可达高度上限 + 墨盘横移余量（跑完即删）
====================================================================
目的：盘沿目测比墨面高 22mm 以上，而 transit_z=-87.88 贴着盘沿，
      必须找出"机械臂到底能抬多高"，再用最大余量横移到墨盘上方。

步骤（前两步只往上走，安全）：
  1) 在当前位置逐步往上抬，找到逆解算能接受的最高点
  2) 在这个最高点横移到墨盘正上方   ★ 唯一有风险的一步：如果盘沿比
     机械臂能抬的最高点还高就会碰上 —— 请盯着、手放急停
  3) 在墨盘位置再测一次最高点
"""

# --- path bootstrap: this script now lives in 绘画/tools/, put 绘画/ on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import time

from config import ReachConfig, dip_position, transit_z
from robot import E6Robot

STEP = 5.0
MAX_STEPS = 14

robot = E6Robot()
ok, message = robot.connect()
print("[连接] %s" % message)
if not ok:
    raise SystemExit(3)

cur = robot.get_pose()
if cur is None:
    print("[!] 读不到位姿")
    robot.close(disable=False)
    raise SystemExit(4)
x0, y0, z0 = cur[0], cur[1], cur[2]
tx, ty = dip_position(0)
print("[当前]   (%.3f, %.3f, %.3f)  RPY=%s" % (x0, y0, z0, tuple(cur[3:])))
print("[墨盘]   (%.3f, %.3f)   当前 transit_z = %.2f" % (tx, ty, transit_z()))
print("[限制]   ReachConfig.Z_MIN = %.1f（最高不许超过它）" % ReachConfig.Z_MIN)
print("")


def move(x, y, z, v=20, note=""):
    reply = robot.move_linear(x, y, z, v, 0)
    if not robot._ok(reply):
        print("      x 被拒: %s" % reply)
        return False
    if not robot.wait_idle(60):
        print("      x mode=%d %s" % (robot.get_mode(), robot.get_error_id()))
        robot.clear_error()
        time.sleep(1.2)
        return False
    got = robot.get_pose()
    print("      OK z=%.2f  %s" % (got[2] if got else float("nan"), note))
    return True


print("=" * 74)
print(" 步骤 1：在当前位置往上找上限（只往上，安全）")
print("=" * 74)
best_z = z0
limit = max(ReachConfig.Z_MIN, z0 - STEP * MAX_STEPS)
z = z0
while z - STEP >= limit - 1e-6:
    z -= STEP
    print("  [%.1f] 目标 z=%.2f  (比起点高 %.1f)" % (STEP, z, z0 - z))
    if move(x0, y0, z, 15, "(抬高)"):
        best_z = z
    else:
        print("      -> 上限在 %.2f ~ %.2f 之间" % (best_z, z))
        break
print("")
print(">>> 当前位置可达最高 Z = %.2f   （比原 transit_z %.2f 高 %.2f mm）"
      % (best_z, transit_z(), transit_z() - best_z))

print("")
print("=" * 74)
print(" 步骤 2：在这个最高点横移到墨盘正上方")
print("=" * 74)
print("  ★ 这一步可能碰到盘沿。请盯着机械臂、手放急停！")
print("  目标 (%.3f, %.3f, %.2f)  横移距离 %.0f mm"
      % (tx, ty, best_z, ((tx - x0) ** 2 + (ty - y0) ** 2) ** 0.5))
time.sleep(2.0)
lateral_ok = move(tx, ty, best_z, 12, "(横移到墨盘上方)")

print("")
print("=" * 74)
print(" 步骤 3：在墨盘位置再测一次上限")
print("=" * 74)
tray_ceiling = best_z if lateral_ok else None
if lateral_ok:
    z = best_z
    while z - STEP >= limit - 1e-6:
        z -= STEP
        print("  [%.1f] 目标 z=%.2f" % (STEP, z))
        if move(tx, ty, z, 15, "(墨盘处抬高)"):
            tray_ceiling = z
        else:
            print("      -> 墨盘处上限在 %.2f ~ %.2f 之间" % (tray_ceiling, z))
            break

print("")
print("=" * 74)
print(" 汇总")
print("=" * 74)
print("  当前位置最高可达 Z      = %.2f" % best_z)
print("  横移到墨盘上方          = %s" % ("成功" if lateral_ok else "失败/被拒"))
if tray_ceiling is not None:
    print("  墨盘位置最高可达 Z      = %.2f" % tray_ceiling)
    ink = -65.5916
    print("")
    print("  墨面 = %.2f" % ink)
    print("  机械臂在墨盘处最高可抬到 %.2f -> 距墨面 %.2f mm" % (tray_ceiling, ink - tray_ceiling))
    print("")
    print("  ★ 可用的 TRANSIT_LIFT_MM（相对纸面 %.2f）:" % -52.88)
    print("     最大可用 = %.1f  （transit_z = %.2f，正好到上限）"
          % (-52.88 - tray_ceiling, tray_ceiling))
    print("     留 5mm 余量建议 = %.1f" % (-52.88 - tray_ceiling - 5.0))
    print("     ★★ 前提：盘沿必须低于 %.2f，否则还是撞" % tray_ceiling)

print("")
print("[机器人] 最后位置: %s" % (robot.get_pose(),))
robot.close(disable=False)
