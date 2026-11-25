# -*- coding: utf-8 -*-
"""诊断（跑完即删）：在几个关键位姿上读关节角，看是不是"接近奇异/接近限位"。

为什么查这个：现场反馈"去墨盘那一段老是被保护性停止（黄灯 + mode 11 + 无报警码）"，
而且"关节一直抖"。这两件事的共同候选原因是【手腕接近奇异位形】—— 笛卡尔指令只动
一点点，关节却要转很多，伺服顶住 -> 力矩保护停 + 抖动。

做法：只走到我们本来就会用到的位姿（纸面上方 20mm / 杯口接近位 / 蘸笔位 / 墨盘与纸之间
的横移中点），每个点读 GetPose() 和 GetAngle()，打印 J1..J6 以及"离奇异/离限位"的余量。

安全：不使能、不清报警；任何一步 mode 不是 5/7/8 立刻停手；不碰纸（纸面上方 20mm）。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import math

from config import ArmConfig, CanvasConfig, station_approach_z, z_for_lift
from park_color_loading import target_pose
from project_paths import PAINTING_DIR
from robot import E6Robot
from run_mountain_v4_robot import no_errors

m = json.loads((PAINTING_DIR / 'pigment_measurements.json').read_text(encoding='utf-8-sig'))
black, red = m['black'], m['red']
park = target_pose()
corners = CanvasConfig.ROBOT_CORNERS
cx = sum(p[0] for p in corners) / 4.0
cy = sum(p[1] for p in corners) / 4.0
above = ArmConfig.PAPER_Z - 20.0          # 纸面上方 20mm（安全，不碰纸）
appr = station_approach_z()
mid = ((cx + black['x']) / 2.0, (cy + black['y']) / 2.0)

POSES = [
    ("停车点/墨盘接近位（cup-high 就在这）", (park[0], park[1], park[2])),
    ("墨盘蘸笔位", (black['x'], black['y'], black['surface_z'] + black['dip_mm'])),
    ("红杯接近位", (red['x'], red['y'], appr)),
    ("红杯蘸笔位", (red['x'], red['y'], red['surface_z'] + red['dip_mm'])),
    ("纸中心上方 20mm", (cx, cy, above)),
    ("纸左上角上方 20mm", (corners[0][0], corners[0][1], above)),
    ("纸右下角上方 20mm", (corners[2][0], corners[2][1], above)),
    ("墨盘↔纸 横移中点", (mid[0], mid[1], appr)),
]

robot = E6Robot()
ok, msg = robot.connect()
print("[连接] %s" % msg)
if not ok:
    raise SystemExit(3)

def angles():
    reply = robot.send('GetAngle()')
    if not robot._ok(reply):
        return None
    body = reply[reply.find('{') + 1:reply.rfind('}')]
    try:
        return [float(v) for v in body.replace('[', '').replace(']', '').split(',') if v.strip() != '']
    except ValueError:
        return None

def safe_move(x, y, z, v=30):
    cur = robot.get_pose()
    if cur is None:
        print("    读不到当前位姿，停"); return False
    for name, tx, ty, tz, tv in (("垂直抬到接近高度", cur[0], cur[1], appr, v),
                                 ("水平移到目标", x, y, appr, v),
                                 ("垂直下降到目标", x, y, z, max(10, v // 2))):
        reply = robot.move_linear(tx, ty, tz, tv, 0)
        if not robot._ok(reply):
            print("    x %s 被拒: %s" % (name, reply)); return False
        if not robot.wait_idle(60):
            print("    x %s 等待到位失败 mode=%s" % (name, robot.get_mode())); return False
        mode = robot.get_mode()
        if mode not in (5, 7, 8) or not no_errors(robot.get_error_id()):
            print("    x %s 后状态异常 mode=%s alarms=%s" % (name, mode, robot.get_error_id())); return False
    return True

print("")
print("%-30s %8s %8s %8s %8s %8s %8s   判定" % ("位姿", "J1", "J2", "J3", "J4", "J5", "J6"))
print("-" * 108)
print("(停车点先用安全三步走过去)")
results = []
for label, (x, y, z) in POSES:
    print(">>> %s  -> (%.2f, %.2f, %.2f)" % (label, x, y, z))
    if not safe_move(x, y, z):
        print("    中途停止，不再继续探点")
        break
    a = angles()
    if a is None or len(a) < 6:
        print("    读不到关节角:", a); continue
    j5 = abs(a[4])
    warn = []
    if min(j5, 180.0 - j5) < 10.0:
        warn.append("★J5=%.1f° 接近手腕奇异" % a[4])
    for i, v in enumerate(a):
        if abs(v) > 165.0:
            warn.append("★J%d=%.1f° 接近限位" % (i + 1, v))
    print("%-30s %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f   %s"
          % (label, a[0], a[1], a[2], a[3], a[4], a[5], " ".join(warn) or "OK"))
    results.append((label, a, warn))

print("")
print("=" * 108)
bad = [(l, w) for l, _a, w in results if w]
if bad:
    print(" 有风险的位姿：")
    for l, w in bad:
        print("   %s：%s" % (l, " ".join(w)))
else:
    print(" 所有探点都没接近奇异/限位 —— 那一段（去墨盘）的问题更可能是线缆或摩擦。")
print("=" * 108)

print("")
print(">>> 回停车点")
safe_move(park[0], park[1], park[2], 30)
robot.close(disable=False)