# -*- coding: utf-8 -*-
"""
逐点试探（跑完即删）
====================================================================
把 --probe 的 5 个目标拆成 7 小步，每步单独报告，失败就停：
  A 原位抬笔 5mm        -> 验证 Z 方向约定 + 基本运动通路（最小风险）
  B 当前XY 升到站点间高度 -> 验证 transit_z
  C/D/E/F 四个角         -> 逐个验证可达性
  G 中心

★ 会真的运动！执行时请盯着机械臂、手放在急停上。
★ 逆解算错误（报警17）是在运动之前就被拒绝的，不会产生危险动作，
  但"被接受"的指令会真实移动，所以仍然要有人看着。
"""

# --- path bootstrap: this script now lives in 绘画/tools/, put 绘画/ on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import time

from config import ArmConfig, CanvasConfig, transit_z
from robot import E6Robot

robot = E6Robot()
ok, message = robot.connect()
print("[连接] %s" % message)
if not ok:
    raise SystemExit(3)

print("[姿态] RPY = %s" % (ArmConfig.RPY,))
print("[速度] SpeedFactor = %d" % ArmConfig.SPEED_FACTOR)
print("[高度] transit_z = %.2f   PAPER_Z = %.2f" % (transit_z(), ArmConfig.PAPER_Z))

pose = robot.get_pose()
print("[当前] %s" % (pose,))
if pose is None:
    robot.close(disable=False)
    raise SystemExit(4)
cur = list(pose)

corners = [tuple(p) for p in CanvasConfig.ROBOT_CORNERS]
cx = sum(p[0] for p in corners) / 4.0
cy = sum(p[1] for p in corners) / 4.0
tz = transit_z()

STEPS = [
    ("A 原位抬笔 5mm  (验证 Z 方向)", cur[0], cur[1], cur[2] - 5.0),
    ("B 当前XY -> 站点间高度",        cur[0], cur[1], tz),
    ("C 左下角",                      corners[3][0], corners[3][1], tz),
    ("D 左上角",                      corners[0][0], corners[0][1], tz),
    ("E 右上角",                      corners[1][0], corners[1][1], tz),
    ("F 右下角",                      corners[2][0], corners[2][1], tz),
    ("G 画布中心",                    cx, cy, tz),
]

failed = None
for name, x, y, z in STEPS:
    print("")
    print(">>> %s" % name)
    print("    目标 (%.2f, %.2f, %.2f)  v=20 cp=0" % (x, y, z))
    reply = robot.move_linear(x, y, z, 20, 0)
    print("    应答 %s" % reply)
    if not robot._ok(reply):
        print("    x 指令被拒（控制器直接返回错误）")
        failed = name
        break
    if not robot.wait_idle(60):
        print("    x 等待到位失败: RobotMode=%d" % robot.get_mode())
        print("      GetErrorID = %s" % robot.get_error_id())
        failed = name
        break
    print("    OK 到位 -> %s" % (robot.get_pose(),))

print("")
if failed:
    print("[!] 在「%s」失败。清报警..." % failed)
    robot.clear_error()
    time.sleep(1.5)
    print("    RobotMode = %d" % robot.get_mode())
    print("    GetErrorID = %s" % robot.get_error_id())
    print("    GetPose   = %s" % (robot.get_pose(),))
    print("    机械臂未离开原位（逆解算错误发生在运动之前）" if
          failed.startswith(("A", "B", "C", "D", "E", "F", "G")) else "")
else:
    print("=== 全部 %d 步通过：坐标与姿态都可用 ===" % len(STEPS))

robot.close(disable=False)
