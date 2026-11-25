# -*- coding: utf-8 -*-
"""
接触标定（跑完即删）：在纸上画 4 条压力递增的线
====================================================================
目的：查清"笔尖从多深开始真正接触纸面并留下痕迹"。
  - 如果 4 条都留痕         -> PAPER_Z 偏低（笔尖一开始就压着了）
  - 如果第 2/3 条开始留痕    -> 正常，把那个压力当作 PRESS_MIN_MM
  - 如果 4 条都不留痕        -> PAPER_Z 偏高（笔尖根本没碰到纸），要往大调

★ 必须在纸上操作，且笔上要有【带颜色的颜料】。
"""

# --- path bootstrap: this script now lives in 绘画/tools/, put 绘画/ on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import time

from config import ArmConfig, z_for_press, z_for_lift, transit_z
from robot import E6Robot

# 4 条线的下压量（mm）—— 覆盖"很轻"到"明显压开"
PRESSES = [0.2, 0.5, 1.0, 1.6]

# 画在纸的左下角空处（避开刚才那幅兰花所在的中间区域）
X0, X1 = 20.0, 80.0
Y_START = 50.0
Y_GAP = 17.0

robot = E6Robot()
ok, message = robot.connect()
print("[连接] %s" % message)
if not ok:
    raise SystemExit(3)

pose = robot.get_pose()
print("[当前] %s" % (pose,))
print("[纸面] PAPER_Z = %.2f   RPY = %s" % (ArmConfig.PAPER_Z, ArmConfig.RPY))
print("")
print("即将画 %d 条线：X %.0f -> %.0f，Y 从 %.0f 起每 %.0f mm 一条"
      % (len(PRESSES), X0, X1, Y_START, Y_GAP))
for i, p in enumerate(PRESSES):
    print("   第 %d 条: Y=%.0f   下压 %.1f mm   ->  Z=%.3f"
          % (i + 1, Y_START + i * Y_GAP, p, z_for_press(p)))
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
    return True


try:
    for i, press in enumerate(PRESSES):
        y = Y_START + i * Y_GAP
        z = z_for_press(press)
        print(">>> 第 %d 条  Y=%.1f  下压 %.1f mm  Z=%.3f" % (i + 1, y, press, z))
        if not move(X0, y, transit_z(), 30, "移动到起点上方"):
            continue
        if not move(X0, y, z, 15, "落笔"):
            continue
        if not move(X1, y, z, 15, "行笔"):
            continue
        if not move(X1, y, z_for_lift(10.0), 30, "抬笔"):
            continue
        print("    OK")
    print("")
    print("=" * 66)
    print(" 4 条线画完了。请看纸上：")
    print("   从下往上第 1 条 = 下压 0.2mm")
    print("   第 2 条 = 0.5mm    第 3 条 = 1.0mm    第 4 条 = 1.6mm")
    print("")
    print(" 判断：")
    print("   4 条都有明显痕迹   -> PAPER_Z 偏低，往上调（Z 变小）")
    print("   第 2~3 条开始留痕   -> 正常！把那个压力填进 ShuimoConfig.PRESS_MIN_MM")
    print("   4 条都不留痕        -> PAPER_Z 偏高（笔尖没碰到纸），往下调（Z 变大）")
    print("=" * 66)
finally:
    try:
        robot.move_safe(pose[0], pose[1], transit_z(), 40)
    except Exception:
        pass
    robot.close(disable=False)
