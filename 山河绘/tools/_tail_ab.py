# -*- coding: utf-8 -*-
"""
出锋 A/B 对照（跑完即删）
====================================================================
在同一次运行里画 4 条线，交替开/关"收笔出锋"，直接对比收笔处的墨迹：

    Y=50   有出锋  (TAIL_MM=4)      <- 收笔应该是一段渐细的"锋"
    Y=68   无出锋  (TAIL_MM=0)      <- 收笔应该是一个墨疙瘩（原行为）
    Y=86   有出锋  (TAIL_MM=4)
    Y=104  无出锋  (TAIL_MM=0)

★ 关键：这里走的是**真实的 job.stroke_waypoints**（用一个恒等 mapper 把
  已经是机器人坐标的点原样传进去），所以出锋逻辑和正式绘制完全一致。

★ 需要在纸上操作，笔上要有【带颜色的颜料】。
"""

# --- path bootstrap: this script now lives in 绘画/tools/, put 绘画/ on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import time

import numpy as np

import job as jobmod
from config import ArmConfig, ShuimoConfig, transit_z
from robot import E6Robot


class IdentityMapper:
    """把已是机器人坐标的点原样返回，让标定脚本也能走真实的 stroke_waypoints"""

    @staticmethod
    def to_robot(points):
        return np.asarray(points, dtype=np.float64).reshape(-1, 2)


PRESS = 1.6          # 4 条线都用同样的笔压，只对比收笔
                     # （用 1.6mm = 实测最饱满的那一档；1.0 太轻，差异看不出来）
X0, X1 = 20.0, 130.0  # 110mm 长，尾巴看得清楚些
ROWS = [
    (50.0, ShuimoConfig.TAIL_MM, "有出锋"),
    (68.0, 0.0, "无出锋（对照）"),
    (86.0, ShuimoConfig.TAIL_MM, "有出锋"),
    (104.0, 0.0, "无出锋（对照）"),
]

robot = E6Robot()
ok, message = robot.connect()
print("[连接] %s" % message)
if not ok:
    raise SystemExit(3)

start_pose = robot.get_pose()
print("[当前] %s" % (start_pose,))
print("[纸面] PAPER_Z = %.2f" % ArmConfig.PAPER_Z)
print("")


def send(wps):
    for w in wps:
        reply = robot.move_linear(w["x"], w["y"], w["z"], w["velocity"], 0)
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
    for y, tail, label in ROWS:
        n = 12
        pts = [(X0 + (X1 - X0) * i / float(n - 1), y) for i in range(n)]
        stroke = {
            "points": pts,
            "press": [PRESS] * n,
            "speed": [18.0] * n,
            "pal_index": 10,
            "name": "标定",
            "lift_mm": float(ArmConfig.LIFT_MM),
            "brush": ShuimoConfig.DEFAULT_BRUSH,
        }
        # 临时改 TAIL_MM，让 stroke_waypoints 按本次要求生成
        saved = ShuimoConfig.TAIL_MM
        ShuimoConfig.TAIL_MM = float(tail)
        wps = jobmod.stroke_waypoints(IdentityMapper(), stroke)
        ShuimoConfig.TAIL_MM = saved

        tags = [w["tag"].split(":")[0] for w in wps]
        print(">>> Y=%-6.1f %-14s TAIL_MM=%.1f  轨迹点=%d (tail=%d)"
              % (y, label, tail, len(wps), tags.count("tail")))

        # 先抬到安全高度再横移到起点
        if not send([{"x": X0, "y": y, "z": transit_z(),
                      "velocity": 30, "tag": "transit"}]):
            continue
        if not send(wps):
            continue
        print("    OK")

    print("")
    print("=" * 70)
    print(" 画完了。请对比 4 条线的【右端】收笔处：")
    print("   Y=50 （下往上第1条）有出锋  -> 右端应该渐细收尾")
    print("   Y=68 （第2条）      无出锋  -> 右端应该有个墨疙瘩")
    print("   Y=86 （第3条）      有出锋")
    print("   Y=104（第4条）      无出锋")
    print("")
    print(" 如果第 1/3 条的右端明显比第 2/4 条干净 -> 出锋修好了")
    print("=" * 70)
finally:
    try:
        if start_pose:
            robot.move_safe(start_pose[0], start_pose[1], transit_z(), 40)
    except Exception:
        pass
    robot.close(disable=False)
