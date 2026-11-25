# -*- coding: utf-8 -*-
"""导出当前配置（跑完即删）"""

# --- path bootstrap: this script now lives in 绘画/tools/, put 绘画/ on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import (
    ArmConfig, CanvasConfig, ReachConfig, StationConfig, JobConfig,
    ShuimoConfig, VisionConfig, PALETTE, transit_z, station_approach_z,
    dip_position, palette_surface_z, dip_depth_of, dip_time_of,
)

P = ArmConfig.PAPER_Z

print("##### ArmConfig")
for k in ("IP", "PORT", "USER_INDEX", "TOOL_INDEX", "PAPER_Z", "Z_INCREASES_DOWN",
          "LIFT_MM", "TRANSIT_LIFT_MM", "SPEED_FACTOR", "RPY",
          "CONNECT_TIMEOUT", "REPLY_TIMEOUT", "ENABLE_DELAY"):
    print("  %-20s = %r" % (k, getattr(ArmConfig, k, None)))
print("  %-20s = %.2f" % ("transit_z()", transit_z()))

print("##### CanvasConfig")
print("  ROBOT_CORNERS = %r" % (CanvasConfig.ROBOT_CORNERS,))
print("  IMAGE_CORNERS = %r" % (CanvasConfig.IMAGE_CORNERS,))
print("  FIT_MODE      = %r" % (getattr(CanvasConfig, "FIT_MODE", None),))

print("##### ReachConfig")
print("  Z_MIN=%.1f  Z_MAX=%.1f" % (ReachConfig.Z_MIN, ReachConfig.Z_MAX))

print("##### StationConfig")
for k in ("STATIONS_ARE_PLACEHOLDER", "APPROACH_LIFT_MM", "DIP_DRIP_PAUSE_S",
          "DIP_MM", "DIP_PAUSE_S", "PALETTE_SURFACE_Z"):
    print("  %-26s = %r" % (k, getattr(StationConfig, k, None)))
print("  %-26s = %.2f" % ("station_approach_z()", station_approach_z()))
print("  WASH       = %r" % (getattr(StationConfig, "WASH", None),))
print("  WIPE       = %r" % (getattr(StationConfig, "WIPE", None),))
print("  TEST_STRIP = %r" % (getattr(StationConfig, "TEST_STRIP", None),))

print("##### PALETTE")
for i, e in enumerate(PALETTE):
    print("  [%d] %r" % (i, e))
    print("      dip=%r  surface_z=%.4f  dip_mm=%.1f  dip_time=%.1f"
          % (dip_position(i), palette_surface_z(i), dip_depth_of(i), dip_time_of(i)))

print("##### ShuimoConfig")
for k in sorted(dir(ShuimoConfig)):
    if k.isupper():
        print("  %-28s = %r" % (k, getattr(ShuimoConfig, k)))

print("##### JobConfig")
for k in ("CONTINUOUS_PATH_RATIO", "MOVE_DELAY", "MOTION_TIMEOUT", "TRANSIT_SPEED",
          "DOWN_SPEED", "SKIP_PIGMENT_ACTIONS", "TEST_AFTER_DIP",
          "WASH_BEFORE_FIRST_COLOR"):
    print("  %-28s = %r" % (k, getattr(JobConfig, k, None)))

print("##### 关键推导高度（越负越高）")
ink = palette_surface_z(0)
for nm, z in (("站点接近高度（横移）", station_approach_z()),
              ("站点间高度", transit_z()),
              ("抬笔高度", P - ArmConfig.LIFT_MM),
              ("纸面 PAPER_Z", P),
              ("最深笔压", P + ShuimoConfig.PRESS_MAX_MM),
              ("墨面 surface_z", ink),
              ("蘸墨插入点", ink + dip_depth_of(0))):
    print("  %-22s %9.2f" % (nm, z))
