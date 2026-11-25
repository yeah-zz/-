"""Measured per-cup coordinates for supervised automatic replenishment."""
from dataclasses import dataclass, asdict
import math
import cv2
import numpy as np
from config import ArmConfig, CanvasConfig, station_approach_z, station_dip_z
from geometry import check_waypoints

_measurements_override = None


def read_measurements():
    """Return a frozen web-run override, or the normal shared measurements."""
    import copy
    import json
    from pathlib import Path
    if _measurements_override is not None:
        return copy.deepcopy(_measurements_override)
    return json.loads(Path(__file__).with_name('pigment_measurements.json').read_text(encoding='utf-8-sig'))


def set_measurements_override(values):
    """Process-local only: no shared configuration or running process is changed."""
    import copy
    global _measurements_override
    _measurements_override = copy.deepcopy(values)


@dataclass(frozen=True)
class PigmentStation:
    x: float
    y: float
    surface_z: float
    rim_z: float
    # 2026-09-25 实画反馈两轮：
    #   第一轮 dip_mm=0.6 / dwell_s=0.6 只湿了笔尖，纸上又淡又断；
    #   第二轮 1.8 / 1.5 又蘸太多（尤其颜料，挂得厚、落笔就积一坨）。
    # 现在改成【每杯独立】：实际数值写在 pigment_measurements.json 的
    # black/red 里（run_plum_experiment.read_station 会读），下面是兜底默认值。
    #   surface_z 是"笔尖刚好触到液面"的高度，所以这里每 1 mm 就是多扎 1 mm。
    dip_mm: float = 1.0
    dwell_s: float = 1.0
    drip_s: float = 2.0
    strokes_per_load: int = 2
    level_compensation_enabled: bool = False
    level_every_dips: int = 10
    level_step_mm: float = 0.0
    level_max_mm: float = 0.0

    level_every_length_mm: float = 0.0  # Positive selects distance instead of dip count.

    def level_offset(self, completed_dips=0, completed_length_mm=0.0):
        if type(self.level_compensation_enabled) is not bool:
            raise ValueError('Liquid compensation enabled must be true/false')
        if type(self.level_every_dips) is not int or self.level_every_dips < 1:
            raise ValueError('Liquid compensation interval must be a positive integer')
        if (not all(math.isfinite(v) and v >= 0 for v in (self.level_step_mm, self.level_max_mm))
                or (self.level_compensation_enabled and not 0 < self.level_step_mm <= self.level_max_mm)):
            raise ValueError('Enabled liquid compensation requires 0 < step <= maximum')
        if type(completed_dips) is not int or completed_dips < 0:
            raise ValueError('Completed dip count must be a nonnegative integer')
        if not math.isfinite(self.level_every_length_mm) or self.level_every_length_mm < 0:
            raise ValueError('Invalid liquid compensation distance interval')
        if not math.isfinite(completed_length_mm) or completed_length_mm < 0:
            raise ValueError('Invalid completed drawing length')
        if not self.level_compensation_enabled:
            return 0.0
        steps = (completed_length_mm // self.level_every_length_mm if self.level_every_length_mm > 0
                 else completed_dips // self.level_every_dips)
        return min(self.level_max_mm, steps * self.level_step_mm)

    def validate(self):
        if not all(math.isfinite(float(v)) for v in asdict(self).values()): raise ValueError('Invalid station values')
        if not 0<self.dip_mm<=2 or not 0<=self.dwell_s<=3 or not .5<=self.drip_s<=5:
            raise ValueError('Dip depth/time outside experiment limits')
        if type(self.strokes_per_load) is not int or not 1 <= self.strokes_per_load <= 4:
            raise ValueError('Reload every 1 to 4 strokes')
        sign=1 if ArmConfig.Z_INCREASES_DOWN else -1
        high=station_approach_z()
        if sign*(self.rim_z-high)<15 or sign*(self.surface_z-self.rim_z)<0:
            raise ValueError('Cup rim needs >=15 mm clearance below high travel; liquid below rim')
        if cv2.pointPolygonTest(np.asarray(CanvasConfig.ROBOT_CORNERS,np.float32),(self.x,self.y),True)>-20:
            raise ValueError('Pigment cup must be >=20 mm outside paper')
        poses=self.poses()
        if self.level_compensation_enabled:
            poses.append((self.x,self.y,station_dip_z(self.surface_z,self.dip_mm+self.level_max_mm),*ArmConfig.RPY))
        if not check_waypoints([dict(x=p[0],y=p[1],z=p[2]) for p in poses])[0]:
            raise ValueError('Pigment cup outside configured reach')
        return self

    def poses(self, completed_dips=0, completed_length_mm=0.0):
        pose=lambda z:(self.x,self.y,z,*ArmConfig.RPY)
        return [pose(station_approach_z()),pose(station_dip_z(self.surface_z,self.dip_mm+self.level_offset(completed_dips, completed_length_mm)))]
