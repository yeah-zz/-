"""Color-only candidate motion settings; never mutate ink/recovery settings."""
from dataclasses import asdict, dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class ColorProfile:
    # All speeds/accelerations below are controller percentages, not mm/s.
    # Historical successful runs used AccJ/AccL=20 and CP=70. Those runs do
    # not establish a universal cause of vibration or a safe minimum speed.
    press_mm: float = 2.50
    draw_velocity_percent: int = 20
    travel_velocity_percent: int = 30  # Legacy fallback; explicit axes below take precedence.
    dip_velocity_percent: int = 20
    global_speed_factor: int = 70
    # Zero means inherited controller setting for legacy optional setup only.
    accel_joint_percent: int = 20
    accel_linear_percent: int = 20
    draw_velocity_max: int = 45
    ink_draw_speed_scale: float = 1.4
    # CP is used only between drawing segments. Cup/air/contact transitions
    # remain exact-stop points to preserve vertical and horizontal clearance.
    continuous_path_ratio: int = 70
    interstroke_lift_mm: float = 5.0
    acceleration_required: bool = False
    # Reserved, not executed as per-point sleeps.
    touchdown_pause_s: float = 0.
    intermediate_point_pause_s: float = 0.
    calibrated_on_paper: bool = False
    touchdown_velocity_percent: Optional[int] = None
    lift_velocity_percent: Optional[int] = None
    horizontal_velocity_percent: Optional[int] = None
    approach_velocity_percent: Optional[int] = None

    def to_dict(self):
        if type(self.acceleration_required) is not bool:
            raise ValueError('Acceleration confirmation flag must be boolean')
        for value, limit in ((self.touchdown_velocity_percent, 60),
                             (self.lift_velocity_percent, 80),
                             (self.horizontal_velocity_percent, 80)):
            if value is not None and (type(value) is not int or not 1 <= value <= limit):
                raise ValueError('Color motion velocity outside its allowed range')
        if self.approach_velocity_percent is not None and (type(self.approach_velocity_percent) is not int
                or not 1 <= self.approach_velocity_percent <= 60):
            raise ValueError('Color vertical approach velocity outside its allowed range')
        if not math.isfinite(self.interstroke_lift_mm) or not 2.5 <= self.interstroke_lift_mm <= 15:
            raise ValueError('Color interstroke lift must be in [2.5, 15] mm')
        if not math.isfinite(self.press_mm) or not 0 <= self.press_mm <= 2.50:
            raise ValueError('Sparse color pressure must be in [0, 2.50] mm')
        # 2026-10-10: requested doubling of web paper speed (48 -> 96).
        # MovL v supports 1..100 (robot.py); defaults remain unchanged.
        # These are command ratios, not measured physical drawing speeds.
        if not 1 <= self.draw_velocity_percent <= 100:
            raise ValueError('Color drawing velocity must be in [1, 100] percent')
        if not 1 <= self.travel_velocity_percent <= 80:
            raise ValueError('Travel velocity must be in [1, 80] percent')
        if not 1 <= self.dip_velocity_percent <= 30:
            raise ValueError('Dip velocity must be in [1, 30] percent')
        if not 1 <= self.global_speed_factor <= 100:
            raise ValueError('Global speed factor must be in [1, 100] percent')
        if not 5 <= self.draw_velocity_max <= 100:
            raise ValueError('Per-point draw velocity cap must be in [5, 100] percent')
        if not .5 <= self.ink_draw_speed_scale <= 2.0:
            raise ValueError('Ink speed scale must be in [0.5, 2.0]')
        if not 0 <= self.continuous_path_ratio <= 100:
            raise ValueError('Continuous path ratio must be in [0, 100] percent')
        if not 0 <= self.accel_joint_percent <= 100:
            raise ValueError('Joint acceleration ratio must be in [0, 100] percent')
        if not 0 <= self.accel_linear_percent <= 100:
            raise ValueError('Linear acceleration ratio must be in [0, 100] percent')
        if self.acceleration_required and any(type(v) is not int or not 1 <= v <= 100 for v in
                                             (self.accel_joint_percent, self.accel_linear_percent)):
            raise ValueError('Required acceleration must be explicitly set in [1, 100] percent')
        return asdict(self)
