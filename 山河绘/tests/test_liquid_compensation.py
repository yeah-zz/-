from dataclasses import replace
import unittest
from unittest.mock import patch
from pigment_station import PigmentStation
from config import ArmConfig


class LiquidCompensationTests(unittest.TestCase):
    def station(self, **kw):
        return PigmentStation(x=-105.4972,y=152.3188,surface_z=-18.3481,rim_z=-43.7398,**kw)

    def test_disabled_keeps_original_target(self):
        s=self.station(level_step_mm=.1,level_max_mm=.5)
        self.assertEqual(s.poses(10000),s.poses(0))

    def test_boundary_cap_and_both_axis_directions(self):
        s=self.station(level_compensation_enabled=True,level_every_dips=10,level_step_mm=.1,level_max_mm=.25)
        self.assertEqual(s.level_offset(9),0)
        self.assertEqual(s.level_offset(10),.1)
        self.assertEqual(s.level_offset(30),.25)
        self.assertEqual(s.level_offset(1000),.25)
        for down,sign in [(True,1),(False,-1)]:
            with patch.object(ArmConfig,'Z_INCREASES_DOWN',down):
                self.assertAlmostEqual(s.poses(10)[1][2]-s.poses(0)[1][2],sign*.1)
                self.assertEqual(s.poses(10)[0],s.poses(0)[0])

    def test_invalid_parameters_and_unreachable_maximum(self):
        for kw in [dict(level_every_dips=0),dict(level_step_mm=-1),
                   dict(level_compensation_enabled=True),dict(level_max_mm=float('nan'))]:
            with self.assertRaises(ValueError):self.station(**kw).validate()
        with self.assertRaises(ValueError):
            self.station(level_compensation_enabled=True,level_step_mm=.1,level_max_mm=10000).validate()

    def test_instances_do_not_share_counts(self):
        s=self.station(level_compensation_enabled=True,level_step_mm=.1,level_max_mm=.5)
        self.assertEqual(s.level_offset(20),.2)
        self.assertEqual(replace(s).level_offset(0),0)
