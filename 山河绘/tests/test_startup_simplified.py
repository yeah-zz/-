import unittest
from unittest.mock import patch
import contextlib
import io
from config import ArmConfig, CanvasConfig, ShuimoConfig, station_approach_z
from experiment_start import validate_start
from park_color_loading import target_pose
from robot import RobotError
import unified_painting


class SimplifiedStartTests(unittest.TestCase):
    def center(self, z=None):
        corners = CanvasConfig.ROBOT_CORNERS
        return (sum(p[0] for p in corners)/4, sum(p[1] for p in corners)/4,
                ArmConfig.PAPER_Z if z is None else z, *ArmConfig.RPY)

    def test_paper_contact_start_lifts_vertically(self):
        pose = self.center()
        end = validate_start(pose, require_park=False)
        self.assertEqual(end[:2], pose[:2])
        self.assertEqual(end[3:], pose[3:])
        self.assertEqual(end[2], station_approach_z())

    def test_already_high_start_never_lowers(self):
        sign = 1 if ArmConfig.Z_INCREASES_DOWN else -1
        pose = self.center(station_approach_z() - sign * 5)
        self.assertEqual(validate_start(pose, require_park=False), pose)

    def test_deep_start_still_rejected(self):
        sign = 1 if ArmConfig.Z_INCREASES_DOWN else -1
        with self.assertRaises(RobotError):
            validate_start(self.center(ArmConfig.PAPER_Z + sign * (ShuimoConfig.PRESS_MAX_MM + 1)), require_park=False)

    def test_rotation_and_out_of_reach_still_rejected(self):
        pose = self.center()
        for bad in ((1e6, *pose[1:]), (*pose[:3], pose[3]+1, *pose[4:])):
            with self.assertRaises(RobotError):
                validate_start(bad, require_park=False)

    def test_strict_recovery_parking_unchanged(self):
        pose = target_pose()
        validate_start(pose)
        with self.assertRaises(RobotError):
            validate_start(self.center())

    def test_single_start_confirmation(self):
        plan = {'modules': {'color': {'strokes': []}}}
        with contextlib.redirect_stdout(io.StringIO()):
            runner = unittest.mock.Mock(return_value=0)
            ask = unittest.mock.Mock(side_effect=AssertionError('Duplicate prompt'))
            self.assertEqual(unified_painting.execute_stages(plan, 'unused', {},
                ask=ask, ink_runner=runner, start_confirmed=True), 0)
            runner.assert_called_once_with('unused')


if __name__ == '__main__':
    unittest.main()
