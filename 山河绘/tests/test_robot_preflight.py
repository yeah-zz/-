"""Offline preflight checks: useful errors without bypassing motion guards."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import run_mountain_balanced as runner
from config import ArmConfig
from robot import RobotError


class PreflightTests(unittest.TestCase):
    def test_mismatch_reports_actual_target_and_deltas(self):
        wrong = (ArmConfig.RPY[0] - 10.9898, ArmConfig.RPY[1] - 2.4736, ArmConfig.RPY[2] + 36.6726)
        with self.assertRaises(RobotError) as caught:
            runner.check_orientation((36.2728, 97.5738, -97.0668, *wrong))
        message = str(caught.exception)
        for value in ("%.4f" % wrong[0], "%.4f" % ArmConfig.RPY[0], "%.4f" % ArmConfig.RPY[1],
                      "0.1500", "User 2 / Tool 0", "no motion sent"):
            self.assertIn(value, message)

    def test_valid_rounded_pose_and_small_error_still_pass(self):
        runner.check_orientation((35.7, 80.36, -100.08, *(v + .02 for v in ArmConfig.RPY)))
        runner.check_orientation((0.,0.,0.,*(v+.14 for v in ArmConfig.RPY)))
        with self.assertRaises(RobotError):
            runner.check_orientation((0.,0.,0.,*(v+.16 for v in ArmConfig.RPY)))

    def test_missing_short_and_nonfinite_poses_rejected(self):
        for pose in (None, (0,0,0), (0,0,0,float('nan'),0,0), (0,0,0,0,0,float('inf'))):
            with self.assertRaises(RobotError): runner.check_orientation(pose)

    def fake_robot(self, correct=False):
        robot = MagicMock()
        robot.get_mode.return_value = 5
        robot.get_error_id.return_value = '0,{[[],[],[],[],[],[],[]]},GetErrorID()'
        robot.get_pose.return_value = (0,0,-100,*(ArmConfig.RPY if correct
                                                  else tuple(v + 10. for v in ArmConfig.RPY)))
        robot.send.side_effect = AssertionError("Must not send setup or motion commands")
        return robot

    def test_readonly_probe_never_sets_or_moves(self):
        for correct, code in ((False,4),(True,0)):
            robot = self.fake_robot(correct)
            with patch.object(runner, 'E6Robot', return_value=robot), \
                 patch.object(runner, 'connect_readonly'), \
                 patch.object(runner, 'load_reviewed', return_value=([{}],None,None,{})), \
                 patch.object(runner, 'Motion', side_effect=AssertionError('No motion')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.check_robot_readonly('unused.json'), code)
            robot.send.assert_not_called()
            robot.close.assert_called_once_with(disable=False)

    def test_execute_mismatch_stops_before_setup_and_motion(self):
        robot = self.fake_robot()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner, 'E6Robot', return_value=robot), \
             patch.object(runner, 'connect_readonly'), \
             patch.object(runner, 'load_reviewed', return_value=([{}],None,None,{})), \
             patch.object(runner, 'Motion', side_effect=AssertionError('No motion')), \
             contextlib.redirect_stdout(io.StringIO()):
            job_path = Path(directory)/'test.json'
            legacy = Path(directory)/'test_execution_log.json'
            legacy.write_text('historical success', encoding='utf-8')
            self.assertEqual(runner.execute_reviewed_job(job_path),4)
            self.assertEqual(runner.execute_reviewed_job(job_path),4)
            self.assertEqual(legacy.read_text(encoding='utf-8'),'historical success')
            self.assertEqual(len(list((Path(directory)/'test_runs').glob('*_execution_log.json'))),2)
            self.assertFalse((Path(directory)/'test_actual.json').exists())
        robot.send.assert_not_called()
        self.assertEqual(robot.close.call_count, 2)
        robot.close.assert_any_call(disable=False)


if __name__ == '__main__': unittest.main()
