"""Recovery motion checks with mocked hardware only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from config import ArmConfig, station_approach_z
from robot import RobotError
import prepare_robot as prep

# 现场实测的"纸内、已经抬起来"的起始位姿。高度和姿态角一律相对当前标定生成：
# 写死旧值时，换笔/重标姿态后这些夹具会变成"已经在站点高位/差 30° 以上"，
# 把 raise / rotate 两步的回归测反。
START = (36.2728, 97.5738, station_approach_z() + 20.,
         ArmConfig.RPY[0] - 6., ArmConfig.RPY[1] - 1., ArmConfig.RPY[2] + 8.)


class PrepareTests(unittest.TestCase):
    def test_aligned_four_mm_pose_only_lifts_without_any_rotation(self):
        # 只比"纸面上方 4 mm"高一点点、姿态已对齐 -> 只允许竖直抬到站点高位
        pose = (121.57, 157.81, ArmConfig.PAPER_Z - 4.5, *ArmConfig.RPY)
        moves = prep.plan_recovery(pose)
        self.assertEqual(moves,[("raise-with-current-orientation",
                                (*pose[:2],station_approach_z(),*pose[3:]))])

    def test_near_paper_branch_rejects_misalignment_and_lower_start(self):
        for pose in ((121.57,157.81,-44.08,*START[3:]),
                     (121.57,157.81,ArmConfig.PAPER_Z-3.9,*ArmConfig.RPY),
                     (121.57,157.81,ArmConfig.PAPER_Z,*ArmConfig.RPY)):
            with self.assertRaises(RobotError): prep.plan_recovery(pose)
        # Just inside the preflight tolerance must never add a rotation step.
        pose = (121.57,157.81,-44.08,*(v+.14 for v in ArmConfig.RPY))
        self.assertEqual(len(prep.plan_recovery(pose)),1)
        self.assertEqual(prep.plan_recovery(pose)[0][1][3:],pose[3:])

    def test_vertical_lift_preserves_actual_orientation_then_rotates(self):
        moves = prep.plan_recovery(START)
        self.assertEqual(len(moves),2)
        self.assertEqual(moves[0][1], (*START[:2],station_approach_z(),*START[3:]))
        self.assertEqual(moves[1][1], (*START[:2],station_approach_z(),*ArmConfig.RPY))

    def test_no_lowering_from_higher_start_and_ready_is_noop(self):
        high = (*START[:2],station_approach_z()-5,*START[3:])
        moves = prep.plan_recovery(high)
        self.assertEqual(len(moves),1)
        self.assertEqual(moves[0][1][2],high[2])
        self.assertEqual(prep.plan_recovery((*high[:3],*ArmConfig.RPY)),[])

    def test_low_bowl_large_rotation_and_invalid_poses_rejected(self):
        for p in (None, START[:3], (*START[:2],ArmConfig.PAPER_Z-5,*START[3:]),
                  (-79.2551,30.9546,START[2],*START[3:]),
                  (*START[:3],ArmConfig.RPY[0]+21,*ArmConfig.RPY[1:]),
                  (*START[:3],float('nan'),0,0), (1000,1000,-100,*ArmConfig.RPY)):
            with self.assertRaises(RobotError): prep.plan_recovery(p)

    def test_default_only_reads(self):
        robot = MagicMock()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(prep,'load_reviewed'), patch.object(prep,'E6Robot',return_value=robot), \
             patch.object(prep,'connect_readonly'), patch.object(prep,'read_idle_pose',return_value=START), \
             patch.object(prep,'move_to') as move, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prep.prepare('job.json',log_dir=directory),0)
        robot.send.assert_not_called()
        move.assert_not_called()
        robot.close.assert_called_once_with(disable=False)

    def test_execute_requires_clearance_before_connect(self):
        with patch.object(prep,'connect_readonly') as connect:
            with self.assertRaises(ValueError): prep.prepare('job.json',execute=True)
            connect.assert_not_called()

    def test_execute_sequence_and_arrival_failure_stops_rotation(self):
        high = prep.plan_recovery(START)[0][1]
        final = prep.plan_recovery(START)[1][1]
        robot = MagicMock()
        robot.send.return_value = '0,{},SpeedFactor(10);'
        robot._ok.return_value = True
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(prep,'load_reviewed'), patch.object(prep,'E6Robot',return_value=robot), \
             patch.object(prep,'connect_readonly'), \
             patch.object(prep,'read_idle_pose',side_effect=[START,START,START,high,final]), \
             patch.object(prep,'move_to') as move, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prep.prepare('job.json',True,True,log_dir=directory),0)
            self.assertEqual([call.args[1] for call in move.call_args_list],[high,final])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(prep,'load_reviewed'), patch.object(prep,'E6Robot',return_value=robot), \
             patch.object(prep,'connect_readonly'), patch.object(prep,'read_idle_pose',return_value=START), \
             patch.object(prep,'move_to',side_effect=RobotError('not arrived')) as move, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prep.prepare('job.json',True,True,log_dir=directory),4)
            self.assertEqual(move.call_count,1)

    def test_first_idle_does_not_count_as_arrival(self):
        target = prep.plan_recovery(START)[0][1]
        robot = MagicMock()
        robot.get_mode.return_value = 5
        robot.get_error_id.return_value = '0,{[]},GetErrorID()'
        robot.get_pose.side_effect = [START,target]
        with patch.object(prep.time,'sleep') as sleep:
            prep.wait_arrival(robot,target)
        self.assertEqual(sleep.call_count,1)

    def test_explicit_current_orientation_command_not_global_rpy(self):
        high = prep.plan_recovery(START)[0][1]
        robot = MagicMock()
        robot._ok.return_value = True
        with patch.object(prep,'wait_arrival') as wait:
            prep.move_to(robot,high)
        command = robot.send.call_args.args[0]
        # 抬笔阶段必须原样带上"当前实际姿态"（不能用标定姿态去替换）
        self.assertIn(','.join('%.4f' % v for v in high[3:]),command)
        self.assertNotIn(','.join('%.4f' % v for v in ArmConfig.RPY),command)
        self.assertIn('user=2,tool=0,v=10,cp=0',command)
        wait.assert_called_once_with(robot,high)

    def test_mode_and_alarm_are_visible_and_block_motion(self):
        for mode, alarms, phrase in (
            (5, '1,{[123]},GetErrorID()', '报警存在'),
            (4, '0,{[]},GetErrorID()', 'RobotMode=4'),
            (-1, '1,{},Control Mode Is Not Tcp', 'TCP 模式'),
            (5, 'broken', '报警存在'),
        ):
            robot = MagicMock()
            robot.get_mode.return_value = mode
            robot.get_error_id.return_value = alarms
            robot.get_pose.return_value = START
            output = io.StringIO()
            with contextlib.redirect_stdout(output), self.assertRaises(RobotError) as caught:
                prep.read_idle_pose(robot)
            self.assertIn(phrase, str(caught.exception))
            self.assertIn(str(mode), output.getvalue())
            self.assertIn(str(START), output.getvalue())
            robot.send.assert_not_called()

    def test_arrival_alarm_stops_without_later_move(self):
        robot = MagicMock()
        robot.get_mode.return_value = 7
        robot.get_error_id.return_value = '1,{[42]},GetErrorID()'
        with self.assertRaisesRegex(RobotError, 'Recovery alarm'):
            prep.wait_arrival(robot, prep.plan_recovery(START)[0][1])
        robot.get_pose.assert_not_called()

    def test_readonly_record_is_unique_and_contains_plan(self):
        robot = MagicMock()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(prep, 'load_reviewed'), patch.object(prep, 'E6Robot', return_value=robot), \
             patch.object(prep, 'connect_readonly'), patch.object(prep, 'read_idle_pose', return_value=START), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prep.prepare('job.json', log_dir=directory), 0)
            self.assertEqual(prep.prepare('job.json', log_dir=directory), 0)
            files = list(Path(directory).glob('prepare_*.json'))
            self.assertEqual(len(files), 2)
            data = json.loads(files[0].read_text(encoding='utf-8'))
            self.assertTrue(data['success'])
            self.assertEqual(len(data['plan']), 2)
            self.assertEqual(data['action'], 'read-only-diagnosis')
        robot.send.assert_not_called()

    def test_final_pose_drift_fails_after_motion(self):
        start = (*START[:2], prep.station_approach_z()-5, *ArmConfig.RPY)
        drift = (start[0]+1, *start[1:])
        robot = MagicMock()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(prep, 'load_reviewed'), patch.object(prep, 'E6Robot', return_value=robot), \
             patch.object(prep, 'connect_readonly'), \
             patch.object(prep, 'read_idle_pose', side_effect=[start,start,drift]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prep.prepare('job.json',True,True,log_dir=directory),4)
            data = json.loads(next(Path(directory).glob('prepare_*.json')).read_text(encoding='utf-8'))
            self.assertIn('Final pose differs', data['error'])
            self.assertFalse(data['success'])
        robot.send.assert_not_called()


if __name__ == '__main__': unittest.main()
