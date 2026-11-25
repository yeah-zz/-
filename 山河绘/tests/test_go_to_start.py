"""go_to_start.py 的模拟测试：不连真机，只验证门限、动作顺序和记录。"""
import contextlib
import io
import json
import math
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from config import ArmConfig, CanvasConfig, station_approach_z
from park_color_loading import target_pose
from robot import RobotError
import go_to_start as gts

PARK = target_pose()
# 高度从当前标定推导：换笔/挪纸后 PAPER_Z 会平移，"比站点高位低/高"的语义必须跟着走，
# 否则这些夹具会悄悄变成"已经在高位"的用例，把 raise 步骤的回归测没了。
LOW_Z = station_approach_z() + 20.        # 比站点高位低 20 mm（离纸面仍有约 55 mm）
HIGH_Z = station_approach_z() - 20.       # 比站点高位再高 20 mm
ALIGNED_LOW = (100., 150., LOW_Z, *ArmConfig.RPY)                     # 纸内、已对齐、低于高位
MISALIGNED_LOW = (100., 150., LOW_Z, ArmConfig.RPY[0], ArmConfig.RPY[1] + 5., ArmConfig.RPY[2])
MISALIGNED_HIGH = (100., 150., HIGH_Z, ArmConfig.RPY[0], ArmConfig.RPY[1] + 5., ArmConfig.RPY[2])
# 2026-09-25 现场实测：纸内但离纸边只有 15.7 mm，姿态差 13.17°
NEAR_EDGE_LIVE = (1.6641, 50.7705, -108.74, -2.0323, 6.9059, 17.6357)


def names(moves):
    return [name for name, _ in moves]


@contextlib.contextmanager
def temp_log_dir():
    """把临时日志目录放在 tests/ 下：项目原有的 TemporaryDirectory() 在受限沙箱里
    无法在系统临时目录创建文件，这里只换位置，语义不变。"""
    base = Path(tempfile.mkdtemp(prefix='_tmp_logs_', dir=Path(__file__).resolve().parent))
    try:
        yield str(base)
    finally:
        shutil.rmtree(base, ignore_errors=True)


class PlanTests(unittest.TestCase):
    def test_aligned_low_pose_raises_then_transfers_to_park(self):
        moves = gts.plan(ALIGNED_LOW)
        self.assertEqual(names(moves),
                         ['raise-with-current-orientation', 'transfer-to-outside-paper-park'])
        self.assertEqual(moves[0][1], (100., 150., station_approach_z(), *ArmConfig.RPY))
        self.assertEqual(moves[1][1], PARK)

    def test_raise_keeps_current_orientation_before_rotating(self):
        moves = gts.plan(MISALIGNED_LOW)
        self.assertEqual(names(moves),
                         ['raise-with-current-orientation', 'restore-drawing-orientation',
                          'transfer-to-outside-paper-park'])
        self.assertEqual(moves[0][1][:3], (100., 150., station_approach_z()))
        self.assertEqual(moves[0][1][3:], MISALIGNED_LOW[3:])   # 抬笔阶段绝不动姿态
        self.assertNotEqual(moves[0][1][3:], ArmConfig.RPY)
        self.assertEqual(moves[1][1], (100., 150., station_approach_z(), *ArmConfig.RPY))

    def test_misaligned_high_rotates_transfers_then_descends_vertically(self):
        moves = gts.plan(MISALIGNED_HIGH)
        self.assertEqual(names(moves),
                         ['restore-drawing-orientation', 'transfer-to-outside-paper-park',
                          'descend-to-approach-level'])
        self.assertEqual(moves[1][1], (PARK[0], PARK[1], HIGH_Z, *ArmConfig.RPY))
        self.assertEqual(moves[2][1], PARK)

    def test_already_at_park_is_a_noop(self):
        self.assertEqual(gts.plan(PARK), [])

    def test_small_rotation_at_outside_paper_station_is_allowed(self):
        pose = (PARK[0] + 5, PARK[1] + 5, station_approach_z(),
                ArmConfig.RPY[0], ArmConfig.RPY[1] + 2., ArmConfig.RPY[2])
        self.assertEqual(names(gts.plan(pose)),
                         ['restore-drawing-orientation', 'transfer-to-outside-paper-park'])

    def test_near_edge_rotation_is_refused_with_minimal_jog_hint(self):
        # 纸内但只离边 5 mm、姿态又不一致 -> 必须拒绝，并给出"只点 X/Y 多少毫米"的提示
        right = (CanvasConfig.ROBOT_CORNERS[1][0] + CanvasConfig.ROBOT_CORNERS[2][0]) / 2.0
        pose = (right + 5., 135., HIGH_Z, 0., 0., 0.)
        with self.assertRaises(RobotError) as caught:
            gts.plan(pose)
        message = str(caught.exception)
        self.assertIn('离纸边', message)
        self.assertIn('只点动', message)

    def test_rotation_beyond_twenty_degrees_per_axis_is_refused(self):
        pose = (100., 150., HIGH_Z, ArmConfig.RPY[0], ArmConfig.RPY[1] + 21., ArmConfig.RPY[2])
        with self.assertRaisesRegex(RobotError, '20°/轴'):
            gts.plan(pose)

    def test_too_low_pose_is_refused_before_any_plan(self):
        with self.assertRaisesRegex(RobotError, '起点太低'):
            gts.plan((100., 150., ArmConfig.PAPER_Z - 3., *ArmConfig.RPY))
        with self.assertRaisesRegex(RobotError, '起点太低'):
            gts.plan((100., 150., ArmConfig.PAPER_Z - 30., *MISALIGNED_LOW[3:]))

    def test_invalid_pose_is_refused(self):
        for pose in (None, (100., 150., HIGH_Z), (100., 150., HIGH_Z, float('nan'), 0., 0.)):
            with self.assertRaises(RobotError):
                gts.plan(pose)

    def test_no_step_changes_xy_and_z_at_the_same_time(self):
        """纸外横移必须保持高度、下降必须原地：不允许斜着穿过盘沿。"""
        previous = MISALIGNED_LOW
        for name, target in gts.plan(MISALIGNED_LOW):
            xy_changed = math.hypot(target[0] - previous[0], target[1] - previous[1]) > .25
            z_changed = abs(target[2] - previous[2]) > .01
            self.assertFalse(xy_changed and z_changed, name)
            previous = target


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.robot = MagicMock()
        self.robot.send.return_value = '0,{},SpeedFactor(10);'
        self.robot._ok.return_value = True

    def _run(self, directory, poses, execute=True, confirm=True, answer='GO',
             move_side_effect=None):
        with patch.object(gts, 'E6Robot', return_value=self.robot), \
             patch.object(gts, 'connect_readonly'), \
             patch.object(gts, 'read_idle_pose', side_effect=list(poses)), \
             patch.object(gts, 'move_to', side_effect=move_side_effect) as move, \
             contextlib.redirect_stdout(io.StringIO()):
            code = gts.go(execute=execute, confirm_clearance=confirm,
                          log_dir=directory, ask=lambda prompt: answer)
        return code, move

    @staticmethod
    def _record(directory):
        return json.loads(next(Path(directory).glob('go_to_start_*.json')).read_text(encoding='utf-8'))

    def test_readonly_prints_plan_without_touching_the_robot(self):
        with temp_log_dir() as directory:
            code, move = self._run(directory, [ALIGNED_LOW], execute=False, confirm=False)
            self.assertEqual(code, 0)
            self.robot.send.assert_not_called()
            move.assert_not_called()
            self.assertFalse((Path(directory) / 'blue_wash_motion.lock').exists())
            record = self._record(directory)
            self.assertEqual(record['action'], 'check')
            self.assertTrue(record['success'])
            self.assertEqual([step['name'] for step in record['plan']],
                             ['raise-with-current-orientation', 'transfer-to-outside-paper-park'])

    def test_execute_requires_clearance_before_connecting(self):
        with patch.object(gts, 'connect_readonly') as connect, \
             self.assertRaises(ValueError):
            gts.go(execute=True, confirm_clearance=False)
        connect.assert_not_called()

    def test_execute_moves_in_planned_order_and_only_sets_speed(self):
        t1, t2 = (moves := gts.plan(ALIGNED_LOW))[0][1], moves[1][1]
        with temp_log_dir() as directory:
            code, move = self._run(directory, [ALIGNED_LOW, ALIGNED_LOW, ALIGNED_LOW, t1, t2])
            self.assertEqual(code, 0)
            self.assertEqual([call.args[1] for call in move.call_args_list], [t1, t2])
            self.assertEqual([call.args[0] for call in self.robot.send.call_args_list],
                             ['SpeedFactor(%d)' % gts.GLOBAL_SPEED_FACTOR])   # 不使能、不清报警、不复位
            self.assertFalse((Path(directory) / 'blue_wash_motion.lock').exists())
            record = self._record(directory)
            self.assertTrue(record['success'])
            self.assertEqual(record['final_pose'], list(t2))
            self.assertEqual([step['name'] for step in record['steps']],
                             ['raise-with-current-orientation', 'transfer-to-outside-paper-park'])
            self.assertNotIn('pending_move', record)

    def test_move_to_uses_the_requested_velocity_and_verifies_arrival(self):
        robot = MagicMock()
        robot.send.return_value = '0,{},MovL(...);'
        robot._ok.return_value = True
        target = (-1.0, 2.0, -3.0, *ArmConfig.RPY)
        with patch.object(gts, 'wait_arrival') as wait:
            gts.move_to(robot, target, 33)
        command = robot.send.call_args.args[0]
        self.assertIn('v=33', command)
        self.assertIn('user=2,tool=0,v=33,cp=0', command)
        self.assertIn('%.4f' % ArmConfig.RPY[0], command)
        wait.assert_called_once_with(robot, target)

    def test_custom_speed_flags_are_sent_and_recorded(self):
        t1, t2 = (moves := gts.plan(ALIGNED_LOW))[0][1], moves[1][1]
        with temp_log_dir() as directory:
            with patch.object(gts, 'E6Robot', return_value=self.robot), \
                 patch.object(gts, 'connect_readonly'), \
                 patch.object(gts, 'read_idle_pose',
                              side_effect=[ALIGNED_LOW, ALIGNED_LOW, ALIGNED_LOW, t1, t2]), \
                 patch.object(gts, 'move_to') as move, \
                 contextlib.redirect_stdout(io.StringIO()):
                code = gts.go(execute=True, confirm_clearance=True, log_dir=directory,
                              ask=lambda prompt: 'GO', speed_factor=40, velocity=35)
            self.assertEqual(code, 0)
            self.assertEqual([call.args[2] for call in move.call_args_list], [35, 35])
            self.assertEqual([call.args[0] for call in self.robot.send.call_args_list],
                             ['SpeedFactor(40)'])
            record = self._record(directory)
            self.assertEqual(record['global_speed_factor'], 40)
            self.assertEqual(record['movl_velocity_percent'], 35)

    def test_cli_rejects_out_of_range_speed(self):
        for argv in (['--speed-factor', '0'], ['--velocity', '101']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                gts.main(argv + ['--execute', '--confirm-clearance'])

    def test_cancel_answer_sends_nothing_and_releases_lock(self):
        with temp_log_dir() as directory:
            code, move = self._run(directory, [ALIGNED_LOW], answer='NO')
            self.assertEqual(code, 0)
            move.assert_not_called()
            self.robot.send.assert_not_called()
            self.assertFalse((Path(directory) / 'blue_wash_motion.lock').exists())

    def test_near_edge_live_pose_is_refused_without_motion(self):
        with temp_log_dir() as directory:
            code, move = self._run(directory, [NEAR_EDGE_LIVE])
            self.assertEqual(code, 4)
            move.assert_not_called()
            self.robot.send.assert_not_called()
            self.assertFalse((Path(directory) / 'blue_wash_motion.lock').exists())
            self.assertIn('离纸边', self._record(directory)['error'])

    def test_existing_motion_lock_blocks_a_second_script(self):
        with temp_log_dir() as directory:
            lock = Path(directory) / 'blue_wash_motion.lock'
            lock.write_text('other-script')
            code, move = self._run(directory, [ALIGNED_LOW])
            self.assertEqual(code, 4)
            move.assert_not_called()
            self.robot.send.assert_not_called()
            self.assertEqual(lock.read_text(), 'other-script')

    def test_pose_change_during_confirmation_stops_before_motion(self):
        moved = (101., 150., LOW_Z, *ArmConfig.RPY)
        with temp_log_dir() as directory:
            code, move = self._run(directory, [ALIGNED_LOW, moved])
            self.assertEqual(code, 4)
            move.assert_not_called()
            self.assertFalse((Path(directory) / 'blue_wash_motion.lock').exists())
            self.assertIn('确认期间', self._record(directory)['error'])

    def test_unconfirmed_move_keeps_lock_and_pending_move(self):
        with temp_log_dir() as directory:
            code, move = self._run(directory, [ALIGNED_LOW] * 3,
                                   move_side_effect=RobotError('arrival not confirmed'))
            self.assertEqual(code, 4)
            self.assertEqual(move.call_count, 1)
            self.assertTrue((Path(directory) / 'blue_wash_motion.lock').exists())
            record = self._record(directory)
            self.assertEqual(record['pending_move']['name'], 'raise-with-current-orientation')
            self.assertFalse(record['success'])

    def test_execute_at_park_is_a_noop_and_releases_lock(self):
        with temp_log_dir() as directory:
            code, move = self._run(directory, [PARK])
            self.assertEqual(code, 0)
            move.assert_not_called()
            self.robot.send.assert_not_called()
            self.assertFalse((Path(directory) / 'blue_wash_motion.lock').exists())
            self.assertTrue(self._record(directory)['success'])


if __name__ == '__main__':
    unittest.main()
