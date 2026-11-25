"""Offline controller model: bounded refill, path preservation and fault injection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from motion_diagnostics import MotionJournal
from queued_motion import send_batch
from robot import RobotError


class Controller:
    def __init__(self, advance=16):
        self.current = 100
        self.last = 100
        self.ids = []
        self.moves = []
        self.commands = []
        self.advance = advance
        self.peak = 0
        self.mode_override = None
        self.alarm = '0,{[]},GetErrorID()'

    def get_mode(self):
        if self.mode_override is not None:
            return self.mode_override
        return 5 if self.current == self.last else 7

    def get_error_id(self):
        return self.alarm

    def _ok(self, reply):
        return reply.startswith('0,')

    def send(self, command):
        self.commands.append(command)
        if command == 'GetCurrentCommandID()':
            later = [i for i in self.ids if i > self.current]
            if later and self.advance:
                self.current = later[min(self.advance, len(later))-1]
            return '0,{%d},GetCurrentCommandID()' % self.current
        assert command.startswith('MovL('), command
        self.last += 3  # IDs need not be consecutive; count commands, not gaps.
        self.ids.append(self.last)
        self.moves.append(command)
        # Include the executing command in the outstanding queue count.
        outstanding = sum(i >= self.current for i in self.ids)
        self.peak = max(self.peak, outstanding)
        assert outstanding <= 32, 'Overfilled controller queue'
        return '0,{%d},MovL()' % self.last


def motions(n=100):
    return [((i, 100, -36.1657, 0, 0, 0), 96, 0 if i in (0, 1, n-2, n-1) else 70)
            for i in range(n)]


class ContinuousMotionTests(unittest.TestCase):
    def test_long_stroke_keeps_path_speed_and_blends_across_old_boundaries(self):
        for advance in (1, 16, 200):
            with self.subTest(advance=advance), patch('queued_motion.time.sleep'):
                robot = Controller(advance)
                source = motions()
                arrivals = []
                send_batch(robot, source, lambda r,p: arrivals.append((len(r.moves),p)), continuous=True)
                self.assertEqual(arrivals, [(100, source[-1][0])])
                self.assertLessEqual(robot.peak, 32)
                for i, command in enumerate(robot.moves):
                    self.assertIn('pose={%.4f,100.0000,-36.1657' % i, command)
                    self.assertIn(',v=96,cp=%d)' % source[i][2], command)
                for i in (31, 63, 95):
                    self.assertIn('cp=70)', robot.moves[i])

    def test_precontact_offset_does_not_insert_stop_in_contact_path(self):
        robot = Controller()
        source = motions(67)[2:]
        wait = MagicMock()
        send_batch(robot, source, wait, sequence_offset=2, continuous=True)
        self.assertIn('cp=70)', robot.moves[29])
        wait.assert_called_once_with(robot, source[-1][0])

    def test_short_motion_keeps_existing_arrival_barrier(self):
        robot = MagicMock()
        robot._ok.return_value = True
        wait = MagicMock()
        send_batch(robot, motions(5), wait, continuous=True)
        self.assertEqual(robot.send.call_count, 5)
        robot.get_mode.assert_not_called()
        wait.assert_called_once()

    def test_bad_initial_state_sends_no_motion(self):
        for mode in (7, 8, 9, 10, 11, -1):
            robot = Controller()
            robot.mode_override = mode
            with self.subTest(mode=mode), self.assertRaises(RobotError):
                send_batch(robot, motions(), MagicMock(), continuous=True)
            self.assertEqual(robot.moves, [])

    def test_rejection_or_unreadable_ack_does_not_refill(self):
        for response in ('-1,{},MovL()', '0,{},MovL()', '0,{103},MovL()', '0,{106},WrongCommand()'):
            robot = Controller()
            original = robot.send
            def send(command):
                reply = original(command)
                return response if command.startswith('MovL(') and len(robot.moves) == 2 else reply
            robot.send = send
            wait = MagicMock()
            with self.subTest(response=response), self.assertRaises(RobotError):
                send_batch(robot, motions(), wait, continuous=True)
            self.assertEqual(len(robot.moves), 2)
            wait.assert_not_called()

    def test_invalid_progress_never_releases_slots(self):
        for reply in ('0,{99},GetCurrentCommandID()', '0,{9999},GetCurrentCommandID()',
                      '0,{104},GetCurrentCommandID()', '0,{},GetCurrentCommandID()',
                      '-1,{120},GetCurrentCommandID()', '0,{120},MovL()'):
            robot = Controller()
            original = robot.send
            def send(command):
                if command == 'GetCurrentCommandID()' and robot.moves:
                    return reply
                return original(command)
            robot.send = send
            wait = MagicMock()
            with self.subTest(reply=reply), self.assertRaises(RobotError):
                send_batch(robot, motions(), wait, continuous=True)
            self.assertEqual(len(robot.moves), 32)
            wait.assert_not_called()

    def test_alarm_pause_disconnect_or_stall_stops_refill(self):
        for fault in ('alarm', 'unreadable_alarm', 'pause', 'disconnect', 'stall'):
            robot = Controller(advance=0)
            original = robot.send
            def send(command):
                reply = original(command)
                if len(robot.moves) == 32:
                    if fault == 'alarm': robot.alarm = '0,{[1445]},GetErrorID()'
                    if fault == 'unreadable_alarm': robot.alarm = 'garbled'
                    if fault == 'pause': robot.mode_override = 10
                    if fault == 'disconnect': raise RobotError('disconnected')
                return reply
            robot.send = send
            wait = MagicMock()
            with self.subTest(fault=fault), patch('queued_motion.time.sleep'), \
                    patch('queued_motion.time.monotonic', side_effect=range(10000)), \
                    self.assertRaises(RobotError):
                send_batch(robot, motions(), wait, continuous=True)
            self.assertEqual(len(robot.moves), 32)
            wait.assert_not_called()

    def test_final_id_alone_does_not_claim_arrival(self):
        robot = Controller(advance=200)
        wait = MagicMock(side_effect=RobotError('pose mismatch'))
        with self.assertRaisesRegex(RobotError, 'pose mismatch'):
            send_batch(robot, motions(), wait, continuous=True)
        wait.assert_called_once()

    def test_final_queue_stall_never_claims_completion(self):
        robot = Controller(advance=200)
        original = robot.send
        def send(command):
            reply = original(command)
            if len(robot.moves) == 40:
                robot.advance = 0
            return reply
        robot.send = send
        wait = MagicMock()
        with patch('queued_motion.time.sleep'), \
                patch('queued_motion.time.monotonic', side_effect=range(10000)), \
                self.assertRaisesRegex(RobotError, 'completion timed out'):
            send_batch(robot, motions(40), wait, continuous=True)
        wait.assert_not_called()

    def test_journal_records_refills_without_inventing_physical_arrivals(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'journal.jsonl'
            journal = MotionJournal(path)
            robot = Controller()
            robot._diagnostic = journal
            def wait(r, target):
                journal.arrival(target, target)
            send_batch(robot, motions(), wait, continuous=True,
                       metadata=[dict(tag='draw:title', point=i) for i in range(100)])
            journal.close()
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            self.assertGreater(sum(r['event'] == 'queue_refill' for r in rows), 1)
            arrivals = [r for r in rows if r['event'] == 'arrival_confirmed']
            self.assertEqual(len(arrivals), 1)
            self.assertEqual(arrivals[0]['context']['waypoint']['point'], 99)
            self.assertEqual(rows[-1]['event'], 'stream_complete')

    def test_ink_stroke_opts_in_but_station_return_does_not(self):
        from run_mountain_balanced import Motion
        from config import ArmConfig, station_approach_z
        from geometry import CanvasMap
        mapper = CanvasMap((1000,1000))
        stroke = dict(points=[[400,400],[410,410]], press=[.1,.1], speed=[40,40], tail_mm=0)
        motion = Motion(MagicMock(), mapper, (100,100,station_approach_z(),*ArmConfig.RPY))
        motion.at_station = True
        with patch.object(motion, 'send') as send:
            motion.stroke(stroke)
        self.assertTrue(send.call_args.kwargs['continuous'])
        self.assertEqual(send.call_args_list[0].kwargs, {})

    def test_color_contact_opts_in_after_approach_confirmation(self):
        from tests.test_web_motion_profile import MotionProfileTests
        from queued_motion import send_batch as actual
        calls = []
        def spy(robot, commands, wait, **kwargs):
            calls.append((kwargs.get('continuous', False), len(commands)))
            return actual(robot, commands, wait, **kwargs)
        with tempfile.TemporaryDirectory() as folder, patch('queued_motion.send_batch', side_effect=spy):
            code, _, _ = MotionProfileTests().run_color(folder)
        self.assertEqual(code, 0)
        self.assertEqual(calls[:2], [(False, 2), (True, 3)])


if __name__ == '__main__':
    unittest.main()
