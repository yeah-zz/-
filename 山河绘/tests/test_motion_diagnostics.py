"""Offline failure injection; these tests never connect to a robot."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock

from motion_diagnostics import MotionJournal, queue_id
from prepare_robot import wait_arrival
from queued_motion import send_batch
from robot import E6Robot, RobotError


class DiagnosticTests(unittest.TestCase):
    def test_alarm_mode_preserves_error_before_stopping_and_last_arrival(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'trace.jsonl'
            journal = MotionJournal(path)
            robot = MagicMock()
            robot._diagnostic = journal
            robot.connected = True
            robot.get_mode.side_effect = [5, 9]
            robot.get_error_id.side_effect = ['0,{[]},GetErrorID()',
                                             '0,{[1445,1446,1447,1448,1449,1450]},GetErrorID()']
            robot.get_pose.return_value = (1,2,3,0,0,0)
            robot.send.return_value = '0,{102},GetCurrentCommandID()'
            wait_arrival(robot, (1,2,3,0,0,0))
            with self.assertRaisesRegex(RobotError, 'mode=9; alarms=.*1445'):
                wait_arrival(robot, (9,9,9,0,0,0))
            journal.close()
            rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
            fault = next(r for r in rows if r['event'] == 'fault')
            self.assertEqual(fault['last_confirmed']['target'][:3], [1,2,3])
            self.assertFalse(fault['stop_confirmed'])
            self.assertEqual(robot.get_error_id.call_count, 2)
            self.assertEqual([c.args[0] for c in robot.send.call_args_list], ['GetCurrentCommandID()'])

    def test_failure_in_batch_wait_never_submits_next_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'trace.jsonl'
            journal = MotionJournal(path)
            robot = E6Robot()
            robot._diagnostic = journal
            submitted = []
            def transaction(command, wait=True):
                submitted.append(command)
                return '0,{%d},MovL()' % (100+len(submitted))
            robot._send_transaction = transaction
            def failed_wait(r, target):
                journal.fault(r, 'injected alarm', 9, '1445')
                raise RobotError('injected alarm')
            motions = [((i,0,0,0,0,0),40,20) for i in range(65)]
            metadata = [dict(tag='draw:s122',point=i) for i in range(65)]
            with self.assertRaisesRegex(RobotError, 'injected alarm'):
                send_batch(robot, motions, failed_wait, metadata=metadata)
            self.assertEqual(len(submitted), 32)
            with self.assertRaisesRegex(RobotError, 'latched'):
                robot.send('MovL()')
            self.assertEqual(len(submitted), 32)
            journal.close()
            rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
            replies = [r for r in rows if r['event'] == 'command_reply']
            self.assertEqual(replies[-1]['controller_queue_id'], 132)
            self.assertEqual(replies[-1]['batch_command'],32)
            self.assertEqual(replies[-1]['waypoint']['point'],31)
            self.assertIn('cp=0)',replies[-1]['command'])
            self.assertFalse(any(r['event']=='arrival_confirmed' for r in rows))

    def test_dashboard_exchanges_are_serialized(self):
        robot = E6Robot()
        active = 0
        peak = 0
        def transaction(command, wait=True):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            time.sleep(.01)
            active -= 1
            return command
        robot._send_transaction = transaction
        threads = [threading.Thread(target=robot.send,args=('RobotMode()',)) for _ in range(6)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertEqual(peak,1)

    def test_snapshot_failure_keeps_original_alarm(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'trace.jsonl'
            journal=MotionJournal(path)
            robot=MagicMock()
            robot._diagnostic=journal
            robot.connected=True
            robot.get_mode.return_value=9
            robot.get_error_id.return_value='0,{[1445]},GetErrorID()'
            robot.get_pose.side_effect=RobotError('disconnected')
            with self.assertRaisesRegex(RobotError,'mode=9; alarms=.*1445'):
                wait_arrival(robot,(0,0,0,0,0,0))
            journal.close()
            rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(r['event']=='fault_snapshot_error' for r in rows))

    def test_queue_id_never_invents_a_value(self):
        self.assertEqual(queue_id('0,{123},MovL()'),123)
        for reply in ('-1,{123},MovL()', '', '0,{},MovL()', '0,{[]},GetErrorID()'):
            self.assertIsNone(queue_id(reply))
