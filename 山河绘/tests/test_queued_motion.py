import unittest
from unittest.mock import MagicMock
from queued_motion import send_batch
from robot import RobotError


class QueueTests(unittest.TestCase):
    def test_batch_bound_and_completion_after_ack(self):
        robot=MagicMock(); robot._ok.return_value=True
        motions=[((i,0,0,0,0,0),20,70) for i in range(65)]
        boundaries=[]
        send_batch(robot,motions,lambda r,p: boundaries.append((r.send.call_count,p[0])))
        self.assertEqual(boundaries,[(32,31),(64,63),(65,64)])
        robot.get_mode.assert_not_called()
        for index in (31,63,64):
            self.assertIn('cp=0)',robot.send.call_args_list[index].args[0])

    def test_rejection_stops_submission_without_claiming_completion(self):
        robot=MagicMock();robot._ok.side_effect=[True,False]
        wait=MagicMock()
        with self.assertRaises(RobotError):
            send_batch(robot,[((0,0,0,0,0,0),20,0)]*3,wait)
        self.assertEqual(robot.send.call_count,2)
        wait.assert_not_called()

    def test_failed_arrival_does_not_submit_next_batch(self):
        robot=MagicMock();robot._ok.return_value=True
        with self.assertRaises(RobotError):
            send_batch(robot,[((0,0,0,0,0,0),20,0)]*33,MagicMock(side_effect=RobotError('alarm')))
        self.assertEqual(robot.send.call_count,32)

    def test_precontact_check_preserves_original_draw_stop_boundaries(self):
        motions=[((i,0,0,0,0,0),40,0 if i<2 else 70) for i in range(67)]
        original=MagicMock(); original._ok.return_value=True
        split=MagicMock(); split._ok.return_value=True
        send_batch(original,motions,lambda r,p:None)
        send_batch(split,motions[:2],lambda r,p:None)
        boundaries=[]
        send_batch(split,motions[2:],lambda r,p:boundaries.append(p[0]),sequence_offset=2)
        self.assertEqual(boundaries,[31,63,66])
        self.assertEqual(original.send.call_args_list,split.send.call_args_list)

    def test_invalid_sequence_offset_never_submits(self):
        for offset in (-1,True,.5):
            robot=MagicMock()
            with self.assertRaises(ValueError):
                send_batch(robot,[((0,0,0,0,0,0),20,0)],lambda r,p:None,sequence_offset=offset)
            robot.send.assert_not_called()
