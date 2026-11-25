"""Controller acknowledgement is required before using web acceleration."""
import unittest
from unittest.mock import Mock
from motion_settings import apply_acceleration
from robot import RobotError

class AccelerationTests(unittest.TestCase):
    def robot(self):
        robot=Mock();robot._ok.return_value=True;robot.wait_idle.return_value=True
        return robot

    def test_both_settings_are_submitted_and_confirmed(self):
        robot=self.robot();apply_acceleration(robot,dict(joint=25,linear=30))
        self.assertEqual([c.args[0] for c in robot.send.call_args_list],['AccJ(25)','AccL(30)'])
        self.assertEqual(robot.wait_idle.call_count,2)

    def test_rejection_or_timeout_aborts_setup(self):
        for kind in ('reject','timeout'):
            robot=self.robot()
            if kind=='reject':robot._ok.return_value=False
            else:robot.wait_idle.return_value=False
            with self.assertRaises(RobotError):apply_acceleration(robot,dict(joint=20,linear=20))
            self.assertEqual(robot.send.call_count,1)

    def test_invalid_acceleration_does_not_contact_robot(self):
        for values in ({'joint':20},{'joint':0,'linear':20},{'joint':True,'linear':20}):
            robot=self.robot()
            with self.assertRaises(ValueError):apply_acceleration(robot,values)
            robot.send.assert_not_called()
