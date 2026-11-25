import unittest
from experiment_start import validate_start
from park_color_loading import target_pose
from robot import RobotError


class StartTests(unittest.TestCase):
    def test_accepts_nearby_and_higher_start_preserving_orientation(self):
        p=target_pose()
        q=(p[0]+3,p[1]+4,p[2]-10,*p[3:])
        self.assertEqual(validate_start(q),(q[0],q[1],p[2],*q[3:]))
        validate_start((p[0],p[1],p[2]+.4,*p[3:]))

    def test_rejects_far_low_and_rotated(self):
        p=target_pose()
        for q in [(p[0]+11,*p[1:]),(p[0],p[1],p[2]+1,*p[3:]),
                  (*p[:4],p[4]+1.27,p[5]),(p[0],p[1],p[2]-21,*p[3:])]:
            with self.assertRaises(RobotError): validate_start(q)

    def test_drawing_start_allows_paper_contact_but_rejects_rotation(self):
        from config import ArmConfig, CanvasConfig, station_approach_z
        corners = CanvasConfig.ROBOT_CORNERS
        pose = (sum(p[0] for p in corners)/4, sum(p[1] for p in corners)/4,
                ArmConfig.PAPER_Z, *ArmConfig.RPY)
        self.assertEqual(validate_start(pose, require_park=False),
                         (*pose[:2], station_approach_z(), *pose[3:]))
        with self.assertRaises(RobotError):
            validate_start((*pose[:3], pose[3]+1, *pose[4:]), require_park=False)

if __name__=='__main__': unittest.main()
