import unittest
from restore_drawing_pose import target_for
from config import ArmConfig
from park_color_loading import target_pose
from robot import RobotError

# 纸外高位停车点：直接取当前标定，换笔/挪盘/重标姿态后这一组夹具自动跟随。
PARK = target_pose()
NEAR = (PARK[0] + 2., PARK[1] + 2., PARK[2] - 10., *(v + .5 for v in ArmConfig.RPY))


class RestoreTests(unittest.TestCase):
    def test_screenshot_target_preserves_xyz(self):
        # 现场截图里读到的姿态：XYZ 原样保留，只把三个姿态角换成标定值
        self.assertEqual(target_for(NEAR), (*NEAR[:3], *ArmConfig.RPY))

    def test_low_far_or_large_rotation_rejected(self):
        for p in [(NEAR[0], NEAR[1], ArmConfig.PAPER_Z - 5., *ArmConfig.RPY),      # 太低，贴纸面
                  (NEAR[0] + 50., NEAR[1], NEAR[2], *ArmConfig.RPY),                # 不在停车点附近
                  (NEAR[0], NEAR[1], NEAR[2], ArmConfig.RPY[0] - 10., *ArmConfig.RPY[1:])]:  # 姿态差 > 3°
            with self.assertRaises((ValueError,RobotError)): target_for(p)


if __name__=='__main__': unittest.main()
