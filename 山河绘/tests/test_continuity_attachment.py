import unittest

import cv2
import numpy as np

from gongbi_training.continuity_attachment import route_between_areas, source_ridge_support


class ContinuityAttachmentTest(unittest.TestCase):
    def setUp(self):
        self.crop = (4, 7, 58, 29)
        self.start = (6, 14, 12, 21)
        self.end = (50, 14, 55, 21)
        self.rgb = np.full((36, 62, 3), 180, np.uint8)
        self.weak = np.zeros(self.rgb.shape[:2], np.uint8)
        cv2.line(self.rgb, (8, 17), (53, 17), (158, 158, 158), 1)
        self.weak[17, 8:54] = 1

    def test_weak_ink_connects_without_a_strong_seed(self):
        contrast = np.float32(self.weak * 11)
        support = source_ridge_support(self.rgb, contrast, self.weak, self.start, self.end)
        result = route_between_areas(self.rgb, support, crop=self.crop,
                                     start_box=self.start, end_box=self.end)
        self.assertEqual(result['status'], 'route_found')
        self.assertTrue((result['points'][:, 1] == 17).all())

    def test_blank_gap_is_preserved(self):
        self.rgb[:, 26:37] = 180
        self.weak[:, 26:37] = 0
        contrast = np.float32(self.weak * 11)
        support = source_ridge_support(self.rgb, contrast, self.weak, self.start, self.end)
        self.assertFalse(support[:, 26:37].any())
        result = route_between_areas(self.rgb, support, crop=self.crop,
                                     start_box=self.end, end_box=self.start)
        self.assertEqual(result['status'], 'no_owned_connection')
