import unittest
import cv2
import numpy as np
from reference_lines import extract


class ReferenceLineTests(unittest.TestCase):
    def test_beige_paper_not_a_solid_ink_region(self):
        rng = np.random.default_rng(12)
        paper = np.clip(np.full((220, 300, 3), [219, 231, 240]) + rng.normal(0, 1.5, (220, 300, 1)), 0, 255).astype(np.uint8)
        lines, _ = extract(paper, .2)
        self.assertLess(np.mean(lines[:, :, 0] < 128), .005)

    def test_broad_branch_keeps_outer_shape_not_solid_fill(self):
        source = np.full((200, 200, 3), 255, np.uint8)
        cv2.rectangle(source, (60, 25), (100, 175), (20, 20, 20), -1)
        lines, _ = extract(source, .2)
        self.assertEqual(int(lines[100, 80, 0]), 255)
        self.assertLess(int(lines[100, 60, 0]), 128)
        self.assertLess(int(lines[100, 100, 0]), 128)

    def test_thin_lines_remain_and_white_gap_is_not_joined(self):
        source = np.full((200, 200, 3), 255, np.uint8)
        cv2.line(source, (25, 80), (85, 80), (60, 60, 60), 2)
        cv2.line(source, (110, 80), (175, 80), (60, 60, 60), 2)
        lines, _ = extract(source, .2)
        self.assertLess(int(lines[80, 50, 0]), 128)
        self.assertTrue(np.all(lines[76:85, 92:103] == 255))

    def test_saturated_petals_get_boundaries_not_black_fill(self):
        source = np.full((200, 200, 3), 255, np.uint8)
        cv2.circle(source, (100, 100), 40, (20, 30, 220), -1)
        lines, _ = extract(source, .2)
        self.assertTrue(np.all(lines[90:110, 90:110] == 255))
        self.assertTrue(np.any(lines[58:63, 95:105] < 128))


if __name__ == '__main__': unittest.main()
