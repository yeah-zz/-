import unittest

import cv2
import numpy as np

from gongbi_training.branch_attachment import attachment_candidates, trace_branch_attachment, woody_support


class BranchAttachmentTest(unittest.TestCase):
    def scene(self):
        wood = np.zeros((55, 75), np.uint8)
        cv2.line(wood, (20, 4), (20, 47), 1, 5)
        cv2.line(wood, (20, 25), (69, 25), 1, 3)
        image = np.full((55, 75, 3), 248, np.uint8)
        image[wood > 0] = [115, 86, 58]
        return image, wood

    def test_locates_attachment_and_routes_to_branch_on_either_side(self):
        rgb, mask = self.scene()
        for flipped in (False, True):
            image, ink = (rgb[:, ::-1], mask[:, ::-1]) if flipped else (rgb, mask)
            result = trace_branch_attachment(image, ink, np.ones_like(ink), crop=(11 if not flipped else 4, 16,
                                             71 if not flipped else 64, 36))
            self.assertEqual(result['status'], 'route_found')
            self.assertEqual(result['trunk_side'], 'right' if flipped else 'left')
            x, y = result['junction']
            self.assertLessEqual(np.hypot(x - (54 if flipped else 20), y - 25), 3)
            self.assertGreater(np.ptp(result['points'][:, 0]), 35)

    def test_parallel_lines_do_not_make_a_junction(self):
        mask = np.zeros((30, 55), np.uint8)
        mask[13, 3:52] = mask[15, 3:52] = 1
        self.assertFalse(attachment_candidates(mask, trunk_side='left'))

    def test_false_ink_in_white_gap_does_not_connect_branch(self):
        rgb, mask = self.scene()
        rgb[:, 42:47] = 248
        allowed = woody_support(rgb, mask, np.ones_like(mask))
        self.assertFalse(allowed[:, 42:47].any())
        result = trace_branch_attachment(rgb, mask, np.ones_like(mask), crop=(11, 16, 71, 36))
        self.assertEqual(result['status'], 'no_supported_branch')
