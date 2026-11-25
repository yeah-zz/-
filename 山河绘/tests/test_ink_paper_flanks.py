import unittest

import numpy as np

from gongbi_training.ink_paper_flanks import separate_flank_paper


class FlankPaperTests(unittest.TestCase):
    def case(self, level):
        y, x = np.indices((64, 64))
        gray = np.where(np.abs(x - (23 + y // 3)) <= 8, level, 225).astype(np.uint8)
        image = np.repeat(gray[..., None], 3, axis=2)
        roles = np.zeros((64, 64), np.uint8)
        rows = np.arange(10, 54)
        center = 23 + rows // 3
        for offset in range(-2, 3):
            roles[rows, center + offset] = 2
        image[rows, center - 2] = image[rows, center + 2] = 80
        return image, roles, rows, center

    def test_local_paper_recovers_an_oblique_gap_partly_in_shadow(self):
        image, roles, rows, center = self.case(204)
        result, removed, added = separate_flank_paper(image, roles, strict=False)
        self.assertGreater(int(added[rows, center].sum()), 30)
        self.assertFalse(removed[rows, center - 2].any() or removed[rows, center + 2].any())
        np.testing.assert_array_equal(result[~removed], roles[~removed])

    def test_dark_wash_is_not_cleared_by_local_shoulder_evidence(self):
        image, roles, rows, center = self.case(135)
        result, _, added = separate_flank_paper(image, roles, strict=False)
        self.assertFalse(added.any())
        self.assertTrue((result[rows, center] == 2).all())
