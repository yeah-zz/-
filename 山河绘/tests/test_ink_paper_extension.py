import unittest

import numpy as np

from gongbi_training.ink_paper_extension import extend_narrow_paper


class PaperExtensionTests(unittest.TestCase):
    def sample(self):
        gray = np.full((48, 48), 230, dtype=np.uint8)
        rows = np.arange(10, 29)
        center = 14 + rows // 3
        roles = np.zeros_like(gray)
        roles[rows, center] = 2
        for side in (-1, 1):
            roles[rows, center + side] = 2
            gray[rows, center + side] = 75
        narrow = roles.copy()
        narrow[rows[:6], center[:6]] = 0
        return gray, roles, narrow, rows, center

    def test_extends_only_one_center_up_to_the_configured_limit(self):
        gray, roles, narrow, rows, center = self.sample()
        image = np.repeat(gray[..., None], 3, axis=2)
        result, edits = extend_narrow_paper(image, roles, narrow, limit=4)
        self.assertTrue(edits[rows[6:10], center[6:10]].all())
        self.assertEqual(int(edits.sum()), 4)
        self.assertTrue((result[rows, center - 1] == 2).all())
        self.assertTrue((result[rows, center + 1] == 2).all())

    def test_stops_at_a_missing_flank(self):
        gray, roles, narrow, rows, center = self.sample()
        gray[rows[8:11], center[8:11] + 1] = 230
        _, edits = extend_narrow_paper(np.repeat(gray[..., None], 3, axis=2), roles, narrow)
        self.assertTrue(edits[rows[6:8], center[6:8]].all())
        self.assertEqual(int(edits.sum()), 2)

    def test_does_not_extend_an_isolated_seed(self):
        gray, roles, _, rows, center = self.sample()
        narrow = roles.copy()
        narrow[rows[3], center[3]] = 0
        result, edits = extend_narrow_paper(np.repeat(gray[..., None], 3, axis=2), roles, narrow)
        self.assertFalse(edits.any())
        np.testing.assert_array_equal(result, narrow)
