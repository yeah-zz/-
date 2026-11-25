import unittest

import numpy as np

from gongbi_training.ink_paper_tracking import track_narrow_paper


class PaperTrackingTests(unittest.TestCase):
    def make_gap(self):
        gray = np.full((48, 48), 232, dtype=np.uint8)
        roles = np.zeros_like(gray)
        rows = np.arange(9, 32)
        center = 14 + rows // 3
        gray[rows, center - 1] = gray[rows, center + 1] = 80
        roles[rows, center - 1] = roles[rows, center + 1] = 2
        roles[rows, center] = 2
        seeds = roles.copy()
        for index in (slice(3, 8), slice(12, 17)):
            seeds[rows[index], center[index]] = 0
        gray[rows[8:12], center[8:12]] = 205
        return gray, roles, seeds, rows, center

    def test_recovers_a_shaded_interval_on_same_pair(self):
        gray, roles, seeds, rows, center = self.make_gap()
        image = np.repeat(gray[..., None], 3, axis=2)
        for nearest in (False, True):
            with self.subTest(nearest=nearest):
                result, edits = track_narrow_paper(image, roles, seeds, nearest=nearest)
                self.assertTrue(edits[rows[8:12], center[8:12]].all())
                self.assertEqual(int(edits.sum()), 4)
                self.assertTrue((result[rows, center - 1] == 2).all())
                self.assertTrue((result[rows, center + 1] == 2).all())

    def test_does_not_cross_dark_wash_or_follow_one_anchor(self):
        gray, roles, seeds, rows, center = self.make_gap()
        gray[rows[9:11], center[9:11]] = 155
        image = np.repeat(gray[..., None], 3, axis=2)
        for nearest in (False, True):
            with self.subTest(nearest=nearest):
                result, edits = track_narrow_paper(image, roles, seeds, nearest=nearest)
                self.assertFalse(edits.any())
                np.testing.assert_array_equal(result, seeds)

    def test_recovers_the_same_interval_when_transposed(self):
        gray, roles, seeds, rows, center = self.make_gap()
        _, edits = track_narrow_paper(np.repeat(gray.T[..., None], 3, axis=2), roles.T, seeds.T)
        self.assertTrue(edits[center[8:12], rows[8:12]].all())
        self.assertEqual(int(edits.sum()), 4)
