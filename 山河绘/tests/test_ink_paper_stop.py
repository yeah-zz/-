import unittest

import numpy as np

from gongbi_training.ink_paper_extension import extend_narrow_paper
from gongbi_training.ink_paper_stop import guarded_paper_extension


class PaperStopTests(unittest.TestCase):
    def scene(self):
        gray = np.full((48, 48), 232, np.uint8)
        roles = np.zeros_like(gray)
        rows = np.arange(10, 34)
        centers = 13 + rows // 3
        for offset in (-1, 0, 1):
            roles[rows, centers + offset] = 2
        gray[rows, centers - 1] = gray[rows, centers + 1] = 80
        narrow = roles.copy()
        narrow[rows[:6], centers[:6]] = 0
        return gray, roles, narrow, rows, centers

    def test_keeps_proposals_between_two_continuing_ink_lines(self):
        gray, roles, narrow, _, _ = self.scene()
        image = np.repeat(gray[..., None], 3, axis=2)
        baseline, proposed = extend_narrow_paper(image, roles, narrow)
        output, accepted, rejected = guarded_paper_extension(image, roles, narrow)
        self.assertEqual(int(accepted.sum()), 6)
        self.assertFalse(rejected.any())
        np.testing.assert_array_equal(proposed, accepted)
        np.testing.assert_array_equal(baseline, output)

    def test_stops_before_ink_lines_merge(self):
        gray, roles, narrow, rows, centers = self.scene()
        gray[rows[10:14], centers[10:14]] = 90
        image = np.repeat(gray[..., None], 3, axis=2)
        _, accepted, rejected = guarded_paper_extension(image, roles, narrow)
        self.assertTrue(accepted[rows[6:8], centers[6:8]].all())
        self.assertTrue(rejected[rows[8:10], centers[8:10]].all())
        self.assertFalse(accepted[rows[10:], centers[10:]].any())

    def test_stops_before_a_fading_flank_even_after_transpose(self):
        gray, roles, narrow, rows, centers = self.scene()
        gray[rows[10:13], centers[10:13] + 1] = 180
        image = np.repeat(gray.T[..., None], 3, axis=2)
        _, accepted, rejected = guarded_paper_extension(image, roles.T, narrow.T)
        self.assertTrue(accepted[centers[6:8], rows[6:8]].all())
        self.assertTrue(rejected[centers[8:10], rows[8:10]].all())
        self.assertFalse(accepted[centers[10:], rows[10:]].any())
