import unittest

import numpy as np

from gongbi_training.ink_paper_seams import separate_visible_paper


class PaperSeamTests(unittest.TestCase):
    def pair(self):
        image = np.full((48, 48, 3), 210, np.uint8)
        image[8:40, [21, 25]] = 80
        roles = np.zeros((48, 48), np.uint8)
        roles[8:40, 21:26] = 2
        return image, roles

    def test_supported_parallel_seam(self):
        image, roles = self.pair()
        adjusted, removed = separate_visible_paper(image, roles)
        self.assertTrue(removed[15:33, 23].all())
        self.assertTrue((adjusted[8:40, [21, 25]] == 2).all())

    def test_dark_join_retained_between_parallel_strokes(self):
        image, roles = self.pair()
        image[24, 21:26] = 80
        adjusted, removed = separate_visible_paper(image, roles)
        self.assertTrue((adjusted[24, 21:26] == 2).all())
        self.assertTrue(removed[16, 23])

    def test_faint_third_stroke_remains(self):
        image, roles = self.pair()
        image[8:40, 23] = 197
        adjusted, _ = separate_visible_paper(image, roles)
        self.assertTrue((adjusted[8:40, 23] == 2).all())

    def test_short_dry_break_does_not_become_a_seam(self):
        image = np.full((48, 48, 3), 210, np.uint8)
        image[8:40, 23] = 80
        image[22:24, 23] = 210
        roles = np.zeros((48, 48), np.uint8)
        roles[8:40, 23] = 1
        adjusted, removed = separate_visible_paper(image, roles)
        self.assertFalse(removed.any())
        np.testing.assert_array_equal(adjusted, roles)
