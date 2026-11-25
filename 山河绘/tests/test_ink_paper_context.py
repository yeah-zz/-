import unittest

import numpy as np

from gongbi_training.ink_paper_context import separate_contextual_paper
from gongbi_training.ink_paper_seams import separate_visible_paper


class ContextualPaperTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((64, 64, 3), 210, np.uint8)
        self.image[12:52, [30, 34]] = 95
        self.roles = np.zeros((64, 64), np.uint8)
        self.roles[12:52, 30:35] = 2

    def test_relatively_bright_channel_in_wash_is_withheld(self):
        self.image[:, 22:43] = 176
        self.image[12:52, [30, 34]] = 95
        _, proposed = separate_visible_paper(self.image, self.roles)
        revised, removed, withheld = separate_contextual_paper(self.image, self.roles)
        self.assertTrue(proposed[20:44, 32].all())
        self.assertTrue(withheld[20:44, 32].all())
        self.assertFalse(removed[20:44, 32].any())
        self.assertTrue((revised[20:44, 32] == 2).all())

    def test_seam_on_paper_with_smooth_shading_is_retained(self):
        self.image[:] = (196 + np.arange(64) // 3)[None, :, None]
        self.image[12:52, [30, 34]] = 95
        revised, removed, withheld = separate_contextual_paper(self.image, self.roles)
        self.assertTrue(removed[20:44, 32].all())
        self.assertFalse(withheld[20:44, 32].any())
        self.assertTrue((revised[12:52, [30, 34]] == 2).all())
