import unittest

import numpy as np

from gongbi_training.ink_paper_connected import separate_connected_paper
from gongbi_training.ink_paper_context import separate_contextual_paper


class ConnectedPaperTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((64, 64, 3), 210, np.uint8)
        self.image[12:52, [30, 34]] = 95
        self.roles = np.zeros((64, 64), np.uint8)
        self.roles[12:52, 30:35] = 2

    def test_open_seam_keeps_its_removal(self):
        revised, removed = separate_connected_paper(self.image, self.roles)
        self.assertTrue(removed[20:44, 32].all())
        self.assertTrue((revised[12:52, [30, 34]] == 2).all())

    def test_enclosed_dry_paper_marks_are_preserved(self):
        self.image[[12, 51], 30:35] = 95
        self.roles[32, 32] = 0  # A pre-existing tiny hole must not become a seed.
        _, local_removal, _ = separate_contextual_paper(self.image, self.roles)
        self.assertTrue(local_removal[30, 32])
        revised, removed = separate_connected_paper(self.image, self.roles)
        self.assertFalse(removed.any())
        np.testing.assert_array_equal(revised, self.roles)
