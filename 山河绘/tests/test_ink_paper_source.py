import unittest

import numpy as np

from gongbi_training.ink_paper_connected import separate_connected_paper
from gongbi_training.ink_paper_source import separate_source_paper


class SourcePaperTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((64, 64, 3), 210, np.uint8)
        self.image[12:52, [30, 34]] = 95
        # Reproduce a model predicting ink even where the paper is visible.
        self.roles = np.full((64, 64), 2, np.uint8)

    def test_recovers_gap_when_model_has_no_background(self):
        _, old_removed = separate_connected_paper(self.image, self.roles)
        adjusted, removed = separate_source_paper(self.image, self.roles)
        self.assertFalse(old_removed.any())
        self.assertTrue(removed[20:44, 32].all())
        self.assertTrue((adjusted[12:52, [30, 34]] == 2).all())

    def test_narrow_closed_fleck_still_has_no_source_entrance(self):
        self.image[[12, 51], 30:35] = 95
        adjusted, removed = separate_source_paper(self.image, self.roles)
        self.assertFalse(removed[20:44, 32].any())
        self.assertTrue((adjusted[20:44, 32] == 2).all())
