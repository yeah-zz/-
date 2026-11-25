import unittest

import numpy as np

from gongbi_training.faint_ink_bridge_sections import bridge_faint_ink_sections


class FaintInkBridgeSectionsTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((34, 36), 222, np.uint8)
        self.roles = np.zeros_like(self.gray)
        self.proposed = self.roles.copy()
        self.target = [(col, 16 + offset) for col, offset in
                       enumerate((0, 0, 1, 1, 1, 1, 0, 0, 0), 10)]
        for i, (col, row) in enumerate(self.target):
            self.gray[row, col] = 207
            self.gray[row+1, col] = 212
            self.proposed[row+1, col] = 1
            if i < 2 or i > 6:
                self.roles[row+1, col] = 1

    def bridge(self):
        image = np.repeat(self.gray[..., None], 3, axis=2)
        return bridge_faint_ink_sections(image, self.roles, self.proposed)

    def test_follows_curved_gray_band_offset_from_proposal(self):
        mask = self.bridge()
        self.assertEqual(int(mask.sum()), 5)
        for col, row in self.target[2:7]:
            self.assertTrue(mask[row+1,col])

    def test_rejects_blank_interior(self):
        col, row = self.target[4]
        self.gray[row:row+2,col] = 222
        self.assertFalse(self.bridge().any())

    def test_rejects_single_ended_texture(self):
        for col, row in self.target[-2:]:
            self.roles[row+1,col] = 0
        self.assertFalse(self.bridge().any())

    def test_rejects_an_intensity_step_without_an_ink_band(self):
        for col, row in self.target[2:7]:
            self.gray[row:,col] = 212
        self.assertFalse(self.bridge().any())
