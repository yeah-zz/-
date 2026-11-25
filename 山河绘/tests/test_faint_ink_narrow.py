import unittest

import numpy as np

from gongbi_training.faint_ink_narrow import bridge_faint_ink_narrow


class FaintInkNarrowTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((42, 42), 220, np.uint8)
        self.gray[16, 5:37] = 105
        self.gray[18, 5:37] = 175
        self.roles = np.zeros_like(self.gray)
        self.roles[16, 5:37] = 1
        self.roles[18, 6:18] = self.roles[18, 24:36] = 2
        self.direction = self.roles.copy()
        self.direction[18, 19:24] = 2

    def run_bridge(self, allow_new=False):
        rgb = np.repeat(self.gray[...,None],3,axis=2)
        return bridge_faint_ink_narrow(rgb,self.roles,self.direction,allow_new=allow_new)

    def test_keeps_the_gap_separate_from_neighboring_dark_ink(self):
        patch = self.run_bridge(allow_new=True)
        np.testing.assert_array_equal(patch[18,18:24],np.full(6,2))
        self.assertFalse(patch[17].any())

    def test_preserves_the_unproposed_pixel_without_generation(self):
        self.assertEqual(int(self.run_bridge()[18,18]),0)

    def test_does_not_continue_through_blank_section(self):
        self.gray[18,20] = 220
        self.assertFalse(self.run_bridge(allow_new=True)[18,20])
