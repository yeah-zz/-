import unittest

import numpy as np

from gongbi_training.faint_ink_bridge import bridge_faint_ink


class FaintInkBridgeTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((36, 40), 225, np.uint8)
        self.gray[18, 7:32] = 213
        self.roles = np.zeros_like(self.gray)
        self.roles[18, 7:17] = self.roles[18, 22:32] = 2
        self.proposals = self.roles.copy()
        self.proposals[18, 17:22] = 2

    def bridge(self, limit=6):
        rgb = np.repeat(self.gray[..., None], 3, axis=2)
        return bridge_faint_ink(rgb, self.roles, self.proposals, limit=limit)

    def test_crosses_faint_gap_of_same_role(self):
        mask = self.bridge()
        self.assertTrue(mask[18, 17:22].all())
        self.assertEqual(int(mask.sum()), 5)

    def test_respects_entire_gap_length(self):
        self.assertFalse(self.bridge(limit=4).any())

    def test_rejects_single_anchor_and_competing_role(self):
        self.roles[18, 22:32] = 0
        self.assertFalse(self.bridge().any())
        self.roles[18, 22:32] = 1
        self.assertFalse(self.bridge().any())

    def test_requires_raw_ink_inside_gap(self):
        self.gray[18, 19] = 225
        self.assertFalse(self.bridge().any())

    def test_requires_two_pixels_at_ending_anchor(self):
        self.roles[18, 23:32] = 0
        self.assertFalse(self.bridge().any())
