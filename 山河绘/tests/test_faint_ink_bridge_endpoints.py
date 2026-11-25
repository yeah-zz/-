import unittest

import numpy as np

from gongbi_training.faint_ink_bridge_endpoints import bridge_faint_ink_verified_endpoints
from gongbi_training.faint_ink_bridge_sections import bridge_faint_ink_sections


class FaintInkEndpointTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((40, 45), 223, np.uint8)
        self.gray[20, 6:38] = 204
        self.baseline = np.zeros_like(self.gray)
        self.baseline[20, 13:19] = self.baseline[20, 24:30] = 2
        self.proposed = self.baseline.copy()
        self.proposed[20, 19:24] = 2

    def evaluate(self, bridge=bridge_faint_ink_verified_endpoints):
        image = np.repeat(self.gray[..., None], 3, axis=2)
        return bridge(image, self.baseline, self.proposed)

    def test_uses_original_band_beyond_detected_ends(self):
        self.baseline[20, 13:17] = self.baseline[20, 26:30] = 0
        mask = self.evaluate()
        self.assertTrue(mask[20, 19:24].all())
        self.assertEqual(int(mask.sum()), 5)

    def test_rejects_ends_backed_by_only_two_dark_sections(self):
        self.gray[20, :17] = self.gray[20, 26:] = 223
        self.assertTrue(self.evaluate(bridge_faint_ink_sections).any())
        self.assertFalse(self.evaluate().any())

    def test_rejects_missing_ink_beyond_one_end(self):
        self.gray[20, 26:] = 223
        self.assertFalse(self.evaluate().any())
