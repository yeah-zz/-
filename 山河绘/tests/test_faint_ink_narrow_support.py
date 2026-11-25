import unittest

import numpy as np

from gongbi_training.faint_ink_narrow_support import supported_proposals


class FaintInkNarrowSupportTests(unittest.TestCase):
    def test_only_keeps_components_with_a_longitudinal_span(self):
        roles = np.zeros((20, 20), np.uint8)
        roles[10, 1] = 2
        proposals = roles.copy()
        proposals[5, 2:4] = 2
        proposals[8, 2:5] = 1
        proposals[13:15, 2:4] = 2
        mask = supported_proposals(roles,proposals,minimum_span=3)
        np.testing.assert_array_equal(mask[8, 2:5], [1, 1, 1])
        self.assertFalse(mask[5].any())
        self.assertFalse(mask[13:15].any())
        self.assertEqual(int(mask[10,1]),2)

    def test_different_roles_cannot_make_up_a_component(self):
        roles = np.zeros((10, 10), np.uint8)
        proposals = roles.copy()
        proposals[5, 3:6] = [1, 2, 1]
        self.assertFalse(supported_proposals(roles,proposals,minimum_span=3).any())
