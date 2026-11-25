import unittest

import numpy as np

from gongbi_training.faint_ink_direction_filter import _anchored_candidates, filter_faint_ink_direction


class FaintInkDirectionFilterTests(unittest.TestCase):
    def test_keeps_diagonal_continuation_and_excludes_an_isolated_spot(self):
        gray = np.full((48, 48), 225, np.uint8)
        path = np.arange(8, 35)
        gray[path, path] = 180
        gray[10, 30] = 180
        roles = np.zeros_like(gray)
        roles[path[:10], path[:10]] = 2
        proposed = roles.copy()
        proposed[path[10:14], path[10:14]] = 2
        proposed[10, 30] = 2

        result, accepted, rejected = filter_faint_ink_direction(
            np.repeat(gray[..., None],3,axis=2), roles, proposed,
        )

        self.assertTrue(accepted[path[10:14], path[10:14]].all())
        self.assertEqual(int(accepted.sum()), 4)
        self.assertTrue(rejected[10, 30])
        np.testing.assert_array_equal(result[roles>0], roles[roles>0])

    def test_does_not_keep_pixels_beyond_a_rejected_section_or_other_role(self):
        roles = np.zeros((12, 12), np.uint8)
        roles[4, 4] = 2
        roles[4, 8] = 1
        candidates = np.zeros((12, 12, 2), dtype=bool)
        candidates[5, 4, 1] = True
        candidates[7:9, 4, 1] = True
        candidates[5:8, 8, 1] = True

        result = _anchored_candidates(candidates, roles, 6)

        self.assertTrue(result[5, 4, 1])
        self.assertEqual(int(result.sum()), 1)
