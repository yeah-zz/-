import unittest

import numpy as np

from gongbi_training.faint_ink_continuation import recover_faint_ink


class FaintInkContinuationTests(unittest.TestCase):
    def test_recovers_short_break_without_filling_between_parallel_lines(self):
        gray = np.full((48, 48), 225, np.uint8)
        gray[[17, 22], 10:34] = 199
        roles = np.zeros_like(gray)
        roles[[17, 22], 10:34] = 2
        roles[22, 20:23] = 0
        image = np.repeat(gray[..., None], 3, axis=2)
        result, restored = recover_faint_ink(image, roles)
        self.assertTrue(restored[22, 20:23].all())
        self.assertTrue((result[22, 20:23] == 2).all())
        self.assertEqual(int(restored.sum()), 3)
        self.assertFalse(restored[18:22].any())

    def test_one_sided_recovery_keeps_the_existing_role_after_transpose(self):
        gray = np.full((48, 48), 225, np.uint8)
        gray[19, 10:33] = 197
        roles = np.zeros_like(gray)
        roles[19, 10:25] = 1
        image = np.repeat(gray.T[..., None], 3, axis=2)
        result, restored = recover_faint_ink(image, roles.T, allow_tips=True, min_path_contrast=12)
        self.assertTrue(restored[25:30, 19].all())
        self.assertTrue((result[25:30, 19] == 1).all())
        self.assertFalse(restored[30:, 19].any())

    def test_weak_segment_and_blank_paper_do_not_pass_whole_path_check(self):
        gray = np.full((48, 48), 225, np.uint8)
        gray[19, 10:33] = 217
        roles = np.zeros_like(gray)
        roles[19, 10:25] = 2
        image = np.repeat(gray[..., None], 3, axis=2)
        result, restored = recover_faint_ink(image, roles, allow_tips=True, min_path_contrast=12)
        self.assertFalse(restored.any())
        np.testing.assert_array_equal(result, roles)
