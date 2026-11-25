import unittest

import numpy as np

from gongbi_training.ink_paper_join_guard import protect_ink_joins


class InkJoinGuardTests(unittest.TestCase):
    def scene(self):
        gray = np.full((48, 48), 232, np.uint8)
        original = np.zeros_like(gray)
        rows = np.arange(10, 34)
        centers = 13 + rows // 3
        for offset in (-1, 0, 1):
            original[rows, centers + offset] = 2
        gray[rows, centers - 1] = gray[rows, centers + 1] = 80
        narrow = original.copy()
        narrow[rows[:6], centers[:6]] = 0
        persistent = narrow.copy()
        persistent[rows[6:12], centers[6:12]] = 0
        return gray, original, narrow, persistent, rows, centers

    def test_keeps_a_shadowed_gap_bracketed_by_paper(self):
        gray, original, narrow, persistent, rows, centers = self.scene()
        gray[rows[8], centers[8]] = 192
        result, accepted, rejected = protect_ink_joins(
            np.repeat(gray[..., None], 3, axis=2), original, narrow, persistent,
            check_dark_edge=True,
        )
        self.assertEqual(int(accepted.sum()), 6)
        self.assertFalse(rejected.any())
        np.testing.assert_array_equal(result, persistent)

    def test_restores_dark_proposals_without_paper_after_them(self):
        gray, original, narrow, persistent, rows, centers = self.scene()
        gray[rows[8:], centers[8:]] = 192
        result, accepted, rejected = protect_ink_joins(
            np.repeat(gray.T[..., None], 3, axis=2), original.T, narrow.T, persistent.T,
        )
        self.assertTrue(accepted[centers[6:8], rows[6:8]].all())
        self.assertTrue(rejected[centers[8:12], rows[8:12]].all())
        self.assertTrue((result[centers[8:12], rows[8:12]] == 2).all())

    def test_does_not_restart_past_a_disconnected_pair(self):
        gray, original, narrow, persistent, rows, centers = self.scene()
        gray[rows[8], centers[8] + 1] = 232
        _, accepted, rejected = protect_ink_joins(
            np.repeat(gray[..., None], 3, axis=2), original, narrow, persistent,
        )
        self.assertTrue(accepted[rows[6:8], centers[6:8]].all())
        self.assertTrue(rejected[rows[8:12], centers[8:12]].all())

    def test_keeps_dark_edge_ink_even_between_bright_sections(self):
        gray, original, narrow, persistent, rows, centers = self.scene()
        gray[rows[8], centers[8]] = 180
        image = np.repeat(gray[..., None], 3, axis=2)
        _, bracket_only, _ = protect_ink_joins(image, original, narrow, persistent)
        result, accepted, rejected = protect_ink_joins(
            image, original, narrow, persistent, check_dark_edge=True,
        )
        self.assertTrue(bracket_only[rows[8], centers[8]])
        self.assertTrue(accepted[rows[6:8], centers[6:8]].all())
        self.assertTrue(rejected[rows[8:12], centers[8:12]].all())
        self.assertTrue((result[rows[8:12], centers[8:12]] == 2).all())
