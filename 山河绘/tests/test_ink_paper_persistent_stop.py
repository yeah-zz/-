import unittest

import numpy as np

from gongbi_training.ink_paper_persistent_stop import persistent_paper_extension


class PersistentPaperStopTests(unittest.TestCase):
    def scene(self):
        gray = np.full((48, 48), 232, np.uint8)
        roles = np.zeros_like(gray)
        rows = np.arange(10, 34)
        centers = 13 + rows // 3
        for offset in (-1, 0, 1):
            roles[rows, centers + offset] = 2
        gray[rows, centers - 1] = gray[rows, centers + 1] = 80
        narrow = roles.copy()
        narrow[rows[:6], centers[:6]] = 0
        return gray, roles, narrow, rows, centers

    def test_keeps_one_shadowed_section_when_the_same_rails_continue(self):
        gray, roles, narrow, rows, centers = self.scene()
        gray[rows[8], centers[8]] = 192
        image = np.repeat(gray[..., None], 3, axis=2)
        result, accepted, rejected = persistent_paper_extension(image, roles, narrow)
        self.assertTrue(accepted[rows[6:12], centers[6:12]].all())
        self.assertFalse(rejected.any())
        self.assertTrue((result[rows, centers - 1] == 2).all())
        self.assertTrue((result[rows, centers + 1] == 2).all())

    def test_stops_when_the_same_brightness_drop_persists(self):
        gray, roles, narrow, rows, centers = self.scene()
        gray[rows[8:14], centers[8:14]] = 192
        image = np.repeat(gray[..., None], 3, axis=2)
        _, accepted, rejected = persistent_paper_extension(image, roles, narrow)
        self.assertTrue(rejected[rows[8:12], centers[8:12]].all())
        self.assertFalse(accepted[rows[8:], centers[8:]].any())

    def test_stops_at_a_fading_flank_or_a_merge_when_transposed(self):
        for merge in (False, True):
            with self.subTest(merge=merge):
                gray, roles, narrow, rows, centers = self.scene()
                gray[rows[10:13], centers[10:13] + int(not merge)] = 90 if merge else 180
                image = np.repeat(gray.T[..., None], 3, axis=2)
                _, accepted, rejected = persistent_paper_extension(image, roles.T, narrow.T)
                self.assertTrue(accepted[centers[6:8], rows[6:8]].all())
                self.assertTrue(rejected[centers[8:10], rows[8:10]].all())
                self.assertFalse(accepted[centers[10:], rows[10:]].any())
