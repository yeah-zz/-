import unittest

import numpy as np

from gongbi_training.ink_paper_width import separate_narrow_paper


class PaperWidthTests(unittest.TestCase):
    def test_keeps_sides_of_oblique_gap(self):
        rows = np.arange(10, 54)
        centers = 21 + rows // 3
        y, x = np.indices((64, 64))
        gray = np.where(np.abs(x - (21 + y // 3)) <= 8, 204, 225).astype(np.uint8)
        roles = np.zeros_like(gray)
        for offset in range(-2, 3):
            roles[rows, centers + offset] = 2
        gray[rows, centers - 2] = gray[rows, centers + 2] = 80
        image = np.repeat(gray[..., None], 3, axis=2)

        result, added, joined = separate_narrow_paper(image, roles, bridge=True)

        self.assertGreater(added[rows, centers].sum(), 20)
        self.assertTrue((result[rows, centers - 2] == 2).all())
        self.assertTrue((result[rows, centers + 2] == 2).all())
        self.assertFalse((added & joined).any())

    def test_preserves_a_broad_flat_valley_in_this_pass(self):
        gray = np.full((64, 64), 204, dtype=np.uint8)
        gray[:, 12:52] = 198
        gray[10:54, [25, 33]] = 70
        roles = np.zeros_like(gray)
        roles[10:54, 25:34] = 1
        image = np.repeat(gray[..., None], 3, axis=2)

        _, added, joined = separate_narrow_paper(image, roles, bridge=True)

        self.assertFalse(added[12:52, 28:31].any())
        self.assertFalse(joined[12:52, 28:31].any())
