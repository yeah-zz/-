import unittest

import numpy as np

from gongbi_training.faint_ink_color_texture import _printing_pattern, _texture_budget, bridge_faint_ink_color_texture


class FaintInkColorTextureTests(unittest.TestCase):
    def test_periodic_checkerboard_differs_from_oriented_ink(self):
        y, x = np.indices((35, 35))
        checkered = ((-1.)**(x+y)).astype(np.float32)
        stroke = np.zeros_like(checkered)
        stroke[:, 17:20] = [1, -2, 1]
        self.assertTrue(_printing_pattern(checkered)[17,17])
        self.assertFalse(_printing_pattern(stroke)[17,17])

    def test_does_not_relax_texture_across_a_pigment_boundary(self):
        gray = np.full((25, 25), 210, np.float32)
        gray[:, 12] = 194
        chroma = np.zeros((25, 25, 2), np.float32)
        isotropic = np.full_like(gray, 7)
        self.assertLess(_texture_budget(gray, chroma, isotropic)[12,12], 7)
        chroma[:, 13:, 0] = 16
        self.assertEqual(_texture_budget(gray, chroma, isotropic)[12,12], 7)

    def test_restores_a_faint_mark_on_consistent_pigment(self):
        rgb = np.empty((40, 45, 3), np.uint8)
        rgb[:] = [127, 169, 120]
        rgb[20, 7:38] = [110, 145, 104]
        roles = np.zeros(rgb.shape[:2], np.uint8)
        roles[20, 11:19] = roles[20, 24:32] = 2
        proposals = roles.copy()
        proposals[20, 19:24] = 2
        mask = bridge_faint_ink_color_texture(rgb, roles, proposals)
        self.assertTrue(mask[20, 19:24].all())
        self.assertEqual(int(mask.sum()), 5)
