import unittest

import numpy as np

from gongbi_training.faint_ink_anchor_guard import _reach_from_source, guard_faint_ink_anchors
from gongbi_training.faint_ink_continuation import recover_faint_ink


class FaintInkAnchorGuardTests(unittest.TestCase):
    def test_requires_source_support_despite_a_directional_prediction(self):
        gray = np.full((44,44),225,np.uint8)
        gray[20,8:33] = 212
        roles = np.zeros_like(gray)
        roles[20,8:20] = 2
        image = np.repeat(gray[...,None],3,axis=2)
        proposed, old = recover_faint_ink(image,roles,limit=6,allow_tips=True,min_path_contrast=12)
        result, accepted, rejected = guard_faint_ink_anchors(
            image,roles,proposed,min_anchor_length=2,min_ink_depth=20,
            texture_weight=2.5,strong_ridge_floor=18,dark_core_floor=55,
        )
        self.assertGreater(int(old.sum()), 0)
        self.assertFalse(accepted.any())
        np.testing.assert_array_equal(rejected,old)
        np.testing.assert_array_equal(result,roles)

    def test_keeps_continuation_from_supported_faint_anchor(self):
        gray = np.full((44,44),225,np.uint8)
        gray[20,8:33] = 188
        roles = np.zeros_like(gray)
        roles[20,8:20] = 1
        image = np.repeat(gray[...,None],3,axis=2)
        proposed, old = recover_faint_ink(image,roles,limit=6,allow_tips=True,min_path_contrast=12)
        result, accepted, rejected = guard_faint_ink_anchors(
            image,roles,proposed,min_anchor_length=2,min_ink_depth=20,
            texture_weight=2.5,strong_ridge_floor=18,dark_core_floor=55,
        )
        self.assertTrue(accepted[20,20:26].all())
        self.assertFalse(rejected.any())
        np.testing.assert_array_equal(old,accepted)
        np.testing.assert_array_equal(result,proposed)

    def test_a_one_pixel_or_different_role_anchor_cannot_start_recovery(self):
        roles = np.zeros((13,13),np.uint8)
        roles[4,4] = 2
        roles[3:5,8] = 1
        evidence = roles>0
        candidates = np.zeros((13,13,2),bool)
        candidates[5:8,4,1] = True
        candidates[5:8,8,1] = True
        self.assertFalse(_reach_from_source(candidates,roles,evidence,2).any())
