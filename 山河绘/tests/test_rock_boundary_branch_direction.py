import unittest

import numpy as np

from gongbi_training.rock_boundary_branch_direction import check_branch_entry, rock_boundary_support


class BoundaryDirectionTest(unittest.TestCase):
    def test_rock_edge_uses_signed_cross_section_and_rejects_uniform_paper(self):
        rgb = np.full((44,60,3),174,np.uint8)
        rgb[22:] = 162
        left,right = (4,17,12,27),(48,17,56,27)
        detected,peaks,_ = rock_boundary_support(rgb,left,right,lighter_side=-1)
        reversed_evidence,_,_ = rock_boundary_support(rgb,left,right,lighter_side=1)
        self.assertTrue(peaks[21:23,10:50].any())
        self.assertFalse(reversed_evidence.any())
        self.assertFalse(detected[:12].any())
        self.assertFalse(detected[33:].any())

    def test_branch_direction_allows_a_continuation_and_rejects_a_rock_crossing(self):
        trunk=(19,19,25,25)
        downstream=((20,17),(24,31))
        branch=[[16,8],[17,13],[19,17],[21,22]]
        rock=[[36,21],[31,21],[26,21],[21,22]]
        self.assertEqual(check_branch_entry(branch,trunk,downstream)['status'],'direction_supported')
        self.assertEqual(check_branch_entry(branch[::-1],trunk,downstream)['status'],'direction_supported')
        self.assertEqual(check_branch_entry(rock,trunk,downstream)['status'],'direction_conflict')
