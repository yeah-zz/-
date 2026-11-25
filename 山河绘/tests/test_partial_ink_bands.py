import unittest
import numpy as np
from gongbi_training.partial_ink_bands import rasterize_band,merge_bands


class PartialBandChecks(unittest.TestCase):
    def band(self,role=1,rows=()):
        return rasterize_band((20,30),[[4,12,13,10,15],[12,13,14,11,16]],role,rows)

    def test_fringe_and_unreviewed_area_are_unknown(self):
        b=self.band();self.assertEqual(int(b['ink_targets'][4,11]),255)
        self.assertEqual(int(b['ink_targets'][0,0]),255)
        self.assertEqual(int(b['role_targets'][4,7]),255)

    def test_uncertain_gap_is_never_interpolated_into_label(self):
        b=self.band(rows=[(7,9)])
        self.assertTrue(np.all(b['ink_targets'][7:10]==255))

    def test_overlap_conflict_is_excluded(self):
        b=merge_bands([self.band(1),self.band(2)])
        self.assertTrue(b['role_conflicts'].any())
        self.assertTrue(np.all(b['role_targets'][b['role_conflicts']]==255))
        self.assertTrue(np.all(b['ink_targets'][b['role_conflicts']]==255))

    def test_paper_cannot_overwrite_another_possible_band(self):
        a=self.band();b=rasterize_band((20,30),[[4,7,8,6,9],[12,7,8,6,9]],2)
        c=merge_bands([a,b]);self.assertFalse((c['paper']&c['possible']).any())

    def test_invalid_nested_bounds_rejected(self):
        with self.assertRaises(ValueError):
            rasterize_band((20,30),[[4,12,17,10,15],[12,13,14,11,16]],1)


if __name__=='__main__':unittest.main()
