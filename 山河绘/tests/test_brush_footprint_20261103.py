import unittest
import numpy as np
from scripts.brush_footprint_20261103 import footprint_audit


class FootprintTests(unittest.TestCase):
    def run_audit(self,ps,w=.6):
        return footprint_audit(ps,w,(30,30),ppm=10,start_factor=1)

    def test_one_stroke_many_points_counts_once(self):
        s,_,c,_=self.run_audit([[[2,5],[5,5],[10,5],[15,5]]])
        self.assertEqual(int(c.max()),1)
        self.assertEqual(s['repeated_coverage_area_mm2'],0)

    def test_reversed_duplicate_adds_no_new_coverage(self):
        s,r,c,_=self.run_audit([[[2,5],[15,5]],[[15,5],[2,5]]])
        self.assertEqual(r[1]['new_coverage_mm2'],0)
        self.assertEqual(int(c.max()),2)
        self.assertAlmostEqual(s['excess_exposure_fraction'],.5)

    def test_parallel_separated_are_distinct(self):
        s,_,_,_=self.run_audit([[[2,5],[15,5]],[[2,7],[15,7]]])
        self.assertEqual(s['repeated_coverage_area_mm2'],0)

    def test_intersection_is_not_called_entire_duplicate(self):
        s,r,_,_=self.run_audit([[[2,10],[20,10]],[[10,2],[10,20]]])
        self.assertGreater(s['repeated_coverage_area_mm2'],0)
        self.assertFalse(r[1]['review_only_high_overlap'])

    def test_single_stroke_self_crossing_not_distinct_exposures(self):
        _,_,c,_=self.run_audit([[[2,2],[20,20],[2,20],[20,2]]])
        self.assertEqual(int(c.max()),1)

    def test_width_sensitivity_preserves_paths_and_never_deletes(self):
        paths=[[[2,5],[15,5]],[[2,5.7],[15,5.7]]]
        a,_,_,_=self.run_audit(paths,.3);b,_,_,_=self.run_audit(paths,1.2)
        self.assertGreater(b['repeated_coverage_area_mm2'],a['repeated_coverage_area_mm2'])
        self.assertEqual(a['paths'],b['paths'])
        self.assertFalse(b['ready_for_robot'])
        self.assertTrue(b['no_path_deleted'])

    def test_invalid_path_fails_instead_of_clipping(self):
        with self.assertRaises(ValueError):self.run_audit([[[-1,2],[3,2]]])


if __name__=='__main__':unittest.main()
