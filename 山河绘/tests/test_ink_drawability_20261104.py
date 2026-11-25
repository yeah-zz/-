# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ink_drawability_20261104 import map_saved_job,exact_segment_repeats,drawability_gate,sensitivity

class TestDrawability(unittest.TestCase):
    def test_mapping_uses_saved_corners(self):
        d=dict(meta=dict(image_size=[11,11],canvas_corners=[[90,40],[70,40],[70,10],[90,10]],
                         ink_execution=dict(base_strokes=[dict(points=[[0,0],[10,10]])])))
        paths,m=map_saved_job(d)
        np.testing.assert_allclose(paths[0],[[0,0],[20,30]],atol=1e-5)
        np.testing.assert_allclose(m['canvas_mm'],[20,30],atol=1e-5)

    def test_reverse_duplicate_is_identical_coverage(self):
        p=np.array([[1,2],[5,2]],float)
        out=exact_segment_repeats([p,p[::-1]])
        self.assertEqual(out['repeated_identical_segments_across_curves'],1)

    def test_near_parallel_is_not_reported_as_exact_duplicate(self):
        p=np.array([[1,2],[5,2]],float)
        self.assertEqual(exact_segment_repeats([p,p+[0,.2]])['repeated_identical_segments'],0)

    def test_legacy_width_cannot_substitute_for_current_measurement(self):
        out=drawability_gate(dict(legacy_estimated_width_mm=2.04))
        self.assertFalse(out['ready_for_robot'])
        self.assertFalse(out['legacy_estimate_fallback_permitted'])
        self.assertIn('width_p95_mm',out['missing_fields'])

    def test_straight_line_profile_without_dense_patch_is_insufficient(self):
        p=dict(brush_id='b',paper_id='p',ink_condition='i',width_p95_mm=.5,
               start_diameter_p95_mm=.8,end_diameter_p95_mm=.7,dense_patch_passed=False)
        self.assertEqual(drawability_gate({},p)['status'],'dense_patch_not_passed')

    def test_separate_centerlines_can_merge_as_finite_bands(self):
        paths=[np.array([[2,2],[8,2]],float),np.array([[2,3],[8,3]],float)]
        rows,_=sensitivity(paths,[10,6],widths=[.2,2.],start_factors=[1.],ppm=12.)
        self.assertEqual(rows[0]['summary']['repeated_coverage_area_mm2'],0)
        self.assertGreater(rows[1]['summary']['repeated_coverage_area_mm2'],0)

if __name__=='__main__':unittest.main()
