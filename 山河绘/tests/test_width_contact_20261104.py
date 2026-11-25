# -*- coding: utf-8 -*-
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from width_contact_core_20261104 import bounded_simplify,contact_envelope,plan_for_width,length

class WidthContactTests(unittest.TestCase):
    def test_corner_and_endpoints_not_rounded_away(self):
        p=np.array([[1,1],[2,1],[3,1],[3,2],[3,3]],float)
        q=bounded_simplify(p)
        np.testing.assert_allclose(q,[[1,1],[3,1],[3,3]])
        self.assertAlmostEqual(length(p),length(q))

    def test_taper_stays_inside_original_curve(self):
        p=np.array([[1,2],[3,2],[3,4]],float);e=contact_envelope(p)
        np.testing.assert_allclose(e['points'][[0,-1]],p[[0,-1]])
        self.assertLess(e['relative_width'][0],1)
        self.assertLess(e['relative_width'][-1],1)
        self.assertEqual(e['physical_compression_mm'],None)
        self.assertTrue(e['no_extension_outside_source_curve'])

    def test_closed_contour_no_fake_gap(self):
        e=contact_envelope([[1,1],[2,1],[2,2],[1,1]])
        self.assertTrue(e['closed']);self.assertTrue((e['relative_width']==1).all())

    def test_short_isolated_detail_is_not_deleted_by_length(self):
        curves=[dict(points=[[1,1],[1.3,1]],module=0)]
        r=plan_for_width(curves,[dict(curve=0)],2.)
        self.assertEqual(r['selected'],[0]);self.assertEqual(r['deferred'],[])

    def test_crowded_detail_is_explicitly_deferred(self):
        curves=[dict(points=[[1,1],[11,1]],module=0),dict(points=[[2,1],[3,1]],module=0)]
        r=plan_for_width(curves,[dict(curve=0),dict(curve=1)],2.)
        self.assertEqual(r['selected'],[0]);self.assertEqual(r['deferred'],[1])
        self.assertTrue(r['complete_extraction_preserved']);self.assertFalse(r['ready_for_robot'])

    def test_overlapping_long_candidates_flag_conflict_not_pass(self):
        curves=[dict(points=[[1,1],[11,1]],module=0),dict(points=[[1,1.1],[11,1.1]],module=0)]
        r=plan_for_width(curves,[dict(curve=0),dict(curve=1)],2.)
        self.assertEqual(set(r['selected']),{0,1});self.assertGreater(r['structural_conflicts'],0)

    def test_module_contiguity_and_complete_partition(self):
        curves=[dict(points=[[2+i,1],[2+i,4]],module=i%2) for i in range(4)]
        r=plan_for_width(curves,[dict(curve=i) for i in [0,2,1,3]],.3)
        self.assertEqual(sorted(r['selected']+r['deferred']),list(range(4)))
        labels=[curves[i]['module'] for i in r['selected']]
        self.assertLessEqual(sum(a!=b for a,b in zip(labels,labels[1:])),1)

if __name__=='__main__':unittest.main()
