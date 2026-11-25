# -*- coding: utf-8 -*-
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cv2,numpy as np
from landscape_stroke_refine_20261104 import deduplicate,join_curves,refine,same_band
from width_contact_core_20261104 import max_deviation

class LandscapeStrokeTests(unittest.TestCase):
    def image(self):return np.full((80,80,3),255,np.uint8)
    def draw(self,a,paths):
        for p in paths:cv2.polylines(a,[np.rint(p).astype(np.int32)],False,(0,0,0),1)
    def rows(self,paths):return [dict(points=np.array(p,float),source_curve_ids=[i],module=0) for i,p in enumerate(paths)]
    def test_reversed_contained_duplicate_accounted(self):
        a=self.image();p=[[[5,20],[65,20]],[[50,20],[15,20]]];self.draw(a,p)
        r,_=deduplicate(a,p);self.assertEqual(len(r),1);self.assertEqual(r[0]['source_curve_ids'],[0,1])
    def test_parallel_real_lines_stay_separate(self):
        a=self.image();p=[[[5,20],[65,20]],[[15,22],[50,22]]];self.draw(a,p)
        r,_=deduplicate(a,p);self.assertEqual(len(r),2)
    def test_bright_separation_is_explicit_evidence(self):
        g=np.ones((80,80),np.float32);g[20,5:66]=0;g[22,5:66]=0
        q=np.array([[x,20] for x in range(10,50)],float)
        self.assertFalse(same_band(g,q,q+[0,2]).any())
    def test_short_pine_needles_and_crossing_do_not_disappear(self):
        a=self.image();p=[[[5,20],[65,20]],[[30,17],[30,20]],[[40,20],[42,23]]];self.draw(a,p)
        r,_=deduplicate(a,p);self.assertEqual(len(r),3)
    def test_closed_motif_not_absorbed_by_open_line(self):
        a=self.image();p=[[[5,20],[65,20]],[[20,20],[21,20],[21,21],[20,20]]];self.draw(a,p)
        r,_=deduplicate(a,p);self.assertEqual(len(r),2)
    def test_join_supported_gap(self):
        a=self.image();p=[[[5,20],[25,20]],[[27,20],[65,20]]];self.draw(a,[[[5,20],[65,20]]])
        r,d=join_curves(a,self.rows(p));self.assertEqual(len(r),1);self.assertEqual(len(d['accepted']),1)
    def test_white_gap_not_connected(self):
        a=self.image();p=[[[5,20],[25,20]],[[27,20],[65,20]]];self.draw(a,p)
        r,_=join_curves(a,self.rows(p));self.assertEqual(len(r),2)
    def test_owners_prevent_cross_object_join(self):
        a=self.image();p=[[[5,20],[25,20]],[[27,20],[65,20]]];self.draw(a,[[[5,20],[65,20]]])
        rows=self.rows(p);rows[1]['module']=1
        r,_=join_curves(a,rows);self.assertEqual(len(r),2)
    def test_competing_fork_stays_pending(self):
        a=self.image();p=[[[5,30],[25,30]],[[26,30],[50,25]],[[26,30],[50,35]]];self.draw(a,p)
        r,d=join_curves(a,self.rows(p));self.assertTrue(d['pending']);self.assertEqual(len(r),3)
    def test_bent_rock_line_not_straightened_or_deleted(self):
        a=self.image();p=np.array([[8,40],[20,10],[35,35],[50,14],[65,45]],float);self.draw(a,[p])
        r,d=refine(a,[p]);self.assertEqual(len(r),1)
        self.assertGreater(len(r[0]['points']),3);self.assertLess(max_deviation(p,r[0]['points']),1.)
        self.assertEqual(r[0]['source_curve_ids'],[0]);self.assertIsNone(d['physical_pressure_mm'])
    def test_tiny_open_curve_remains_nonzero(self):
        a=self.image();r,_=refine(a,[[[20,20],[20.1,20]]]);self.assertGreater(np.linalg.norm(r[0]['points'][-1]-r[0]['points'][0]),0)
    def test_closed_small_motif_survives_simplification(self):
        a=self.image();p=np.array([[20,20],[20.1,20],[20.1,20.1],[20,20]],float)
        r,_=refine(a,[p]);self.assertGreater(len(r[0]['points']),3)
        self.assertGreater(np.linalg.norm(np.diff(r[0]['points'],axis=0),axis=1).sum(),0)
    def test_recentering_does_not_create_long_jagged_detour(self):
        a=self.image();p=np.array([[5,20],[20,20],[35,20],[65,20]],float)
        self.draw(a,[p]);r,d=refine(a,[p]);l=np.linalg.norm(np.diff(r[0]['points'],axis=0),axis=1).sum()
        self.assertLessEqual(l,60.6)

if __name__=='__main__':unittest.main()
