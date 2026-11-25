# -*- coding: utf-8 -*-
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cv2,numpy as np
from source_stroke_width_20261104 import reconstruct_lines,width_profile

class SourceStrokeTests(unittest.TestCase):
    def image(self):return np.full((64,64,3),255,np.uint8)
    def test_supported_straight_fragments_become_one(self):
        a=self.image();cv2.line(a,(5,20),(45,20),(0,0,0),1)
        curves=[np.array([[5,20],[20,20]],float),np.array([[22,20],[45,20]],float)]
        out,d=reconstruct_lines(a,curves)
        self.assertEqual(len(out),1);self.assertEqual(sorted(out[0]['source_curve_ids']),[0,1])
    def test_parallel_real_lines_stay_separate(self):
        a=self.image();curves=[np.array([[5,y],[45,y]],float) for y in [20,22]]
        for p in curves:cv2.polylines(a,[p.astype(np.int32)],False,(0,0,0),1)
        out,_=reconstruct_lines(a,curves);self.assertEqual(len(out),2)
    def test_white_gap_not_bridged(self):
        a=self.image();curves=[np.array([[5,20],[20,20]],float),np.array([[23,20],[45,20]],float)]
        for p in curves:cv2.polylines(a,[p.astype(np.int32)],False,(0,0,0),1)
        out,_=reconstruct_lines(a,curves);self.assertEqual(len(out),2)
    def test_crossing_lines_not_merged_as_same_stroke(self):
        a=self.image();curves=[np.array([[5,20],[45,20]],float),np.array([[25,5],[25,45]],float)]
        for p in curves:cv2.polylines(a,[p.astype(np.int32)],False,(0,0,0),1)
        out,_=reconstruct_lines(a,curves);self.assertEqual(len(out),2)
    def test_width_responds_to_source_not_fixed_pressure(self):
        a=self.image();cv2.line(a,(5,18),(55,18),(0,0,0),1);cv2.line(a,(5,42),(55,42),(0,0,0),3)
        thin=width_profile(a,[[8,18],[50,18]]);thick=width_profile(a,[[8,42],[50,42]])
        self.assertGreater(np.median(thick['width_px']),np.median(thin['width_px'])+1.)
        self.assertIsNone(thin['pressure_mm'])
    def test_wide_black_junction_is_uncertain(self):
        a=np.zeros((64,64,3),np.uint8);p=width_profile(a,[[8,32],[50,32]])
        self.assertFalse(p['observed'].any());self.assertEqual(p['fallback'],'unmeasured_1px_display_placeholder')
    def test_neighboring_basin_does_not_inflate_single_line_width(self):
        a=self.image();cv2.line(a,(5,20),(55,20),(0,0,0),1);cv2.line(a,(5,23),(55,23),(0,0,0),1)
        p=width_profile(a,[[8,20],[50,20]])
        self.assertLess(np.median(p['width_px']),2.);self.assertTrue(p['neighboring_basin'].any())
    def test_edge_biased_path_recenters_toward_its_own_band(self):
        a=self.image();cv2.line(a,(5,20),(55,20),(0,0,0),1)
        p=width_profile(a,[[8,20.5],[50,20.5]])
        self.assertLess(abs(np.median(p['points'][:,1])-20),.5)
        self.assertLessEqual(np.max(abs(p['normal_shift_px'])),.75)
    def test_measurement_can_leave_already_reconstructed_path_fixed(self):
        a=self.image();cv2.line(a,(5,20),(55,20),(0,0,0),1)
        p=width_profile(a,[[8,20.5],[50,20.5]],recenter=False)
        np.testing.assert_allclose(p['points'],p['original_sample_points'])
        self.assertFalse(p['normal_recentering_applied'])

if __name__=='__main__':unittest.main()
