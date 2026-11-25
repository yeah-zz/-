# -*- coding: utf-8 -*-
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np,cv2
from module_stroke_core_20261104 import length,samples
from long_stroke_recovery_20261105 import recover_long_strokes

def row(p,module=0,uncertain=False):return dict(points=np.array(p,float),module=module,uncertain_module=uncertain)
def image(paths,value=0,background=255,width=1):
    rgb=np.full((60,70,3),background,np.uint8)
    for p in paths:cv2.polylines(rgb,[np.rint(p).astype(np.int32)],False,(value,)*3,width,cv2.LINE_AA)
    return rgb

class LongRecovery(unittest.TestCase):
    def test_source_supported_quarter_turn_is_not_forced_straight(self):
        rows=[row([[5,15],[20,15]]),row([[25,20],[25,40]])]
        rgb=image([r['points'] for r in rows]);cv2.ellipse(rgb,(20,20),(5,5),0,270,360,(0,0,0),1,cv2.LINE_AA)
        out,d=recover_long_strokes(rgb,rows)
        self.assertEqual(len(out),1);self.assertEqual(d['types'].get('curved_source_supported_gap'),1)
        # Original incoming and outgoing straight portions are left untouched.
        self.assertGreater(length(out[0]['points']),35)

    def test_overlap_is_stitched_without_drawing_backwards(self):
        rows=[row([[5,20],[23,20]]),row([[20,20],[42,20]])]
        out,d=recover_long_strokes(image([[[5,20],[42,20]]]),rows)
        self.assertEqual(len(out),1);self.assertEqual(d['types'].get('same_band_overlap'),1)
        q=out[0]['points'];self.assertLess(abs(length(q)-37),.5)
        self.assertTrue((np.diff(q[:,0])>=-1e-6).all() or (np.diff(q[:,0])<=1e-6).all())

    def test_supported_gap_in_uncertain_same_region_can_continue(self):
        rows=[row([[5,20],[20,20]],uncertain=True),row([[25,20],[42,20]],uncertain=True)]
        out,d=recover_long_strokes(image([[[5,20],[42,20]]]),rows)
        self.assertEqual(len(out),1);self.assertEqual(d['joins_in_uncertain_same_region'],1)

    def test_faint_line_is_not_globally_thresholded_away(self):
        rows=[row([[5,20],[20,20]]),row([[24,20],[42,20]])]
        out,_=recover_long_strokes(image([[[5,20],[42,20]]],value=215),rows)
        self.assertEqual(len(out),1)

    def test_white_gap_and_flat_wash_not_connected(self):
        rows=[row([[5,20],[20,20]]),row([[25,20],[42,20]])]
        out,_=recover_long_strokes(image([r['points'] for r in rows]),rows);self.assertEqual(len(out),2)
        rows=[{**r,'uncertain_module':True} for r in rows]
        rgb=image([r['points'] for r in rows],value=20,background=60,width=1)
        out,_=recover_long_strokes(rgb,rows);self.assertEqual(len(out),2)

    def test_parallel_strokes_do_not_become_one_zigzag(self):
        rows=[row([[5,20],[30,20]]),row([[7,22],[32,22]])]
        out,_=recover_long_strokes(image([r['points'] for r in rows]),rows);self.assertEqual(len(out),2)

    def test_different_modules_still_forbidden(self):
        rows=[row([[5,20],[20,20]],0),row([[20,20],[42,20]],1)]
        out,_=recover_long_strokes(image([r['points'] for r in rows]),rows);self.assertEqual(len(out),2)

    def test_crossing_continuation_keeps_the_branch_separate(self):
        rows=[row([[5,25],[25,25]]),row([[25,25],[45,25]]),row([[25,25],[25,45]])]
        out,_=recover_long_strokes(image([r['points'] for r in rows]),rows)
        self.assertEqual(len(out),2)
        self.assertIn([0,1],[sorted(r['source_curve_ids']) for r in out])

    def test_ambiguous_y_junction_not_guessed(self):
        rows=[row([[5,25],[25,25]]),row([[25,25],[45,20]]),row([[25,25],[45,30]])]
        out,d=recover_long_strokes(image([r['points'] for r in rows]),rows)
        self.assertEqual(len(out),3);self.assertTrue(d['pending'])

    def test_y_ambiguity_survives_one_branch_failing_image_evidence(self):
        rows=[row([[5,25],[25,25]]),row([[25,25],[45,20]]),row([[25,25],[45,30]])]
        rgb=image([rows[0]['points'],rows[1]['points']])
        out,d=recover_long_strokes(rgb,rows)
        self.assertEqual(len(out),3);self.assertTrue(d['pending'])

    def test_new_loop_and_closed_motif_are_not_synthesized(self):
        rows=[row([[10,10],[25,10],[25,25]]),row([[25,25],[10,25],[10,10]])]
        out,_=recover_long_strokes(image([r['points'] for r in rows]),rows)
        self.assertEqual(len(out),2)
        closed=row([[35,10],[45,10],[45,20],[35,10]])
        out,_=recover_long_strokes(image([closed['points']]),[closed])
        np.testing.assert_array_equal(out[0]['points'],closed['points'])

    def test_three_fragment_chain_keeps_provenance_and_original_coverage(self):
        rows=[row([[5,20],[20,20]]),row([[18,20],[35,20]]),row([[33,20],[50,20]])]
        out,_=recover_long_strokes(image([[[5,20],[50,20]]]),rows)
        self.assertEqual(len(out),1);self.assertEqual(sorted(out[0]['source_curve_ids']),[0,1,2])
        self.assertLess(abs(length(out[0]['points'])-45),.5)

if __name__=='__main__':unittest.main()
