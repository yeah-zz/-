import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np,cv2
from source_evidence_filter_20261104 import filter_curves

class EvidenceFilterTests(unittest.TestCase):
    def image(self):return np.full((96,96,3),245,np.uint8)
    def rows(self,p):return [dict(points=np.asarray(q,float),source_curve_ids=[i],module=0) for i,q in enumerate(p)]
    def draw(self,a,p,color=(20,20,20),width=1):cv2.polylines(a,[np.rint(p).astype(np.int32)],False,color,width,cv2.LINE_AA)
    def test_empty_paper_candidate_rejected(self):
        a=self.image();r=filter_curves(a,self.rows([[[15,40],[70,40]]]))
        self.assertEqual(r['removed'],[0])
    def test_bent_main_outline_preserved(self):
        a=self.image();p=[[8,65],[25,15],[45,55],[65,10],[85,70]];self.draw(a,p)
        r=filter_curves(a,self.rows([p]));self.assertEqual(r['selected'],[0])
    def test_faint_texture_preserved(self):
        a=self.image();p=[[15,65],[22,50],[27,36],[35,20]];self.draw(a,p,(238,238,238))
        r=filter_curves(a,self.rows([p]));self.assertEqual(r['selected'],[0])
    def test_short_real_needle_preserved(self):
        a=self.image();p=[[35,42],[39,37]];self.draw(a,p)
        r=filter_curves(a,self.rows([p]));self.assertEqual(r['selected'],[0])
    def test_true_parallel_cun_lines_not_collapsed(self):
        a=self.image();p=[[[15,40],[75,40]],[[15,42],[75,42]]]
        for q in p:self.draw(a,q)
        r=filter_curves(a,self.rows(p));self.assertEqual(r['selected'],[0,1]);self.assertEqual(r['duplicates'],[])
    def test_broad_single_band_duplicate_suppressed(self):
        a=self.image();self.draw(a,[[15,40],[75,40]],width=5)
        p=[[[15,40],[75,40]],[[17,41],[73,41]]]
        r=filter_curves(a,self.rows(p));self.assertEqual(r['selected'],[0]);self.assertEqual(r['decisions'][1]['disposition'],'duplicate')
    def test_one_sided_color_outline_preserved(self):
        a=self.image();a[40:]=[95,165,175];p=[[10,40],[85,40]]
        r=filter_curves(a,self.rows([p]));self.assertEqual(r['selected'],[0])
    def test_true_closed_rock_contour_preserved(self):
        a=self.image();p=[[20,65],[30,15],[55,20],[75,65],[20,65]];self.draw(a,p)
        r=filter_curves(a,self.rows([p]));self.assertEqual(r['selected'],[0])
    def test_selected_geometry_is_not_replaced_by_straight_chord(self):
        a=self.image();p=np.array([[8,65],[25,15],[45,55],[65,10],[85,70]],float);self.draw(a,p)
        rows=self.rows([p]);r=filter_curves(a,rows);np.testing.assert_array_equal(rows[r['selected'][0]]['points'],p)
    def test_moss_dot_is_not_rejected_just_for_shortness(self):
        a=self.image();cv2.circle(a,(30,30),2,(20,20,20),-1)
        r=filter_curves(a,self.rows([[[30,29],[30,31]]]))
        self.assertEqual(r['selected'],[0])

if __name__=='__main__':unittest.main()
