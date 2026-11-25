import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cv2,numpy as np
from source_cycle_repair_20261104 import inspect_cycles,apply_cycles

class CycleTests(unittest.TestCase):
    def geometry(self):return [np.array([[20,21],[23,21],[23,23],[20,23],[20,21]],float),np.array([[5,22],[20,22]],float),np.array([[23,22],[45,22]],float)]
    def test_source_bright_hole_is_preserved(self):
        a=np.full((60,60,3),255,np.uint8);p=self.geometry()
        for q in p:cv2.polylines(a,[q.astype(np.int32)],False,(0,0,0),1)
        c,accepted=inspect_cycles(a,p);self.assertTrue(c);self.assertEqual(accepted,[])
    def test_dark_filled_junction_can_collapse(self):
        a=np.full((60,60,3),255,np.uint8);p=self.geometry()
        # Smaller loop stays in the fixed 8-square-pixel conservative budget.
        p[0]=np.array([[20,21],[23,21],[23,23],[20,23],[20,21]],float)
        p[2]=np.array([[23,22],[45,22]],float)
        cv2.rectangle(a,(19,20),(24,24),(0,0,0),-1)
        for q in p[1:]:cv2.polylines(a,[q.astype(np.int32)],False,(0,0,0),1)
        rows=[dict(points=q,source_curve_ids=[i],module=0) for i,q in enumerate(p)]
        out,d=apply_cycles(a,rows);self.assertTrue(d['applied_junctions'])
        self.assertEqual(sorted(v for r in out for v in r['source_curve_ids']),[0,1,2])
    def test_white_background_not_accepted_as_dark_junction(self):
        a=np.full((60,60,3),255,np.uint8);p=self.geometry();p[0][:,0]=20+(p[0][:,0]-20)*.5
        _,accepted=inspect_cycles(a,p);self.assertEqual(accepted,[])
    def test_isolated_closed_mark_is_not_turned_into_dot(self):
        a=np.full((60,60,3),255,np.uint8);cv2.rectangle(a,(19,20),(24,24),(0,0,0),-1)
        _,accepted=inspect_cycles(a,self.geometry()[:1]);self.assertEqual(accepted,[])

if __name__=='__main__':unittest.main()
