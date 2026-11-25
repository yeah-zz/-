import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cv2,numpy as np
from native_tree_restore_20261104 import restore

class NativeTreeTests(unittest.TestCase):
    def test_clear_missing_needle_can_be_recovered(self):
        a=np.full((80,80,3),255,np.uint8)
        cv2.line(a,(50,5),(50,70),(0,0,0),1)
        cv2.line(a,(30,25),(50,25),(0,0,0),1,cv2.LINE_AA)
        rows,d=restore(a,[np.array([[50,5],[50,70]],float)],np.ones((80,80)),[0,0,80,80])
        self.assertTrue(rows);self.assertTrue(any(abs(np.array(r['points'])[:,1].mean()-25)<1 for r in rows))
    def test_blank_source_does_not_get_needles(self):
        a=np.full((80,80,3),255,np.uint8)
        rows,_=restore(a,[np.array([[50,5],[50,70]],float)],np.ones((80,80)),[0,0,80,80])
        self.assertEqual(rows,[])
    def test_tree_route_is_required(self):
        a=np.full((80,80,3),255,np.uint8);cv2.line(a,(30,25),(50,25),(0,0,0),1)
        rows,_=restore(a,[np.array([[50,5],[50,70]],float)],np.zeros((80,80)),[0,0,80,80])
        self.assertEqual(rows,[])

if __name__=='__main__':unittest.main()
