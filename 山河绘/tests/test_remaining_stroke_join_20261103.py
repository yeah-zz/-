import unittest
import numpy as np,cv2
from scripts.remaining_stroke_join_20261103 import join_remaining,raster

class RemainingJoinTests(unittest.TestCase):
    def image(self,paths):
        a=np.ones((60,60),np.float32)
        for p in paths:cv2.polylines(a,[np.array(p,np.int32)],False,.1,2)
        return a
    def test_source_supported_continuation(self):
        ps=[[[5,20],[15,20]],[[17,20],[27,20]]]
        a=self.image([[[5,20],[27,20]]]);out,s=join_remaining(ps,a,np.zeros(a.shape),.3)
        self.assertEqual(len(out),1);self.assertEqual(len(s['accepted']),1)
        self.assertEqual(sorted(out[0]['old_remaining_indices']),[0,1])
    def test_white_gap_not_joined(self):
        ps=[[[5,20],[15,20]],[[18,20],[28,20]]]
        a=np.ones((60,60),np.float32);a[20,5:16]=.1;a[20,18:29]=.1
        out,s=join_remaining(ps,a,np.zeros(a.shape),.3)
        self.assertEqual(len(out),2)
    def test_parallel_neighbors_not_joined(self):
        ps=[[[5,20],[15,20]],[[5,22],[15,22]]];a=self.image(ps)
        self.assertEqual(len(join_remaining(ps,a,np.zeros(a.shape),.3)[0]),2)
    def test_drawn_region_protected(self):
        ps=[[[5,20],[15,20]],[[17,20],[27,20]]];a=self.image([[[5,20],[27,20]]])
        done=np.zeros(a.shape);done[20,16]=1
        self.assertEqual(len(join_remaining(ps,a,done,.3)[0]),2)
    def test_maximum_length_and_closed_contour(self):
        ps=[[[5,20],[15,20]],[[17,20],[27,20]],[[40,40],[50,40],[50,50],[40,40]]]
        a=self.image([[[5,20],[27,20]],ps[-1]])
        out,s=join_remaining(ps,a,np.zeros(a.shape),1.,maximum_mm=19.8)
        self.assertEqual(len(out),3)

if __name__=='__main__':unittest.main()
