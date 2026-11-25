import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cv2,numpy as np,torch
from context_relation_20261105 import eligible,suppress

class AlwaysDuplicate(torch.nn.Module):
    def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(1))
    def forward(self,x):return torch.tensor([[0.,15.,0.]],device=x.device).repeat(len(x),1)+self.dummy*0

class RelationTests(unittest.TestCase):
    def image(self):return np.full((80,80,3),255,np.uint8)
    def test_contained_same_band_eligible(self):
        a=self.image();cv2.line(a,(5,30),(65,30),(0,0,0),5)
        self.assertIsNotNone(eligible(a,[[15,31],[55,31]],[[5,30],[65,30]]))
    def test_independent_parallel_bright_gap_preserved(self):
        a=self.image();cv2.line(a,(5,30),(65,30),(0,0,0),1);cv2.line(a,(5,32),(65,32),(0,0,0),1)
        self.assertIsNone(eligible(a,[[15,32],[55,32]],[[5,30],[65,30]]))
    def test_short_continuation_not_duplicate_at_endpoint(self):
        a=self.image();cv2.line(a,(5,30),(65,30),(0,0,0),1)
        self.assertIsNone(eligible(a,[[23.5,30],[25,30]],[[25,30],[65,30]]))
    def test_closed_motif_protected(self):
        a=self.image();p=[[20,20],[22,20],[22,22],[20,20]]
        self.assertIsNone(eligible(a,p,p))
    def test_representatives_are_retained_and_input_not_mutated(self):
        a=self.image();cv2.line(a,(5,30),(65,30),(0,0,0),5)
        ps=[np.array([[5,30+y],[65,30+y]],float) for y in [0,.5,1.]]
        rows=[dict(points=p.copy(),module=0,source_curve_ids=[i]) for i,p in enumerate(ps)]
        result=suppress(AlwaysDuplicate(),.99,a,rows)
        self.assertTrue(result['removed']);self.assertTrue(all(r['representative'] in result['selected'] for r in result['events']))
        for a,b in zip(ps,rows):np.testing.assert_array_equal(a,b['points'])

if __name__=='__main__':unittest.main()
