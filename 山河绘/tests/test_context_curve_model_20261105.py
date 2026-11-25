import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np,cv2,torch
from context_curve_model_20261105 import pack,ContextCurveNet
from context_curve_data_20261105 import candidate_label
from context_curve_apply_20261105 import metrics,select

class AlwaysNoise(torch.nn.Module):
    def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(1))
    def forward(self,x):return torch.tensor([[-10.,10.]],device=x.device).repeat(len(x),1)+self.dummy*0

class ContextTests(unittest.TestCase):
    def test_pack_has_whole_curve_and_three_local_views(self):
        rgb=np.full((80,80,3),250,np.uint8);p=np.array([[20,60],[40,15],[65,60]],float)
        z=pack(rgb,p,[]);self.assertEqual(z.shape,(4,5,48,48));self.assertEqual(z.dtype,np.uint8)
        self.assertGreater(z[:,3].max(),0);self.assertEqual(z[:,4].max(),0)
    def test_unmatched_unmarked_is_unknown(self):
        label,_=candidate_label([[10,10],[30,10]],[np.array([[50,40],[60,40]])],np.zeros((80,80),np.uint8))
        self.assertEqual(label,-1)
    def test_explicit_background_can_label_negative(self):
        bg=np.zeros((80,80),np.uint8);bg[5:20,5:40]=1
        label,_=candidate_label([[10,10],[30,10]],[np.array([[50,40],[60,40]])],bg)
        self.assertEqual(label,1)
    def test_actual_predictions_coverage_not_teacher_passthrough(self):
        ref=[np.array([[10,20],[50,20]],float)];bg=np.zeros((80,80),np.uint8)
        good=metrics([ref[0]],ref,bg);miss=metrics([np.array([[10,35],[50,35]],float)],ref,bg)
        self.assertEqual(good['mean_reference_coverage'],1.);self.assertEqual(miss['mean_reference_coverage'],0.)
    def test_background_length_integration(self):
        bg=np.ones((80,80),np.uint8);ref=[np.array([[10,50],[50,50]],float)]
        d=metrics([np.array([[10,10],[50,10]],float)],ref,bg)
        self.assertAlmostEqual(d['trace_length_inside_explicit_background_px'],40.,places=5)
    def test_source_main_outline_guard_can_override_classifier(self):
        rgb=np.full((96,96,3),245,np.uint8);p=np.array([[10,70],[25,15],[45,55],[70,20],[85,70]],float)
        cv2.polylines(rgb,[p.astype(np.int32)],False,(20,20,20),1)
        rows=[dict(points=p,module=0,source_curve_ids=[0])]
        result=select(AlwaysNoise(),.97,rgb,rows)
        self.assertEqual(result['selected'],[0]);self.assertEqual(result['contour_guards_restored'],1)
    def test_forward_uses_two_classes(self):
        torch.set_num_threads(2);m=ContextCurveNet().eval()
        with torch.no_grad():z=m(torch.zeros(2,4,5,48,48))
        self.assertEqual(tuple(z.shape),(2,2));self.assertTrue(torch.isfinite(z).all())

if __name__=='__main__':unittest.main()
