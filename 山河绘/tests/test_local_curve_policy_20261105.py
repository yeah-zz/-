# -*- coding: utf-8 -*-
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from local_curve_policy_20261105 import *
from local_curve_replacement_20261105 import whole_guard

class LocalPolicy(unittest.TestCase):
    def source(self):
        rgb=np.full((64,64,3),255,np.uint8);cv2.line(rgb,(5,24),(55,24),(0,0,0),1,cv2.LINE_AA);return rgb
    def rows(self):
        return [dict(points=np.array(p,float),phase=1,module=0,source_curve_ids=[i],length_px=15)
            for i,p in enumerate([[[5,24],[25,24]],[[25,24],[55,24]]])]

    def test_real_partial_ends_never_become_stop_targets(self):
        x,y,k=examples(self.source(),[[5,24],[55,24]],np.random.default_rng(1),40,False,True)
        self.assertNotIn(STOP,y);self.assertTrue(all(t=='next_point' for t in k))

    def test_recovery_states_require_more_than_fixed_forward_motion(self):
        x,y,k=examples(self.source(),[[5,24],[55,24]],np.random.default_rng(2),80,False,True)
        self.assertGreater(len(set(y)),15);self.assertEqual(x[0].shape,(5,48,48))

    def test_network_and_loss_finite(self):
        net=LocalCurveNet();x=torch.zeros((2,5,48,48),dtype=torch.uint8);out=net(x)
        value=loss(out,torch.tensor([STOP,8*GRID+12]));value.backward()
        self.assertEqual(out.shape,(2,STOP+1));self.assertTrue(torch.isfinite(value))

    def test_supported_whole_curve_can_replace_two_inputs(self):
        rows=self.rows();proposal=dict(points=np.array([[5,24],[55,24]],float),stop_forward='model_stop',stop_backward='model_stop')
        out,d=whole_guard(self.source(),rows,[proposal])
        self.assertEqual(len(out),1);self.assertEqual(d['selected_proposals'],[0])
        self.assertEqual(out[0]['source_curve_ids'],[0,1])

    def test_blank_crossing_is_rejected_whole_without_pointwise_fragments(self):
        rows=self.rows();bad=dict(points=np.array([[5,24],[25,24],[45,45]],float),stop_forward='model_stop',stop_backward='model_stop')
        out,d=whole_guard(self.source(),rows,[bad]);self.assertEqual(len(out),2);self.assertEqual(d['selected_proposals'],[])
        for actual,expected in zip(out,rows):np.testing.assert_array_equal(actual['points'],expected['points'])

    def test_cross_module_and_unfinished_neural_trace_are_rejected(self):
        rows=self.rows();rows[1]['module']=1
        p=dict(points=np.array([[5,24],[55,24]],float),stop_forward='model_stop',stop_backward='model_stop')
        out,d=whole_guard(self.source(),rows,[p]);self.assertEqual(d['reasons'].get('cross_module_candidate'),1)
        p['stop_forward']='step_limit';out,d=whole_guard(self.source(),rows,[p]);self.assertEqual(d['selected_proposals'],[])

if __name__=='__main__':unittest.main()
