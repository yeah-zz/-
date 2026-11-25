# -*- coding: utf-8 -*-
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from whole_stroke_instance_20261105 import *

class WholeInstance(unittest.TestCase):
    def test_query_crop_does_not_need_whole_reference_geometry(self):
        rgb=np.full((120,120,3),250,np.uint8)
        x,origin=crop_query(rgb,[60,60],80,[55,60],[65,60])
        self.assertEqual(x.shape,(5,96,96));np.testing.assert_array_equal(origin,[20,20])
        self.assertNotEqual(np.unravel_index(x[3].argmax(),x[3].shape),np.unravel_index(x[4].argmax(),x[4].shape))

    def test_partial_real_ownership_leaves_unmarked_pixels_unknown(self):
        rgb=np.full((120,120,3),250,np.uint8);path=np.array([[30,60],[90,60]],float)
        sample=make_example(rgb,path,[],np.zeros((120,120),np.uint8),np.random.default_rng(1),False)
        self.assertIsNotNone(sample);image,mask,valid,curve,presence,ep=sample
        self.assertLess(np.mean(valid>0),.1);self.assertTrue((valid[mask>100]>0).all());self.assertEqual(ep,0.)
        self.assertEqual(curve.shape,(33,2))

    def test_joint_network_has_simultaneous_complete_curve_and_mask_outputs(self):
        net=WholeInstanceNet();x=np.zeros((1,5,96,96),np.uint8)
        yy,xx=np.mgrid[:96,:96];x[0,3]=np.uint8(np.exp(-((xx-44)**2+(yy-48)**2)/4)*255)
        x[0,4]=np.uint8(np.exp(-((xx-52)**2+(yy-48)**2)/4)*255)
        out=net(torch.tensor(x));self.assertEqual(out['curve'].shape,(1,33,2));self.assertEqual(out['mask'].shape,(1,1,96,96))
        self.assertTrue(torch.isfinite(out['curve']).all())

    def test_unknown_pixel_targets_do_not_change_loss(self):
        net=WholeInstanceNet();x=torch.zeros((1,5,96,96));out=net(x)
        mask=torch.zeros((1,96,96));valid=mask.clone();valid[:,45:50,20:70]=255
        curve=torch.stack([torch.linspace(.2,.8,33),torch.full((33,),.5)],-1)[None]
        batch=[x,mask,valid,curve,torch.ones(1),torch.zeros(1)]
        first=losses(out,batch);altered=mask.clone();altered[:,0:20,:]=255
        batch[1]=altered;second=losses(out,batch)
        self.assertAlmostEqual(float(first.detach()),float(second.detach()),places=6)

if __name__=='__main__':unittest.main()
