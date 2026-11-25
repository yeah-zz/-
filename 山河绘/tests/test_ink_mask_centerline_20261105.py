# -*- coding: utf-8 -*-
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ink_mask_centerline_20261105 import *

class InkReadout(unittest.TestCase):
    def image(self,paths):
        rgb=np.full((80,80,3),250,np.uint8)
        for p in paths:cv2.polylines(rgb,[np.array(p,np.int32)],False,(25,25,25),2,cv2.LINE_AA)
        return rgb
    def read(self,rgb,back=(25,35),front=(35,35),mode='model_source'):
        return read_instance(rgb,np.full((96,96),.95,np.float32),[0,0],80,back,front,mode)

    def test_uniform_paper_does_not_create_centerlines(self):
        out=self.read(np.full((80,80,3),250,np.uint8));self.assertEqual(len(out['points']),0)
        self.assertFalse(out['eligible_for_replacement'])
        dark=self.read(np.full((80,80,3),30,np.uint8));self.assertEqual(len(dark['points']),0)

    def test_broad_opaque_ink_does_not_become_a_hollow_double_edge(self):
        rgb=np.full((80,80,3),250,np.uint8);cv2.line(rgb,(10,35),(68,35),(15,15,15),7,cv2.LINE_AA)
        out=self.read(rgb);self.assertGreater(length(out['points']),45)
        self.assertLess(np.ptp(out['points'][:,1]),2)

    def test_real_ink_is_followed_instead_of_mask_center(self):
        rgb=self.image([[[12,35],[67,35]]]);out=self.read(rgb)
        self.assertGreater(length(out['points']),48)
        self.assertLess(abs(np.median(out['points'][:,1])-35),1.)

    def test_bright_gap_not_bridged_even_when_model_mask_spans_it(self):
        rgb=self.image([[[10,35],[35,35]],[[41,35],[67,35]]]);out=self.read(rgb,back=(20,35),front=(26,35))
        self.assertLess(out['points'][:,0].max(),38)

    def test_parallel_bands_do_not_become_a_zigzag(self):
        rgb=self.image([[[10,35],[68,35]],[[10,41],[68,41]]]);out=self.read(rgb)
        self.assertLess(np.ptp(out['points'][:,1]),2)
        self.assertLess(out['points'][:,1].max(),38)

    def test_ambiguous_y_is_marked_not_adopted(self):
        rgb=self.image([[[10,35],[35,35]],[[35,35],[65,29]],[[35,35],[65,41]]])
        out=self.read(rgb,back=(20,35),front=(28,35))
        self.assertEqual(out['status'],'ambiguous_branch');self.assertFalse(out['eligible_for_replacement'])

    def test_no_model_source_control_agrees_for_uniform_high_owner(self):
        rgb=self.image([[[10,35],[68,35]]]);a=self.read(rgb);b=self.read(rgb,mode='source_only')
        np.testing.assert_allclose(a['points'],b['points'])

    def test_source_control_does_not_depend_on_probability_mask(self):
        rgb=self.image([[[10,35],[68,35]]]);a=self.read(rgb,mode='source_only')
        b=read_instance(rgb,np.zeros((96,96),np.float32),[0,0],80,(25,35),(35,35),'source_only')
        np.testing.assert_allclose(a['points'],b['points'])
        c=read_instance(rgb,np.zeros((96,96),np.float32),[0,0],80,(25,35),(35,35),'model_source')
        self.assertEqual(len(c['points']),0)

if __name__=='__main__':unittest.main()
