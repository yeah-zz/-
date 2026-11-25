# -*- coding: utf-8 -*-
"""Image-space safety/regression fixtures. Never runs a robot executor."""
from pathlib import Path
import sys
from unittest import TestCase
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stroke_unit_core_20261102 import process,pair_features,match_whole_band,union_raster,FEATURES


def image(paths,width=1):
    gray=np.full((128,128),255,np.uint8)
    for p in paths:cv2.polylines(gray,[np.array(p,np.int32)],False,20,width)
    return np.repeat(gray[:,:,None],3,axis=2)


def score(label):
    def predict(x):
        p=np.zeros((len(x),3),np.float32);p[:,label]=1;return p
    return predict


class StrokeUnitTests(TestCase):
    def run_case(self,paths,label,rgb=None):
        p=[np.array(x,float) for x in paths]
        return process(rgb if rgb is not None else image(paths),p,score(label),dict(continuous=.9,duplicate=.9))

    def test_feature_vector_is_finite_and_symmetric_length(self):
        p=np.array([[20,30],[30,30],[40,30]],float);q=p+[0,4]
        f=pair_features(p,q,image([p,q])[:,:,0].astype(float)/255)
        self.assertEqual(len(f),len(FEATURES));self.assertTrue(np.isfinite(f).all())

    def test_white_separated_parallel_strokes_cannot_be_deduplicated(self):
        p=np.array([[20,30],[40,30]],float);q=p+[0,2]
        rgb=image([p,q]);okay,_=match_whole_band(p,q,rgb[:,:,0].astype(float)/255)
        self.assertFalse(okay)
        out,decisions,_=self.run_case([p,q],1,rgb)
        self.assertEqual(len(out),2);self.assertFalse(decisions['duplicate_decisions'])

    def test_same_ink_ribbon_duplicate_requires_neural_and_source_agreement(self):
        p=np.array([[20,30],[40,30]],float);q=p+[0,.2]
        rgb=image([p],5)
        out,decisions,_=self.run_case([p,q],1,rgb)
        self.assertEqual(len(out),1);self.assertEqual(len(decisions['duplicate_decisions']),1)
        out,decisions,_=self.run_case([p,q],2,rgb)
        self.assertEqual(len(out),2)

    def test_source_backed_bending_junction_can_form_one_curve(self):
        p=np.array([[20,50],[40,50]],float);q=np.array([[40,50],[55,65]],float)
        out,decisions,stats=self.run_case([p,q],0)
        self.assertEqual(len(out),1)
        self.assertEqual(stats['learned_joins'],1)
        self.assertTrue(np.array_equal(union_raster((128,128),[p,q]),union_raster((128,128),[r['points'] for r in out])))

    def test_white_gap_cannot_be_bridged_by_overconfident_model(self):
        # Native 128px scale makes this 1.16 normalized pixels, within proposal radius.
        p=np.array([[20,50],[40,50]],float);q=np.array([[40.3,50],[55,50]],float)
        rgb=np.full((128,128,3),255,np.uint8)
        out,decisions,_=self.run_case([p,q],0,rgb)
        self.assertFalse(decisions['join_decisions']);self.assertEqual(len(out),2)

    def test_ambiguous_y_has_no_forced_pairing(self):
        center=np.array([60.,60.]);paths=[]
        for angle in [0,120,240]:
            unit=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
            paths.append(np.array([center,center+unit*20]))
        out,decisions,_=self.run_case(paths,0)
        self.assertEqual(len(out),3);self.assertFalse(decisions['join_decisions'])

    def test_closed_contour_is_retained(self):
        p=np.array([[30,30],[40,30],[40,40],[30,40],[30,30]],float)
        out,_,stats=self.run_case([p],0)
        self.assertEqual(len(out),1);self.assertEqual(stats['output_closed_paths'],1)

    def test_short_detail_not_silently_deleted_or_marked_robot_ready(self):
        p=np.array([[30,30],[31,30]],float)
        out,_,stats=self.run_case([p],2)
        self.assertEqual(len(out),1);self.assertFalse(stats['ready_for_robot'])
        self.assertEqual(stats['original_candidate_coverage_at_2_normalized_px'],1.)
