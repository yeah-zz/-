# -*- coding: utf-8 -*-
"""Topology/provenance/scheduling invariants, not painting accuracy claims."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from module_stroke_core_20261104 import assemble, schedule, route_metrics, raster, propose_modules

def own(n, module=0):
    return [dict(module=module,uncertain=False) for _ in range(n)]

class TestPlanner(unittest.TestCase):
    def test_straight_join_preserves_all_sources(self):
        p=[np.array([[5,10],[10,10]]),np.array([[11,10],[18,10]])]
        gray=np.ones((30,30));gray[10,5:19]=0
        out,d=assemble(p,gray,own(2))
        self.assertEqual(len(out),1)
        self.assertEqual(sorted(out[0]['source_curve_ids']),[0,1])
        self.assertEqual(d['original_raster_missing_within_one_pixel'],0)

    def test_large_degree_two_corner(self):
        p=[np.array([[5,10],[12,10]]),np.array([[12,10],[12,18]])]
        gray=1-raster(p,(30,30))
        out,d=assemble(p,gray.astype(float),own(2))
        self.assertEqual(len(out),1)
        self.assertTrue(d['accepted'][0]['unique_corner'])

    def test_parallel_strokes_stay_separate(self):
        p=[np.array([[5,10],[18,10]]),np.array([[5,12],[18,12]])]
        out,_=assemble(p,(1-raster(p,(30,30))).astype(float),own(2))
        self.assertEqual(len(out),2)

    def test_white_gap_is_not_filled(self):
        p=[np.array([[5,10],[10,10]]),np.array([[13,10],[19,10]])]
        out,_=assemble(p,(1-raster(p,(30,30))).astype(float),own(2))
        self.assertEqual(len(out),2)

    def test_different_objects_are_not_joined(self):
        p=[np.array([[5,10],[10,10]]),np.array([[10,10],[18,10]])]
        owners=own(2);owners[1]['module']=1
        out,_=assemble(p,np.zeros((30,30)),owners)
        self.assertEqual(len(out),2)

    def test_ambiguous_branch_stays_pending(self):
        p=[np.array([[5.,10.],[10.,10.]]),np.array([[10.,10.],[17.,9.]]),np.array([[10.,10.],[17.,11.]])]
        out,d=assemble(p,np.zeros((30,30)),own(3))
        self.assertEqual(len(out),3)
        self.assertGreater(len(d['pending']),0)

    def test_module_lock_and_no_curve_repeated(self):
        curves=[dict(points=np.array(p,float),module=m) for p,m in [
            ([[1,1],[4,1]],0),([[25,25],[29,25]],0),
            ([[5,1],[8,1]],1),([[25,22],[29,22]],1)]]
        modules=[dict(sam_key='a'),dict(sam_key='b')]
        plan=schedule(curves,modules,(35,35))
        self.assertEqual(sorted(r['curve'] for r in plan),list(range(4)))
        self.assertEqual(route_metrics(plan,curves)['module_reentries'],0)

    def test_crossing_mask_does_not_split_input_curve(self):
        image=np.full((100,100,3),180,np.uint8)
        mask=np.zeros((100,100),bool);mask[10:40,10:40]=True
        p=[np.array([[15,20],[85,20]],float)]
        modules,ownership=propose_modules(image,p,{'a':mask},[dict(key='a',predicted_iou=.9,stability_score=.95)])
        self.assertEqual(len(ownership),1)
        self.assertTrue(ownership[0]['uncertain'])
        self.assertEqual(modules[ownership[0]['module']]['kind'],'unresolved local region')

    def test_nested_parts_use_whole_object_proposal(self):
        image=np.full((100,100,3),180,np.uint8)
        roof=np.zeros((100,100),bool);roof[20:30,20:65]=True
        whole=np.zeros((100,100),bool);whole[20:60,20:65]=True
        paths=[np.array([[25,25],[60,25]],float),np.array([[25,50],[60,50]],float)]
        records=[dict(key=k,predicted_iou=.9,stability_score=.95) for k in ['roof','whole']]
        modules,ownership=propose_modules(image,paths,{'roof':roof,'whole':whole},records)
        self.assertEqual(ownership[0]['module'],ownership[1]['module'])
        self.assertEqual(modules[ownership[0]['module']]['sam_key'],'whole')

if __name__=='__main__':
    unittest.main()
