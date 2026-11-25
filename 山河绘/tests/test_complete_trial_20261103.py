import unittest
import numpy as np
from scripts.prepare_complete_trial_20261103 import split_length,full_scope
from config import ArmConfig,PALETTE
from pigment_station import read_measurements
from copy import deepcopy

class CompleteTrialTests(unittest.TestCase):
    def test_split_preserves_length_and_shared_endpoints(self):
        p=np.array([[0.,0.],[12.,0.],[12.,15.]])
        qs=split_length(p,10)
        self.assertEqual(len(qs),3)
        self.assertAlmostEqual(sum(np.linalg.norm(np.diff(q,axis=0),axis=1).sum() for q in qs),27)
        self.assertTrue(all(np.allclose(a[-1],b[0]) for a,b in zip(qs[:-1],qs[1:])))
        self.assertTrue(all(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()<=10+1e-8 for q in qs))
    def test_exact_boundary_no_zero_piece(self):
        qs=split_length([[0,0],[10,0],[20,0]],10)
        self.assertEqual(len(qs),2)
        self.assertTrue(all(len(q)>=2 for q in qs))
    def test_curved_polyline_and_duplicate_vertices(self):
        p=np.array([[0.,0.],[0.,0.],[3.,4.],[7.,4.],[7.,12.]])
        qs=split_length(p,6)
        self.assertAlmostEqual(sum(np.linalg.norm(np.diff(q,axis=0),axis=1).sum() for q in qs),17.)
    def test_task_scope_restores_shared_values(self):
        original=(ArmConfig.RPY,ArmConfig.PAPER_Z,deepcopy(PALETTE),read_measurements())
        with full_scope():
            self.assertAlmostEqual(ArmConfig.PAPER_Z,-23.2686)
            self.assertEqual(read_measurements()['black']['dwell_s'],2.)
            self.assertFalse(read_measurements()['black']['level_compensation_enabled'])
        self.assertEqual(original,(ArmConfig.RPY,ArmConfig.PAPER_Z,PALETTE,read_measurements()))

if __name__=='__main__':unittest.main()
