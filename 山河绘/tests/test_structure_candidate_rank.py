import unittest

import numpy as np

from gongbi_training.structure_candidate_rank import rank_owned_candidates


class StructureCandidateRankTests(unittest.TestCase):
    def setUp(self):
        self.owner=np.ones((26,45),np.uint8)
        self.start=(4,10,10,19)
        self.end=(35,10,41,19)

    def route(self,row,cost):
        cols=np.arange(7,39)
        return dict(points=np.column_stack((cols,np.full(len(cols),row))),cost=cost)

    def test_attachment_center_can_outweigh_lower_ink_cost(self):
        pool=dict(routes=[self.route(10,.6),self.route(14,1.0)])
        result=rank_owned_candidates([pool],self.owner,self.start,self.end)
        self.assertEqual(result['status'],'route_found')
        self.assertTrue((result['points'][:,1]==14).all())

    def test_rejects_a_candidate_that_crosses_another_owner(self):
        self.owner[14,22]=0
        pool=dict(routes=[self.route(14,.8)])
        result=rank_owned_candidates([pool],self.owner,self.start,self.end)
        self.assertEqual(result['status'],'no_owned_candidate')
