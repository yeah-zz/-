import unittest

import numpy as np

from gongbi_training.automatic_ink_routing import automatic_ink_route


class AutomaticInkRoutingTests(unittest.TestCase):
    def test_reports_similar_parallel_routes(self):
        ink = np.zeros((28,42),np.uint8)
        ink[8,2:40] = ink[18,2:40] = 1
        rgb = np.full((*ink.shape,3),225,np.uint8)
        rgb[ink>0] = 175
        out = automatic_ink_route(rgb,ink,crop=(0,0,42,28))
        self.assertEqual(out['status'],'ambiguous')

    def test_prioritizes_source_supported_role_hint(self):
        ink = np.zeros((28,42),np.uint8)
        ink[8,2:40] = ink[18,2:40] = 1
        rgb = np.full((*ink.shape,3),225,np.uint8)
        rgb[ink>0] = 175
        role = np.zeros_like(ink)
        role[18,2:40] = 1
        out = automatic_ink_route(rgb,ink,crop=(0,0,42,28),hints=role)
        self.assertEqual(out['status'],'route_found')
        self.assertTrue((out['points'][:,1]==18).all())
