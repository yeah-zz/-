import unittest

import numpy as np

from gongbi_training.structural_ink_routing import route_owned_ink


class StructuralInkRoutingTests(unittest.TestCase):
    def setUp(self):
        self.ink = np.zeros((31,45),np.uint8)
        self.ink[9,2:43] = self.ink[20,2:43] = 1
        self.rgb = np.full((*self.ink.shape,3),224,np.uint8)
        self.rgb[9,2:43] = 110
        self.rgb[20,2:43] = 165
        self.owner = np.ones_like(self.ink)
        self.start=np.zeros_like(self.ink)
        self.end=np.zeros_like(self.ink)
        self.start[16:24,4:8]=1
        self.end[16:24,37:41]=1

    def test_attachment_zones_exclude_a_darker_neighbor(self):
        route=route_owned_ink(self.rgb,self.ink,self.owner,crop=(0,0,45,31),
                              start_zone=self.start,end_zone=self.end)
        self.assertEqual(route['status'],'route_found')
        self.assertTrue((route['points'][:,1]==20).all())

    def test_a_route_cannot_leave_its_owner(self):
        self.owner[:,21]=0
        route=route_owned_ink(self.rgb,self.ink,self.owner,crop=(0,0,45,31),
                              start_zone=self.start,end_zone=self.end)
        self.assertEqual(route['status'],'no_owned_connection')
