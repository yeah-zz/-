import unittest

import numpy as np

from gongbi_training.ink_band_reconstruction import reconstruct_ink_path


class InkBandReconstructionTests(unittest.TestCase):
    def test_stays_in_one_connected_band(self):
        mask = np.zeros((24, 25), np.uint8)
        mask[5:10, 3:19] = 1
        mask[13:17, 3:19] = 1
        image = np.full((*mask.shape, 3), 220, np.uint8)
        image[7, 3:19] = 170
        path = reconstruct_ink_path(image,mask,[4,7],[17,7],crop=(0,0,25,24))
        self.assertTrue(len(path)>10)
        self.assertTrue(mask[path[:,1],path[:,0]].all())
        self.assertTrue((path[:,1]<=9).all())

    def test_rejects_missing_ink_between_endpoints(self):
        mask = np.zeros((24, 25), np.uint8)
        mask[7, 3:10] = 1
        mask[7, 12:19] = 1
        image = np.full((*mask.shape,3),220,np.uint8)
        path = reconstruct_ink_path(image,mask,[4,7],[17,7],crop=(0,0,25,24))
        self.assertEqual(len(path),0)
