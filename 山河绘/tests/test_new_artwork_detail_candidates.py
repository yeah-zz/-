import unittest
from unittest.mock import patch

import numpy as np

from gongbi_training.new_artwork_detail_candidates import guarded_tree_hints, supplemental_ink


class DetailCandidateTest(unittest.TestCase):
    def test_weak_pass_keeps_new_long_ink_and_rejects_existing_ink(self):
        bgr = np.full((40, 45, 3), 180, np.uint8)
        old = np.zeros(bgr.shape[:2], np.uint8)
        old[12, 8:33] = 255
        def line(y):
            return dict(points=[[8, y], [32, y]], length_px=24., closed=False)
        contrast = np.zeros_like(old)
        contrast[[12, 22], 8:33] = 12
        weaker = dict(paths=[line(12), line(22), line(30)], contrast=contrast)
        with patch('gongbi_training.new_artwork_detail_candidates.extract', return_value=weaker):
            _, additions = supplemental_ink(bgr, dict(raster=old))
        self.assertEqual(additions, [line(22)])
        self.assertEqual(np.count_nonzero(old), 25)

    def test_tree_hints_require_agreement_and_abstain_from_confident_rock(self):
        full = np.array([[.05,.05,.05], [.54,.85,.54], [.37,.06,.37], [.04,.04,.04]])[:,None,:]
        local_a = np.array([[.05,.05,.05], [.19,.19,.19], [.72,.72,.72], [.04,.04,.04]])[:,None,:]
        local_b = local_a.copy()
        local_b[:,0,2] = [.05,.72,.19,.04]
        candidate, accepted = guarded_tree_hints(full,local_a,local_b)
        self.assertEqual(candidate.tolist(), [[True,True,False]])
        self.assertEqual(accepted.tolist(), [[True,False,False]])
