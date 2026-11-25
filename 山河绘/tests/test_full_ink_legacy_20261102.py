# -*- coding: utf-8 -*-
"""Binding checks: no sockets, no physical fixtures, no configuration writes."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from config import ArmConfig
from geometry import CanvasMap
from run_mountain_balanced import load_reviewed

SPEC=importlib.util.spec_from_file_location('legacy_bind',Path(__file__).resolve().parents[1]/'scripts'/'bind_full_ink_legacy_20261102.py')
m=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class LegacyBindingTests(unittest.TestCase):
    def test_contain_mapping_preserves_physical_aspect_and_margins(self):
        for source in [(1254,1254),(600,1000),(1600,600)]:
            size,mult,off,scale=m.padded_mapping(*source,296.81485,206.8166)
            physical_per_px=np.array([296.81485/(size[0]-1),206.8166/(size[1]-1)])
            np.testing.assert_allclose(mult*physical_per_px,[scale,scale])
            self.assertGreaterEqual(float(np.min(off*physical_per_px)),10-1e-9)
            extent=(np.array(source)-1)*mult
            self.assertTrue(np.all(extent+off <= np.array(size)-1))

    def test_invalid_margin_cannot_export(self):
        with self.assertRaises(ValueError):
            m.padded_mapping(1254,1254,20,15)

    def test_generated_job_preflight_has_no_connection_or_added_tails(self):
        target=m.OUT/'N02_full_black_legacy_job.json'
        if not target.exists():
            self.skipTest('Offline artifact not generated')
        with patch('socket.socket',side_effect=AssertionError('No connection allowed')):
            strokes,mapper,_,meta=load_reviewed(str(target))
        self.assertEqual(len(strokes),1194)
        self.assertTrue(meta['calibration_reuse']['user_requested'])
        self.assertFalse(meta['calibration_reuse']['site_match_verified'])
        for s in strokes:
            self.assertEqual(s['tail_mm'],0)
            self.assertTrue(all(abs(p-.1)<1e-9 for p in s['press']))
            self.assertEqual(s['pal_index'],0)
        d=json.loads(target.read_text('utf-8'))
        self.assertTrue(d['waypoints'])
        self.assertFalse(any(w['tag'].startswith(('tail:','test:')) for w in d['waypoints']))

    def test_changed_paper_calibration_rejected_by_existing_executor(self):
        target=m.OUT/'N02_full_black_legacy_job.json'
        if not target.exists():
            self.skipTest('Offline artifact not generated')
        with patch('socket.socket',side_effect=AssertionError('No connection allowed')):
            with patch.object(ArmConfig,'PAPER_Z',ArmConfig.PAPER_Z+.25):
                with self.assertRaisesRegex(ValueError,'Calibration/settings changed'):
                    load_reviewed(str(target))


if __name__=='__main__':
    unittest.main()
