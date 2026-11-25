"""Planner integration: no hardware, preserved vertices and valid stroke data."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from config import ShuimoConfig
from source_ink import extract, build_strokes


class SourceInkIntegrationTests(unittest.TestCase):
    def fixture(self):
        image = np.full((260, 380, 3), 255, np.uint8)
        cv2.rectangle(image, (35, 45), (345, 230), (225, 185, 130), -1)
        cv2.polylines(image, [np.array([[50, 220], [165, 60], [250, 130], [330, 75]], np.int32)],
                      False, (90, 35, 15), 3)
        return image

    def test_adapter_keeps_all_fitted_vertices_and_bounded_pressure(self):
        image = self.fixture()
        source = extract(image)
        strokes, stats = build_strokes(image, .2)
        self.assertTrue(strokes)
        for s in strokes:
            pts = np.asarray(s['points'])
            for vertex in source['paths'][s['source_path_id']]['points']:
                self.assertLess(np.min(np.linalg.norm(pts - vertex, axis=1)), 1e-6)
            self.assertEqual(len(pts), len(s['press']))
            self.assertEqual(len(pts), len(s['speed']))
            self.assertEqual(len(pts), len(s['source_width_mm']))
            self.assertGreaterEqual(min(s['press']), ShuimoConfig.PRESS_MIN_MM)
            self.assertLessEqual(max(s['press']), min(.3, ShuimoConfig.PRESS_MAX_MM))
            self.assertLessEqual(np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1))*.2, .350001)
        self.assertEqual(stats['strokes'], len(strokes))
        self.assertFalse(stats['brush_width_calibrated_for_this_job'])

    def test_source_widths_stay_aligned_after_ink_budget_splitting(self):
        from ink_control import InkPolicy, split_long_strokes, cost
        strokes, _ = build_strokes(self.fixture(), .2)
        longest = max(strokes, key=lambda s: cost(s, .2))
        self.assertGreater(len(longest['points']), 20)
        budget = cost(longest, .2) / 3
        split = split_long_strokes([longest], .2, InkPolicy(max_segment_eq_mm=budget))
        self.assertGreater(len(split), 1)
        for s in split:
            self.assertEqual(len(s['points']), len(s['source_width_mm']))
            self.assertEqual(len(s['points']), len(s['press']))

    def test_new_plan_uses_v2_and_shared_image_coordinates_without_robot(self):
        import unified_painting as flow
        root = Path(__file__).resolve().parents[1]/'offline_previews'
        with tempfile.TemporaryDirectory(prefix='ink_integration_', dir=root) as folder:
            self.assertEqual(Path(folder).resolve().parent, root.resolve())
            path = Path(folder)/'fixture.png'
            cv2.imencode('.png', self.fixture())[1].tofile(str(path))
            with patch('robot.E6Robot.__init__', side_effect=AssertionError('Hardware forbidden')):
                with contextlib.redirect_stdout(io.StringIO()):
                    plan, work, lineart, waypoints, meta = flow.build_plan(path)
        self.assertEqual(meta['quality_stats']['algorithm'], 'source-constrained-ink-v2')
        self.assertFalse(plan['ready_for_execution'])
        self.assertEqual(lineart.shape, work.shape)
        self.assertTrue(waypoints)
        for key in ('image_size', 'mm_per_px', 'image_corners'):
            self.assertEqual(plan['modules']['outline'][key], plan['modules']['color'][key])
        self.assertTrue(plan['modules']['outline']['strokes'])
        for s in plan['modules']['outline']['strokes']:
            self.assertEqual(len(s['points']), len(s['source_width_mm']))

    def test_invalid_scale_rejected(self):
        for scale in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                build_strokes(self.fixture(), scale)


if __name__ == '__main__':
    unittest.main()
