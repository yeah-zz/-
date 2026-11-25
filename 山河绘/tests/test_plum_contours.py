import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from color_profile import ColorProfile
from plum_contours import build_plan, color_plan, ink_plan, masks
from project_paths import PAINTING_DIR
from run_color_experiment import module_lines


class PlumContourTests(unittest.TestCase):
    def test_pale_pink_is_retained_without_turning_white_paper_red(self):
        image = np.full((12, 12, 3), 255, np.uint8)
        image[2:8, 2:8] = (239, 242, 255)
        image[9:11, 9:11] = (30, 30, 30)
        ink, red = masks(image)
        self.assertEqual(int(red.sum()), 36)
        self.assertEqual(int(ink.sum()), 4)

    def test_large_lobes_are_outlined_and_small_bud_is_dabbed(self):
        red = np.zeros((300, 400), np.uint8)
        for angle in np.arange(5)*2*np.pi/5:
            center = tuple(np.rint([135+40*np.cos(angle), 150+40*np.sin(angle)]).astype(int))
            cv2.circle(red, center, 33, 1, -1)
        cv2.circle(red, (340, 140), 8, 1, -1)
        strokes, regions = color_plan(red, .2)
        self.assertEqual([r['kind'] for r in regions].count('flower'), 1)
        self.assertEqual([r['kind'] for r in regions].count('bud'), 1)
        self.assertTrue(any(s['kind']=='petal_contour' for s in strokes))
        self.assertEqual(sum(s['kind']=='bud_dab' for s in strokes), 2)
        for s in strokes:
            if s['kind']=='bud_dab':
                self.assertLess(s['length_mm'], 1.)
                self.assertEqual(s['press'][-1], 0.)

    def test_thick_branch_has_empty_center_and_source_twigs_remain(self):
        ink = np.zeros((160, 240), np.uint8)
        cv2.rectangle(ink, (30, 60), (150, 100), 1, -1)
        cv2.line(ink, (150, 80), (210, 40), 1, 2)
        strokes = ink_plan(ink, .2)
        self.assertTrue(any(s['kind']=='branch_contour' for s in strokes))
        self.assertTrue(any(s['kind']=='ink_detail' for s in strokes))
        for s in strokes:
            for x, y in s['points']:
                self.assertFalse(50<x<125 and 70<y<90, (s['id'], x, y))

    def test_close_spaced_stamens_survive_branch_envelope_closing(self):
        ink = np.zeros((140, 140), np.uint8)
        for x in (55, 62, 69, 76, 83):
            cv2.line(ink, (70, 90), (x, 55), 1, 1)
        flower = np.ones_like(ink)
        strokes = ink_plan(ink, .2, flower)
        self.assertFalse(any(s['kind']=='branch_contour' for s in strokes))
        for x in (55, 62, 69, 76, 83):
            points = np.asarray([p for s in strokes for p in s['points']])
            self.assertLess(float(np.linalg.norm(points-[x, 55], axis=1).min()), 3.)

    def test_attached_reference_count_fit_and_executor_compatibility_offline(self):
        with patch('socket.socket', side_effect=AssertionError('Offline planner opened a socket')):
            plan, work = build_plan(PAINTING_DIR/'参考图'/'plum_contour_reference_20260930.png',
                                    work_size=1000)
            regions = plan['modules']['color']['regions']
            self.assertEqual(sum(r['kind']=='flower' for r in regions), 4)
            self.assertEqual(sum(r['kind']=='bud' for r in regions), 8)
            for key, color in [('outline', 'ink'), ('color', 'red')]:
                module = plan['modules'][key]
                lines, skipped = module_lines(module, color, ColorProfile().to_dict())
                self.assertFalse(skipped)
                self.assertEqual(len(lines), module['stroke_count'])
                self.assertFalse(module['mass_fill'])
                self.assertEqual(len({s['id'] for s in module['strokes']}), len(lines))
                for s in module['strokes']:
                    self.assertEqual(len(s['points']), len(s['press']))
                    self.assertEqual(len(s['points']), len(s['speed']))
                    self.assertLessEqual(len(s['points']), 80)
                    self.assertTrue(np.isfinite(s['points']).all())
            json.dumps(plan, allow_nan=False)


if __name__=='__main__':
    unittest.main()
