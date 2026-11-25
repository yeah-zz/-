"""Offline regression checks: no robot construction is permitted."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import unified_painting as flow


class UnifiedTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch('robot.E6Robot.__init__', side_effect=AssertionError('Hardware forbidden'))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def plan(self):
        return {'modules': {'color': {'strokes': [{'color': 'red'}],
                                     'motion_candidate': flow.ColorProfile(press_mm=.22).to_dict()}}}

    def execute(self, answers, ink_result=0, color_result=0):
        events = []
        def ink(path):
            events.append('ink')
            return ink_result
        def color(**kwargs):
            events.append(kwargs['color'])
            self.assertIsNotNone(kwargs['station'])
            return color_result
        replies = iter(answers)
        with patch('run_color_experiment.module_lines', return_value=([{'id': 'test'}], [])), contextlib.redirect_stdout(io.StringIO()):
            result = flow.execute_stages(self.plan(), 'unused', {'red': object()},
                ask=lambda _: next(replies), ink_runner=ink, color_runner=color)
        return result, events

    def test_black_before_dry_before_color(self):
        self.assertEqual(self.execute(['BLACK', 'red']), (0, ['ink', 'red']))

    def test_failed_black_never_colors(self):
        self.assertEqual(self.execute(['BLACK'], ink_result=4), (4, ['ink']))

    def test_cancel_dry_never_colors(self):
        self.assertEqual(self.execute(['BLACK', '']), (0, ['ink']))

    def test_preview_only_never_moves(self):
        self.assertEqual(self.execute(['']), (0, []))

    def test_failed_color_propagates(self):
        self.assertEqual(self.execute(['BLACK', 'red'], color_result=4), (4, ['ink', 'red']))

    def test_source_color_boundary_is_hollow_and_black_line_preserved(self):
        picture = np.full((200, 200, 3), 255, np.uint8)
        cv2.rectangle(picture, (70, 60), (140, 140), (30, 48, 245), -1)
        cv2.line(picture, (30, 30), (30, 150), (0, 0, 0), 3)
        lineart, _ = flow.outline_image(picture, .3)
        self.assertTrue(np.all(lineart[100, 100] == 255))
        self.assertTrue(np.all(lineart[90, 30] == 0))
        self.assertTrue(np.any(lineart[55:65, 80:130] == 0))

    def test_subject_labels_produce_identical_plans_and_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / 'mixed.png'
            picture = np.full((180, 240, 3), 255, np.uint8)
            cv2.line(picture, (45, 130), (160, 45), (0, 0, 0), 3)
            cv2.circle(picture, (145, 95), 25, (160, 180, 240), -1)
            flow.imgio.imwrite(str(reference), picture)
            landscape = flow.build_plan(reference, '山水')
            blossom = flow.build_plan(reference, '花')
            self.assertEqual(landscape[0], blossom[0])
            self.assertEqual(landscape[3:], blossom[3:])
            np.testing.assert_array_equal(landscape[2], blossom[2])
            self.assertTrue(landscape[0]['modules']['color']['strokes'])

    def test_blackwhite_shared_coordinates_and_existing_executor_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / 'reference.png'
            image = np.full((240, 320, 3), 255, np.uint8)
            cv2.line(image, (60, 170), (160, 60), (0, 0, 0), 3)
            cv2.line(image, (160, 60), (260, 170), (0, 0, 0), 3)
            flow.imgio.imwrite(str(reference), image)
            plan, work, _, waypoints, meta = flow.build_plan(reference)
            self.assertEqual(plan['stages'], ['outline', 'color'])
            self.assertFalse(plan['modules']['color']['strokes'])
            for key in ('image_size', 'image_corners', 'mm_per_px'):
                self.assertEqual(plan['modules']['outline'][key], plan['modules']['color'][key])
            ink_path = Path(folder) / 'ink.json'
            flow.job.save_job(ink_path, waypoints, meta)
            from run_mountain_balanced import load_reviewed
            load_reviewed(ink_path)
            flow.save_outputs(plan, work, Path(folder) / 'plan.json', Path(folder) / 'preview.png')


if __name__ == '__main__':
    unittest.main()
