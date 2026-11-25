"""Synthetic, offline interface checks for the subject planner registry."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import imgio
from multi_painting import PLANNERS, analyze, build_plan, save_outputs


class MultiPaintingTests(unittest.TestCase):
    def fixture(self, folder):
        image = np.full((240, 320, 3), 255, np.uint8)
        cv2.polylines(image, [np.array([[15, 135], [95, 45], [160, 125],
                                        [230, 55], [300, 130]], np.int32)],
                      False, (25, 25, 25), 4)
        for x, y in ((55, 190), (90, 175), (125, 195)):
            cv2.circle(image, (x, y), 9, (35, 35, 210), -1)
        cv2.ellipse(image, (220, 190), (24, 12), 25, 0, 360,
                    (170, 100, 220), -1)
        cv2.line(image, (250, 170), (270, 225), (50, 150, 50), 8)
        path = Path(folder) / 'synthetic.png'
        self.assertTrue(imgio.imwrite(str(path), image))
        return path

    def test_shared_evidence_and_subject_contracts(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.fixture(folder)
            evidence = analyze(source, 320)
            self.assertGreater(int(evidence.ink.sum()), 0)
            self.assertGreater(int(evidence.colors['red'].sum()), 0)
            self.assertGreater(int(evidence.colors['purple'].sum()), 0)
            self.assertGreater(int(evidence.colors['green'].sum()), 0)
            expected = {'generic': {'color', 'outline'},
                        'plum': {'outline', 'petals'},
                        'mountain': {'mountain_ink'},
                        'lotus': {'petals', 'leaves_stems'}}
            self.assertEqual(set(PLANNERS), set(expected))
            for subject, stages in expected.items():
                with self.subTest(subject=subject):
                    plan, work = build_plan(source, subject, 320)
                    self.assertFalse(plan['ready_for_execution'])
                    self.assertEqual(set(plan['modules']), stages)
                    self.assertEqual(plan['image_size'], [work.shape[1], work.shape[0]])
                    for module in plan['modules'].values():
                        self.assertFalse(module['ready_for_execution'])
                        self.assertEqual(module['stroke_count'], len(module['strokes']))
                        for stroke in module['strokes']:
                            points = np.asarray(stroke['points'])
                            self.assertTrue(np.isfinite(points).all())
                            self.assertTrue((points[:, 0] >= 0).all())
                            self.assertTrue((points[:, 1] >= 0).all())
                            self.assertTrue((points[:, 0] < work.shape[1]).all())
                            self.assertTrue((points[:, 1] < work.shape[0]).all())
                    if subject == 'lotus':
                        self.assertEqual(plan['status'], 'experimental_no_lotus_reference')
                        self.assertGreater(plan['modules']['petals']['stroke_count'], 0)
                        self.assertGreater(plan['modules']['leaves_stems']['stroke_count'], 0)

    def test_export_refuses_overwrite_and_unknown_subject(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.fixture(folder)
            with self.assertRaises(ValueError):
                build_plan(source, 'unknown', 320)
            plan, work = build_plan(source, 'lotus', 320)
            output = Path(folder) / 'design.json'
            preview = Path(folder) / 'paths.png'
            save_outputs(plan, work, output, preview)
            self.assertTrue(preview.is_file())
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['subject'], 'lotus')
            with self.assertRaises(FileExistsError):
                save_outputs(plan, work, output, preview)


if __name__ == '__main__':
    unittest.main()
