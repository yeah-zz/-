"""Behavioral checks against known geometry, independent of real robot code."""
import unittest
import cv2
import numpy as np

from source_ink import extract, InkConfig


def distance_to(raster):
    return cv2.distanceTransform((raster == 0).astype(np.uint8), cv2.DIST_L2, 5)


class SourceInkTests(unittest.TestCase):
    def blank(self, h=360, w=540):
        return np.full((h, w, 3), 255, np.uint8)

    def test_saturated_blue_ink_recovered_without_outlining_flat_wash(self):
        source = self.blank()
        cv2.rectangle(source, (20, 25), (520, 330), (215, 165, 105), -1)
        expected = np.zeros(source.shape[:2], np.uint8)
        points = np.array([[60, 280], [175, 70], [250, 170], [330, 100], [480, 260]], np.int32)
        cv2.polylines(source, [points], False, (100, 45, 15), 3)
        cv2.polylines(expected, [points], False, 255, 1)
        result = extract(source)
        self.assertGreater(np.mean(distance_to(result['raster'])[expected > 0] <= 2), .96)
        self.assertGreater(np.mean(distance_to(expected)[result['raster'] > 0] <= 3), .96)
        self.assertEqual(np.count_nonzero(result['raster'][23:28, 50:490]), 0)

    def test_real_white_gap_is_not_bridged(self):
        source = self.blank()
        cv2.line(source, (40, 180), (250, 180), (25, 25, 25), 3)
        cv2.line(source, (270, 180), (500, 180), (25, 25, 25), 3)
        # Even when the allowed search radius spans the gap, direction alone
        # cannot justify painting a connection through unsupported white paper.
        for cfg in (InkConfig(), InkConfig(max_gap=80.)):
            result = extract(source, cfg)
            self.assertEqual(np.count_nonzero(result['raster'][177:184, 255:265]), 0)
            self.assertEqual(len(result['paths']), 2)

    def test_junction_and_sharp_roof_corner_are_retained(self):
        source = self.blank()
        points = np.array([[60, 240], [270, 60], [480, 240]], np.int32)
        cv2.polylines(source, [points], False, (30, 30, 30), 3)
        cv2.line(source, (270, 60), (270, 320), (30, 30, 30), 3)
        result = extract(source)
        d = distance_to(result['raster'])
        for x, y in [(60, 240), (270, 60), (480, 240), (270, 320)]:
            # At a thick three-way junction the medial axis lies inside the
            # corner. Allow the drawn half-width, not an arbitrary exact pixel.
            self.assertLessEqual(d[y, x], 3. if (x, y) == (270, 60) else 2.)
        _, _, stats, _ = cv2.connectedComponentsWithStats(result['raster'], 8)
        self.assertEqual(len(stats) - 1, 1)

    def test_closed_loop_survives_and_does_not_become_a_cross(self):
        source = self.blank()
        cv2.rectangle(source, (120, 80), (420, 280), (30, 30, 30), 3)
        result = extract(source)
        self.assertTrue(any(p['closed'] for p in result['paths']))
        self.assertEqual(np.count_nonzero(result['raster'][100:260, 140:400]), 0)

    def test_tinted_paper_and_small_specks_do_not_become_paths(self):
        rng = np.random.default_rng(42)
        source = np.clip(np.full((360, 540, 3), [219, 231, 240]) + rng.normal(0, 2, (360, 540, 1)), 0, 255).astype(np.uint8)
        for x, y in [(80, 70), (290, 130), (430, 280)]:
            source[y, x] = 0
        result = extract(source)
        self.assertEqual(result['paths'], [])

    def test_pad_and_border_strokes(self):
        source = self.blank()
        cv2.line(source, (0, 20), (0, 340), (10, 10, 10), 3)
        cv2.line(source, (200, 20), (200, 340), (10, 10, 10), 3)
        result = extract(source)
        self.assertGreater(np.count_nonzero(result['raster'][30:330, :4]), 250)
        cropped = extract(source, pad=(0, 0, 20, 0))
        self.assertEqual(np.count_nonzero(cropped['raster'][:, :20]), 0)
        self.assertGreater(np.count_nonzero(cropped['raster'][:, 198:203]), 250)

    def test_deterministic_and_no_input_mutation(self):
        source = self.blank()
        cv2.line(source, (30, 300), (500, 50), (60, 70, 20), 4)
        saved = source.copy()
        a, b = extract(source), extract(source)
        self.assertEqual(a['paths'], b['paths'])
        np.testing.assert_array_equal(source, saved)
        self.assertFalse(a['stats']['ready_for_execution'])


if __name__ == '__main__':
    unittest.main()
