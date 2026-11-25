import unittest
import numpy as np
from gongbi_training.ink_path_diagnostics import arc_cells, score_visible_runs


class VisibleRunChecks(unittest.TestCase):
    def test_corners_keep_arc_length(self):
        _, w = arc_cells([[0, 0], [3, 0], [3, 4]])
        self.assertAlmostEqual(w.sum(), 7)

    def test_disconnected_runs_never_bridge_or_merge_wrong_intervals(self):
        pred = np.zeros((3, 22), dtype=int)
        pred[:, 10:20] = 1
        r = score_visible_runs(pred, [[[0, 1], [2, 1]], [[20, 1], [21, 1]]], 1)
        self.assertAlmostEqual(r['length_px'], 3)
        self.assertAlmostEqual(r['longest_wrong_px'], 2)
        self.assertEqual(len(r['runs']), 2)

    def test_unknown_is_not_negative(self):
        r = score_visible_runs(np.zeros((2, 2)), [], None)
        self.assertIsNone(r['coverage'])

    def test_native_boundary_error_has_measured_length(self):
        pred = np.ones((4, 12), dtype=int)
        pred[:, 5:] = 2
        r = score_visible_runs(pred, [[[1, 2], [9, 2]]], 1)
        self.assertAlmostEqual(r['correct_length_px'], 3.5)
        self.assertAlmostEqual(r['longest_wrong_px'], 4.5)

    def test_invalid_reference_does_not_clip_into_image(self):
        with self.assertRaises(ValueError):
            score_visible_runs(np.zeros((3, 3)), [[[1, 1], [8, 1]]], 1)


if __name__ == '__main__':
    unittest.main()
