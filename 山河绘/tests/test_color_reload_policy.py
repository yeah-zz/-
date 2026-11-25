import unittest
from run_color_experiment import reload_stroke_limit, needs_color_reload, RELOAD_LENGTH_MM


class ReloadPolicyTests(unittest.TestCase):
    def test_legacy_and_task_limits(self):
        self.assertEqual(reload_stroke_limit({}, 4), 4)
        self.assertEqual(reload_stroke_limit({'reload_strokes_per_load': 8}, 4), 8)
        for value in (0, 41, True, 4.5, '8'):
            with self.assertRaises(ValueError):
                reload_stroke_limit({'reload_strokes_per_load': value})

    def test_independent_reload_triggers(self):
        self.assertFalse(needs_color_reload(100, 4, 20, 5, 130, 8))
        self.assertTrue(needs_color_reload(None, 0, 0, 5, 130, 8))
        self.assertTrue(needs_color_reload(100, 8, 20, 5, 130, 8))
        self.assertTrue(needs_color_reload(100, 4, RELOAD_LENGTH_MM-2, 5, 130, 8))
        self.assertTrue(needs_color_reload(100, 4, 20, 5, 161, 8))

    def test_fixed_count_calibration_ignores_length_and_time(self):
        self.assertFalse(needs_color_reload(100,2,1000,30,500,3,stroke_count_only=True))
        self.assertTrue(needs_color_reload(100,3,20,30,101,3,stroke_count_only=True))
        self.assertTrue(needs_color_reload(None,0,0,30,0,3,stroke_count_only=True))
