"""Verify the provisional blue accents remain short and clear of planned ink."""

from pathlib import Path
import unittest

from plan_blue_trial import make_trial
from project_paths import PROJECT_DIR


class BlueTrialTests(unittest.TestCase):
    def test_v6_trial_clearance_and_length(self):
        root = Path(PROJECT_DIR)
        plan, _ = make_trial(root / "artifacts/mountain/mountain_v6_job.json",
                             root / "artifacts/reference/desired_mountain_reference.png",
                             brush_width_mm=1.0, gap_mm=2.0, count=2)
        self.assertFalse(plan["ready_for_execution"])
        self.assertEqual(len(plan["strokes"]), 2)
        for stroke in plan["strokes"]:
            self.assertEqual(stroke["color"], "blue")
            self.assertLessEqual(stroke["length_mm"], 9.2)
            self.assertGreaterEqual(stroke["min_ink_clearance_mm"], 2.5)


if __name__ == "__main__":
    unittest.main()
