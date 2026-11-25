"""No-network tests for the experimental v5 ink coordination."""
from copy import deepcopy
from dataclasses import replace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import ink_control as ink
import main
import mountain
import vision
import job
from config import ShuimoConfig
from geometry import CanvasMap, check_waypoints
from run_mountain_balanced import run_feedback, load_reviewed, Motion
from robot import RobotError


def stroke(length=20, press=.3):
    x = np.arange(0, length+1, dtype=float)
    return dict(points=np.column_stack((x, np.ones_like(x)*100)).tolist(),
                press=[press]*len(x), speed=[28.]*len(x), length_mm=float(length),
                layer=0, pal_index=0, name="test", tail_mm=0, role="silhouette",
                width_mm=1., press_mean=press, press_max=press)


class InkTests(unittest.TestCase):
    def test_missing_palette_refills_legacy_branch_without_mutating_it(self):
        first, branch = stroke(8), stroke(8)
        branch.pop('pal_index')
        branch['id'] = 'legacy-branch'
        source = deepcopy([first, branch])
        now, drawn, dips = [0.], [], []
        policy = replace(ink.InkPolicy(), fresh_press_factor=1.)
        def draw(s):
            drawn.append(s)
            now[0] += 100  # Forces the refill branch on the missing-field stroke.
        rows, count = run_feedback([first, branch], 1, policy, dips.append, draw,
                                  lambda _: None, clock=lambda: now[0])
        self.assertEqual(count, 2)
        self.assertEqual([d['pal_index'] for d in dips], [0, 0])
        self.assertEqual(len(rows), 2)
        for actual, expected in zip(drawn, source):
            for key in ('points', 'press', 'speed'):
                self.assertEqual(actual[key], expected[key])
        self.assertEqual([first, branch], source)

    def test_explicit_invalid_palette_rejected_before_any_callback(self):
        for value in (None, True, -1, 999, '0', 0.0, 1):
            bad = stroke(8)
            bad['pal_index'] = value
            action = MagicMock()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'palette index'):
                run_feedback([stroke(8), bad], 1, ink.InkPolicy(), action, action, action,
                             before_stroke=action)
            action.assert_not_called()

    def test_weighted_cost_and_fresh_compensation(self):
        p = ink.InkPolicy()
        s = stroke()
        self.assertGreater(ink.cost(stroke(20, .39), 1), ink.cost(stroke(20, .15), 1))
        state = ink.InkState(p)
        state.load(0)
        fresh = ink.condition_fresh(s, state, 1)
        self.assertLess(fresh["press"][0], s["press"][0])
        self.assertEqual(fresh["press"][-1], s["press"][-1])
        self.assertEqual(fresh["points"], s["points"])
        self.assertTrue(all(a <= b for a, b in zip(fresh["press"], s["press"])))
        state.consume(fresh, 1, 4)
        self.assertEqual(ink.condition_fresh(s, state, 1)["press"], s["press"])

    def test_split_preserves_geometry_and_budget(self):
        p = ink.InkPolicy()
        original = stroke(400, .39)
        parts = ink.split_long_strokes([original], 1, p)
        self.assertGreater(len(parts), 1)
        reconstructed = parts[0]["points"] + [pt for s in parts[1:] for pt in s["points"][1:]]
        self.assertEqual(reconstructed, original["points"])
        for s in parts:
            self.assertLessEqual(ink.cost(s, 1), p.max_segment_eq_mm + 1e-6)
        self.assertAlmostEqual(sum(s["length_mm"] for s in parts), 400)

    def simulate(self, draw_seconds=2, rest_every=20):
        now, drawn, dips = [0.], [], []
        p = replace(ink.InkPolicy(), rest_every=rest_every)
        def dip(step):
            dips.append(deepcopy(step))
            now[0] += step["dwell_s"] + step["drip_s"] + 8
        def draw(s):
            drawn.append(s)
            now[0] += draw_seconds
        def rest(seconds): now[0] += seconds
        rows, n = run_feedback([stroke(8)]*8, 1, p, dip, draw, rest, clock=lambda: now[0])
        return rows, n, drawn, dips

    def test_disabled_rest_keeps_drawing_and_refilling(self):
        policy = ink.InkPolicy()
        self.assertEqual(policy.rest_every, 0)
        rests, dips, drawn = [], [], []
        rows, count = run_feedback([stroke(8)] * 45, 1, policy,
                                  dips.append, drawn.append, rests.append,
                                  clock=lambda: 0.)
        self.assertEqual(rests, [])
        self.assertEqual(len(drawn), 45)
        self.assertGreater(count, 1)
        steps, _ = ink.build_plan([stroke(8)] * 45, 0, 1, policy)
        self.assertFalse(any(s['kind'] == 'rest' for s in steps))
        self.assertEqual(sum(s['kind'] == 'stroke' for s in steps), 45)
        with self.assertRaises(ValueError):
            replace(policy, rest_every=-1)

    def test_actual_clock_causes_earlier_refresh(self):
        fast = self.simulate(2)
        slow = self.simulate(25)
        self.assertGreater(slow[1], fast[1])
        self.assertEqual(len(slow[2]), 8)
        self.assertTrue(any("elapsed_time" in r["refill_reason"] for r in slow[0]))
        self.assertTrue(all(s["points"] == stroke(8)["points"] for s in slow[2]))

    def test_rest_refresh_and_short_dwell(self):
        rows, n, drawn, dips = self.simulate(2, rest_every=2)
        self.assertEqual(n, 4)
        for i in (2, 4, 6): self.assertIn("idle_or_rest", rows[i]["refill_reason"])
        self.assertEqual(dips[0]["dwell_s"], .6)
        self.assertTrue(all(d["dwell_s"] == .3 for d in dips[1:]))
        # The default follows the measured cup configuration, not the ink policy.
        with patch('config.dip_time_of', return_value=.43):
            self.assertEqual(job.dip_waypoints(0)[1]["pause"], .43)
        self.assertEqual(job.dip_waypoints(0, .6, 1.6)[1]["pause"], .6)
        self.assertEqual(job.dip_waypoints(0, .6, 1.6)[-1]["pause"], 1.6)

    def test_slow_stroke_refills_without_aborting_next_stroke(self):
        now, drawn = [0.], []
        def draw(s):
            drawn.append(s)
            now[0] += 100
            return {"draw_seconds": 100}
        rows, dips = run_feedback([stroke(8), stroke(100)], 1, ink.InkPolicy(), lambda _: None,
                                  draw, lambda _: None, clock=lambda: now[0])
        self.assertEqual(len(drawn), 2)
        self.assertEqual(dips, 2)
        self.assertIn("elapsed_time", rows[1]["refill_reason"])

    def test_dip_dwell_occurs_after_arrival(self):
        events = []
        class FakeRobot:
            def send(self, command): events.append("move"); return "0,{}"
            def move_linear(self, *args): events.append("move"); return "0,{}"
            def _ok(self, result): return True
            def wait_idle(self, *args): events.append("arrived"); return True
        m = Motion(FakeRobot(), CanvasMap((640, 440)), (0, 100, -100))
        with patch("run_mountain_balanced.time.sleep", side_effect=lambda t: events.append(t)), \
             patch('prepare_robot.wait_arrival',side_effect=lambda *a,**k: events.append('arrived')):
            m.send(job.dip_waypoints(0, .6, 1.6))
        for duration in (.6, 1.6): self.assertEqual(events[events.index(duration)-1], "arrived")

    def test_v5_pipeline_and_calibration_guard(self):
        old = ShuimoConfig.PRESS_MIN_MM
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                patch("socket.socket", side_effect=AssertionError("No network")):
            path = str(Path(folder)/"v5.json")
            rc = main.main(["--mode", "shuimo", "--shuimo-profile", "mountain",
                            "--mountain-detail", "enriched", "--ink-policy", "balanced",
                            "--image", "artifacts/reference/desired_mountain_reference.png", "--press-min", ".15",
                            "--no-preview", "--job-out", path])
            self.assertEqual(rc, 0)
            strokes, mapper, policy, meta = load_reviewed(path)
            # Extraction and ink scheduling can change the exact count;
            # the planner must retain every semantic stroke.
            self.assertGreaterEqual(meta["quality_stats"]["semantic_strokes"], 70)
            self.assertGreaterEqual(len(strokes), meta["quality_stats"]["semantic_strokes"])
            self.assertEqual(ShuimoConfig.PRESS_MIN_MM, old)
            planned, _ = job.load_job(path)
            self.assertTrue(check_waypoints(planned)[0])
            self.assertFalse(any(w["tag"].startswith("tail:") for w in planned))
            self.assertEqual(sum(w["tag"] == "rest" for w in planned), 0)
            self.assertLessEqual(max(r["end_age_s"] for r in meta["ink_plan"]["ink_cycles"]), policy.max_age_s)
            for r in meta["ink_plan"]["ink_cycles"]:
                self.assertLessEqual(r["used_before_eq_mm"] + r["demand_eq_mm"] + policy.reserve_eq_mm,
                                     policy.capacity_eq_mm + .02)
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            data["meta"]["ink_execution"]["calibration"]["user"] = 1
            Path(path).write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError): load_reviewed(path)

    def test_enriched_preserves_silhouettes(self):
        work, info = vision.prepare_image(vision.load_image("artifacts/reference/desired_mountain_reference.png"), work_size=1800)
        mapper = CanvasMap(info["prepared_size"])
        standard, _ = mountain.build_strokes(work, mapper.mm_per_px, info["pad"])
        richer, _ = mountain.build_strokes(work, mapper.mm_per_px, info["pad"], detail="enriched")
        def outlines(strokes): return sorted(json.dumps(s["points"]) for s in strokes if s["role"] == "silhouette")
        self.assertEqual(outlines(standard), outlines(richer))
        self.assertGreater(len(richer), len(standard))


if __name__ == "__main__": unittest.main()
