"""Offline regression checks. Run: uv run python -m unittest -v test_mountain"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

import main
import mountain
import shuimo
import scheduler
import job
import vision
from config import ArmConfig, ShuimoConfig, JobConfig, validate
from geometry import CanvasMap, check_waypoints
from trajectory_preview import contact_paths, render_contact


class MountainTests(unittest.TestCase):
    def setUp(self):
        self.press = ShuimoConfig.PRESS_MIN_MM
        self.skip = JobConfig.SKIP_PIGMENT_ACTIONS

    def tearDown(self):
        ShuimoConfig.PRESS_MIN_MM = self.press
        JobConfig.SKIP_PIGMENT_ACTIONS = self.skip

    def build(self, file="artifacts/reference/desired_mountain_reference.png"):
        work, info = vision.prepare_image(vision.load_image(file), work_size=1800)
        mapper = CanvasMap(info["prepared_size"], info["image_corners"])
        strokes, stats = mountain.build_strokes(work, mapper.mm_per_px, info["pad"])
        return work, mapper, strokes, stats

    def test_reference_layers_determinism_and_safety(self):
        ShuimoConfig.PRESS_MIN_MM = .15
        work, mapper, strokes, stats = self.build()
        self.assertEqual(strokes, self.build()[2])
        self.assertTrue(all(stats["layers"][k] > 0 for k in range(3)))
        self.assertGreater(stats["length_mm"] / len(strokes), 15)
        for s in strokes:
            self.assertGreaterEqual(s["length_mm"], 5)
            self.assertLessEqual(max(s["press"]), .39 + 1e-6)
            self.assertEqual(s["press"][-1], 0)
            if s["role"] == "downhill":
                self.assertGreater(s["points"][-1][1], s["points"][0][1])
        steps, plan = scheduler.build_framework_plan(strokes, 0)
        waypoints, _ = job.steps_to_waypoints(steps, mapper)
        self.assertTrue(check_waypoints(waypoints)[0])
        self.assertFalse(any(w["tag"].startswith("tail:") for w in waypoints))
        self.assertEqual(len(contact_paths(waypoints, mapper)), len(strokes))
        actual = sum(np.linalg.norm(np.diff(mapper.to_robot(s["points"]), axis=0), axis=1).sum() for s in strokes)
        self.assertLess(abs(actual - plan["length_mm"]) / actual, .01)
        for path in contact_paths(waypoints, mapper):
            for p, press in path:
                self.assertTrue(np.all(p >= -.001))
                self.assertLessEqual(p[0], mapper.image_size[0] - 1 + .001)
                self.assertLessEqual(p[1], mapper.image_size[1] - 1 + .001)
                self.assertGreaterEqual(press, -1e-6)
        self.assertFalse(validate())

    def test_preview_includes_real_tail_and_excludes_travel(self):
        mapper = CanvasMap((640, 440))
        s = dict(points=[[100, 100], [200, 100]], press=[.3, .3], speed=[20, 20],
                 length_mm=45, name="test", layer=0)
        wps = job.stroke_waypoints(mapper, s)
        paths = contact_paths(wps, mapper)
        self.assertGreater(paths[0][-1][0][0], 200)
        old = render_contact(wps, mapper)
        s["tail_mm"] = 0
        clean = render_contact(job.stroke_waypoints(mapper, s), mapper)
        self.assertGreater(np.count_nonzero(old < 100), np.count_nonzero(clean < 100))
        self.assertTrue(np.all(old[0:50] == 255))

    def test_cli_override_restored_and_segment_matches_full(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                patch("socket.socket", side_effect=AssertionError("Offline only")):
            full = str(Path(folder) / "full.json")
            part = str(Path(folder) / "part.json")
            argv = ["--mode", "shuimo", "--shuimo-profile", "mountain", "--image",
                    "artifacts/reference/desired_mountain_reference.png", "--press-min", ".15", "--no-preview", "--no-pigment"]
            self.assertEqual(main.main(argv + ["--job-out", full]), 0)
            self.assertEqual(ShuimoConfig.PRESS_MIN_MM, self.press)
            self.assertEqual(JobConfig.SKIP_PIGMENT_ACTIONS, self.skip)
            self.assertEqual(main.main(argv + ["--job-out", part, "--start", "3", "--limit", "4"]), 0)
            full_wp, meta = job.load_job(full)
            part_wp, _ = job.load_job(part)
            mapper = CanvasMap(meta["image_size"])
            a, b = contact_paths(full_wp, mapper)[3:7], contact_paths(part_wp, mapper)
            self.assertEqual(len(a), len(b))
            for pa, pb in zip(a, b): np.testing.assert_allclose(np.array([x[0] for x in pa]), np.array([x[0] for x in pb]))

    def test_classic_sample_reproducible(self):
        work, info = vision.prepare_image(vision.load_image("samples/ai/rock_02.png"))
        mapper = CanvasMap(info["prepared_size"])
        a = shuimo.build_framework_strokes(work, mapper.mm_per_px, info["pad"])[0]
        b = shuimo.build_framework_strokes(work, mapper.mm_per_px, info["pad"])[0]
        self.assertEqual(a, b)
        self.assertGreater(len(a), 0)

    def test_no_duplicate_initial_dip_for_overlong_stroke(self):
        JobConfig.SKIP_PIGMENT_ACTIONS = False
        s = dict(length_mm=ShuimoConfig.REDIP_LENGTH_MM + 50, tail_mm=0)
        _, stats = scheduler.build_framework_plan([s], 0)
        self.assertEqual(stats["dips"], 1)
        self.assertEqual(stats["overlong_count"], 1)

    def test_empty_input_and_no_unsupported_gap_bridge(self):
        blank = np.full((200, 300, 3), 255, np.uint8)
        self.assertEqual(mountain.build_strokes(blank, .2)[0], [])
        import cv2
        cv2.line(blank, (20, 90), (110, 30), (0, 0, 0), 2)
        cv2.line(blank, (180, 30), (270, 90), (0, 0, 0), 2)
        strokes, _ = mountain.build_strokes(blank, .2)
        self.assertGreaterEqual(len(strokes), 2)
        for s in strokes:
            x = np.array(s["points"])[:, 0]
            self.assertFalse(x.min() < 120 and x.max() > 170)

    def test_upper_ridge_tracks_ink_middle_without_following_vertical_spur(self):
        region = np.zeros((40, 60), dtype=bool)
        region[10:16, :] = True
        region[10:36, 30] = True
        center = mountain.upper_ink_centerline(region)
        self.assertTrue(np.all(center[np.arange(60) != 30, 1] == 12.5))
        self.assertLessEqual(center[30, 1], 15)
        self.assertTrue(np.all(region[np.rint(center[:, 1]).astype(int),
                                      center[:, 0].astype(int)]))

    def test_outer_contours_heavier_than_face_lines_and_far_hills_lighter(self):
        ShuimoConfig.PRESS_MIN_MM = .15
        _, _, strokes, _ = self.build()
        bands = {b: [s for s in strokes if s["depth_band"] == b]
                 for b in ("far", "middle", "near")}
        self.assertTrue(all(bands.values()))
        for band, group in bands.items():
            outlines = [s for s in group if s["role"] == "silhouette"]
            details = [s for s in group if s["role"] == "downhill"]
            self.assertTrue(outlines)
            if details:
                self.assertGreater(max(s["press_max"] for s in outlines),
                                   max(s["press_max"] for s in details))
        self.assertLess(max(s["press_max"] for s in bands["far"] if s["role"] == "silhouette"),
                        max(s["press_max"] for s in bands["near"] if s["role"] == "silhouette"))


if __name__ == "__main__": unittest.main()
