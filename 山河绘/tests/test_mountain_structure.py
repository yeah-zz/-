"""Meaningful offline checks for source fidelity, pressure and sample scope."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import job
import main
import mountain
import vision
from config import ShuimoConfig
from geometry import CanvasMap, check_waypoints
from mountain_structure import simplify_contour
from run_mountain_balanced import load_reviewed
from review_mountain_v6 import export, probe
from ink_control import InkPolicy


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work, cls.info = vision.prepare_image(vision.load_image("artifacts/reference/desired_mountain_reference.png"), work_size=1800)
        cls.mapper = CanvasMap(cls.info["prepared_size"], cls.info["image_corners"])
        with patch.object(ShuimoConfig, "PRESS_MIN_MM", .15):
            cls.strokes, cls.stats = mountain.build_strokes(cls.work, cls.mapper.mm_per_px,
                                                         cls.info["pad"], "structured")

    def test_natural_source_support_and_downhill_instead_of_horizontal_fill(self):
        white = (cv2.cvtColor(self.work, cv2.COLOR_BGR2GRAY) >= 210).astype(np.uint8)*255
        distance = cv2.distanceTransform(white, cv2.DIST_L2, 5)*self.mapper.mm_per_px
        details = [s for s in self.strokes if s["role"] != "silhouette"]
        # Centering the broad upper ink run absorbs some formerly separate
        # face fragments. Require fidelity as well as a useful detail count.
        self.assertGreaterEqual(len(details), 60)
        source_center = vision._thin(vision.neutral_ink_mask(self.work, gray_limit=210) * 255)
        drawn = np.zeros(source_center.shape, np.uint8)
        for s in self.strokes:
            cv2.polylines(drawn, [np.rint(s["points"]).astype(np.int32)],
                          False, 255, 1)
        near_drawn = cv2.dilate(drawn, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        coverage = np.count_nonzero((source_center > 0) & (near_drawn > 0)) / np.count_nonzero(source_center)
        self.assertGreaterEqual(coverage, .62)
        for s in details:
            p = np.asarray(s["points"])
            self.assertEqual(s["design_origin"], "source-skeleton")
            self.assertGreaterEqual((p[-1,1]-p[0,1])*self.mapper.mm_per_px, .47*s["length_mm"])
            ij = np.rint(p).astype(int)
            self.assertLessEqual(distance[ij[:,1], ij[:,0]].max(), .45)
            self.assertTrue(check_waypoints(job.stroke_waypoints(self.mapper, s))[0])

    def test_summit_endpoints_and_simplification_error(self):
        scale = .2
        p = np.array([[0, 20], [10, 14], [12, 14.1], [20, 0], [25, 10], [40, 20]], float)
        simple = simplify_contour(p, scale)
        np.testing.assert_allclose(simple[[0,-1]], p[[0,-1]])
        self.assertTrue(any(np.allclose(v, [20,0]) for v in simple))
        for point in p:
            a, b = simple[:-1], simple[1:]
            d = b-a
            t = np.clip(np.sum((point-a)*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-12),0,1)
            error = np.linalg.norm(point-(a+t[:,None]*d),axis=1).min()*scale
            self.assertLessEqual(error, .33)

    def test_pressure_hierarchy_taper_and_no_external_tail(self):
        caps = {"far": .12, "middle": .18, "near": .24}
        self.assertEqual({s["depth_band"] for s in self.strokes}, set(caps))
        for s in self.strokes:
            self.assertEqual(s["press"][0], 0.)
            self.assertEqual(s["press"][-1], 0.)
            self.assertLessEqual(max(s["press"]), caps[s["depth_band"]]+1e-9)
            self.assertEqual(s["tail_mm"], 0.)

    def test_generated_job_and_sample_are_same_scale_offline(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                patch("socket.socket", side_effect=AssertionError("Must remain offline")):
            full = str(Path(folder)/"full.json")
            self.assertEqual(main.main(["--mode","shuimo","--shuimo-profile","mountain",
                "--mountain-detail","structured","--ink-policy","balanced","--press-min",".15",
                "--image","artifacts/reference/desired_mountain_reference.png","--no-preview","--job-out",full]),0)
            strokes, mapper, policy, meta = load_reviewed(full)
            selected = [s for s in strokes if s["mountain_group"] in {17,20,22}]
            part = str(Path(folder)/"part.json")
            export(part, selected, meta, policy, "test same-scale subset")
            actual, part_mapper, _, _ = load_reviewed(part)
            self.assertEqual(actual, selected)
            np.testing.assert_equal(mapper.matrix, part_mapper.matrix)
            self.assertEqual({s["mountain_group"] for s in actual}, {17,20,22})
            for pressure in (.08,.12,.18,.24):
                s = probe(strokes[0], mapper.mm_per_px, 40, 40, 40, pressure)
                self.assertAlmostEqual(s["length_mm"],40)
                self.assertEqual(set(s["press"]), {pressure})
                self.assertEqual(s["tail_mm"],0)

    def test_empty_profile_does_not_invent_lines(self):
        blank = np.full((200,300,3),255,np.uint8)
        self.assertEqual(mountain.build_strokes(blank,.2,detail="structured")[0], [])


if __name__ == "__main__": unittest.main()
