"""Offline checks for the uncalibrated second-pass color planner."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import color_pass
from config import CanvasConfig
from geometry import CanvasMap
import imgio
import job
import main
import vision


class ColorPassTests(unittest.TestCase):
    def test_color_classification_ignores_black_and_paper(self):
        swatches = np.array([[[42, 53, 214], [52, 194, 225],
                              [181, 115, 50], [0, 0, 0], [255, 255, 255]]],
                            dtype=np.uint8)
        labels, _ = color_pass.classify_colors(swatches)
        self.assertEqual(labels[0].tolist(), [0, 1, 2, -1, -1])

    def test_strokes_keep_entire_brush_away_from_ink(self):
        labels = np.zeros((100, 100), dtype=np.int8)
        blocked = np.zeros((100, 100), dtype=bool)
        blocked[:, 48:52] = True
        lightness = np.full((100, 100), 75.0, dtype=np.float32)
        strokes = color_pass.plan_strokes(labels, lightness, blocked, .5,
                                          brush_width_mm=3, min_area_mm2=1)
        self.assertTrue(strokes)
        radius_px = 3
        for stroke in strokes:
            x0, y = map(int, stroke["points"][0])
            x1 = int(stroke["points"][1][0])
            self.assertFalse(blocked[max(0, y-radius_px):y+radius_px+1,
                                     max(0, x0-radius_px):min(100, x1+radius_px+1)].any())

    def test_full_plan_and_execution_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ink.png"
            design = root / "color.png"
            ink_image = np.full((100, 100, 3), 255, np.uint8)
            cv2.line(ink_image, (50, 4), (50, 95), (0, 0, 0), 2)
            color_image = np.full_like(ink_image, 255)
            cv2.rectangle(color_image, (5, 15), (42, 85), (42, 53, 214), -1)
            cv2.rectangle(color_image, (58, 15), (95, 85), (181, 115, 50), -1)
            self.assertTrue(imgio.imwrite(str(source), ink_image))
            self.assertTrue(imgio.imwrite(str(design), color_image))
            work, info = vision.prepare_image(ink_image, work_size=1800)
            mapper = CanvasMap(info["prepared_size"], info["image_corners"])
            image_w, image_h = info["prepared_size"]
            meta = {"ink_policy": "balanced", "profile": "mountain",
                    "image": str(source), "image_size": list(info["prepared_size"]),
                    "canvas_corners": [list(p) for p in CanvasConfig.ROBOT_CORNERS],
                    "ink_execution": {"mm_per_px": mapper.mm_per_px,
                                      "base_strokes": [{"points": [[image_w/2, 0],
                                                                   [image_w/2, image_h-1]]}]}}
            ink_job = root / "ink_job.json"
            job.save_job(str(ink_job), [], meta)
            plan, work = color_pass.build_plan(ink_job, design)
            self.assertFalse(plan["ready_for_execution"])
            self.assertNotIn("waypoints", plan)
            self.assertEqual({s["color"] for s in plan["strokes"]}, {"red", "blue"})
            self.assertIn("manual_wash_and_wipe",
                          [s["kind"] for s in plan["sequence"]])
            output_dir = root / "中文输出"
            preview = output_dir / "预览.png"
            color_pass.render_preview(work, plan["strokes"], plan["mm_per_px"], preview)
            self.assertIsNotNone(imgio.imread(str(preview)))
            path = output_dir / "plan.json"
            color_pass.save_plan(path, plan)
            self.assertFalse(json.loads(path.read_text(encoding="utf-8"))["ready_for_execution"])
            with self.assertRaises(FileExistsError):
                color_pass.save_plan(path, plan)
            cli_plan = output_dir / "cli_plan.json"
            cli_preview = output_dir / "cli_preview.png"
            self.assertEqual(main.main(["--mode", "color-pass", "--image", str(design),
                                        "--ink-job", str(ink_job), "--job-out", str(cli_plan),
                                        "--preview-out", str(cli_preview)]), 0)
            self.assertTrue(cli_plan.is_file() and cli_preview.is_file())
            self.assertEqual(main.main(["--mode", "color-pass", "--image", str(design),
                                        "--ink-job", str(ink_job), "--execute"]), 2)


if __name__ == "__main__":
    unittest.main()
