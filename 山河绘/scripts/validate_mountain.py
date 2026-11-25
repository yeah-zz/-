"""Rebuild offline review artifacts and a quantitative trajectory comparison.

Run from the project root: uv run python validate_mountain.py
No robot connection is made; socket creation is explicitly blocked.
"""

# --- path bootstrap: this script lives in scripts/, put the project root on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import imgio
import job
import main
from config import ArmConfig
from geometry import CanvasMap, check_waypoints
from trajectory_preview import contact_paths, render_contact


def metrics(wps, mapper):
    paths = contact_paths(wps, mapper)
    lengths = [sum(float(np.linalg.norm(b[0]-a[0])) * mapper.mm_per_px for a, b in zip(p, p[1:])) for p in paths]
    tails = sum(float(np.hypot(b["x"]-a["x"], b["y"]-a["y"]))
                for a, b in zip(wps, wps[1:]) if b["tag"].startswith("tail:"))
    travel = sum(float(np.hypot(b["x"]-a["x"], b["y"]-a["y"])) for a, b in zip(wps, wps[1:]))
    return dict(strokes=len(paths), waypoints=len(wps), dips=sum(w["tag"] == "dip" for w in wps),
                contact_length_mm=round(sum(lengths), 1), mean_mm=round(float(np.mean(lengths)), 2),
                median_mm=round(float(np.median(lengths)), 2), shorter_than_5mm=sum(x < 5 for x in lengths),
                added_tail_mm=round(tails, 1), xy_travel_mm=round(travel, 1),
                peak_press_mm=round(max(v for p in paths for _, v in p), 3),
                reach_ok=check_waypoints(wps)[0])


def tile(panel, title):
    h = round(panel.shape[0] * 900 / panel.shape[1])
    panel = cv2.resize(panel, (900, h), interpolation=cv2.INTER_AREA)
    bar = np.full((40, 900, 3), 245, np.uint8)
    cv2.putText(bar, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .57, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack((bar, panel))


def run():
    results = {}
    with patch("socket.socket", side_effect=AssertionError("Offline report forbids hardware/network")):
        for source, prefix, profile in [
            ("artifacts/reference/desired_mountain_reference.png", "artifacts/mountain/mountain_v4", "mountain"),
            ("artifacts/reference/mountain_baimiao_v3.png", "artifacts/mountain/mountain_v3_reprocessed", "mountain"),
            ("artifacts/reference/mountain_baimiao.png", "artifacts/mountain/mountain_v1_reprocessed", "mountain"),
            ("samples/ai/rock_02.png", "artifacts/mountain/rock_classic_regression", "classic")]:
            with contextlib.redirect_stdout(io.StringIO()):
                code = main.main(["--mode", "shuimo", "--shuimo-profile", profile, "--image", source,
                                  "--press-min", ".15", "--diagnostics",
                                  "--preview-out", prefix + "_preview.png",
                                  "--job-out", prefix + "_job.json"])
            if code: raise RuntimeError((source, code))
            wps, meta = job.load_job(prefix + "_job.json")
            results[prefix] = metrics(wps, CanvasMap(meta["image_size"]))
            results[prefix]["layers"] = meta["quality_stats"]["layers"]
            print(prefix, results[prefix])

        old = []
        for i in range(1, 5):
            wps, meta = job.load_job("artifacts/mountain/mountain_baimiao_v3_part%d_job.json" % i)
            old.extend(wps)
        mapper = CanvasMap(meta["image_size"])
        results["v3_archived_four_parts"] = metrics(old, mapper)
        print("v3_archived_four_parts", results["v3_archived_four_parts"])
        old_contact = render_contact(old, mapper, output_scale=3.)
        imgio.imwrite("artifacts/mountain/mountain_v3_actual_trajectory.png", old_contact)
        without_tails = render_contact([w for w in old if not w["tag"].startswith("tail:")], mapper, output_scale=3.)
        imgio.imwrite("artifacts/mountain/mountain_v3_without_tails.png", without_tails)
        new, new_meta = job.load_job("artifacts/mountain/mountain_v4_job.json")
        new_contact = render_contact(new, CanvasMap(new_meta["image_size"]), output_scale=2.)
        # Force equal panel height: old prepared image differs by one pixel.
        new_contact = cv2.resize(new_contact, (old_contact.shape[1], old_contact.shape[0]))
        new3, meta3 = job.load_job("artifacts/mountain/mountain_v3_reprocessed_job.json")
        same_source = render_contact(new3, CanvasMap(meta3["image_size"]), output_scale=2.)
        same_source = cv2.resize(same_source, (old_contact.shape[1], old_contact.shape[0]))
        board = np.vstack((np.hstack((tile(without_tails, "V3 without tails (old preview omission)"),
                                     tile(old_contact, "V3 exported contact: includes 4 mm tails"))),
                           np.hstack((tile(same_source, "SAME V3 SOURCE / new mountain profile"),
                                      tile(new_contact, "TARGET REFERENCE / new mountain profile")))))
        imgio.imwrite("artifacts/mountain/mountain_v4_comparison.png", board)
    Path("artifacts/mountain/mountain_validation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": run()
