"""Build visual and numerical offline review from preserved v4 and new v5 jobs."""

# --- path bootstrap: this script lives in scripts/, put the project root on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
from pathlib import Path
import cv2
import numpy as np
import imgio
import job
from config import ArmConfig
from geometry import CanvasMap
from shuimo import press_to_width_mm


def archived_cycles(points):
    cycles = []
    current = None
    previous = None
    for wp in points:
        if wp["tag"] == "dip":
            if current: cycles.append(current)
            current = dict(strokes=0, length_mm=0., equivalent_mm=0.)
        if current is None: continue
        if wp["tag"].startswith("lift:"): current["strokes"] += 1
        if previous and not previous["pen_up"] and not wp["pen_up"]:
            length = float(np.hypot(wp["x"]-previous["x"], wp["y"]-previous["y"]))
            pressures = [(p["z"]-ArmConfig.PAPER_Z)*(1 if ArmConfig.Z_INCREASES_DOWN else -1) for p in (previous, wp)]
            current["length_mm"] += length
            current["equivalent_mm"] += length * sum(press_to_width_mm(p) for p in pressures) / 1.8
        previous = wp
    if current: cycles.append(current)
    return cycles


def tile(panel, label):
    panel = cv2.resize(panel, (900, round(panel.shape[0]*900/panel.shape[1])), interpolation=cv2.INTER_AREA)
    bar = np.full((44, 900, 3), 245, np.uint8)
    cv2.putText(bar, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, .62, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack((bar, panel))


def main():
    old, old_meta = job.load_job("artifacts/mountain/mountain_v4_safe_execution_job.json")
    new, new_meta = job.load_job("artifacts/mountain/mountain_v5_job.json")
    cycles = archived_cycles(old)
    rows = new_meta["ink_plan"]["ink_cycles"]
    new_usage = {}
    for row in rows: new_usage[row["dip_cycle"]] = row["used_before_eq_mm"] + row["demand_eq_mm"]
    stats = dict(v4=dict(strokes=70, dips=9, total_dip_dwell_s=27.,
                        actual_total_seconds=597.2, per_cycle_timing_available=False,
                        ink_cycles=cycles, max_equivalent_mm=max(c["equivalent_mm"] for c in cycles)),
                 v5=dict(semantic_strokes=new_meta["quality_stats"]["semantic_strokes"],
                         strokes=new_meta["strokes"], extra_splits=new_meta["quality_stats"]["ink_split_count"],
                         layers=new_meta["quality_stats"]["layers"],
                         predicted_dips=new_meta["dips"],
                         predicted_total_s=new_meta["ink_plan"]["estimated_total_s"],
                         predicted_dwell_s=new_meta["ink_plan"]["total_dip_dwell_s"],
                         max_predicted_equivalent_mm=max(new_usage.values()),
                         max_predicted_age_s=max(r["end_age_s"] for r in rows),
                         real_paper_verified=False),
                 warning="Ink consumption/age predictions are heuristic, not ink volume or optical density measurements")
    Path("artifacts/mountain/mountain_v5_validation.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    a = imgio.imread("artifacts/mountain/mountain_v4_preview_contact.png")
    b = imgio.imread("artifacts/mountain/mountain_v5_preview_contact.png")
    def closeup(p):
        h, w = p.shape[:2]
        return p[int(.48*h):int(.74*h), int(.20*w):int(.58*w)]
    board = np.vstack((np.hstack((tile(a, "V4: 70 strokes / 31 texture strokes"),
                                 tile(b, "V5: 90 lines / 94 execution strokes / 50 textures"))),
                       np.hstack((tile(closeup(a), "V4 central mountain - contact width estimate"),
                                  tile(closeup(b), "V5 central mountain - added downhill structure")))))
    imgio.imwrite("artifacts/mountain/mountain_v5_comparison.png", board)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
