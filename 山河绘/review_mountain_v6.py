"""Generate offline V6 review, same-scale mountain sample and brush probes.
Never connects to hardware. Individual probes are separate paper trials.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import cv2
import numpy as np
import imgio
import job
import ink_control
from geometry import CanvasMap, check_waypoints
from run_mountain_v4_robot import safe_transfers
from run_mountain_balanced import load_reviewed
from trajectory_preview import render_contact


def export(path, base, template, policy, purpose):
    meta = deepcopy(template)
    mapper = CanvasMap(meta["image_size"])
    steps, plan = ink_control.build_plan(base, base[0]["pal_index"], mapper.mm_per_px, policy)
    points, _ = job.steps_to_waypoints(steps, mapper)
    points = safe_transfers(points)
    if not check_waypoints(points)[0]: raise ValueError("Unreachable sample")
    meta.update(strokes=len(base), length_mm=plan["length_mm"], dips=plan["dips"],
                ink_plan=plan, purpose=purpose)
    meta["ink_execution"].update(base_strokes=base, policy=asdict(policy))
    meta["quality_stats"] = dict(detail="structured", strokes=len(base),
        layers={k: sum(s["layer"] == k for s in base) for k in range(3)},
        length_mm=plan["length_mm"], scope=purpose)
    pressures = [p for step in steps if step["kind"] == "stroke" for p in step["stroke"]["press"]]
    meta["actual_contact_press_range_mm"] = [min(pressures), max(pressures)]
    meta["mean_width_mm"] = float(np.mean([s["width_mm"] for s in base]))
    job.save_job(path, points, meta)
    load_reviewed(path)
    imgio.imwrite(str(Path(path).with_suffix(".png")), render_contact(points, mapper, output_scale=1.5))
    return dict(file=path, strokes=len(base), predicted_dips=plan["dips"],
                length_mm=round(plan["length_mm"], 2), purpose=purpose)


def probe(template, scale, x_mm, y_mm, length_mm, pressure):
    s = deepcopy(template)
    t = np.linspace(0, 1, int(length_mm*2)+1)
    # Vertical test marks, on mapped paper only; no uncalibrated test station.
    s.update(points=np.column_stack((np.full_like(t, x_mm)/scale,
                                    (y_mm+length_mm*t)/scale)).tolist(),
             press=[pressure]*len(t), speed=[28.]*len(t), tail_mm=0.,
             name="笔压供墨测试", role="brush_probe", layer=0,
             design_origin="diagnostic-test-mark")
    for key in ("mountain_group", "depth_band", "face_root_px", "parent_stroke_id", "segment", "segments"):
        s.pop(key, None)
    return ink_control.refresh_stats(s, scale)


def main():
    points, meta = job.load_job("artifacts/mountain/mountain_v6_job.json")
    base = meta["ink_execution"]["base_strokes"]
    scale = meta["ink_execution"]["mm_per_px"]
    policy = ink_control.InkPolicy(**meta["ink_execution"]["policy"])
    # Reference-specific groups: upper-left small mountain, central mountain,
    # lower-left foreground mountain. Preserve their original paper coordinates.
    groups = {20, 17, 22}
    sample = [s for s in base if s["mountain_group"] in groups]
    records = [export("artifacts/mountain/mountain_v6_sample_job.json", sample, meta, policy,
                      "Three source mountain groups at unchanged physical scale")]
    probe_policy = replace(policy, fresh_press_factor=1.)
    for pressure in (.08, .12, .18, .24):
        s = probe(base[0], scale, 40, 40, 40, pressure)
        records.append(export("artifacts/mountain/mountain_v6_pressure_%02d.json" % round(pressure*100),
                              [s], meta, probe_policy, "Separate 40mm pressure trial: %.2f mm" % pressure))
    endurance = [probe(base[0], scale, 40+i*7, 40, 12, .12) for i in range(8)]
    for dwell in (.6, .9, 1.2):
        records.append(export("artifacts/mountain/mountain_v6_ink_%02d.json" % round(dwell*10), endurance,
                              meta, replace(probe_policy, initial_dip_s=dwell),
                              "Separate ink trial: eight 12mm marks, initial dip %.1fs" % dwell))
    old, old_meta = job.load_job("artifacts/mountain/mountain_v5_job.json")
    mapper = CanvasMap(meta["image_size"])
    panels = []
    for title, wps in (("V5 | estimated contact", old), ("V6 | source-guided natural texture", points)):
        panel = render_contact(wps, mapper)
        panel = cv2.resize(panel, (1000, round(panel.shape[0]*1000/panel.shape[1])))
        bar = np.full((45, 1000, 3), 245, np.uint8)
        cv2.putText(bar, title, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, .7, (40,40,40), 1, cv2.LINE_AA)
        panels.append(np.vstack((bar, panel)))
    imgio.imwrite("artifacts/mountain/mountain_v6_comparison.png", np.hstack(panels))
    validation = dict(offline_only=True, brush_width_calibrated=False, selected_groups=sorted(groups),
                      sample_scale_unchanged=True, jobs=records,
                      full_strokes=len(base), natural_interior_strokes=sum(s["role"] != "silhouette" for s in base),
                      full_pressure_max=max(max(s["press"]) for s in base))
    Path("artifacts/mountain/mountain_v6_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=True, indent=2))


if __name__ == "__main__": main()
