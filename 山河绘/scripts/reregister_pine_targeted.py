"""Recompile the reviewed targeted-pine design for the current site calibration.

This is offline only: it reads the retained pixel-space design, rebuilds all
robot-space waypoints from the current config, renders a fresh preview, and
refuses to overwrite an existing revision.
"""

from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imgio  # noqa: E402
import ink_control  # noqa: E402
import job  # noqa: E402
import shuimo  # noqa: E402
import vision  # noqa: E402
import watercolor  # noqa: E402
from config import ArmConfig, CanvasConfig, PALETTE  # noqa: E402
from geometry import CanvasMap, check_waypoints  # noqa: E402
from run_color_from_job import load_plan  # noqa: E402
from run_mountain_balanced import load_reviewed  # noqa: E402
from run_mountain_v4_robot import safe_transfers  # noqa: E402


SOURCE = ROOT / "输出" / "山亭松瀑_松树专项定稿_20260928"
OUTPUT = ROOT / "输出" / "山亭松瀑_松树专项_新高度_20260929"
STEM = "山亭松瀑_松树专项_新高度"


def geometry_dict(mapper):
    return {
        "image_size": list(mapper.image_size),
        "mm_per_px": mapper.mm_per_px,
        "image_corners": [list(point) for point in mapper.image_corners],
    }


def path_length_mm(points, scale):
    points = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() * scale)


def main():
    completed_markers = [OUTPUT / "说明.txt", OUTPUT / (STEM + "_预览.png")]
    if any(path.exists() for path in completed_markers):
        raise FileExistsError("Refusing to overwrite completed revision: %s" % OUTPUT)

    source_job = SOURCE / "山亭松瀑_松树专项_黑线任务.json"
    source_plan = SOURCE / "山亭松瀑_松树专项_计划.json"
    source_image = SOURCE / "参考原图.png"
    _, old_meta = job.load_job(source_job)
    plan = json.loads(source_plan.read_text(encoding="utf-8-sig"))

    mapper = CanvasMap(old_meta["image_size"])
    scale = mapper.mm_per_px
    policy = ink_control.InkPolicy(**old_meta["ink_execution"]["policy"])

    strokes = copy.deepcopy(old_meta["ink_execution"]["base_strokes"])
    for stroke in strokes:
        ink_control.refresh_stats(stroke, scale)
    strokes = ink_control.split_long_strokes(strokes, scale, policy)
    strokes = ink_control.order_for_ink(strokes, scale, policy)
    for index, stroke in enumerate(strokes):
        stroke["id"] = "outline-%05d" % index

    steps, _ = ink_control.build_plan(
        strokes, shuimo.ink_palette_index(), scale, policy
    )
    conditioned = [step["stroke"] for step in steps if step["kind"] == "stroke"]
    waypoints, _ = job.steps_to_waypoints(steps, mapper)
    waypoints = safe_transfers(waypoints)
    reachable, problems = check_waypoints(waypoints)
    if not reachable:
        raise ValueError("Recompiled black path is outside reach: " + "; ".join(problems))

    output_image = OUTPUT / "参考原图.png"
    meta = copy.deepcopy(old_meta)
    meta["image"] = str(output_image.resolve())
    meta["paper_z"] = ArmConfig.PAPER_Z
    meta["canvas_corners"] = [list(point) for point in CanvasConfig.ROBOT_CORNERS]
    meta["strokes"] = len(strokes)
    meta["quality_stats"]["strokes"] = len(strokes)
    meta["ink_execution"]["base_strokes"] = strokes
    meta["ink_execution"]["mm_per_px"] = scale
    meta["ink_execution"]["calibration"] = {
        "rpy": list(ArmConfig.RPY),
        "user": ArmConfig.USER_INDEX,
        "tool": ArmConfig.TOOL_INDEX,
        "speed_factor": ArmConfig.SPEED_FACTOR,
        "palette": copy.deepcopy(PALETTE),
    }
    meta["registration_revision"] = {
        "source_job": str(source_job.resolve()),
        "paper_z": ArmConfig.PAPER_Z,
        "canvas_corners": [list(point) for point in CanvasConfig.ROBOT_CORNERS],
        "note": "Pixel-space artwork retained; physical waypoints rebuilt from current calibration.",
    }

    geom = geometry_dict(mapper)
    plan["reference_image"] = str(output_image.resolve())
    plan.update(geom)
    plan["modules"]["outline"].update(geom)
    plan["modules"]["outline"]["strokes"] = conditioned
    plan["modules"]["outline"]["stroke_count"] = len(conditioned)
    color_module = plan["modules"]["color"]
    color_module.update(geom)
    for stroke in color_module["strokes"]:
        stroke["length_mm"] = path_length_mm(stroke["points"], scale)
    color_module["stroke_count"] = len(color_module["strokes"])
    plan["registration_revision"] = copy.deepcopy(meta["registration_revision"])

    # exist_ok permits recovery from an interrupted offline build.  The
    # completed markers above still protect a finished, reviewable revision.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prefix = OUTPUT / STEM
    job.save_job(str(prefix) + "_黑线任务.json", waypoints, meta)
    Path(str(prefix) + "_计划.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(source_image, output_image)

    raw = vision.load_image(str(source_image))
    target_width, target_height = mapper.image_size
    # The retained reference file is the unpadded 1448x1086 source.  The
    # reviewed pine paths live on the old 1572x1086 centered working canvas,
    # so reproduce that canvas exactly instead of asking today's aspect-ratio
    # settings to choose a new padding width.
    if raw.shape[:2] == (target_height, target_width):
        work = raw
    elif raw.shape[0] == target_height and raw.shape[1] < target_width:
        work = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
        left = (target_width - raw.shape[1]) // 2
        work[:, left:left + raw.shape[1]] = raw
    else:
        work, _ = vision.prepare_image(raw, work_size=1800)
        if work.shape[:2] != (target_height, target_width):
            raise ValueError("Reference image cannot reproduce the reviewed design canvas")
    canvas, _ = watercolor.render_image(work, color_module, include_ink=False)
    for stroke in conditioned:
        points = np.asarray(stroke["points"], dtype=float)
        presses = stroke["press"]
        for start, end, p0, p1 in zip(points, points[1:], presses, presses[1:]):
            width_px = (shuimo.press_to_width_mm(p0) + shuimo.press_to_width_mm(p1)) / (2 * scale)
            cv2.line(
                canvas,
                tuple(np.rint(start).astype(int)),
                tuple(np.rint(end).astype(int)),
                (35, 35, 35),
                max(1, round(width_px)),
                cv2.LINE_AA,
            )
    if not imgio.imwrite(str(prefix) + "_预览.png", canvas):
        raise OSError("Could not write preview")
    if not imgio.imwrite(str(OUTPUT / "松树局部_路径预览.png"), canvas[350:735, 130:710]):
        raise OSError("Could not write pine detail preview")

    explanation = """山亭松瀑“松树专项定稿”当前标定重生成版。

保留原版像素空间内的松树专项构图和逐点笔压；机器人坐标、绝对 Z、画布比例、
耗墨统计及安全转移轨迹均按生成时的当前 config.py 重新计算。

纸面高度：{paper_z:.4f}
画布四角：{corners}
黑线：{black} 笔；彩色：{colors}

本目录是新版本，不覆盖 2026-09-28 旧任务。正式绘制前先检查两张路径预览，
并先用不带 --execute 的命令做离线校验：

uv run --locked python run_mountain_balanced.py --job "输出\\山亭松瀑_松树专项_新高度_20260929\\山亭松瀑_松树专项_新高度_黑线任务.json"
""".format(
        paper_z=ArmConfig.PAPER_Z,
        corners=json.dumps(CanvasConfig.ROBOT_CORNERS, ensure_ascii=False),
        black=len(strokes),
        colors=json.dumps(Counter(s["color"] for s in color_module["strokes"]), ensure_ascii=False),
    )
    (OUTPUT / "说明.txt").write_text(explanation, encoding="utf-8")

    # Final offline loaders exercise the same registration gates used by the runners.
    load_reviewed(str(prefix) + "_黑线任务.json")
    load_plan(str(prefix) + "_计划.json")
    print("Output:", OUTPUT)
    print("Paper Z:", ArmConfig.PAPER_Z)
    print("Scale:", scale)
    print("Black strokes:", len(strokes))
    print("Colors:", dict(Counter(s["color"] for s in color_module["strokes"])))
    print("Offline validation passed; no robot connection or motion.")


if __name__ == "__main__":
    main()
