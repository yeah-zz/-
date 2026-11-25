"""Render source-aligned offline comparison; stroke width is not simulated."""

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("artifacts/legacy_review")


def read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def panel(image, title):
    bar = np.full((34, image.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, title, (9, 23), cv2.FONT_HERSHEY_SIMPLEX, .49,
                (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack((bar, image))


def paint(paths, size, pressure=False):
    width, height = size
    image = np.full((height, width, 3), 255, np.uint8)
    for path in paths:
        points = np.asarray(path, float)
        if len(points) < 2:
            continue
        xy = np.rint(points[:, :2] * [width / 1024, height / 1024]).astype(np.int32)
        if pressure:
            for (a, pa), (b, pb) in zip(zip(xy, points[:, 2]),
                                          zip(xy[1:], points[1:, 2])):
                value = min(1., max(0., (pa + pb) / 4.))  # shared 0–2 mm scale
                color = (int(210 * (1-value)), int(170 * (1-value)), 220)
                cv2.line(image, tuple(a), tuple(b), color, 2, cv2.LINE_AA)
        else:
            cv2.polylines(image, [xy], False, (35, 35, 35), 1, cv2.LINE_AA)
    return image


def norm_length(paths):
    return round(sum(float(np.linalg.norm(np.diff(np.asarray(p)[:, :2], axis=0),
                                           axis=1).sum()) for p in paths), 1)


def main():
    old = json.loads((ROOT / "legacy_sb_strokes.json").read_text(encoding="utf-8"))
    new = json.loads((ROOT / "current_sb_job.json").read_text(encoding="utf-8"))
    source = read_image(ROOT / "source/sb.jpg")
    if source is None:
        raise ValueError("Cannot decode source")
    width, height = 600, round(source.shape[0] * 600 / source.shape[1])
    source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    old_paths = [np.asarray(s["trajectory"], float)[:, :2]
                 for s in old["strokes"]]
    old_press_paths = old["drawn_norm_x_y_press"]
    new_width, new_height = new["meta"]["image_size"]
    source_height, source_width = read_image(ROOT / "source/sb.jpg").shape[:2]
    # Current FIT_MODE=contain pads this portrait-ish source at left and right.
    # Remove that padding before comparing both algorithms in source coordinates.
    source_scale = new_height / source_height
    left_pad = (new_width - source_width * source_scale) / 2
    new_paths, new_press_paths = [], []
    for stroke in new["meta"]["ink_execution"]["base_strokes"]:
        xy = np.asarray(stroke["points"], float)
        xy[:, 0] = (xy[:, 0] - left_pad) / (source_width * source_scale) * 1024
        xy[:, 1] = xy[:, 1] / (source_height * source_scale) * 1024
        new_paths.append(xy)
        new_press_paths.append(np.column_stack((xy, stroke["press"])))
    board = np.vstack((
        np.hstack((panel(source, "SAME INPUT: sb.jpg"),
                   panel(paint(old_paths, (width, height)), "LEGACY geometry"),
                   panel(paint(new_paths, (width, height)), "CURRENT geometry"))),
        np.hstack((panel(np.full_like(source, 255), "COMMAND PRESSURE: 0 to 2 mm"),
                   panel(paint(old_press_paths, (width, height), True),
                         "LEGACY planned pressure"),
                   panel(paint(new_press_paths, (width, height), True),
                         "CURRENT planned pressure"))),
    ))
    cv2.imencode(".png", board)[1].tofile(ROOT / "sb_same_source_comparison.png")
    metrics = {
        "source": "sb.jpg", "canvas_normalization": "each axis mapped to 0–1024",
        "legacy_normalized_centerline_length": norm_length(old_paths),
        "current_normalized_centerline_length": norm_length(new_paths),
        "legacy_strokes": len(old_paths), "current_strokes": len(new_paths),
        "note": "Physical lengths differ because source projects use different canvas scales",
    }
    (ROOT / "sb_same_source_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
