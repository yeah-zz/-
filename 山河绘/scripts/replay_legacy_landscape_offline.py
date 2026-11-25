"""Offline replay of the supplied legacy landscape ink branch; no robot imports.

The supplied file imports sklearn/scipy for its color fallback. Those imports
are omitted here because this replay uses only its white-background ink path.
The copied generator and brush engine functions otherwise run unchanged.
"""

import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np


def load_legacy(source_dir):
    namespace = {"__name__": "legacy_landscape_offline"}
    source = (source_dir / "landscape_painting.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module in
        {"sklearn.cluster", "scipy.ndimage"}
    )]
    exec(compile(tree, str(source_dir / "landscape_painting.py"), "exec"), namespace)
    config_ns = {"__name__": "legacy_config_offline"}
    config_source = (source_dir / "config.py").read_text(encoding="utf-8")
    exec(compile(config_source, str(source_dir / "config.py"), "exec"), config_ns)
    return namespace["LandscapeBrushGenerator"], namespace["LandscapeBrushEngine"], config_ns["Config"]


def percentile(values, q):
    return round(float(np.percentile(values, q)), 3) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path,
                        default=Path("artifacts/legacy_review/source"))
    parser.add_argument("--image", type=Path,
                        default=Path("artifacts/legacy_review/source/sb.jpg"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("artifacts/legacy_review"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    encoded = np.fromfile(args.image, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image")
    h0, w0 = image.shape[:2]
    if max(h0, w0) > 600:
        scale = 600 / max(h0, w0)
        image = cv2.resize(image, (int(w0 * scale), int(h0 * scale)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    generator_type, engine_type, config = load_legacy(args.source_dir)
    generator = generator_type(config)
    # Same operation as the ink branch of image_to_landscape_strokes after resize.
    strokes = generator._extract_continuous_ink_strokes(gray)
    np.random.seed(0)  # Only the engine's ±0.05 mm XY jitter uses randomness here.
    waypoints = engine_type(config).generate_waypoints(
        strokes, offset_x=float(config.START_X), offset_y=float(config.START_Y))

    preview = np.full_like(image, 255)
    colors = [(30, 30, 30), (90, 90, 90), (145, 145, 145)]
    lengths_mm = []
    for stroke in strokes:
        path = np.asarray(stroke["trajectory"], dtype=float)
        if len(path) < 2:
            continue
        points = np.rint(np.column_stack((
            path[:, 0] * image.shape[1] / 1024,
            path[:, 1] * image.shape[0] / 1024,
        ))).astype(np.int32)
        cv2.polylines(preview, [points], False, colors[min(stroke["layer"], 2)],
                      1, cv2.LINE_AA)
        lengths_mm.append(float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()
                                * config.LANDSCAPE_SCALE))

    source = cv2.resize(image, (preview.shape[1], preview.shape[0]))
    board = np.hstack((source, preview))
    cv2.imencode(".png", board)[1].tofile(args.out_dir / "legacy_sb_centerline.png")
    pressures = [float(p["z"] - config.LANDSCAPE_BASE_Z) for p in waypoints
                 if not p["pen_up"]]
    result = {
        "source": str(args.image), "source_size": [w0, h0],
        "processing_size": [image.shape[1], image.shape[0]],
        "replay": "legacy white-background ink branch; color imports omitted",
        "strokes": len(strokes), "layers": {str(k): sum(s["layer"] == k for s in strokes)
                                            for k in range(3)},
        "waypoints": len(waypoints), "contact_points": len(pressures),
        "length_total_mm": round(sum(lengths_mm), 2),
        "length_median_mm": percentile(lengths_mm, 50),
        "strokes_under_5mm": sum(x < 5 for x in lengths_mm),
        "press_mm": {"p10": percentile(pressures, 10),
                     "p50": percentile(pressures, 50),
                     "p90": percentile(pressures, 90),
                     "max": round(max(pressures), 3) if pressures else None},
        "legacy_calibration": {
            "landscape_base_z": config.LANDSCAPE_BASE_Z,
            "landscape_press_min": config.LANDSCAPE_PRESS_MIN,
            "landscape_press_max": config.LANDSCAPE_PRESS_MAX,
            "landscape_scale": config.LANDSCAPE_SCALE,
        },
    }
    (args.out_dir / "legacy_sb_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    drawn, segment = [], []
    for point in waypoints:
        if point["pen_up"]:
            if segment:
                drawn.append(segment)
                segment = []
        else:
            segment.append([
                (point["y"] - config.START_Y) / config.LANDSCAPE_SCALE,
                1024 - (point["x"] - config.START_X) / config.LANDSCAPE_SCALE,
                point["z"] - config.LANDSCAPE_BASE_Z,
            ])
    (args.out_dir / "legacy_sb_strokes.json").write_text(
        json.dumps({"format": "legacy-offline-strokes", "strokes": strokes,
                    "drawn_norm_x_y_press": drawn}, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
