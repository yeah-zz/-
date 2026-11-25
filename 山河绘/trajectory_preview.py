"""Render the exported contact trajectory, including tails, excluding travel.

Widths remain estimates until the loaded brush/ink/paper are measured.
Controller corner blending and wet ink diffusion cannot be inferred from XY/Z.

默认只写一张拼版预览图（源图 / 分层路径 / 接触轨迹 / 加宽敏感性四宫格）。
四张分面板图（`_contact` `_stress` `_layers` `_lineart`）需要 `diagnostics=True` 才另存 ——
它们的内容本来就已经画在拼版图里，单独出图只是便于放大比对。
"""
from pathlib import Path
import cv2
import numpy as np
import imgio
from config import ArmConfig
from shuimo import press_to_width_mm


def contact_paths(waypoints, mapper):
    inverse = np.linalg.inv(mapper.matrix)
    paths, active = [], []
    for wp in waypoints:
        contact = not wp["pen_up"] and wp.get("tag", "").startswith(("down:", "draw:", "tail:"))
        if not contact:
            if active: paths.append(active)
            active = []
        else:
            p = cv2.perspectiveTransform(np.array([[[wp["x"], wp["y"]]]], np.float64), inverse)[0, 0]
            press = (wp["z"] - ArmConfig.PAPER_Z) * (1 if ArmConfig.Z_INCREASES_DOWN else -1)
            active.append((p, press))
    if active: paths.append(active)
    return paths


def render_contact(waypoints, mapper, extra_width_mm=0., output_scale=1.):
    w, h = mapper.image_size
    sim = np.full((round(h * output_scale), round(w * output_scale), 3), 255, np.uint8)
    px_per_mm = output_scale / mapper.mm_per_px
    for path in contact_paths(waypoints, mapper):
        for (a, pa), (b, pb) in zip(path, path[1:]):
            # Short subsegments faithfully interpolate tapering contact widths.
            n = max(1, int(np.ceil(np.linalg.norm(b-a) * output_scale / 1.5)))
            for t in np.linspace(0, 1, n+1):
                p = (a + (b-a) * t) * output_scale
                width = press_to_width_mm(pa + (pb-pa) * t) + extra_width_mm
                cv2.circle(sim, tuple(np.rint(p).astype(int)), max(1, round(width * px_per_mm / 2)),
                           (38, 38, 38), -1, cv2.LINE_AA)
    return sim


def render(source, strokes, mapper, waypoints, out_path, diagnostics=False):
    """写一张拼版预览图；diagnostics=True 时额外另存 4 张分面板图。

    笔痕宽度按 press_to_width_mm（= ShuimoConfig.BRUSHES 标定表）逐点算，
    所以拼版图里的 CONTACT 面板本身就能看出粗细变化。
    """
    scale = max(2., 1400. / source.shape[1])
    source = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    nominal = render_contact(waypoints, mapper, output_scale=scale)
    stress = render_contact(waypoints, mapper, extra_width_mm=.5, output_scale=scale)
    layers = np.full_like(source, 255)
    lineart = np.full_like(source, 255)
    colors = [(45, 45, 45), (175, 100, 35), (70, 155, 195)]
    for stroke in strokes:
        p = np.rint(np.array(stroke["points"]) * scale).astype(np.int32)
        cv2.polylines(layers, [p], False, colors[min(2, stroke.get("layer", 0))], 2, cv2.LINE_AA)
        cv2.polylines(lineart, [p], False, (38, 38, 38),
                      [3, 2, 1][min(2, stroke.get("layer", 0))], cv2.LINE_AA)
    path = Path(out_path)
    if diagnostics:
        for suffix, panel in [("contact", nominal), ("stress", stress), ("layers", layers), ("lineart", lineart)]:
            if not imgio.imwrite(str(path.with_name(path.stem + "_" + suffix + ".png")), panel):
                raise IOError("Cannot write preview: " + str(path))

    def tile(panel, title):
        panel = cv2.resize(panel, (900, round(panel.shape[0] * 900 / panel.shape[1])))
        bar = np.full((38, 900, 3), 245, np.uint8)
        cv2.putText(bar, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .54, (40, 40, 40), 1, cv2.LINE_AA)
        return np.vstack((bar, panel))

    board = np.vstack((np.hstack((tile(source, "SOURCE"), tile(layers, "PATHS: black outline / blue ridge / ochre texture"))),
                       np.hstack((tile(nominal, "EXPORTED XY/Z CONTACT | width estimated, not calibrated"),
                                  tile(stress, "WIDTH SENSITIVITY +0.5 mm | scenario, not a prediction")))))
    if not imgio.imwrite(str(path), board): raise IOError("Cannot write preview: " + str(path))
    return str(path)
