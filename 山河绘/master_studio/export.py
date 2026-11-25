# -*- coding: utf-8 -*-
"""把名家工坊的笔触计划导出为现有机械臂可读取的完整绘画包。"""

import json
from dataclasses import asdict
from pathlib import Path

from config import ArmConfig, CanvasConfig, PALETTE, ShuimoConfig
from color_profile import ColorProfile
from geometry import CanvasMap
from ink_control import InkPolicy, build_plan as build_ink_plan, split_long_strokes, order_for_ink, refresh_stats
from job import steps_to_waypoints
from run_color_experiment import module_lines
from run_mountain_v4_robot import safe_transfers
from shuimo import ink_palette_index, press_to_width_mm


def _save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _thin_by_role(items, max_total=1500):
    """机械臂一次作画不宜落上万颗苔点；按角色配额保留结构笔、压缩碎点。"""
    quotas = {"outline": 520, "tree": 380, "bark": 70, "detail": 90,
              "texture": 460, "needle": 180, "moss": 240, "water": 80}
    buckets = {}
    for item in items:
        buckets.setdefault(item.get("role", "texture"), []).append(item)
    kept = []
    for role, vals in buckets.items():
        q = quotas.get(role, 60)
        if len(vals) <= q:
            kept.extend(vals)
        else:
            step = len(vals) / float(q)
            kept.extend(vals[int(i * step)] for i in range(q))
    # 如果仍然过多，按比例再抽一次，但保证轮廓优先
    if len(kept) > max_total:
        outline = [x for x in kept if x.get("role") == "outline"]
        rest = [x for x in kept if x.get("role") != "outline"]
        room = max(0, max_total - len(outline))
        if len(rest) > room and room > 0:
            step = len(rest) / float(room)
            rest = [rest[int(i * step)] for i in range(room)]
        kept = outline + rest
    return kept


def _image_size(path):
    import cv2
    import numpy as np
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("导出图无法读取。")
    return im.shape[1], im.shape[0]


def _synthetic_color_strokes(image_path, W, H, scale, max_per_color=260):
    """从预览的设色区域生成可执行的宽笔铺色，避免“预览有色、机械臂只有几条”。"""
    import cv2
    import numpy as np
    im = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return []
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    out = []
    ranges = {"blue": ((82, 118), (18, 255), (35, 252)),
              "green": ((32, 86), (16, 255), (35, 248))}
    for color, ((h0, h1), (s0, s1), (v0, v1)) in ranges.items():
        mask = cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
        # 避开墨线与纸边
        mask[gray < 70] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count = 0
        for y in range(12, H - 12, max(8, int(2.6 / max(0.05, scale)))):
            row = mask[y]
            x = 12
            while x < W - 12 and count < max_per_color:
                if row[x] == 0:
                    x += 1
                    continue
                x0 = x
                while x < W - 12 and row[x] > 0:
                    x += 1
                if x - x0 >= max(16, int(4.0 / max(0.05, scale))):
                    x1 = x
                    pts = [[float(x0), float(y)], [float((x0 + x1) / 2), float(y - 1)],
                           [float(x1), float(y)]]
                    n = 3
                    out.append(dict(id="master-fill-%s-%04d" % (color, count), color=color,
                                    role="wash", points=pts, press=[0.85] * n, speed=[40] * n,
                                    layer=0, brush_width_mm_estimate=float(press_to_width_mm(0.85)),
                                    target_opacity=0.55,
                                    target_width_mm=float(press_to_width_mm(0.85)),
                                    length_mm=float((x1 - x0) * scale)))
                    count += 1
        print("export: synthetic", color, count, flush=True)
    return out


def export_bundle(image_path, strokes, out_path, title="山河绘 · 名家笔触"):
    """image_path 与 strokes 使用同一像素坐标系；strokes 为 render 的 plan。"""
    image_path = Path(image_path).resolve()
    out_path = Path(out_path).resolve()
    W, H = _image_size(image_path)
    mapper = CanvasMap((W, H))
    scale = float(mapper.mm_per_px)
    policy = InkPolicy()

    black, colors = [], {"blue": [], "green": []}
    for idx, s in enumerate(strokes):
        pts = [[round(float(x), 3), round(float(y), 3)] for x, y in s.get("points", [])]
        if len(pts) < 2:
            continue
        press = [_clamp(0.06 + float(p) * 0.40, 0.06, float(ShuimoConfig.PRESS_MAX_MM))
                 for p in s.get("press", [0.8] * len(pts))]
        if len(press) != len(pts):
            press = [press[0]] * len(pts)
        raw_speed = s.get("speed", 40)
        if isinstance(raw_speed, (list, tuple)) and len(raw_speed) == len(pts):
            speed = [int(_clamp(v, 5, 100)) for v in raw_speed]
        else:
            speed = [int(_clamp(raw_speed, 5, 100))] * len(pts)
        if s.get("color") == "ink":
            if s.get("role") == "inscription":
                continue
            item = dict(id="master-black-%05d" % idx, points=pts, press=press, speed=speed,
                        layer=0, role=s.get("role", "texture"), name=s.get("role", "texture"),
                        tail_mm=0, lift_mm=2.5, horizontal_velocity_percent=80,
                        lift_velocity_percent=60, touchdown_velocity_percent=30)
            refresh_stats(item, scale)
            black.append(item)
        elif s.get("color") in colors:
            n = len(pts)
            item = dict(id="master-color-%05d" % idx, color=s["color"], role=s.get("role", "wash"),
                        points=pts, press=[0.85] * n, speed=[40] * n, layer=0,
                        brush_width_mm_estimate=float(press_to_width_mm(0.85)),
                        target_opacity=0.55, target_width_mm=float(press_to_width_mm(0.85)))
            item["length_mm"] = float(sum(((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5 for i in range(n - 1)) * scale)
            colors[s["color"]].append(item)

    if not black:
        raise ValueError("没有可导出的黑线笔触。")
    black = _thin_by_role(black, max_total=1500)
    colors = {k: _thin_by_role(v, max_total=180) for k, v in colors.items()}
    black = split_long_strokes(black, scale, policy)
    black = order_for_ink(black, scale, policy)
    for i, s in enumerate(black):
        s["id"] = "master-black-%05d" % i
    print("export: black=%d color=%d/%d" % (len(black), len(colors["blue"]), len(colors["green"])), flush=True)
    steps, ink_stats = build_ink_plan(black, ink_palette_index(), scale, policy)
    waypoints, _ = steps_to_waypoints(steps, mapper)
    waypoints = safe_transfers(waypoints)

    all_speeds = [float(v) for st in black for v in st.get("speed", [])]
    all_press = [float(v) for st in black for v in st.get("press", [])]
    traj = dict(speed_min=round(min(all_speeds), 1) if all_speeds else 0,
                speed_max=round(max(all_speeds), 1) if all_speeds else 0,
                press_min=round(min(all_press), 3) if all_press else 0,
                press_max=round(max(all_press), 3) if all_press else 0,
                points=sum(len(st.get("points", [])) for st in black),
                note="B样条平滑 + 曲率降速 + 压力耦合")
    meta = dict(mode="shuimo-framework", profile="mountain", ink_policy="balanced",
                image=str(image_path), image_size=[W, H],
                canvas_corners=[list(p) for p in CanvasConfig.ROBOT_CORNERS],
                paper_z=ArmConfig.PAPER_Z, strokes=len(black), task_title=title,
                air_transfer_velocity_percent=40, quality_stats={}, trajectory=traj,
                ink_execution=dict(base_strokes=black, policy=dict(ink_stats["ink_policy"]),
                                   mm_per_px=scale,
                                   calibration=dict(rpy=list(ArmConfig.RPY), user=ArmConfig.USER_INDEX,
                                                    tool=ArmConfig.TOOL_INDEX,
                                                    speed_factor=ArmConfig.SPEED_FACTOR, palette=PALETTE)))
    meta["ink_execution"]["policy"] = asdict(InkPolicy())

    profile = ColorProfile(press_mm=0.85, draw_velocity_percent=40,
                           travel_velocity_percent=80, dip_velocity_percent=20,
                           global_speed_factor=70, draw_velocity_max=100,
                           interstroke_lift_mm=2.5, touchdown_velocity_percent=30,
                           lift_velocity_percent=60, horizontal_velocity_percent=80).to_dict()
    color_strokes = []
    for name in ("blue", "green"):
        color_strokes.extend(colors[name])
    color_strokes.extend(_synthetic_color_strokes(image_path, W, H, scale))
    color_module = dict(format="sparse-color-design-v1", palette_bgr={}, motion_candidate=profile,
                        reload_strokes_per_load=8, image_size=[W, H], image_corners=None,
                        mm_per_px=scale, regions=[], strokes=color_strokes,
                        stroke_count=len(color_strokes))
    # 只保留执行器认为在纸面安全距内的色笔。
    allowed = set()
    for name in ("blue", "green"):
        lines, _skipped = module_lines(color_module, name, profile)
        allowed.update(line["id"] for line in lines)
    color_module["strokes"] = [s for s in color_strokes if s["id"] in allowed]
    color_module["stroke_count"] = len(color_module["strokes"])

    plan = dict(format="painting-workflow-design-v1", ready_for_execution=True, subject="master",
                reference_image=str(image_path), stages=["outline", "color"] if color_module["strokes"] else ["outline"],
                modules=dict(outline=dict(format="outline-design-v1", strokes=black,
                                          stroke_count=len(black), image_size=[W, H],
                                          image_corners=None, mm_per_px=scale),
                             color=color_module),
                sequence=[dict(kind="run_module", module="outline"),
                          dict(kind="observe_previous_stage_dry", automatic_seconds=None),
                          dict(kind="manual_wash_wipe_and_load", next_stage="color"),
                          dict(kind="run_module", module="color")],
                execution_order=dict(preferred_stage_order=["black", "blue", "green"],
                                     color_lead="", note="名家笔触导出"),
                image_size=[W, H], image_corners=None, mm_per_px=scale,
                save_progress=False, source_revision=dict(kind="master-studio", title=title),
                note="由山河绘·名家笔触工坊生成；导出后仍须现场检查。")
    bundle = dict(format="arm-painting-bundle-v1", version=1, workflow=plan,
                  ink_job=dict(format="arm-painting job", version=1, meta=meta, waypoints=waypoints))
    _save(out_path, bundle)

    # 离线校验（不连接机械臂）
    from run_json_painting import load_artwork, manifest_for
    ink, p, strokes_out, m = load_artwork(out_path)
    manifest_for(strokes_out, p, m)
    return dict(path=str(out_path), black=len(black), color=len(color_module["strokes"]),
                size=[W, H], waypoints=len(waypoints), trajectory=traj)