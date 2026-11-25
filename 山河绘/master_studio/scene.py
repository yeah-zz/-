# -*- coding: utf-8 -*-
"""画面母题：把不同构图拆成可渲染、可导出的山、树、亭、舟、水。"""

from copy import deepcopy


def M(cx, base, height, width, depth="mid", tone=0.6, color=0.5, jag=0.10, lean=0.0):
    return dict(cx=cx, base=base, height=height, width=width, depth=depth,
                tone=tone, color=color, jag=jag, lean=lean)


def T(x, base, height, kind="pine", depth="near", lean=0.0, scale=1.0):
    return dict(x=x, base=base, height=height, kind=kind, depth=depth,
                lean=lean, scale=scale)


def P(x, base, width, depth="near"):
    return dict(x=x, base=base, width=width, depth=depth)


def B(x, y, width, depth="mid"):
    return dict(x=x, y=y, width=width, depth=depth)


LAYOUTS = {
    "shanhe": dict(
        ratio=(16, 10), water=0.740, horizon=0.50, inscription="top_left",
        mountains=[
            M(0.24, 0.610, 0.12, 0.26, "far", 0.26, 0.18, 0.12),
            M(0.45, 0.600, 0.16, 0.34, "far", 0.32, 0.24, 0.14),
            M(0.70, 0.605, 0.13, 0.30, "far", 0.28, 0.20, 0.12),
            M(0.90, 0.620, 0.10, 0.22, "far", 0.24, 0.15, 0.10),
            M(0.55, 0.800, 0.50, 0.46, "mid", 0.78, 0.66, 0.16),
            M(0.31, 0.740, 0.33, 0.28, "mid", 0.64, 0.50, 0.14),
            M(0.845, 0.700, 0.37, 0.255, "mid", 0.72, 0.58, 0.16),
            M(0.120, 1.040, 0.48, 0.36, "near", 0.84, 0.55, 0.18),
            M(0.885, 0.800, 0.15, 0.30, "near", 0.58, 0.34, 0.12),
        ],
        trees=[T(0.175, 0.760, 0.380, "pine", "near", -0.06),
               T(0.100, 0.850, 0.260, "broadleaf", "near", 0.05),
               T(0.245, 0.820, 0.205, "pine", "near", 0.10)],
        pavilion=P(0.825, 0.705, 0.100),
        boat=B(0.545, 0.865, 0.100),
        waterfall=None,
        colors=dict(blue=[(0.55, 0.34), (0.845, 0.28)], green=[(0.55, 0.16), (0.175, 0.20)]),
    ),
    "fuchun": dict(
        ratio=(16, 9), water=0.740, horizon=0.47, inscription="top_left",
        mountains=[
            M(0.16, 0.600, 0.10, 0.26, "far", 0.26, 0.18),
            M(0.40, 0.590, 0.12, 0.32, "far", 0.30, 0.22),
            M(0.65, 0.605, 0.09, 0.28, "far", 0.26, 0.17),
            M(0.88, 0.610, 0.08, 0.22, "far", 0.22, 0.14),
            M(0.23, 0.735, 0.32, 0.30, "mid", 0.62, 0.44),
            M(0.50, 0.800, 0.52, 0.42, "mid", 0.76, 0.60),
            M(0.79, 0.710, 0.27, 0.30, "mid", 0.56, 0.38),
            M(0.035, 1.050, 0.32, 0.34, "near", 0.80, 0.52),
            M(0.975, 0.870, 0.16, 0.26, "near", 0.54, 0.26),
        ],
        trees=[T(0.100, 0.770, 0.390, "pine", "near", -0.08),
               T(0.175, 0.840, 0.300, "broadleaf", "near", 0.08),
               T(0.030, 0.910, 0.235, "pine", "near", 0.05),
               T(0.915, 0.790, 0.220, "broadleaf", "near", -0.05)],
        pavilion=P(0.760, 0.710, 0.065, "mid"),
        boat=B(0.500, 0.860, 0.078),
        waterfall=None,
        colors=dict(blue=[(0.50, 0.34), (0.79, 0.26)], green=[(0.50, 0.18), (0.23, 0.20)]),
    ),
    "yijiang": dict(
        ratio=(16, 10), water=0.600, horizon=0.52, inscription="top_right",
        mountains=[
            M(0.755, 0.545, 0.13, 0.26, "far", 0.26, 0.18, 0.10),
            M(0.900, 0.535, 0.11, 0.24, "far", 0.24, 0.15, 0.09),
            M(0.085, 0.920, 0.29, 0.235, "near", 0.72, 0.36, 0.15),
        ],
        trees=[T(0.075, 0.700, 0.235, "sparse", "near", 0.10),
               T(0.130, 0.725, 0.210, "sparse", "near", -0.04),
               T(0.190, 0.760, 0.180, "sparse", "near", 0.06)],
        pavilion=P(0.235, 0.735, 0.060, "near"),
        boat=B(0.545, 0.815, 0.050, "far"),
        waterfall=None,
        colors=dict(blue=[], green=[]),
    ),
    "qifeng": dict(
        ratio=(4, 5), water=0.790, horizon=0.50, inscription="left",
        mountains=[
            M(0.12, 0.590, 0.20, 0.20, "far", 0.26, 0.20),
            M(0.88, 0.585, 0.19, 0.19, "far", 0.25, 0.18),
            M(0.500, 0.780, 0.62, 0.52, "mid", 0.82, 0.62, 0.18),
            M(0.245, 0.700, 0.30, 0.27, "mid", 0.60, 0.42),
            M(0.765, 0.690, 0.29, 0.26, "mid", 0.62, 0.46),
            M(0.155, 0.990, 0.23, 0.32, "near", 0.80, 0.48),
            M(0.845, 0.985, 0.21, 0.30, "near", 0.76, 0.44),
        ],
        trees=[T(0.175, 0.810, 0.245, "pine", "near", 0.10),
               T(0.835, 0.800, 0.215, "pine", "near", -0.08)],
        pavilion=P(0.340, 0.850, 0.075, "near"),
        boat=None,
        waterfall=dict(x=0.515, top=0.190, bottom=0.640, width=0.040),
        colors=dict(blue=[(0.50, 0.45)], green=[(0.25, 0.25), (0.76, 0.24)]),
    ),
    "jiyu": dict(
        ratio=(4, 5), water=0.825, horizon=0.45, inscription="top_right",
        mountains=[
            M(0.18, 0.555, 0.24, 0.34, "far", 0.30, 0.18),
            M(0.50, 0.530, 0.30, 0.40, "far", 0.34, 0.22),
            M(0.82, 0.560, 0.23, 0.33, "far", 0.30, 0.18),
            M(0.13, 0.770, 0.36, 0.34, "mid", 0.72, 0.22),
            M(0.38, 0.745, 0.45, 0.40, "mid", 0.82, 0.30),
            M(0.64, 0.760, 0.48, 0.42, "mid", 0.84, 0.30),
            M(0.88, 0.790, 0.34, 0.32, "mid", 0.70, 0.20),
            M(0.24, 1.010, 0.24, 0.42, "near", 0.86, 0.16),
            M(0.72, 1.015, 0.28, 0.46, "near", 0.88, 0.18),
        ],
        trees=[T(0.105, 0.835, 0.250, "broadleaf", "near", 0.04),
               T(0.905, 0.850, 0.225, "broadleaf", "near", -0.06)],
        pavilion=P(0.500, 0.845, 0.050, "mid"),
        boat=None,
        waterfall=dict(x=0.585, top=0.360, bottom=0.710, width=0.018, faint=0.55),
        colors=dict(blue=[], green=[]),
    ),
    "canshan": dict(
        ratio=(16, 10), water=0.665, horizon=0.55, inscription="left",
        mountains=[
            M(0.780, 0.590, 0.10, 0.24, "far", 0.24, 0.12, 0.08),
            M(0.150, 0.950, 0.245, 0.275, "near", 0.78, 0.28, 0.16),
        ],
        trees=[T(0.145, 0.705, 0.255, "sparse", "near", 0.12),
               T(0.215, 0.745, 0.205, "sparse", "near", -0.10)],
        pavilion=P(0.265, 0.715, 0.060, "near"),
        boat=B(0.610, 0.820, 0.042, "far"),
        waterfall=None,
        colors=dict(blue=[], green=[]),
    ),
}


VARIANT_NAMES = {
    0: "平远",
    1: "高远",
    2: "秋霁",
    3: "云起",
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _mirror(lay):
    for m in lay["mountains"]:
        m["cx"] = 1.0 - m["cx"]
        m["lean"] = -m.get("lean", 0)
    for t in lay["trees"]:
        t["x"] = 1.0 - t["x"]
        t["lean"] = -t.get("lean", 0)
    for key in ("pavilion", "boat", "waterfall"):
        if lay.get(key):
            lay[key]["x"] = 1.0 - lay[key]["x"]
    if lay["inscription"] == "top_left":
        lay["inscription"] = "top_right"
    elif lay["inscription"] == "top_right":
        lay["inscription"] = "top_left"


def _variant_mutate(lay, variant):
    """让四幅预览是四张不同的画，而不是同一张图换种子。"""
    import random
    rng = random.Random((lay.get("scene_id") or "scene") + str(variant) + "v7")
    mids = [m for m in lay["mountains"] if m["depth"] != "far"]
    main = max(mids, key=lambda m: m["height"]) if mids else None
    if variant % 4 == 1 and main:
        # 高远：主峰拔高、收窄；添一座侧峰；树换到右岸
        main["height"] *= 1.18
        main["width"] *= 0.90
        main["cx"] = _clamp(main["cx"] + rng.choice([-1, 1]) * 0.06, 0.10, 0.90)
        main["base"] = _clamp(main["base"] + 0.015, 0.30, 1.10)
        lay["mountains"].append(M(_clamp(1.0 - main["cx"] + 0.16, 0.10, 0.90),
                                  main["base"] - 0.02, main["height"] * 0.52,
                                  main["width"] * 0.55, "mid", 0.60, 0.46, 0.14))
        if lay["trees"]:
            lay["trees"][0]["kind"] = "broadleaf"
            lay["trees"][0]["x"] = 0.82
            lay["trees"][0]["height"] *= 1.08
        lay["water"] = _clamp(lay["water"] + 0.03, 0.45, 0.84)
        if lay.get("boat"):
            lay["boat"]["x"] = 0.62
        lay["inscription"] = "top_right" if main["cx"] < 0.5 else "top_left"
    elif variant % 4 == 2 and main:
        # 平远：山压低、江面变宽；删一座中景，添远山
        main["height"] *= 0.82
        main["width"] *= 1.15
        main["base"] = _clamp(main["base"] + 0.02, 0.30, 1.10)
        lay["water"] = _clamp(lay["water"] + 0.055, 0.45, 0.86)
        if len(mids) > 2:
            smallest = min(mids, key=lambda m: m["height"])
            lay["mountains"].remove(smallest)
        lay["mountains"].append(M(0.12, lay["water"] + 0.01, 0.10, 0.22, "far", 0.22, 0.14))
        lay["mountains"].append(M(0.90, lay["water"] + 0.01, 0.085, 0.18, "far", 0.20, 0.12))
        if lay["trees"]:
            lay["trees"][0]["x"] = 0.12
            lay["trees"][0]["kind"] = "mixed"
        lay["inscription"] = "top_left"
    elif variant % 4 == 3:
        # 云起：左右互换，树法/点景换位，水口与孤舟重排
        _mirror(lay)
        for i, t in enumerate(lay["trees"]):
            t["kind"] = "broadleaf" if i % 2 == 0 else "sparse"
            t["height"] *= 0.92
        if lay.get("waterfall"):
            lay["waterfall"]["x"] = _clamp(0.5 + (lay["waterfall"]["x"] - 0.5) * 0.6, 0.25, 0.75)
            lay["waterfall"]["width"] *= 1.15
        else:
            lay["waterfall"] = dict(x=0.58, top=0.20, bottom=0.62, width=0.035, faint=0.55)
        if lay.get("boat"):
            lay["boat"]["x"] = 0.42
        if lay.get("pavilion"):
            lay["pavilion"]["x"] = _clamp(lay["pavilion"]["x"] + 0.05, 0.10, 0.90)
        lay["inscription"] = "left" if lay["inscription"] in ("top_left", "left") else "right"
    # 所有变体都只做很小的整体位移，不再靠位移制造差异
    shift = rng.uniform(-0.008, 0.008)
    for m in lay["mountains"]:
        m["cx"] = _clamp(m["cx"] + shift, 0.04, 0.96)
    for t in lay["trees"]:
        t["x"] = _clamp(t["x"] + shift * 0.8, 0.02, 0.98)
    lay["variant_name"] = VARIANT_NAMES.get(variant % 4, "云起")
    return lay


def layout_for(scene_id, variant=0):
    if scene_id not in LAYOUTS:
        raise ValueError("未知构图。")
    lay = deepcopy(LAYOUTS[scene_id])
    lay["scene_id"] = scene_id
    lay["variant"] = variant
    return _variant_mutate(lay, variant)
