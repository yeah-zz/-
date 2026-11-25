# -*- coding: utf-8 -*-
"""Preview: black-ink pressure reduced in the upper half, official preview style.
Read-only: writes only under artifacts/black_top_relief_20261013."""
import json, io, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from web.catalog import ARTWORKS
from ink_control import refresh_stats

OUT = ROOT / "artifacts/black_top_relief_20261013"
VARIANTS = [("原始", None, 1.0), ("黑线上半减一成", "upper", .90),
            ("黑线上半减一成半", "upper", .85), ("黑线上半减两成", "upper", .80),
            ("黑线全幅减一成", "all", .90)]
TAPER_MM = 3.0


def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def variant(bundle, scope, factor):
    b = json.loads(json.dumps(bundle))
    ink = b["ink_job"]["meta"]["ink_execution"]; scale = ink["mm_per_px"]
    _w, height = b["ink_job"]["meta"]["image_size"]; middle = height / 2.
    for s in ink["base_strokes"]:
        p = np.asarray(s["points"], float); v = np.asarray(s["press"], float)
        if v.size == 1: v = np.full(len(p), float(v))
        if scope == "all":
            new = v * factor
        else:
            t = TAPER_MM / scale
            weight = np.clip((middle - p[:, 1]) / t + .5, 0., 1.)
            new = v * (1. - weight * (1. - factor))
        s["press"] = np.maximum(new, .06).tolist()
        refresh_stats(s, scale)
    return b


def render(bundle):
    w, h = bundle["ink_job"]["meta"]["image_size"]
    scale = bundle["ink_job"]["meta"]["ink_execution"]["mm_per_px"]
    canvas = np.full((h, w, 3), (245, 250, 253), np.uint8)
    module = bundle["workflow"]["modules"]["color"]
    for s in module["strokes"]:
        cv2.polylines(canvas, [np.rint(s["points"]).astype(np.int32)], False,
                      tuple(int(v) for v in module["palette_bgr"][s["color"]]),
                      max(1, round(s["width_mm"] / scale)), cv2.LINE_AA)
    for s in bundle["ink_job"]["meta"]["ink_execution"]["base_strokes"]:
        cv2.polylines(canvas, [np.rint(s["points"]).astype(np.int32)], False, (75, 83, 76),
                      max(1, round(s.get("width_mm", .3) / scale)), cv2.LINE_AA)
    return cv2.imencode(".png", canvas)[1].tobytes()


def ink_mask(bundle):
    w, h = bundle["ink_job"]["meta"]["image_size"]
    scale = bundle["ink_job"]["meta"]["ink_execution"]["mm_per_px"]
    mask = np.zeros((h, w), np.uint8)
    for s in bundle["ink_job"]["meta"]["ink_execution"]["base_strokes"]:
        cv2.polylines(mask, [np.rint(s["points"]).astype(np.int32)], False, 1,
                      max(1, round(s.get("width_mm", .3) / scale)), cv2.LINE_AA)
    return mask


def band_table(bundle):
    """Where the black ink sits: press mean per horizontal twentieth."""
    _w, h = bundle["ink_job"]["meta"]["image_size"]
    ink = bundle["ink_job"]["meta"]["ink_execution"]["base_strokes"]
    pts, prs = [], []
    for s in ink:
        p = np.asarray(s["points"], float); v = np.asarray(s["press"], float)
        if v.size == 1: v = np.full(len(p), float(v))
        pts.append(p); prs.append(v)
    pts = np.concatenate(pts); prs = np.concatenate(prs)
    bands = []
    for i in range(20):
        lo, hi = i * h / 20., (i + 1) * h / 20.
        m = (pts[:, 1] >= lo) & (pts[:, 1] < hi)
        bands.append(dict(band="%02d" % (i + 1), y=[round(lo), round(hi)], points=int(m.sum()),
                          press_mean=round(float(prs[m].mean()), 4) if m.any() else None))
    return bands


summary = {}
for n, (folder, name, _s, rel) in enumerate(ARTWORKS[:5], 1):
    bundle = read(ROOT / rel)
    h = bundle["ink_job"]["meta"]["image_size"][1]; middle = h // 2
    panels, metrics, base_mask = [], [], None
    for label, scope, factor in VARIANTS:
        b = bundle if scope is None else variant(bundle, scope, factor)
        mask = (ink_mask(b) > 0)
        upper = mask[:middle]
        if base_mask is None:
            base_mask = upper
        diff = int((upper != base_mask).sum())
        metrics.append(dict(variant=label, upper_ink_px=int(upper.sum()), changed_px=diff,
                            changed_pct=round(100. * diff / max(1, int(base_mask.sum())), 2)))
        im = Image.open(io.BytesIO(render(b))).convert("RGB")
        crop = im.crop((int(im.width * .03), int(im.height * .02), int(im.width * .66), int(im.height * .50)))
        panels.append((crop, label))
    pw = sum(p.width for p, _ in panels) + 8 * (len(panels) + 1)
    ph = max(p.height for p, _ in panels) + 52
    board = Image.new("RGB", (pw, ph), "#f2efe6"); d = ImageDraw.Draw(board)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
    x = 8
    for p, label in panels:
        board.paste(p, (x, 44)); d.text((x + 4, 10), label, font=font, fill="#22303a"); x += p.width + 8
    board.save(OUT / ("%02d_%s_上半黑线减压_局部.png" % (n, name)))
    summary["%02d_%s" % (n, name)] = dict(metrics=metrics, bands=band_table(bundle))
    print(n, name, json.dumps(metrics, ensure_ascii=False), flush=True)
(OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
