# -*- coding: utf-8 -*-
"""Zoom crops plus pixel metrics for the upper-half black-ink reduction."""
import importlib.util, io, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from web.catalog import ARTWORKS
from ink_control import refresh_stats

OUT = ROOT / "artifacts/black_top_relief_20261013"
SPEC = importlib.util.spec_from_file_location("pr", ROOT / "artifacts/蓝山绿树配色_20261010/update.py")
R = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(R)
VARIANTS = [("原始", None, 1.0), ("上半减一成", "upper", .90), ("上半减一成半", "upper", .85), ("上半减两成", "upper", .80), ("全幅减一成", "all", .90)]
TAPER_MM = 3.0

def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def variant(bundle, scope, factor):
    b = json.loads(json.dumps(bundle))
    ink = b["ink_job"]["meta"]["ink_execution"]; scale = ink["mm_per_px"]
    w, height = b["ink_job"]["meta"]["image_size"]; middle = height / 2.
    for s in ink["base_strokes"]:
        p = np.asarray(s["points"], float); v = np.asarray(s["press"], float)
        if v.size == 1: v = np.full(len(p), float(v))
        if scope == "all":
            new = v * factor
        else:
            t = TAPER_MM / scale
            weight = np.clip((middle - p[:, 1]) / t + .5, 0., 1.)
            new = v * (1. - weight * (1. - factor))
        s["press"] = np.maximum(new, 0.06).tolist()
        refresh_stats(s, scale)
    return b

def ink_mask(bundle):
    """Binary map of where black ink lands, from the same model the preview uses."""
    w, h = bundle["ink_job"]["meta"]["image_size"]
    scale = bundle["ink_job"]["meta"]["ink_execution"]["mm_per_px"]
    mask = np.zeros((h, w), np.uint8)
    for s in bundle["ink_job"]["meta"]["ink_execution"]["base_strokes"]:
        cv2.polylines(mask, [np.rint(s["points"]).astype(np.int32)], False, 1,
                      max(1, round(s.get("width_mm", .3) / scale)), cv2.LINE_AA)
    return mask

summary = {}
for n, (folder, name, _s, rel) in enumerate(ARTWORKS[:5], 1):
    bundle = read(ROOT / rel)
    h = bundle["ink_job"]["meta"]["image_size"][1]; middle = h // 2
    panels, metrics = [], []
    base_mask = None
    for label, scope, factor in VARIANTS:
        b = bundle if scope is None else variant(bundle, scope, factor)
        mask = ink_mask(b)
        upper = mask[:middle]
        covered = int((upper > 127).sum())
        if base_mask is None:
            base_mask = upper
        diff = int(((upper > 127) != (base_mask > 127)).sum())
        metrics.append(dict(variant=label, upper_ink_px=covered,
                            vs_original_px=diff, pct=round(100. * diff / max(1, (base_mask > 127).sum()), 2)))
        png = R.render(b)
        im = Image.open(io.BytesIO(png)).convert("RGB")
        crop = im.crop((int(im.width * .04), int(im.height * .03), int(im.width * .62), int(im.height * .46)))
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        panels.append((crop, label))
    pw = sum(p.width for p, _ in panels) + 8 * (len(panels) + 1)
    ph = max(p.height for p, _ in panels) + 54
    board = Image.new("RGB", (pw, ph), "#f2efe6"); d = ImageDraw.Draw(board)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    x = 8
    for p, label in panels:
        board.paste(p, (x, 44)); d.text((x + 4, 10), label, font=font, fill="#22303a"); x += p.width + 8
    board.save(OUT / ("%02d_%s_上半局部放大.png" % (n, name)))
    summary["%02d_%s" % (n, name)] = metrics
    print(n, name, json.dumps(metrics, ensure_ascii=False), flush=True)
(OUT / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
