# -*- coding: utf-8 -*-
"""Zoomed band comparison saved as separate panels. Read-only."""
import io, json, sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from web.catalog import ARTWORKS

spec = importlib.util.spec_from_file_location("bp", ROOT / "scripts/black_upper_press_20261015.py")
bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)
OUT = bp.OUT / "zoom"
OUT.mkdir(parents=True, exist_ok=True)
DELTA, SCOPE = -.10, "quarter"
MAX_W = 7600

def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def cap(pic, text, path):
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    c = Image.new("RGB", (pic.width, pic.height + 44), "#f2efe6")
    c.paste(pic, (0, 44))
    ImageDraw.Draw(c).text((8, 8), text, font=font, fill="#22303a")
    if c.width > MAX_W:
        r = MAX_W / c.width
        c = c.resize((int(c.width * r), int(c.height * r)), Image.LANCZOS)
    c.save(path)
    return c.size

rows = []
for n, (folder, name, _s, rel) in enumerate(ARTWORKS[:5], 1):
    bundle = read(ROOT / rel)
    after, info = bp.relieve_delta(bundle, DELTA, SCOPE, bp.SCOPE_FRACTION[SCOPE])
    w, h = bundle["ink_job"]["meta"]["image_size"]
    pad = 26
    top = max(0, int(info["content_top_px"]) - pad)
    bottom = min(h, int(round(info["divider_px"])) + pad)
    pics, ref = {}, None
    for key, text, b in (("before", "修改前", bundle), ("after", "修改后（该带内黑线 −0.10mm）", after)):
        crop = Image.open(io.BytesIO(bp.render(b))).convert("RGB").crop((0, top, w, bottom))
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        pics[key] = crop
        cap(crop, "%02d %s · %s" % (n, name, text), OUT / ("%02d_%s_%s.png" % (n, name, key)))
        if key == "before":
            ref = np.asarray(crop).astype(int)
    d = np.abs(np.asarray(pics["after"]).astype(int) - ref).sum(2)
    m = cv2.cvtColor(((d > 24).astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)
    cap(Image.fromarray(m), "%02d %s · 差异（白=下压变轻的位置）" % (n, name), OUT / ("%02d_%s_diff.png" % (n, name)))
    rows.append(dict(number=n, name=name, band_px=[top, bottom],
                     band_mm=[round(top * bundle["ink_job"]["meta"]["ink_execution"]["mm_per_px"], 1),
                              round(bottom * bundle["ink_job"]["meta"]["ink_execution"]["mm_per_px"], 1)],
                     changed_points=info["points_changed"], changed_strokes=info["strokes_changed"],
                     total_strokes=info["strokes_total"],
                     diff_px=int((d > 24).sum())))
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
(OUT / "report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
