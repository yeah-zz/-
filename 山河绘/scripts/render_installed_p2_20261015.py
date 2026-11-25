# -*- coding: utf-8 -*-
"""Render the as-installed preview of painting 02 and report the numbers."""
import importlib.util, io, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from web.catalog import ARTWORKS

spec = importlib.util.spec_from_file_location("bp", ROOT / "scripts/black_upper_press_20261015.py")
bp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bp)

def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))

folder, name, _sub, rel = ARTWORKS[1]
bundle = read(ROOT / rel)
rev = bundle["workflow"]["black_upper_press_revision"]
backup = ROOT / rev["backup"]
old = read(backup / "待画作品/02_松岸曲水_竖题山河绘_绘字修复版/任务资料/painting.json")
out = ROOT / "artifacts/black_upper_press_20261015/已写入_第2幅"
out.mkdir(parents=True, exist_ok=True)

w, h = bundle["ink_job"]["meta"]["image_size"]
scale = bundle["ink_job"]["meta"]["ink_execution"]["mm_per_px"]
ys = []
for group in (bundle["ink_job"]["meta"]["ink_execution"]["base_strokes"],
              bundle["workflow"]["modules"]["color"]["strokes"]):
    for s in group:
        ys.append(np.asarray(s["points"], float)[:, 1])
y = np.concatenate(ys)
pad = 26
top = max(0, int(y.min()) - pad)
bottom = min(h, int(round(rev["divider_px"])) + pad)
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)

def cap(pic, text, path):
    c = Image.new("RGB", (pic.width, pic.height + 44), "#f2efe6")
    c.paste(pic, (0, 44))
    ImageDraw.Draw(c).text((8, 8), text, font=font, fill="#22303a")
    if c.width > 7600:
        r = 7600 / c.width
        c = c.resize((int(c.width * r), int(c.height * r)), Image.LANCZOS)
    c.save(path)

pics = {}
for key, text, b in (("before", "修改前（从副本读取）", old), ("after", "已写入画稿（该带内黑线 −0.10mm）", bundle)):
    crop = Image.open(io.BytesIO(bp.render(b))).convert("RGB").crop((0, top, w, bottom))
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    pics[key] = crop
    cap(crop, "02 %s · %s" % (name, text), out / ("02_%s.png" % key))
ref = np.asarray(pics["before"]).astype(int)
d = np.abs(np.asarray(pics["after"]).astype(int) - ref).sum(2)
diff = Image.fromarray(cv2.cvtColor(((d > 24).astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB))
cap(diff, "02 %s · 差异（白=下压变轻的位置）" % name, out / "02_diff.png")
bp.board([(pics["before"], "修改前"), (pics["after"], "已写入")], out / "02_全幅对照.png")
print(json.dumps(dict(name=name, band_px=[top, bottom], band_mm=[round(top * scale, 1), round(bottom * scale, 1)],
                      changed_points=rev["points_changed"], changed_strokes=rev["strokes_changed"],
                      before=rev["before"], after=rev["after"], diff_px=int((d > 24).sum()),
                      backup=rev["backup"]), ensure_ascii=False, indent=1))
