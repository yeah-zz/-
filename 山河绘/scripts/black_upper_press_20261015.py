# -*- coding: utf-8 -*-
"""Relieve the black-ink press in each painting's upper half. Offline.

Preview: writes only under artifacts/. Install: backs up, then replaces the
painting bundles, their companions, the catalog digests and the previews.
"""
import argparse
import contextlib
import copy
import csv
import datetime
import hashlib
import io
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from web.catalog import ARTWORKS
from ink_control import refresh_stats
from shuimo import press_to_width_mm

OUT = ROOT / "artifacts/black_upper_press_20261015"
TAPER_MM = 6.0          # each side of the divider: no abrupt step between halves
PRESS_FLOOR = 0.06      # ShuimoConfig.PRESS_MIN_MM


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def stroke_data(bundle):
    return {"black": bundle["ink_job"]["meta"]["ink_execution"]["base_strokes"],
            "color": bundle["workflow"]["modules"]["color"]["strokes"]}


def content_span(bundle):
    """Top and bottom of the painted area (the paper margins stay blank)."""
    ys = []
    for group in (bundle["ink_job"]["meta"]["ink_execution"]["base_strokes"],
                  bundle["workflow"]["modules"]["color"]["strokes"]):
        for s in group:
            p = np.asarray(s["points"], float)
            ys.append(p[:, 1])
    y = np.concatenate(ys)
    return float(y.min()), float(y.max())


def content_divider(bundle, fraction=.5):
    """Boundary of the top `fraction` of the painted area."""
    y0, y1 = content_span(bundle)
    return y0 + (y1 - y0) * float(fraction)


def weight(y, divider, taper_px):
    """1 well above the divider, 0 well below, linear in between."""
    return np.clip((divider - y) / taper_px + .5, 0., 1.)


def relieve(bundle, factor):
    """Reduce press (and the modelled width) of black ink in the upper half."""
    b = json.loads(json.dumps(bundle))
    ink = b["ink_job"]["meta"]["ink_execution"]
    scale = ink["mm_per_px"]
    divider = content_divider(b)
    taper_px = TAPER_MM / scale
    points = 0
    changed = []
    for s in ink["base_strokes"]:
        p = np.asarray(s["points"], float)
        v = np.asarray(s["press"], float)
        if v.size == 1:
            v = np.full(len(p), float(v))
        w = weight(p[:, 1], divider, taper_px)
        new = np.maximum(v * (1. - w * (1. - factor)), PRESS_FLOOR)
        points += int(np.count_nonzero(np.abs(new - v) > 1e-12))
        if not np.allclose(new, v):
            changed.append(s)
        s["press"] = new.tolist()
        refresh_stats(s, scale)
    return b, dict(divider_px=round(divider, 2), divider_mm=round(divider * scale, 1),
                   taper_mm=TAPER_MM, factor=factor, points_changed=points,
                   strokes_changed=len(changed))


def relieve_delta(bundle, delta_mm, scope="quarter", fraction=.25):
    """Take delta_mm off the black-ink press inside the top `fraction` of the
    painted area (scope 'half'/'quarter'/'all').

    The subtraction is exact wherever the result stays at or above the
    calibrated lightest press (0.06 mm); points whose press is already that
    light cannot go lower, so they are held at the floor and counted.
    """
    b = json.loads(json.dumps(bundle))
    ink = b["ink_job"]["meta"]["ink_execution"]
    scale = ink["mm_per_px"]
    divider = content_divider(b, .5 if scope == "half" else fraction)
    taper_px = TAPER_MM / scale
    points = 0
    changed = []
    per_stroke = []
    for s in ink["base_strokes"]:
        p = np.asarray(s["points"], float)
        v = np.asarray(s["press"], float)
        if v.size == 1:
            v = np.full(len(p), float(v))
        w = np.ones(len(p)) if scope == "all" else weight(p[:, 1], divider, taper_px)
        new = np.clip(v + delta_mm * w, PRESS_FLOOR, .80)
        n = int(np.count_nonzero(np.abs(new - v) > 1e-12))
        points += n
        if n:
            changed.append(s)
            per_stroke.append(dict(id=s.get("id"), role=s.get("role"), region=s.get("drawing_region"),
                                   points=n, press_before=round(float(v.mean()), 4),
                                   press_after=round(float(new.mean()), 4),
                                   held_at_floor=int(np.count_nonzero(new <= PRESS_FLOOR + 1e-12))))
        s["press"] = new.tolist()
        refresh_stats(s, scale)
    y0, y1 = content_span(b)
    return b, dict(delta_mm=delta_mm, scope=scope, fraction=fraction,
                   content_top_px=round(y0, 2), content_bottom_px=round(y1, 2),
                   divider_px=round(divider, 2), divider_mm=round(divider * scale, 1),
                   taper_mm=TAPER_MM, points_changed=points, strokes_changed=len(changed),
                   strokes_total=len(ink["base_strokes"]), strokes=per_stroke)


def render(bundle, black_in=6, black_out=2):
    """Colour fills under a two-layer black line, so a thinner line shows."""
    w, h = bundle["ink_job"]["meta"]["image_size"]
    scale = bundle["ink_job"]["meta"]["ink_execution"]["mm_per_px"]
    canvas = np.full((h, w, 3), (245, 250, 253), np.uint8)
    module = bundle["workflow"]["modules"]["color"]
    for s in module["strokes"]:
        cv2.polylines(canvas, [np.rint(s["points"]).astype(np.int32)], False,
                      tuple(int(v) for v in module["palette_bgr"][s["color"]]),
                      max(1, round(s["width_mm"] / scale)), cv2.LINE_AA)
    for s in bundle["ink_job"]["meta"]["ink_execution"]["base_strokes"]:
        thick = max(1, round(s.get("width_mm", .3) / scale))
        cv2.polylines(canvas, [np.rint(s["points"]).astype(np.int32)], False, black_out,
                      thick + 2, cv2.LINE_AA)
        cv2.polylines(canvas, [np.rint(s["points"]).astype(np.int32)], False, black_in,
                      thick, cv2.LINE_AA)
    return cv2.imencode(".png", canvas)[1].tobytes()


def press_summary(bundle):
    ink = bundle["ink_job"]["meta"]["ink_execution"]["base_strokes"]
    v = np.concatenate([np.asarray(s["press"], float) for s in ink])
    return dict(points=int(v.size), press_min=round(float(v.min()), 4),
                press_max=round(float(v.max()), 4), press_mean=round(float(v.mean()), 4),
                width_mean_mm=round(float(np.mean([press_to_width_mm(p) for p in v])), 3))


def board(panels, path, font_size=24):
    pw = sum(p.width for p, _ in panels) + 8 * (len(panels) + 1)
    ph = max(p.height for p, _ in panels) + 52
    canvas = Image.new("RGB", (pw, ph), "#f2efe6")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
    x = 8
    for pic, label in panels:
        canvas.paste(pic, (x, 44))
        draw.text((x + 4, 10), label, font=font, fill="#22303a")
        x += pic.width + 8
    canvas.save(path)


SCOPE_NAMES = {"quarter": "上四分之一", "half": "上半", "all": "全幅"}
SCOPE_FRACTION = {"quarter": .25, "half": .5}


def preview(delta_mm, scope="quarter"):
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    label = "黑线%s减%.2fmm" % (SCOPE_NAMES[scope], abs(delta_mm))
    for n, (folder, name, _sub, rel) in enumerate(ARTWORKS[:5], 1):
        bundle = read(ROOT / rel)
        after, info = relieve_delta(bundle, delta_mm, scope, SCOPE_FRACTION.get(scope, .25))
        w, h = bundle["ink_job"]["meta"]["image_size"]
        top = min(h, int(round(info["divider_px"])) + int(h * .08)) if scope != "all" else h
        panels, rows = [], []
        for cap, b in (("修改前", bundle), ("修改后", after)):
            pic = Image.open(io.BytesIO(render(b))).convert("RGB")
            crop = pic.crop((0, 0, w, top))
            scale_to = 1250 / crop.width
            crop = crop.resize((int(crop.width * scale_to), int(crop.height * scale_to)), Image.LANCZOS)
            panels.append((crop, cap))
            rows.append(press_summary(b))
        board(panels, OUT / ("%02d_%s_%s.png" % (n, name, label)))
        report["%02d_%s" % (n, name)] = dict(info=info, before=rows[0], after=rows[1])
        print(n, name.encode("unicode_escape").decode(),
              json.dumps(dict(info={k: v for k, v in info.items() if k != "strokes"},
                              before=rows[0], after=rows[1]), ensure_ascii=False), flush=True)
    (OUT / ("preview_%s.json" % label)).write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="黑线勾勒下压量调节（仅离线；默认只出预览）")
    parser.add_argument("--delta-mm", type=float, default=-.10, help="区域黑线下压增量，-0.10 即调低 0.1mm")
    parser.add_argument("--scope", choices=("quarter", "half", "all"), default="quarter",
                        help="作用区域：画面高度上四分之一 / 上半 / 全幅")
    parser.add_argument("--install", action="store_true", help="校验通过后写入五幅画稿")
    args = parser.parse_args()
    assert -.5 <= args.delta_mm < 0., "delta_mm must be negative"
    if not args.install:
        preview(args.delta_mm, args.scope)
        print("预览完成，未修改任何画稿。", flush=True)
        return
    install(args.delta_mm, args.scope)


def install(delta_mm, scope="quarter"):
    from web.site_settings import prepared_artwork
    from run_json_painting import manifest_for
    from artwork_order import ink_stages
    from scripts.consolidate_selected_artworks import index_for
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "黑线%s减%.2fmm" % (SCOPE_NAMES[scope], abs(delta_mm))
    catalog_path = ROOT / "待画作品/画稿清单.json"
    catalog = read(catalog_path)
    writes, originals, reports = {}, {}, []
    canvases = {}
    for n, (folder, name, _sub, rel) in enumerate(ARTWORKS[:5], 1):
        target = ROOT / rel
        old = read(target)
        bundle, info = relieve_delta(old, delta_mm, scope, SCOPE_FRACTION.get(scope, .25))
        plan = bundle["workflow"]
        baseline = target.read_bytes()
        assert plan.get("black_upper_press_revision") is None, "Already adjusted: " + str(target)
        task, package = target.parent, target.parent.parent
        # Execution invariants must survive a pressure-only edit.
        assert bundle["ink_job"]["waypoints"] == old["ink_job"]["waypoints"]
        assert [s["points"] for s in stroke_data(bundle)["black"]] == [s["points"] for s in stroke_data(old)["black"]]
        assert [s["id"] for s in stroke_data(bundle)["black"]] == [s["id"] for s in stroke_data(old)["black"]]
        assert bundle["workflow"]["modules"]["color"] == old["workflow"]["modules"]["color"]
        for s in stroke_data(bundle)["black"]:
            assert len(s["press"]) == len(s["points"])
            assert all(PRESS_FLOOR - 1e-9 <= p <= .80 + 1e-9 for p in s["press"]), s["id"]
        revision = dict(date=stamp, source_sha256=hashlib.sha256(baseline).hexdigest(),
                        delta_mm=info["delta_mm"], scope=info["scope"],
                        divider_px=info["divider_px"], divider_mm=info["divider_mm"],
                        taper_mm=info["taper_mm"], press_floor_mm=PRESS_FLOOR,
                        points_changed=info["points_changed"], strokes_changed=info["strokes_changed"],
                        strokes_total=info["strokes_total"], strokes=info["strokes"],
                        before=press_summary(old), after=press_summary(bundle),
                        unchanged="彩色上色、黑线路径与坐标、笔序、速度、题字一律未动。",
                        note="黑墨勾勒线%s逐点下压调低 %.2fmm；彩色上色下压仍为 %.2fmm。"
                             "已在下限 %.2fmm 的极轻笔触点保持原值并逐笔记录。下一次启动生效。"
                             % ("（%s，画面内容 %.0f~%.0fmm，分界 %.0fmm 处留 %.0fmm 过渡带）"
                                % (SCOPE_NAMES[scope], info["content_top_px"] * bundle["ink_job"]["meta"]["ink_execution"]["mm_per_px"],
                                   info["content_bottom_px"] * bundle["ink_job"]["meta"]["ink_execution"]["mm_per_px"],
                                   info["divider_mm"], info["taper_mm"]) if scope != "all" else "（全幅统一）",
                                abs(info["delta_mm"]),
                                bundle["workflow"]["modules"]["color"]["motion_candidate"]["press_mm"],
                                PRESS_FLOOR))
        plan["black_upper_press_revision"] = revision
        hashed = digest(stroke_data(bundle))
        for key in ("source_revision", "web_installation"):
            if key in plan:
                plan[key]["stroke_data_sha256"] = hashed
        # Offline validation through the exact web preparation path.
        candidate = task / "painting.upper-press-candidate.json"
        candidate.write_bytes(encode(bundle))
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")):
                prepared = prepared_artwork(candidate, ROOT / "web/data/site_settings.json")
                ink_ready, plan_ready, strokes_ready, meta_ready = prepared
                manifest = manifest_for(strokes_ready, plan_ready, meta_ready)
                stages = ink_stages({"meta": meta_ready}, plan_ready)
                assert len(stages["black"]) == len(ink_stages(old["ink_job"], old["workflow"])["black"])
                assert len(manifest["stages"]["black"]) == len(stages["black"])
                press = np.concatenate([np.asarray(s["press"], float) for s in stages["black"]])
                assert float(press.min()) >= PRESS_FLOOR - 1e-9 and float(press.max()) <= .80 + 1e-9
        finally:
            candidate.unlink(missing_ok=True)
        counts = {k: len(v) for k, v in index_for(bundle).items()}
        report = dict(number=n, title=name, revision=revision, counts=counts,
                      stroke_data_sha256=hashed, offline_check="passed",
                      hardware_started=False, effective_from="next-start")
        reports.append(report)
        writes[target] = encode(bundle)
        if (task / "black_only.json").exists():
            writes[task / "black_only.json"] = encode(bundle["ink_job"])
        for companion in task.glob("*_黑线任务.json"):
            writes[companion] = encode(bundle["ink_job"])
        for companion in task.glob("*_计划.json"):
            writes[companion] = encode(plan)
        index = index_for(bundle)
        order_path = task / "绘画顺序索引.json"
        if order_path.exists():
            value = read(order_path)
            value.update(counts=counts, stroke_data_sha256=hashed, black_upper_press_revision=revision)
            value["stages"] = index
            writes[order_path] = encode(value)
        table = io.StringIO(newline="")
        writer = csv.writer(table)
        writer.writerow(["阶段", "阶段内笔序", "笔触ID", "完整部位", "类别", "原笔触ID", "长度mm"])
        for stage, rows in index.items():
            for row in rows:
                writer.writerow([stage, row["order"], row["id"], row["region"], row["role"],
                                 row["parent_stroke_id"], row["length_mm"]])
        writes[task / "逐笔顺序索引.csv"] = table.getvalue().encode("utf-8-sig")
        applied = task / "Web当前应用记录.json"
        if applied.exists():
            value = read(applied)
            value.update(counts=counts, stroke_data_sha256=hashed, black_upper_press_revision=revision)
            value["black_upper_press_validation"] = report
            writes[applied] = encode(value)
        writes[task / (label + "校验.json")] = encode(report)
        row = next(row for row in catalog["artworks"] if row["folder"] == package.name)
        row.update(counts=counts, stroke_data_sha256=hashed, black_upper_press_revision=revision)
        png = render(bundle)
        for name_ in ("web_preview.png", "执行路径预览_非实测线宽.png", "任务预览_线宽色彩为估计.png",
                      "黑线加蓝绿_实际路径.png", "当前作画阶段预览.png", label + "预览.png"):
            writes[package / "预览与下压标注" / name_] = png
        blank = copy.deepcopy(bundle)
        blank["workflow"]["modules"]["color"]["strokes"] = []
        writes[package / "预览与下压标注" / (label + "_纯勾线.png")] = render(blank)
        readme = package / "阅读说明.md"
        writes[readme] = (readme.read_text(encoding="utf-8") + "\n\n## %s（%s）\n\n%s\n"
                          % (label, datetime.date.today().isoformat(), revision["note"])).encode("utf-8")
        canvases[n] = (old, bundle, revision)
        print(n, name.encode("unicode_escape").decode(), json.dumps(report, ensure_ascii=False), flush=True)
    writes[catalog_path] = encode(catalog)
    backup = ROOT / "archive" / (stamp + "_上半幅黑线减压前")
    originals = {p: (p.read_bytes() if p.exists() else None) for p in writes}
    for path, data in originals.items():
        if data is not None:
            q = backup / path.relative_to(ROOT)
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(data)
    from painting_session import session_lock
    with session_lock(ROOT / "robot_runs"):
        assert not (ROOT / "robot_runs/blue_wash_motion.lock").exists(), "Painting task still active"
        done = []
        try:
            for path, payload in writes.items():
                assert (path.read_bytes() if path.exists() else None) == originals[path], "Material changed: " + str(path)
                tmp = path.with_name(path.name + ".upper-press.tmp")
                tmp.write_bytes(payload)
                tmp.replace(path)
                done.append(path)
            assert all(path.read_bytes() == payload for path, payload in writes.items())
        except Exception:
            for path in reversed(done):
                if originals[path] is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(originals[path])
            raise
    (OUT / "install_report.json").write_text(
        json.dumps(dict(factor=factor, backup=str(backup), date=stamp, reports=reports,
                        status="installed"), ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写入；备份：" + str(backup), flush=True)


if __name__ == "__main__":
    main()
