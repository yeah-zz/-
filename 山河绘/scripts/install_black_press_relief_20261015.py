# -*- coding: utf-8 -*-
"""Install the black-ink press relief for chosen paintings; keep a restorable copy."""
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
import numpy as np

from web.catalog import ARTWORKS
from scripts.black_upper_press_20261015 import (OUT, PRESS_FLOOR, SCOPE_FRACTION, SCOPE_NAMES,
                                               digest, encode, press_summary, read, relieve_delta,
                                               render, stroke_data)

REPORT = OUT / "installs.json"
RESTORE_TEMPLATE = '''# -*- coding: utf-8 -*-
"""Undo the black-line relief of %(stamp)s: restore the files from this folder."""
import hashlib, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(%(root)r)
manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
for entry in manifest["files"]:
    source = HERE / entry["backup"]
    if not source.exists():
        print("missing backup:", source)
        sys.exit(1)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256_before"]
    target = ROOT / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256_before"]
print("已还原 %%d 个文件，内容与修改前逐字节一致。" %% len(manifest["files"]))
'''


def between_stages(url="http://127.0.0.1:8765/api/state"):
    """True when the web workbench sits between stages (no arm motion in flight)."""
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            state = json.load(response)
    except Exception as error:                      # server down: nothing to collide with
        return False, "工作台未响应：%s" % error
    busy = state.get("status") in ("running",) or state.get("active")
    return (not busy), json.dumps({k: state.get(k) for k in ("status", "active", "completed", "notice")},
                                 ensure_ascii=False)


def write_files(writes, originals, allow_live):
    """Replace every file atomically, or via the lock when the session is closed."""
    from painting_session import session_lock
    if not allow_live:
        with session_lock(ROOT / "robot_runs"):
            assert not (ROOT / "robot_runs/blue_wash_motion.lock").exists(), "Painting task still active"
            replace_all(writes, originals)
        return "session_lock"
    ok, detail = between_stages()
    assert ok, "机械臂正在动作，禁止改写画稿：" + detail
    assert not (ROOT / "robot_runs/blue_wash_motion.lock").exists(), "Painting task still active"
    replace_all(writes, originals)
    return "between-stages:" + detail


def replace_all(writes, originals):
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


def install_numbers(numbers, delta_mm, scope, allow_live=False):
    from web.site_settings import prepared_artwork
    from run_json_painting import manifest_for
    from artwork_order import ink_stages
    from scripts.consolidate_selected_artworks import index_for
    from painting_session import session_lock
    fraction = SCOPE_FRACTION.get(scope, .25)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "黑线%s减%.2fmm" % (SCOPE_NAMES[scope], abs(delta_mm))
    backup = ROOT / "archive" / ("%s_%s前_第%s幅" % (stamp, tag, "".join(str(n) for n in numbers)))
    backup.mkdir(parents=True, exist_ok=False)
    catalog_path = ROOT / "待画作品/画稿清单.json"
    catalog = read(catalog_path)
    writes, reports = {}, []
    for n in numbers:
        folder, name, _sub, rel = ARTWORKS[n - 1]
        target = ROOT / rel
        old = read(target)
        assert old["workflow"].get("black_upper_press_revision") is None, "Already adjusted: " + str(target)
        bundle, info = relieve_delta(old, delta_mm, scope, fraction)
        plan = bundle["workflow"]
        task, package = target.parent, target.parent.parent
        # Only the black-ink press may change.
        assert bundle["ink_job"]["waypoints"] == old["ink_job"]["waypoints"]
        assert [s["points"] for s in stroke_data(bundle)["black"]] == [s["points"] for s in stroke_data(old)["black"]]
        assert [s["id"] for s in stroke_data(bundle)["black"]] == [s["id"] for s in stroke_data(old)["black"]]
        assert bundle["workflow"]["modules"]["color"] == old["workflow"]["modules"]["color"]
        assert bundle["workflow"]["sequence"] == old["workflow"]["sequence"]
        for s in stroke_data(bundle)["black"]:
            assert len(s["press"]) == len(s["points"])
            assert all(PRESS_FLOOR - 1e-9 <= p <= .80 + 1e-9 for p in s["press"]), s["id"]
        revision = dict(date=stamp, delta_mm=info["delta_mm"], scope=info["scope"],
                        fraction=info["fraction"], divider_px=info["divider_px"],
                        divider_mm=info["divider_mm"], taper_mm=info["taper_mm"],
                        press_floor_mm=PRESS_FLOOR, points_changed=info["points_changed"],
                        strokes_changed=info["strokes_changed"], strokes_total=info["strokes_total"],
                        strokes=info["strokes"], before=press_summary(old), after=press_summary(bundle),
                        backup=str(backup.relative_to(ROOT)),
                        unchanged="彩色上色、黑线坐标与笔序、速度、题字一律未动。",
                        note="黑墨勾勒线按画面内容高度上 1/4 区域逐点下压调低 %.2fmm（分界 %.0fmm，"
                             "±%.0fmm 过渡带）；彩色上色下压仍为 %.2fmm。"
                             "已在下限 %.2fmm 的极轻笔触点保持原值并逐笔记录。下一次启动生效。"
                             % (abs(info["delta_mm"]), info["divider_mm"], info["taper_mm"] / 2,
                                bundle["workflow"]["modules"]["color"]["motion_candidate"]["press_mm"],
                                PRESS_FLOOR))
        plan["black_upper_press_revision"] = revision
        hashed = digest(stroke_data(bundle))
        for key in ("source_revision", "web_installation"):
            if key in plan:
                plan[key]["stroke_data_sha256"] = hashed
        candidate = task / "painting.upper-press-candidate.json"
        candidate.write_bytes(encode(bundle))
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")):
                ink_ready, plan_ready, strokes_ready, meta_ready = prepared_artwork(
                    candidate, ROOT / "web/data/site_settings.json")
                manifest = manifest_for(strokes_ready, plan_ready, meta_ready)
                stages = ink_stages({"meta": meta_ready}, plan_ready)
                assert len(stages["black"]) == len(ink_stages(old["ink_job"], old["workflow"])["black"])
                assert len(manifest["stages"]["black"]) == len(stages["black"])
                press = np.concatenate([np.asarray(s["press"], float) for s in stages["black"]])
                assert float(press.min()) >= PRESS_FLOOR - 1e-9 and float(press.max()) <= .80 + 1e-9
                from run_color_experiment import module_lines
                from color_profile import ColorProfile
                effective = plan_ready["modules"]["color"]
                profile = ColorProfile(**effective["motion_candidate"]).to_dict()
                for color in ("blue", "green"):
                    lines, skipped = module_lines(effective, color, profile)
                    assert not skipped and lines, color
                assert profile["press_mm"] == old["workflow"]["modules"]["color"]["motion_candidate"]["press_mm"]
        finally:
            candidate.unlink(missing_ok=True)
        counts = {k: len(v) for k, v in index_for(bundle).items()}
        report = dict(number=n, title=name, revision=revision, counts=counts,
                      stroke_data_sha256=hashed, offline_check="passed", hardware_started=False,
                      active_run_unchanged=True, effective_from="next-start", backup=str(backup))
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
        writes[task / (tag + "校验.json")] = encode(report)
        row = next(row for row in catalog["artworks"] if row["folder"] == package.name)
        row.update(counts=counts, stroke_data_sha256=hashed, black_upper_press_revision=revision)
        png = render(bundle)
        for filename in ("web_preview.png", "执行路径预览_非实测线宽.png", "任务预览_线宽色彩为估计.png",
                         "黑线加蓝绿_实际路径.png", "当前作画阶段预览.png", tag + "预览.png"):
            writes[package / "预览与下压标注" / filename] = png
        blank = copy.deepcopy(bundle)
        blank["workflow"]["modules"]["color"]["strokes"] = []
        writes[package / "预览与下压标注" / (tag + "_纯勾线.png")] = render(blank)
        before_png = render(old)
        writes[package / "预览与下压标注" / (tag + "_修改前.png")] = before_png
        readme = package / "阅读说明.md"
        writes[readme] = (readme.read_text(encoding="utf-8") + "\n\n## %s（%s）\n\n%s\n副本：%s\n"
                          % (tag, datetime.date.today().isoformat(), revision["note"],
                             revision["backup"])).encode("utf-8")
        print(n, name.encode("unicode_escape").decode(), json.dumps(report, ensure_ascii=False), flush=True)
    writes[catalog_path] = encode(catalog)
    originals = {path: (path.read_bytes() if path.exists() else None) for path in writes}
    for path, data in originals.items():
        if data is None:
            continue
        saved = backup / path.relative_to(ROOT)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(data)
    manifest = dict(stamp=stamp, delta_mm=delta_mm, scope=scope, artworks=numbers, tag=tag,
                    files=[dict(path=str(path.relative_to(ROOT)),
                                backup=str((backup / path.relative_to(ROOT)).relative_to(backup)),
                                sha256_before=hashlib.sha256(data).hexdigest())
                           for path, data in originals.items() if data is not None])
    (backup / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (backup / "还原.py").write_text(RESTORE_TEMPLATE % dict(stamp=stamp, root=str(ROOT)), encoding="utf-8")
    (backup / "还原说明.md").write_text(
        "# 还原副本 · %s\n\n本次改动：%s（第 %s 幅）。彩色上色、黑线坐标与笔序、速度均未改动。\n\n"
        "本目录下按项目相对路径保存了 %d 个文件的原件；`manifest.json` 记录每个文件的相对路径与修改前 SHA256。\n\n"
        "## 还原方法（二选一，先停掉机械臂作业）\n\n"
        "1. 双击运行本目录的 `还原.py`（或命令行 `python 还原.py`）—— 会逐文件校验 SHA256 后覆盖回去；\n"
        "2. 或手动把本目录里的文件按相同相对路径覆盖回项目根目录。\n\n"
        "还原后所有被改文件与修改前逐字节一致。\n"
        % (stamp, tag, "".join(str(v) for v in numbers), len(manifest["files"])), encoding="utf-8")
    mode = write_files(writes, originals, allow_live)
    history = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else []
    history.append(dict(stamp=stamp, tag=tag, artworks=numbers, delta_mm=delta_mm, scope=scope,
                        backup=str(backup.relative_to(ROOT)), write_mode=mode,
                        reports=reports, status="installed"))
    REPORT.write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写入；副本：" + str(backup), flush=True)
    return backup


def main():
    parser = argparse.ArgumentParser(description="黑线减压写入指定画稿，并留可还原副本")
    parser.add_argument("--numbers", default="2", help="画稿编号，如 2 或 2,3")
    parser.add_argument("--delta-mm", type=float, default=-.10)
    parser.add_argument("--scope", choices=("quarter", "half", "all"), default="quarter")
    parser.add_argument("--allow-live-session", action="store_true",
                        help="工作台停在阶段之间时也写入（先确认没有正在进行的机械臂动作）")
    args = parser.parse_args()
    assert -.5 <= args.delta_mm < 0.
    numbers = [int(v) for v in args.numbers.replace("，", ",").split(",") if v.strip()]
    assert numbers and all(1 <= v <= 5 for v in numbers), numbers
    install_numbers(numbers, args.delta_mm, args.scope, args.allow_live_session)


if __name__ == "__main__":
    main()
