"""Respace existing straight color fills in paintings 02–04 to 1 mm."""
import copy
import csv
from datetime import datetime
import hashlib
import io
from pathlib import Path
import socket
import sys
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scripts.first_color_edge_gap_20261012 import h, read, encode
from scripts.consolidate_selected_artworks import digest, stroke_data, index_for
from ink_control import refresh_stats
from painting_session import session_lock
from shuimo import press_to_width_mm
from web.catalog import ARTWORKS

OUT = ROOT / 'artifacts/第二至第四幅中心线间距1mm_20261012'


def state():
    with urllib.request.urlopen('http://127.0.0.1:8765/api/state', timeout=5) as r:
        return __import__('json').load(r)


def unlocked():
    s = state()
    assert not s.get('parametersLocked') and s['status'] in ('idle', 'complete', 'stopped', 'saved', 'error'), s


def rebuild(number, old):
    b = copy.deepcopy(old)
    module = b['workflow']['modules']['color']
    colors = module['strokes']
    eligible = lambda s: s.get('nominal_pitch_mm') is not None
    assert {s['nominal_pitch_mm'] for s in colors if eligible(s)} == {.9}
    scale = module['mm_per_px']
    w, height = module['image_size']
    groups = list(dict.fromkeys((s['color'], s['drawing_region'], s['path_style']) for s in colors if eligible(s)))
    replace, removed, audits = {}, set(), []
    for gi, key in enumerate(groups):
        originals = [s for s in colors if eligible(s) and (s['color'], s['drawing_region'], s['path_style']) == key]
        template = originals[0]
        assert all(len(s['points']) == 2 and set(s['press']) == {.3} for s in originals)
        horizontal = key[2] == 'left-to-right-lift-return'
        assert all(abs(s['points'][0][1 if horizontal else 0] - s['points'][1][1 if horizontal else 0]) < 1e-6 for s in originals)
        area = h.footprint(originals, (height, w), scale)
        # Use the current brush footprint as the approved envelope. Eroding by
        # the same discrete brush cap guarantees new strokes cannot expand it.
        thickness = max(1, int(np.ceil(press_to_width_mm(.3) / scale)))
        radius = int(np.ceil(thickness / 2)) + 1
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), np.uint8)
        cv2.line(kernel, (radius, radius), (radius, radius), 1, thickness)
        safe = cv2.erode(area, kernel)
        scan = safe.T.copy() if horizontal else safe
        ys, xs = np.nonzero(scan)
        assert len(xs), key
        # Each subject/color gets one lattice, also across disconnected parts.
        columns = np.arange(float(xs.min()), float(xs.max()) + 1e-6, 1. / scale)
        assert len(columns) < 2 or np.allclose(np.diff(columns) * scale, 1., atol=1e-10, rtol=0)
        paths = []
        for x in columns:
            yy = np.arange(int(ys.min()), int(ys.max()) + .2, .2)
            q = np.column_stack((np.full(len(yy), x), yy))
            for part in h.segments(q, scan):
                length = float(part[-1, 1] - part[0, 1]) * scale
                if length < .08:
                    continue
                maximum = 55 if horizontal else 20
                ends = np.linspace(part[0, 1], part[-1, 1], max(1, int(np.ceil(length / maximum))) + 1)
                for start, end in zip(ends[:-1], ends[1:]):
                    points = [[float(x), float(start)], [float(x), float(end)]]
                    if horizontal:
                        points = [p[::-1] for p in points]
                    s = copy.deepcopy(template)
                    s.update(id=f'pitch1-{number}-{gi}-{len(paths):04d}', points=points,
                             press=[.3, .3], speed=[template['speed'][0]] * 2,
                             nominal_pitch_mm=1., parent_stroke_id=template['id'])
                    s.pop('target_edge_gap_mm', None)
                    refresh_stats(s, scale)
                    paths.append(s)
        assert paths, key
        footprint = h.footprint(paths, (height, w), scale)
        assert not np.any(footprint & (1 - area)), (number, key, 'expanded footprint')
        axes = sorted(set(s['points'][0][1 if horizontal else 0] for s in paths))
        assert all(abs((z - a) * scale - round((z - a) * scale)) < 1e-8 for a, z in zip(axes[:-1], axes[1:]))
        audits.append(dict(color=key[0], region=key[1], before=len(originals), after=len(paths),
                           center_pitch_mm=1., actual_lattice_checked=True,
                           modeled_footprint_within_previous_envelope=True,
                           before_area_mm2=float(area.sum() * scale**2), after_area_mm2=float(footprint.sum() * scale**2)))
        replace[originals[0]['id']] = paths
        removed.update(s['id'] for s in originals)
    result = []
    for s in colors:
        if s['id'] in replace:
            result.extend(replace[s['id']])
        if s['id'] not in removed:
            result.append(copy.deepcopy(s))
    assert len({s['id'] for s in result}) == len(result)
    assert [s for s in result if not eligible(s)] == [s for s in colors if not eligible(s)]
    assert b['ink_job'] == old['ink_job']
    module.update(strokes=result, stroke_count=len(result))
    revision = dict(date=datetime.now().isoformat(timespec='seconds'), previous_center_pitch_mm=.9,
                    center_pitch_mm=1., pressure_mm=.3, regions=audits,
                    note='第2至4幅成片上色中心线间距从0.9mm改为1mm，与第一幅一致。实际路径已重排；下压保持0.3mm，黑线、零散叶片与水纹保留。')
    plan = b['workflow']
    plan['center_pitch_revision'] = revision
    plan['color_geometry'].update(pitch_mm=1., pitch_px=1. / scale, pressure_range_mm=[.3, .3])
    if any('canopy' in s.get('role', '') for s in result):
        plan['color_geometry']['canopy_pitch_mm'] = 1.
    hashed = digest(stroke_data(b))
    for key in ('source_revision', 'web_installation'):
        plan[key]['stroke_data_sha256'] = hashed
    return b, revision, hashed


def main():
    unlocked()
    OUT.mkdir(parents=True, exist_ok=True)
    catalog_path = ROOT / '待画作品/画稿清单.json'
    catalog = read(catalog_path)
    protected = {ROOT / a[3]: (ROOT / a[3]).read_bytes() for i, a in enumerate(ARTWORKS, 1) if i not in (2, 3, 4)}
    writes, reports = {}, []
    board = Image.new('RGB', (1600, 3 * 630), '#faf8f0')
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
    from web.site_settings import prepared_artwork
    from run_json_painting import manifest_for
    for number in (2, 3, 4):
        target = ROOT / ARTWORKS[number - 1][3]
        old = read(target)
        b, revision, hashed = rebuild(number, old)
        task, package = target.parent, target.parent.parent
        candidate = task / 'painting.pitch1-candidate.json'
        candidate.write_bytes(encode(b))
        try:
            with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline only')):
                prepared = prepared_artwork(candidate, ROOT / 'web/data/site_settings.json')
                manifest_for(prepared[2], prepared[1], prepared[3])
        finally:
            candidate.unlink(missing_ok=True)
        order = index_for(b)
        counts = {k: len(v) for k, v in order.items()}
        report = dict(number=number, counts=counts, revision=revision, stroke_data_sha256=hashed,
                      offline_check='passed', hardware_started=False, effective_from='next-start')
        reports.append(report)
        writes[target] = encode(b)
        writes[task / '中心线间距1mm校验.json'] = encode(report)
        for p in task.glob('*_计划.json'):
            writes[p] = encode(b['workflow'])
        for name in ('绘画顺序索引.json', 'Web当前应用记录.json'):
            p = task / name
            v = read(p)
            v.update(counts=counts, stroke_data_sha256=hashed, center_pitch_revision=revision)
            if name == '绘画顺序索引.json':
                v['stages'] = order
            writes[p] = encode(v)
        table = io.StringIO(newline='')
        writer = csv.writer(table)
        writer.writerow(['阶段', '阶段内笔序', '笔触ID', '完整部位', '类别', '原笔触ID', '长度mm'])
        for stage, rows in order.items():
            for row in rows:
                writer.writerow([stage, row['order'], row['id'], row['region'], row['role'], row['parent_stroke_id'], row['length_mm']])
        writes[task / '逐笔顺序索引.csv'] = table.getvalue().encode('utf-8-sig')
        row = next(row for row in catalog['artworks'] if row['folder'] == package.name)
        row.update(counts=counts, stroke_data_sha256=hashed, center_pitch_revision=revision)
        png = h.render(b)
        for name in ('web_preview.png', '执行路径预览_非实测线宽.png', '任务预览_线宽色彩为估计.png', '黑线加蓝绿_实际路径.png', '当前作画阶段预览.png', '完整色面黑轮廓.png', '中心线间距1mm预览.png'):
            writes[package / '预览与下压标注' / name] = png
        for x, bb, label in ((20, old, '修改前：0.9 mm'), (820, b, '修改后：1 mm')):
            pic = Image.open(io.BytesIO(h.render(bb)))
            pic.thumbnail((760, 545))
            y = (number - 2) * 630
            board.paste(pic, (x, y + 55))
            draw.text((x, y + 12), f'第{number}幅 · {label}', font=font, fill='#34463a')
        p = package / '阅读说明.md'
        writes[p] = (p.read_text(encoding='utf-8') + '\n\n## 上色中心线间距1mm（2026-10-12）\n\n' + revision['note'] + '\n通过离线校验，下次启动生效。\n').encode('utf-8')
        print(encode(dict(number=number, counts=counts, offline_check='passed')).decode(), flush=True)
    writes[catalog_path] = encode(catalog)
    board.save(OUT / '第二至第四幅间距调整前后.png')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = ROOT / 'archive' / (stamp + '_第二至第四幅中心线1mm前')
    originals = {p: p.read_bytes() if p.exists() else None for p in writes}
    for p, data in originals.items():
        if data is not None:
            q = backup / p.relative_to(ROOT)
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(data)
    unlocked()
    with session_lock(ROOT / 'robot_runs'):
        assert not (ROOT / 'robot_runs/blue_wash_motion.lock').exists()
        done = []
        try:
            for p, data in writes.items():
                assert (p.read_bytes() if p.exists() else None) == originals[p]
                tmp = p.with_name(p.name + '.pitch1.tmp')
                tmp.write_bytes(data)
                tmp.replace(p)
                done.append(p)
            assert all(p.read_bytes() == data for p, data in writes.items())
            assert all(p.read_bytes() == data for p, data in protected.items())
        except Exception:
            for p in reversed(done):
                if originals[p] is None:
                    p.unlink(missing_ok=True)
                else:
                    p.write_bytes(originals[p])
            raise
    (OUT / 'report.json').write_bytes(encode(dict(reports=reports, backup=str(backup), status='installed')))
    print('Installed; backup: ' + str(backup))


if __name__ == '__main__':
    main()
