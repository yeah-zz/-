"""Align paintings 02–05 with painting 01; never move hardware."""
import contextlib
import copy
import csv
from datetime import datetime
import hashlib
import io
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scripts.first_color_edge_gap_20261012 import h, read, encode
from scripts.set_paintings_2_to_4_center_pitch1_20261012 import rebuild, unlocked
from scripts.consolidate_selected_artworks import digest, stroke_data, index_for
from ink_control import refresh_stats
from painting_session import session_lock
from shuimo import press_to_width_mm

OUT = ROOT / 'artifacts/其余四幅对齐第一幅_20261013'


def remove_accents(b):
    module = b['workflow']['modules']['color']
    strokes = module['strokes']
    w, height = module['image_size']
    shape = (height, w)
    scale = module['mm_per_px']
    # Rock-face accents only: trees, grassy slopes and terrace tops keep their
    # established green subjects even if the wider brush meets nearby blue.
    green = [s for s in strokes if s['color'] == 'green' and s.get('role') == 'terrain_wash'
             and not any(v in s.get('drawing_region', '') for v in ('树', '草', '田', '松', '柳'))]
    blue = h.footprint([s for s in strokes if s['color'] == 'blue'], shape, scale)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(h.footprint(green, shape, scale))
    selected, components = [], []
    for i in range(1, count):
        x, y, ww, hh, area = stats[i]
        area_mm2 = float(area * scale**2)
        overlap = float(np.count_nonzero((labels == i) & (blue > 0)) / area)
        if area_mm2 <= 80 and overlap >= .85:
            selected.append(i)
            components.append(dict(component=i, box=[int(x), int(y), int(ww), int(hh)],
                                   area_mm2=area_mm2, existing_blue_coverage=overlap))
    removed = []
    for s in green:
        q = np.rint(s['points']).astype(int)
        if np.any(np.isin(labels[q[:, 1], q[:, 0]], selected)):
            removed.append(s['id'])
    module['strokes'] = [s for s in strokes if s['id'] not in removed]
    module['stroke_count'] = len(module['strokes'])
    assert [s for s in module['strokes'] if s['color'] == 'blue'] == [s for s in strokes if s['color'] == 'blue']
    return dict(removed_ids=removed, removed_green_strokes=len(removed), components=components,
                selection_rule=dict(max_green_component_area_mm2=80, min_existing_blue_coverage=.85),
                trees_grass_terraces_preserved=True, large_green_slopes_preserved=True)


def fifth_pitch(b):
    original = copy.deepcopy(b)
    design = copy.deepcopy(b)
    # Reconstruct the current designed envelope at the pressure used when the
    # faces were created; a wider trial brush must not enlarge that envelope.
    for s in design['workflow']['modules']['color']['strokes']:
        if s.get('nominal_pitch_mm') is not None:
            s['press'] = [.3] * len(s['points'])
            refresh_stats(s, design['workflow']['modules']['color']['mm_per_px'])
    result, revision, _ = rebuild(5, design)
    module = result['workflow']['modules']['color']
    width = press_to_width_mm(.85)
    for s in module['strokes']:
        s['press'] = [.85] * len(s['points'])
        for key in ('pressure_mm', 'press_mean', 'press_max'):
            if key in s:
                s[key] = .85
        for key in ('width_mm', 'brush_width_mm_estimate'):
            if key in s:
                s[key] = width
        if 'target_width_mm' in s:
            s['target_width_mm'] = [width] * len(s['points'])
    module['motion_candidate']['press_mm'] = .85
    revision.update(date='2026-10-13', pressure_mm=.85, design_pressure_mm=.3,
                    note='第五幅实际铺色路径重排为1mm中心线间距，下压保持0.85mm。设计边界按原0.3mm规划笔宽重建；较宽试画笔触未用来扩大设计边界。')
    result['workflow']['color_geometry'].update(pressure_range_mm=[.85, .85])
    assert result['ink_job'] == original['ink_job']
    assert [s for s in module['strokes'] if s.get('nominal_pitch_mm') is None] == [s for s in original['workflow']['modules']['color']['strokes'] if s.get('nominal_pitch_mm') is None]
    return result


def shown(b):
    v = copy.deepcopy(b)
    # Show the user's blue-then-green execution view, including small accents.
    v['workflow']['modules']['color']['strokes'].sort(key=lambda s: s['color'] == 'green')
    return h.render(v)


def main():
    from web.catalog import ARTWORKS
    from web.site_settings import prepared_artwork
    from run_json_painting import manifest_for
    from run_color_experiment import module_lines
    from color_profile import ColorProfile
    from config import z_for_press
    unlocked()
    OUT.mkdir(parents=True, exist_ok=True)
    catalog_path = ROOT / '待画作品/画稿清单.json'
    catalog_baseline = catalog_path.read_bytes()
    catalog = read(catalog_path)
    protected = {ROOT / ARTWORKS[n-1][3]: (ROOT / ARTWORKS[n-1][3]).read_bytes() for n in (1, 6)}
    writes, reports, baselines = {}, [], {}
    board = Image.new('RGB', (1600, 4 * 630), '#faf8f0')
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
    for number in (2, 3, 4, 5):
        target = ROOT / ARTWORKS[number-1][3]
        baselines[target] = target.read_bytes()
        old = read(target)
        assert not old['workflow'].get('first_color_alignment_revision'), 'Already aligned'
        b = copy.deepcopy(old)
        accents = remove_accents(b)
        if number == 5:
            b = fifth_pitch(b)
        m = b['workflow']['modules']['color']
        assert {s['nominal_pitch_mm'] for s in m['strokes'] if s.get('nominal_pitch_mm') is not None} == {1.}
        assert {p for s in m['strokes'] for p in s['press']} == {.85}
        assert m['motion_candidate']['press_mm'] == .85
        assert b['ink_job'] == old['ink_job']
        revision = dict(date='2026-10-13', center_pitch_mm=1., pressure_mm=.85,
                        source_sha256=hashlib.sha256(baselines[target]).hexdigest(), **accents,
                        all_mountain_outlines_black=True,
                        note='第2至5幅对齐第一幅：铺色中心线间距1mm、下压0.85mm，按第一幅相同规则去掉已被蓝色覆盖的零碎绿色岩面点缀。保留树木、草坡、梯田与成片绿色坡面；山体外轮廓沿用黑墨。')
        plan = b['workflow']
        plan['first_color_alignment_revision'] = revision
        plan['green_accent_removal_revision'] = accents | dict(date='2026-10-13')
        plan['color_geometry'].update(pitch_mm=1., pitch_px=1./m['mm_per_px'], pressure_range_mm=[.85, .85])
        hashed = digest(stroke_data(b))
        for key in ('source_revision', 'web_installation'):
            plan[key]['stroke_data_sha256'] = hashed
        task, package = target.parent, target.parent.parent
        candidate = task / 'painting.alignment-candidate.json'
        candidate.write_bytes(encode(b))
        try:
            with contextlib.redirect_stdout(io.StringIO()), patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline only')):
                prepared = prepared_artwork(candidate, ROOT / 'web/data/site_settings.json')
                manifest_for(prepared[2], prepared[1], prepared[3])
                effective = prepared[1]['modules']['color']
                profile = ColorProfile(**effective['motion_candidate']).to_dict()
                assert profile['press_mm'] == .85
                for color in ('blue', 'green'):
                    lines, skipped = module_lines(effective, color, profile)
                    assert not skipped
                    assert all(abs(pose[2] - z_for_press(.85)) < 1e-9 for line in lines for kind, pose in line['sequence'] if kind in ('touch', 'draw'))
        finally:
            candidate.unlink(missing_ok=True)
        order = index_for(b)
        counts = {k: len(v) for k, v in order.items()}
        report = dict(number=number, counts=counts, revision=revision, stroke_data_sha256=hashed,
                      offline_check='passed', hardware_started=False, effective_from='next-start')
        reports.append(report)
        writes[target] = encode(b)
        writes[task / '对齐第一幅上色校验.json'] = encode(report)
        for p in task.glob('*_计划.json'):
            writes[p] = encode(plan)
        for name in ('绘画顺序索引.json', 'Web当前应用记录.json'):
            p = task / name
            v = read(p)
            v.update(counts=counts, stroke_data_sha256=hashed, first_color_alignment_revision=revision)
            if name == '绘画顺序索引.json':
                v['stages'] = order
            else:
                v['color_pressure_validation'] = report
            writes[p] = encode(v)
        table = io.StringIO(newline='')
        writer = csv.writer(table)
        writer.writerow(['阶段', '阶段内笔序', '笔触ID', '完整部位', '类别', '原笔触ID', '长度mm'])
        for stage, rows in order.items():
            for row in rows:
                writer.writerow([stage, row['order'], row['id'], row['region'], row['role'], row['parent_stroke_id'], row['length_mm']])
        writes[task / '逐笔顺序索引.csv'] = table.getvalue().encode('utf-8-sig')
        row = next(row for row in catalog['artworks'] if row['folder'] == package.name)
        row.update(counts=counts, stroke_data_sha256=hashed, first_color_alignment_revision=revision)
        png = h.render(b)
        for name in ('web_preview.png', '执行路径预览_非实测线宽.png', '任务预览_线宽色彩为估计.png', '黑线加蓝绿_实际路径.png', '当前作画阶段预览.png', '完整色面黑轮廓.png', '中心线间距1mm预览.png', '上色下压0.85mm预览.png', '对齐第一幅上色预览.png'):
            writes[package / '预览与下压标注' / name] = png
        for x, bb, label in ((20, old, '修改前'), (820, b, '对齐第一幅后')):
            pic = Image.open(io.BytesIO(shown(bb)))
            pic.thumbnail((760, 545))
            y = (number-2) * 630
            board.paste(pic, (x, y+55))
            draw.text((x, y+12), f'第{number}幅 · {label}', font=font, fill='#34463a')
        p = package / '阅读说明.md'
        writes[p] = (p.read_text(encoding='utf-8') + '\n\n## 对齐第一幅上色（2026-10-13）\n\n' + revision['note'] + '\n通过离线校验，下次启动生效。\n').encode('utf-8')
        print(encode(dict(number=number, counts=counts, removed_green_strokes=accents['removed_green_strokes'], removed_components=len(accents['components']), offline_check='passed')).decode(), flush=True)
    writes[catalog_path] = encode(catalog)
    board.save(OUT / '四幅对齐第一幅前后.png')
    backup = ROOT / 'archive' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_其余四幅对齐第一幅前')
    originals = {p: p.read_bytes() if p.exists() else None for p in writes}
    assert originals[catalog_path] == catalog_baseline
    assert all(originals[p] == data for p, data in baselines.items())
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
                tmp = p.with_name(p.name + '.alignment.tmp')
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
