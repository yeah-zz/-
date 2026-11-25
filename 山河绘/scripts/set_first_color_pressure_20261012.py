"""Set one catalog painting's color pressure for the next run, offline."""
import copy
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
from datetime import datetime
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web.catalog import ARTWORKS
from scripts.consolidate_selected_artworks import digest, stroke_data
from shuimo import press_to_width_mm


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description='修改指定画稿全部上色下压量，仅离线操作。')
    parser.add_argument('--press-mm', type=float, default=1.0)
    parser.add_argument('--painting', type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
    from color_profile import ColorProfile
    pressure = args.press_mm
    ColorProfile(press_mm=pressure).to_dict()
    label = format(pressure, 'g')
    number = args.painting
    target = ROOT / ARTWORKS[number - 1][3]
    task = target.parent
    package = task.parent
    baseline = target.read_bytes()
    old = read(target)
    bundle = copy.deepcopy(old)
    module = bundle['workflow']['modules']['color']
    assert module['strokes'], 'No color strokes to update'
    module['motion_candidate']['press_mm'] = pressure
    width = press_to_width_mm(pressure)
    for stroke in module['strokes']:
        stroke['press'] = [pressure] * len(stroke['points'])
        for key in ('pressure_mm', 'press_mean', 'press_max'):
            if key in stroke:
                stroke[key] = pressure
        for key in ('width_mm', 'brush_width_mm_estimate'):
            if key in stroke:
                stroke[key] = width
        if 'target_width_mm' in stroke:
            stroke['target_width_mm'] = [width] * len(stroke['points'])
    changed = {'press', 'pressure_mm', 'press_mean', 'press_max', 'width_mm',
               'brush_width_mm_estimate', 'target_width_mm'}
    assert bundle['ink_job'] == old['ink_job']
    for before, after in zip(old['workflow']['modules']['color']['strokes'], module['strokes']):
        assert {k: v for k, v in before.items() if k not in changed} == {k: v for k, v in after.items() if k not in changed}
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    note = f'全部蓝色、绿色上色点统一下压{pressure:.2f}mm；下一次启动生效，当前执行器已载入的参数保持不变。'
    revision = dict(source_sha256=hashlib.sha256(baseline).hexdigest(), pressure_mm=pressure,
                    note=note, date=stamp)
    plan = bundle['workflow']
    plan['color_pressure_revision'] = revision
    geometry = plan.get('color_geometry', {})
    geometry['pressure_range_mm'] = [pressure, pressure]
    for key in ('terrain_pressure_mm', 'leaf_pressure_mm'):
        if key in geometry:
            geometry[key] = pressure
    if 'modeled_brush_width_mm' in geometry:
        geometry['modeled_brush_width_mm'] = width
    stroke_hash = digest(stroke_data(bundle))
    for key in ('source_revision', 'web_installation'):
        if key in plan:
            plan[key]['stroke_data_sha256'] = stroke_hash
    candidate = task / 'painting.pressure-candidate.json'
    candidate.write_bytes(encode(bundle))
    try:
        from web.site_settings import prepared_artwork
        from run_color_experiment import module_lines
        from color_profile import ColorProfile
        from config import z_for_press
        from run_json_painting import manifest_for
        # Verify the exact web preparation path without any hardware connection.
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline validation only')):
            prepared = prepared_artwork(candidate, ROOT / 'web/data/site_settings.json')
            effective = prepared[1]['modules']['color']
            profile = ColorProfile(**effective['motion_candidate']).to_dict()
            assert profile['press_mm'] == pressure
            manifest_for(prepared[2], prepared[1], prepared[3])
            counts = {}
            for color in ('blue', 'green'):
                lines, skipped = module_lines(effective, color, profile)
                assert not skipped
                counts[color] = len(lines)
                for line in lines:
                    assert all(abs(pose[2] - z_for_press(pressure)) < 1e-9
                               for kind, pose in line['sequence'] if kind in ('touch', 'draw'))
        report = dict(pressure_mm=pressure, color_strokes=counts,
                      color_points=sum(len(s['points']) for s in module['strokes']),
                      ink_unchanged=True, geometry_order_speed_unchanged=True,
                      offline_check='passed', hardware_started=False,
                      active_run_unchanged=True, effective_from='next-start',
                      stroke_data_sha256=stroke_hash, revision=revision)
        writes = {target: encode(bundle), task / f'上色{label}mm校验.json': encode(report)}
        for companion in task.glob('*_计划.json'):
            writes[companion] = encode(plan)
        for name in ('绘画顺序索引.json', 'Web当前应用记录.json'):
            path = task / name
            value = read(path)
            value['stroke_data_sha256'] = stroke_hash
            value['color_pressure_revision'] = revision
            if name == 'Web当前应用记录.json':
                value['color_pressure_validation'] = report
            writes[path] = encode(value)
        library_path = ROOT / '待画作品/画稿清单.json'
        library = read(library_path)
        row = next(row for row in library['artworks'] if row['painting_json'] == ARTWORKS[number - 1][3])
        row.update(stroke_data_sha256=stroke_hash, color_pressure_revision=revision)
        writes[library_path] = encode(library)
        from scripts.first_color_edge_gap_20261012 import h
        image = h.render(bundle)
        preview = package / '预览与下压标注'
        for name in ('web_preview.png', '执行路径预览_非实测线宽.png', '任务预览_线宽色彩为估计.png', '黑线加蓝绿_实际路径.png', '当前作画阶段预览.png', '完整色面黑轮廓.png', f'上色下压{label}mm预览.png'):
            writes[preview / name] = image
        readme = package / '阅读说明.md'
        writes[readme] = (readme.read_text(encoding='utf-8') + f'\n\n## 上色{label}mm试画（2026-10-12）\n\n' + note + f'\n蓝色{counts["blue"]}笔、绿色{counts["green"]}笔，共{len(module["strokes"])}笔、{report["color_points"]}个上色点。已通过网页准备与绝对Z轨迹离线检查。预览笔宽按现有模型估计。\n').encode('utf-8')
        backup = package / '历史版本' / (stamp + f'_上色{label}mm前')
        backup.mkdir(parents=True, exist_ok=False)
        originals = {}
        for path in writes:
            originals[path] = path.read_bytes() if path.exists() else None
            if path.exists():
                relative = path.relative_to(ROOT)
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, saved)
        (backup / 'painting.json').write_bytes(baseline)
        assert target.read_bytes() == baseline, 'Artwork changed during preparation'
        done = []
        try:
            for path, payload in writes.items():
                assert (path.read_bytes() if path.exists() else None) == originals[path]
                temp = path.with_name(path.name + '.pressure.tmp')
                temp.write_bytes(payload)
                temp.replace(path)
                done.append(path)
            installed = read(target)
            assert installed == bundle
            assert all(v == pressure for s in installed['workflow']['modules']['color']['strokes'] for v in s['press'])
        except Exception:
            for path in reversed(done):
                if originals[path] is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(originals[path])
            raise
        print(json.dumps(report | dict(target=str(target), backup=str(backup)), ensure_ascii=False))
    finally:
        candidate.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
