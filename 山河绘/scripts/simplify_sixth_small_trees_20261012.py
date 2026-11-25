"""Remove five small trees from artwork 06, preserving two framing pines."""
import copy
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import socket
import sys
import urllib.request
from datetime import datetime
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from web.catalog import ARTWORKS
from ink_control import InkPolicy, refresh_stats, build_plan
from geometry import CanvasMap, check_waypoints
from painting_session import session_lock
from scripts.consolidate_selected_artworks import digest, stroke_data
from run_mountain_v4_robot import safe_transfers
import job
import shuimo

ART = ROOT / '待画作品/06_松瀑云山_简化单线稿'
TASK = ART / '任务资料'
OUT = ROOT / 'artifacts/simplify_small_trees_20261012'
DROP = {58, 59, 60, 190, 191, 192, 193, 228, 229, 230, 231, 232,
        243, 244, 245, 306, 307, 308, 312, 313, 314, 315, 316, 317,
        318, 319, 320, 341, 342, 343, 344}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def payload(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def center_preview(bundle):
    canvas = Image.new('RGB', tuple(bundle['ink_job']['meta']['image_size']), 'white')
    draw = ImageDraw.Draw(canvas)
    for stroke in bundle['ink_job']['meta']['ink_execution']['base_strokes']:
        draw.line([tuple(p) for p in stroke['points']], fill='#46524c', width=2)
    return canvas


def main():
    target = TASK / 'painting.json'
    baseline = target.read_bytes()
    old = read(target)
    assert not old['workflow'].get('small_tree_revision'), 'Already simplified'
    def unrelated_active_run():
        locked = any((ROOT / 'robot_runs' / name).exists() for name in ('painting_session.lock', 'blue_wash_motion.lock'))
        if not locked:
            return False
        with urllib.request.urlopen('http://127.0.0.1:8765/api/state', timeout=5) as response:
            state = json.load(response)
        # An executor has already loaded its artwork into memory. Editing this
        # inactive sixth artwork cannot change another artwork's running motions.
        assert state.get('mode') == 'live' and state.get('id') not in (None, 'painting-06'), state
        return True
    unrelated_active_run()
    bundle = copy.deepcopy(old)
    meta = bundle['ink_job']['meta']
    before = old['ink_job']['meta']['ink_execution']['base_strokes']
    assert len(before) == 394 and before[195]['id'] == 'outline-simplified-00195'
    scale = meta['ink_execution']['mm_per_px']
    strokes = []
    modified = []

    def make(stroke, points, suffix=''):
        result = copy.deepcopy(stroke)
        assert len(set(stroke['speed'])) == 1
        # Remove raster stair steps only on the four edited contours, <=0.15mm.
        p = np.array(points, dtype=np.float32)
        p = cv2.approxPolyDP(p, .15 / scale, False).reshape(-1, 2).astype(float).tolist()
        nearest = np.argmin(np.linalg.norm(np.asarray(p)[:,None,:] - np.asarray(stroke['points'])[None,:,:], axis=2), axis=1)
        result.update(points=p, press=[stroke['press'][i] for i in nearest],
                      speed=[stroke['speed'][0]] * len(p))
        if suffix:
            result.update(id=stroke['id'] + suffix, revision_parent_id=stroke['id'])
        return refresh_stats(result, scale)

    for number, stroke in enumerate(before, 1):
        if number in DROP:
            continue
        points = stroke['points']
        if number == 305:
            # The crown is attached to the mountain contour; reconnect its shoulder.
            result = make(stroke, points[:10] + [[618.,108.],[621.,115.]])
        elif number == 188:
            # Keep the cliff and root-level terrain, discard the branch loop.
            result = make(stroke, points[:46] + [[565.,655.]])
        elif number == 218:
            # Replace the foreground crown silhouette with a low rock ridge.
            result = make(stroke, points[:31] + [[446.,914.],[469.,924.],[490.,928.]] + points[54:])
        elif number == 196:
            # Keep pavilion walls and rock contours as three separate motions.
            parts = [points[:34], points[33:51] + [[338.,560.]] + points[63:78], points[77:]]
            results = [make(stroke, part, '-terrain-%d' % (i+1)) for i, part in enumerate(parts)]
            for i, result in enumerate(results):
                result.update(segment=i, segments=len(results))
            strokes.extend(results)
            modified.append(number)
            continue
        else:
            strokes.append(copy.deepcopy(stroke))
            continue
        strokes.append(result)
        modified.append(number)
    assert len(strokes) == 394 - len(DROP) + 2
    edited_ids = {before[n-1]['id'] for n in modified}
    retained = {s['id']: s for s in strokes}
    for number, original in enumerate(before, 1):
        if number not in DROP and original['id'] not in edited_ids:
            assert retained[original['id']] == original
    meta['ink_execution']['base_strokes'] = strokes
    meta['strokes'] = len(strokes)
    meta['quality_stats'].update(execution_strokes=len(strokes), small_trees_removed=5,
                                 retained_trees=2, source_paths=len(strokes))
    plan = bundle['workflow']
    plan['modules']['outline'].update(strokes=copy.deepcopy(strokes), stroke_count=len(strokes))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    revision = dict(date=stamp, retained_trees=['左侧大松树','右岸松树'], removed_small_trees=5,
                    removed_original_stroke_numbers=sorted(DROP), edited_original_stroke_numbers=modified,
                    old_strokes=len(before), new_strokes=len(strokes), original_196_split_into=3,
                    original_196_points=len(before[195]['points']), new_196_points=sum(len(s['points']) for s in strokes if s.get('revision_parent_id')==before[195]['id']),
                    pressure_mm=.10, speeds_unchanged=True,
                    note='删除山顶、山腰、亭旁、瀑布旁和前景五处小树；保留左侧与右岸两棵松树。原196笔去树后拆成三笔。简化不等于确认解决现场总线报警。')
    plan['small_tree_revision'] = revision
    # Fix the moved drive reference without changing registration.
    meta['image'] = plan['reference_image'] = '../参考原图.png'
    mapper = CanvasMap(meta['image_size'])
    steps, _ = build_plan(strokes, shuimo.ink_palette_index(), scale, InkPolicy(**meta['ink_execution']['policy']))
    waypoints, _ = job.steps_to_waypoints(steps, mapper)
    bundle['ink_job']['waypoints'] = safe_transfers(waypoints)
    assert check_waypoints(bundle['ink_job']['waypoints'])[0]
    candidate = TASK / 'painting.trees-candidate.json'
    candidate.write_bytes(payload(bundle))
    try:
        from web.site_settings import prepared_artwork
        from run_json_painting import manifest_for
        from web.server import artwork_data
        with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline validation only')):
            prepared = prepared_artwork(candidate, ROOT / 'web/data/site_settings.json')
            manifest = manifest_for(prepared[2], prepared[1], prepared[3])
            assert len(manifest['stages']['black']) == len(strokes)
            assert artwork_data(candidate)['counts'] == {'black': len(strokes)}
        stroke_hash = digest(stroke_data(bundle))
        report = dict(revision=revision, stroke_data_sha256=stroke_hash, offline_check='passed',
                      robot_connected=False, pressure_speed_order_preserved_outside_edits=True,
                      all_waypoints_rebuilt=True, alarm_resolved=False)
        writes = {target: payload(bundle), TASK/'黑线任务.json': payload(bundle['ink_job']),
                  TASK/'绘画计划.json': payload(plan), TASK/'小树精简校验.json': payload(report)}
        rows = [dict(order=i+1, id=s['id'], source_path=s['source_path_index'], length_mm=s['length_mm']) for i,s in enumerate(strokes)]
        writes[TASK/'绘画顺序索引.json'] = payload(dict(strokes=rows, stroke_data_sha256=stroke_hash))
        table = io.StringIO(newline=''); writer = csv.writer(table)
        writer.writerow(['笔序','笔触ID','原路径编号','长度毫米','下压毫米'])
        for i,s in enumerate(strokes): writer.writerow([i+1,s['id'],s['source_path_index']+1,round(s['length_mm'],3),round(s['press_mean'],3)])
        writes[TASK/'逐笔顺序索引.csv'] = table.getvalue().encode('utf-8-sig')
        paths = [s['points'] for s in strokes]
        writes[TASK/'landscape-paths.json'] = payload(dict(coordinate_system='prepared image pixels; x right; y down', canvas_width_px=meta['image_size'][0], canvas_height_px=meta['image_size'][1], paths=paths, revision=revision))
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d">' % tuple(meta['image_size']), '<g fill="none" stroke="#46524c" stroke-width="2">']
        svg += ['<polyline points="%s"/>' % ' '.join('%g,%g' % tuple(p) for p in points) for points in paths]
        writes[TASK/'landscape-centerline.svg'] = ('\n'.join(svg+['</g></svg>'])).encode('utf-8')
        writes[TASK/'conversion-stats.json'] = payload(dict(canvas=meta['image_size'], strokes=len(strokes),vertices=sum(map(len,paths)), source='reviewed execution paths after small tree removal'))
        image = center_preview(bundle); png = io.BytesIO(); image.save(png,format='PNG')
        preview = ART / '预览与下压标注'
        writes[ART/'参考原图.png'] = png.getvalue()
        for name in ('单线路径预览.png','web_preview.png','小树精简后.png'):
            writes[preview/name] = png.getvalue()
        from trajectory_preview import render_contact
        contact = render_contact(bundle['ink_job']['waypoints'], mapper, output_scale=2.)
        ok, encoded = cv2.imencode('.png', contact); assert ok
        writes[preview/'实际任务笔宽估计预览.png'] = encoded.tobytes()
        board = Image.new('RGB',(1600,680),'#f8f7f0'); draw = ImageDraw.Draw(board)
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
        for x,b,label in [(20,old,'修改前：多处小树 · 394笔'),(820,bundle,f'修改后：保留两棵松树 · {len(strokes)}笔')]:
            im = center_preview(b); im.thumbnail((760,550)); board.paste(im,(x,65)); draw.text((x,20),label,font=font,fill='#34463a')
        draw.text((20,620),'原196笔去掉树冠、树枝与树干细节并拆成三笔；0.10mm下压与原速度设置保留。',font=font,fill='#34463a')
        comparison = io.BytesIO(); board.save(comparison,format='PNG')
        writes[preview/'小树精简前后对照.png'] = comparison.getvalue()
        writes[ART/'阅读说明.md'] = (f'# 松瀑云山 · 简化单线稿\n\n当前执行{len(strokes)}笔，仅黑墨勾线，下压0.10mm。\n\n2026-10-12按用户要求保留左侧大松树与右岸松树，删除五处小树。原第196笔去除亭旁小树后拆成三条独立笔触，保持亭子和山石轮廓。\n\n任务资料/painting.json是唯一执行画稿；已同步黑线任务、计划、索引、SVG、路径JSON和预览。已通过网页准备、画布与完整轨迹离线校验，没有启动机械臂。\n\n精简可减少细碎笔触和指令，但不证明已排除1445–1450现场总线同步报警。异常任务不支持自动续画。\n').encode('utf-8')
        admission = read(TASK/'入库校验.json'); admission.update(date='2026-10-12', source_paths=len(strokes), execution_strokes=len(strokes), length_mm=sum(s['length_mm'] for s in strokes), painting_sha256=hashlib.sha256(writes[target]).hexdigest(), small_tree_revision=revision)
        writes[TASK/'入库校验.json'] = payload(admission)
        library_path = ROOT / '待画作品/画稿清单.json'; library = read(library_path)
        row = next(row for row in library['artworks'] if row['folder']==ART.name)
        row.update(counts={'black':len(strokes)}, source_paths=len(strokes), painting_sha256=admission['painting_sha256'], stroke_data_sha256=stroke_hash, small_tree_revision=revision)
        writes[library_path] = payload(library)
        OUT.mkdir(parents=True,exist_ok=True)
        backup = ART/'历史版本'/(stamp+'_小树精简前'); backup.mkdir(parents=True,exist_ok=False)
        originals = {p:p.read_bytes() if p.exists() else None for p in writes}
        for p,data in originals.items():
            if data is not None:
                saved = backup/p.relative_to(ROOT); saved.parent.mkdir(parents=True,exist_ok=True); saved.write_bytes(data)
        (backup/'painting.json').write_bytes(baseline)
        other_run = unrelated_active_run()
        with (contextlib.nullcontext() if other_run else session_lock(ROOT/'robot_runs')):
            if not other_run:
                assert not (ROOT/'robot_runs/blue_wash_motion.lock').exists()
            assert target.read_bytes() == baseline
            done=[]
            try:
                for p,data in writes.items():
                    assert (p.read_bytes() if p.exists() else None)==originals[p]
                    tmp=p.with_name(p.name+'.trees.tmp'); tmp.write_bytes(data); tmp.replace(p); done.append(p)
                assert read(target)==bundle
            except Exception:
                for p in reversed(done):
                    if originals[p] is None:p.unlink(missing_ok=True)
                    else:p.write_bytes(originals[p])
                raise
        (OUT/'小树精简校验.json').write_bytes(payload(report))
        print(json.dumps(report|dict(backup=str(backup)),ensure_ascii=False))
    finally:
        candidate.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
