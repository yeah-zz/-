# -*- coding: utf-8 -*-
"""Full-frame image-space trial preparation. No robot/config imports or motion.

Exact graph edges are retained; only unambiguous existing junctions can be
traversed without a pen lift. Calibration is collected separately, never guessed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022')
SOURCE = ROOT / 'faint_center_direction_20261101'
OUT = ROOT / 'full_ink_trial_20261102'


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def point_key(p):
    return tuple(float(x) for x in p)


def edge_key(a, b):
    return tuple(sorted((point_key(a), point_key(b))))


def edges(paths):
    return Counter(edge_key(a, b) for p in paths for a, b in zip(p[:-1], p[1:]) if np.linalg.norm(b-a) > 1e-9)


def deduplicate(paths):
    """Remove exact repeated edges only, splitting at removed portions."""
    seen, out, removed = set(), [], 0
    for source_id, p in enumerate(paths):
        chunk = []
        for a, b in zip(p[:-1], p[1:]):
            if np.linalg.norm(b-a) <= 1e-9:
                continue
            key = edge_key(a, b)
            if key in seen:
                removed += 1
                if len(chunk) >= 2:
                    out.append(dict(points=np.array(chunk), source_ids=[source_id]))
                chunk = []
            else:
                seen.add(key)
                if not chunk:
                    chunk = [a]
                chunk.append(b)
        if len(chunk) >= 2:
            out.append(dict(points=np.array(chunk), source_ids=[source_id]))
    return out, removed


def tangent(p, end, distance=7.):
    q = p if end == 0 else p[::-1]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    target = min(distance, arc[-1])
    at = np.array([np.interp(target, arc, q[:, j]) for j in range(2)])
    v = at - q[0]
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-9 else None


def join_existing(records):
    """Pair only exact endpoints with strong tangent evidence and a clear margin.

    No gap bridging, no nearest-path snapping, no forced pairing at ambiguous Y's.
    This is geometric routing, not a claim of the artist's true stroke identity.
    """
    junctions = defaultdict(list)
    for i, row in enumerate(records):
        p = row['points']
        if point_key(p[0]) == point_key(p[-1]):
            continue
        for e in (0, 1):
            junctions[point_key(p[0 if e == 0 else -1])].append((i, e))
    pairs, audit = {}, []
    for location, ends in sorted(junctions.items()):
        if len(ends) not in (2, 3, 4):
            continue
        options = {}
        for a in ends:
            ta = tangent(records[a[0]]['points'], a[1])
            candidates = []
            for b in ends:
                if a[0] == b[0]:
                    continue
                tb = tangent(records[b[0]]['points'], b[1])
                if ta is not None and tb is not None:
                    angle = math.degrees(math.acos(float(np.clip(-ta @ tb, -1, 1))))
                    candidates.append((angle, b))
            candidates.sort()
            if candidates and candidates[0][0] <= 22.5:
                margin = candidates[1][0] - candidates[0][0] if len(candidates) > 1 else 180.
                if margin >= 25.:
                    options[a] = (candidates[0][1], candidates[0][0], margin)
        for a, (b, angle, margin) in sorted(options.items()):
            if a not in pairs and b not in pairs and b in options and options[b][0] == a:
                pairs[a], pairs[b] = b, a
                audit.append(dict(at=list(location), ends=[list(a), list(b)], turn_deg=angle, margin_deg=margin))
    out, visited = [], set()

    def walk(start):
        current = start
        arrays, ids = [], []
        while current[0] not in visited:
            i, e = current
            visited.add(i)
            p = records[i]['points'] if e == 0 else records[i]['points'][::-1]
            arrays.append(p if not arrays else p[1:])
            ids.extend(records[i]['source_ids'])
            exit_end = (i, 1-e)
            if exit_end not in pairs:
                break
            current = pairs[exit_end]
        out.append(dict(points=np.concatenate(arrays), source_ids=sorted(set(ids))))

    for i in range(len(records)):
        for e in (0, 1):
            if i not in visited and (i, e) not in pairs:
                walk((i, e))
    for i in range(len(records)):
        if i not in visited:
            walk((i, 0))
    return out, audit


def collinear(p):
    """Exact geometry compression: preserve every turn, no curve displacement."""
    if len(p) < 3:
        return p.copy()
    a, b = np.diff(p, axis=0)[:-1], np.diff(p, axis=0)[1:]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    same = (np.abs(cross) < 1e-10) & ((a*b).sum(axis=1) > 0)
    return p[np.r_[True, ~same, True]]


def air_length(records, start=(0., 0.)):
    pos, total = np.array(start), 0.
    for row in records:
        p = row['points']
        total += float(np.linalg.norm(p[0]-pos))
        pos = p[-1]
    return total


def order_paths(records):
    remaining = set(range(len(records)))
    out, pos = [], np.array([0., 0.])
    while remaining:
        options = [(float(np.linalg.norm(records[i]['points'][0 if e == 0 else -1]-pos)), i, e)
                   for i in sorted(remaining) for e in (0, 1)]
        _, i, e = min(options)
        row = dict(records[i])
        row['points'] = row['points'] if e == 0 else row['points'][::-1].copy()
        row['reversed_for_travel'] = bool(e)
        out.append(row)
        remaining.remove(i)
        pos = row['points'][-1]
    # Retain original ordering if this particular greedy route is worse.
    return out if air_length(out) < air_length(records) else records


def contain_transform(image_size, paper_size, margin_mm):
    w, h = map(float, image_size)
    pw, ph = map(float, paper_size)
    if not all(math.isfinite(v) and v > 0 for v in (w, h, pw, ph)) or not math.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError('Invalid sizes/margin')
    if 2*margin_mm >= min(pw, ph):
        raise ValueError('Paper too small for margins')
    scale = min((pw-2*margin_mm)/w, (ph-2*margin_mm)/h)
    return scale, np.array([(pw-w*scale)/2, (ph-h*scale)/2])


def draw_paths(shape, records, colored=False, air=False):
    canvas = np.full((*shape, 3), 255, np.uint8)
    pos = np.array([0., 0.])
    for i, row in enumerate(records):
        p = np.rint(row['points']).astype(np.int32)
        if air:
            cv2.line(canvas, tuple(np.rint(pos).astype(int)), tuple(p[0]), (225, 225, 225), 1, cv2.LINE_AA)
        color = tuple(int(x) for x in cv2.cvtColor(np.uint8([[[i*47 % 180, 190, 155]]]), cv2.COLOR_HSV2RGB)[0, 0]) if colored else (0, 0, 0)
        cv2.polylines(canvas, [p], False, color, 1, cv2.LINE_AA)
        pos = p[-1]
    return Image.fromarray(canvas)


def calibration_template():
    return dict(schema='full_ink_site_calibration_v1', measured_at=None,
        user_index=None, tool_index=None, rpy_deg=[None]*3, z_increases_down=None,
        paper_corners_touch_xyz={k:[None]*3 for k in ('TL','TR','BR','BL')},
        paper_center_touch_xyz=[None]*3,
        black_station=dict(x=None,y=None,surface_z=None,rim_z=None,dip_mm=None,dwell_s=None,drip_s=None),
        brush_test=dict(samples=[], selected_press_mm=None, draw_velocity_percent=None, global_speed_factor=None),
        paper_lift_mm=None, station_clearance_mm=None, reach_limits_xyz=[[None,None]]*3,
        site_clearance_checked=False, notes='换笔/夹持/纸面/墨碗后重新测量；禁止抄旧配置。samples每项填写press_mm、width_mm、continuous。')


def validate_calibration(doc):
    errors = []
    def numeric(v, name):
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
            raise ValueError(name+' must be a finite number')
        return float(v)
    try:
        if doc.get('schema') != 'full_ink_site_calibration_v1':
            raise ValueError('Calibration schema mismatch')
        if not isinstance(doc.get('measured_at'), str) or not doc['measured_at'].strip():
            raise ValueError('measured_at required')
        for name in ('user_index','tool_index'):
            if type(doc.get(name)) is not int or doc[name] < 0:
                raise ValueError(name+' must be a nonnegative integer')
        if type(doc.get('z_increases_down')) is not bool:
            raise ValueError('z_increases_down required')
        if len(doc.get('rpy_deg', [])) != 3:
            raise ValueError('Three orientation values required')
        for v in doc['rpy_deg']:
            numeric(v, 'rpy')
        corners = np.array([[numeric(v,k) for v in doc['paper_corners_touch_xyz'][k]] for k in ('TL','TR','BR','BL')])
        center = np.array([numeric(v,'center') for v in doc['paper_center_touch_xyz']])
        if corners.shape != (4,3) or center.shape != (3,):
            raise ValueError('Paper touch poses require XYZ')
        xy = corners[:,:2].astype(np.float32)
        if not cv2.isContourConvex(xy) or abs(cv2.contourArea(xy)) < 100:
            raise ValueError('Paper corners must form ordered nondegenerate TL/TR/BR/BL quadrilateral')
        if cv2.pointPolygonTest(xy, tuple(map(float,center[:2])), False) < 0:
            raise ValueError('Center touch point must lie inside paper')
        spread = float(np.ptp(np.r_[corners[:,2],center[2]]))
        if spread > .2:
            raise ValueError('Paper contact Z spread >0.20 mm; flatten/recheck or use a fitted paper plane before export')
        limits = np.array([[numeric(v,'reach') for v in r] for r in doc['reach_limits_xyz']])
        if limits.shape != (3,2) or np.any(limits[:,0] >= limits[:,1]):
            raise ValueError('Invalid configured reach bounds')
        for p in [*corners,center]:
            if np.any(p < limits[:,0]) or np.any(p > limits[:,1]):
                raise ValueError('Paper touch coordinate outside supplied reach')
        station = doc['black_station']
        for name in ('x','y','surface_z','rim_z','dip_mm','dwell_s','drip_s'):
            numeric(station[name], 'black_station.'+name)
        sign = 1 if doc['z_increases_down'] else -1
        if sign*(station['surface_z']-station['rim_z']) < 0:
            raise ValueError('Cup surface/rim ordering inconsistent with Z direction')
        if min(station[k] for k in ('dip_mm','dwell_s','drip_s')) <= 0:
            raise ValueError('Positive dip depth/dwell/drain required')
        brush = doc['brush_test']
        pressure = numeric(brush['selected_press_mm'],'selected_press_mm')
        if pressure <= 0:
            raise ValueError('Positive measured pressure required')
        good = [r for r in brush['samples'] if r.get('continuous') is True
                and numeric(r['press_mm'],'sample pressure') > 0 and numeric(r['width_mm'],'sample width') > 0]
        if not any(abs(r['press_mm']-pressure) < 1e-9 for r in good):
            raise ValueError('Selected pressure must have a measured continuous-line sample')
        for k in ('draw_velocity_percent','global_speed_factor'):
            if type(brush[k]) is not int or not 1 <= brush[k] <= 100:
                raise ValueError(k+' must be an integer in [1,100]')
        lift = numeric(doc['paper_lift_mm'],'paper_lift_mm')
        clearance = numeric(doc['station_clearance_mm'],'station_clearance_mm')
        if lift <= 0 or clearance <= 0:
            raise ValueError('Positive measured clearances required')
        paper_z = float(center[2])
        high_z = min(paper_z, station['rim_z'])-clearance if sign > 0 else max(paper_z,station['rim_z'])+clearance
        targets = [np.array([station['x'],station['y'],z]) for z in (station['surface_z']+sign*station['dip_mm'], high_z)]
        targets += [np.array([p[0],p[1],p[2]+sign*pressure]) for p in corners]
        targets += [np.array([p[0],p[1],p[2]-sign*lift]) for p in corners]
        for p in targets:
            if np.any(p < limits[:,0]) or np.any(p > limits[:,1]):
                raise ValueError('Press/lift/station envelope outside supplied reach')
        if doc.get('site_clearance_checked') is not True:
            raise ValueError('Operator site clearance review still required')
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(str(exc))
    return dict(valid=not errors, errors=errors, robot_connected=False, executable_job_generated=False,
                note='仅检查填写值的一致性，不能验证现场碰撞、标定准确度或毛笔效果。')


def make_report(output, doc):
    strokes = json.dumps([r['points'] for r in doc['paths']], separators=(',',':'))
    stats = doc['statistics']
    page = '''<!doctype html><meta charset="utf-8"><title>完整黑线山水试画准备</title>
<style>body{font:17px system-ui;max-width:1250px;margin:24px auto;padding:0 20px;background:#f5f3eb;color:#292c29}h1{font-size:27px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}img,canvas{width:100%;background:white}section{background:white;padding:18px;margin:18px 0;border-radius:10px}input{width:70%}small{color:#555}a{color:#17694c}button{padding:8px 20px}li{margin:9px 0}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<h1>完整黑线山水 · 上机前轨迹准备</h1><p>整幅 N02，保留原始留白与构图。当前为像素坐标轨迹；新标定尚未填写，没有连接机械臂。</p>
<section><b>本轮整理</b><p>STATS</p><p>交点续接仅改变几何行笔分段；不代表已识别画家的真实笔顺。所有原有非重复线段均保留，未跨空白补线，也未删去短松针。</p></section>
<div class="grid"><section><b>完整原图</b><img src="source.png"></section><section><b>完整黑线轨迹（等细线示意）</b><img src="full_black_lines.png"></section></div>
<section><b>落笔顺序回放</b><p><button id="play">播放 / 暂停</button> <input id="progress" type="range" min="0" max="COUNT" value="COUNT"><span id="count"></span></p><canvas id="c" width="WIDTH" height="HEIGHT"></canvas><small>颜色表示轨迹段，灰线为空行程；尚未包含现场蘸墨行程及实际耗时。当前预览不模拟笔宽、浓淡或真实墨水。</small></section>
<section><b>上机前：重新标定</b><ol><li>在示教器固定本次 User / Tool 与绘画姿态；夹笔后重新记录。</li><li>记录纸张四角和中心的首次触纸 XYZ，保持同一姿态；四角顺序 TL、TR、BR、BL。</li><li>重测黑墨碗中心、液面与杯沿，填写实际蘸深和沥墨时间。</li><li>用当前纸笔确认一档连续细线，记录下压量、实测线宽及所用速度。本次先用该档画黑线，不把图像笔宽直接换算成下压。</li><li>将结果填入 <a href="calibration_to_fill.json">新标定表</a>，按 <a href="上机准备说明.md">操作说明</a>离线校验。收到新值后才能生成绝对坐标与蘸墨任务。</li></ol><p><a href="trial_plan_unbound.json">完整未绑定轨迹 JSON</a> · <a href="full_black_lines.svg">完整 SVG</a> · <a href="statistics.json">整理统计</a></p></section>
<script>const ps=STROKES,c=document.getElementById('c'),ctx=c.getContext('2d'),r=document.getElementById('progress');let playing=false;function render(){let n=+r.value;ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);let last=[0,0];for(let i=0;i<n;i++){let p=ps[i];ctx.strokeStyle='#ddd';ctx.lineWidth=.6;ctx.beginPath();ctx.moveTo(...last);ctx.lineTo(...p[0]);ctx.stroke();ctx.strokeStyle=`hsl(${i*137.5%360},55%,34%)`;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(...p[0]);for(let j=1;j<p.length;j++)ctx.lineTo(...p[j]);ctx.stroke();last=p[p.length-1];}document.getElementById('count').textContent=` ${n}/${ps.length}`;}r.oninput=()=>{playing=false;render()};document.getElementById('play').onclick=()=>{if(+r.value===ps.length)r.value=0;playing=!playing};setInterval(()=>{if(playing){r.value=Math.min(ps.length,+r.value+8);render();if(+r.value===ps.length)playing=false}},70);render();</script>'''
    words = f"{stats['source_paths']} 条原始图结构段 → {stats['planned_paths']} 条几何行笔段；合并 {stats['joined_junctions']} 处明确交点。精确去除 {stats['duplicate_edges_removed']} 条重复边。点数 {stats['source_points']} → {stats['planned_points']}。图内空行程减少 {stats['air_reduction_fraction']:.1%}（以图像左上角为共同起点，未含墨碗往返）。"
    for key, value in [('STATS',words),('COUNT',str(len(doc['paths']))),('WIDTH',str(doc['width'])),('HEIGHT',str(doc['height'])),('STROKES',strokes)]:
        page = page.replace(key,value)
    (output/'index.html').write_text(page,encoding='utf-8')


def prepare(sample='N02', output=OUT):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    source = SOURCE/'fresh_inputs'/f'{sample}.png'
    pathfile = SOURCE/'fresh_outputs'/sample/'paths.json'
    doc = json.loads(pathfile.read_text('utf-8'))
    w,h = doc['width'],doc['height']
    raw = [np.array(r['points'],dtype=float) for r in doc['paths']]
    if any(p.ndim != 2 or p.shape[1] != 2 or len(p)<2 or not np.isfinite(p).all()
           or (p<0).any() or (p[:,0]>w-1).any() or (p[:,1]>h-1).any() for p in raw):
        raise ValueError('Invalid source paths')
    records, removed = deduplicate(raw)
    joined,audit = join_existing(records)
    if set(edges(raw)) != set(edges([r['points'] for r in joined])):
        raise AssertionError('Source edge coverage changed')
    ordered = order_paths(joined)
    simplified = [dict(r,points=collinear(r['points'])) for r in ordered]
    stats = dict(source_paths=len(raw),planned_paths=len(simplified),source_points=sum(map(len,raw)),
        planned_points=sum(len(r['points']) for r in simplified),duplicate_edges_removed=removed,
        joined_junctions=len(audit),draw_length_px=sum(length(r['points']) for r in simplified),
        source_air_px=air_length([dict(points=p) for p in raw]),planned_air_px=air_length(simplified),
        source_unique_edges=len(edges(raw)),all_unique_edges_preserved=True,
        missing_strokes_recovered=0,deleted_short_strokes=0,added_gap_bridges=0,new_training_steps=0)
    stats['air_reduction_fraction']=1-stats['planned_air_px']/stats['source_air_px']
    plan = dict(schema='full_ink_trial_unbound_v1',sample=sample,width=w,height=h,
        coordinate_system='native_image_pixels',image_fit='contain_no_crop_no_stretch',
        source_image_sha256=digest(source),source_paths_sha256=digest(pathfile),
        ready_for_robot=False,calibration_required=True,actually_drawn=False,
        physical_z=None,physical_speed=None,pressure_profile=None,
        stroke_identity_verified=False,direction_verified=False,statistics=stats,
        paths=[dict(id=i,points=r['points'].tolist(),source_ids=r['source_ids'],
                    reversed_for_travel=r.get('reversed_for_travel',False),tail_mm=0) for i,r in enumerate(simplified)])
    save(output/'trial_plan_unbound.json',plan)
    save(output/'statistics.json',stats)
    save(output/'junction_decisions.json',audit)
    template=output/'calibration_to_fill.json'
    if not template.exists():
        save(template,calibration_template())
    source_image=Image.open(source).convert('RGB')
    if source_image.size != (w,h):
        raise ValueError('Image/trajectory size mismatch')
    source_image.save(output/'source.png')
    black=draw_paths((h,w),simplified)
    black.save(output/'full_black_lines.png')
    draw_paths((h,w),simplified,colored=True).save(output/'stroke_order.png')
    paths=''.join('<path d="M '+' L '.join(f'{x:g},{y:g}' for x,y in r['points'])+'"/>' for r in simplified)
    (output/'full_black_lines.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><g fill="none" stroke="black" stroke-width="1" stroke-linejoin="round" stroke-linecap="round">{paths}</g></svg>',encoding='utf-8')
    preview=Image.new('RGB',(1400,758),'#f5f3eb')
    font_path=Path(r'C:\Windows\Fonts\msyh.ttc')
    font=ImageFont.truetype(str(font_path),24) if font_path.exists() else ImageFont.load_default()
    pen=ImageDraw.Draw(preview)
    pen.text((22,12),'完整黑线山水 · N02（保留整幅构图）',font=font,fill='#252a25')
    pen.text((22,53),'原图',font=font,fill='#252a25')
    pen.text((718,53),'整理后中心线 · 未标定 / 非实画',font=font,fill='#252a25')
    preview.paste(source_image.resize((668,668)),(22,88))
    preview.paste(black.resize((668,668)),(718,88))
    preview.save(output/'full_preview.jpg',quality=94)
    make_report(output,plan)
    (output/'上机准备说明.md').write_text('''# 本次完整黑线山水

完整 N02 构图，不裁剪，不上色。index.html 可看黑线与顺序回放。
trial_plan_unbound.json 是图像坐标方案，没有 XYZ、下压或控制器速度；不能交给实机执行器。

## 新标定填写

用当前夹持、当前纸笔，在示教器人工点动记录本次实际值，填 calibration_to_fill.json。
全程采用相同 User/Tool、相同 RPY。不得从旧 config.py 或旧说明中抄坐标。

1. measured_at：测量时间；user_index/tool_index/rpy_deg：本次坐标系及姿态。
2. paper_corners_touch_xyz：左上 TL → 右上 TR → 右下 BR → 左下 BL，每角首次触纸 XYZ；另填中心首次触纸 XYZ。画布使用这四角内的区域，等比例居中留边，不把正方图拉伸成长方图。
3. z_increases_down：沿 Z 正方向是否更压向纸面。黑碗 x/y、surface_z/rim_z 是同一姿态下笔尖所对应的位置；dip_mm 是实际蘸入深度，不是杯底坐标。
4. brush_test.samples：填写本次小样，例如 {"press_mm": 实测下压, "width_mm": 线宽, "continuous": true}；selected_press_mm 只选已得到连续细线的一档。同时记录该小样 draw_velocity_percent/global_speed_factor。
5. paper_lift_mm / station_clearance_mm 与 reach_limits_xyz 必须依据本次现场及现有控制配置填写；核对纸面、杯沿与路径空间后将 site_clearance_checked 设为 true。

只做离线表格校验（该命令不连接机械臂，也不修改配置）：

```powershell
$env:TEMP='E:\\Codex_Workspace_Data\\CLIPascene_20261021\\temp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR='E:\\Codex_Workspace_Data\\CLIPascene_20261021\\cache\\pip'
$env:TORCH_HOME='E:\\Codex_Workspace_Data\\CLIPascene_20261021\\cache\\torch'
$env:HF_HOME='E:\\Codex_Workspace_Data\\CLIPascene_20261021\\cache\\huggingface'
$env:XDG_CACHE_HOME='E:\\Codex_Workspace_Data\\CLIPascene_20261021\\cache'
$env:CUDA_CACHE_PATH='E:\\Codex_Workspace_Data\\CLIPascene_20261021\\cache\\cuda'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'E:\\Anaconda\\python.exe' -B 'F:\\山河绘\\山河绘\\scripts\\prepare_full_ink_trial_20261102.py' --validate-calibration 'E:\\Codex_Workspace_Data\\Gongbi_Line_Model_20261022\\full_ink_trial_20261102\\calibration_to_fill.json'
```

校验通过仅表示填写值基本一致，不能证明现场无碰撞。拿到新标定后，需重新生成图像到纸面的等比例映射、带抬落笔/蘸墨的绝对轨迹，并通过既有执行器的配置/姿态检查。本轮不改旧 config.py、墨碗测量或旧实机任务。

## 预期与限制

本次可比较山体轮廓、皴线和松针的落笔位置及纸笔连续性。黑线先用实测细线档；图上中心线粗细不是物理笔宽。仍有漏线、分笔及误识别，整理不会补出模型没有提取到的线。本次没有新增训练，没有采集实画，不给出准确率或实画效果承诺。
''',encoding='utf-8')
    return stats


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',choices=['N01','N02','N03'],default='N02')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--validate-calibration',type=Path)
    args=parser.parse_args()
    if args.validate_calibration:
        report=validate_calibration(json.loads(args.validate_calibration.read_text('utf-8-sig')))
        print(json.dumps(report,ensure_ascii=False,indent=2))
        raise SystemExit(0 if report['valid'] else 2)
    print(json.dumps(prepare(args.sample,args.output),ensure_ascii=False,indent=2))
