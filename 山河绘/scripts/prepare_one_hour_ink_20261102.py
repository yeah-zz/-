# -*- coding: utf-8 -*-
"""Offline preparation for a simple complete composition with per-stroke ink.

The speed factor is process-local; shared calibration and defaults are intact.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from config import ArmConfig,CanvasConfig,ShuimoConfig,PALETTE,validate
from geometry import CanvasMap,check_waypoints
from pigment_station import read_measurements,PigmentStation
from ink_control import InkPolicy,split_long_strokes,build_plan
from run_mountain_balanced import load_reviewed
from run_mountain_v4_robot import safe_transfers
from prepare_full_ink_trial_20261102 import deduplicate,join_existing,order_paths,collinear,length,draw_paths,edges
from bind_full_ink_legacy_20261102 import padded_mapping
import job

LAB=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022')
SOURCE=LAB/'simple_landscape_specialist_20261029'/'round2'
OUT=LAB/'one_hour_ink_trial_20261102'
SAMPLE='G05_03'


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


@contextmanager
def session_speed(factor=40):
    if type(factor) is not int or not 1<=factor<=100:
        raise ValueError('Invalid explicit session speed factor')
    original=ArmConfig.SPEED_FACTOR
    ArmConfig.SPEED_FACTOR=factor
    try:yield
    finally:ArmConfig.SPEED_FACTOR=original


def simplified(p):
    p=np.asarray(p,float)
    q=cv2.approxPolyDP(p.astype(np.float32),.35,False).reshape(-1,2).astype(float)
    if len(q)<2 or not np.allclose(p[[0,-1]],q[[0,-1]],atol=1e-7,rtol=0):
        return collinear(p)
    # RDP's one-sided .35px bound is small relative to the source-pixel spacing.
    return q


def estimate_from_reference():
    ref=LAB/'full_ink_trial_20261102'/'contact_recalibration'/'ink_each_stroke_probe'/'two_line_pressure_probe_job_runs'/'20261102_203413_410807_a8def08b_execution_log.json'
    d=json.loads(ref.read_text('utf-8'))
    diag=Path(d['diagnostic_path']); starts={};dip=[]
    for line in diag.read_text('utf-8').splitlines():
        row=json.loads(line)
        if row['event']=='action_start':starts[row.get('action')]=row['monotonic_s']
        if row['event']=='action_complete' and row.get('action')=='dip':
            if 'dip' in starts:dip.append(row['monotonic_s']-starts['dip'])
    return dict(log=str(ref),total_seconds=d['elapsed_seconds'],dips=d['dips'],
                mean_transfer_and_stroke_s=float(np.mean([r['actual_transfer_and_stroke_s'] for r in d['ink_cycles']])),
                mean_paper_stroke_s=float(np.mean([r['actual_draw_s'] for r in d['ink_cycles']])),
                dip_seconds=dip,reference_global_factor=20,new_global_factor=40,
                note='不同速度下的实际用时不保证按倍率线性变化；需用本次前两笔校验。')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    source=SOURCE/'imagegen_holdouts'/f'{SAMPLE}.png'
    pathfile=SOURCE/'outputs'/SAMPLE/'v4'/'paths.json'
    doc=json.loads(pathfile.read_text('utf-8'))
    raw=[np.array(r['points'],float) for r in doc['paths']]
    chosen=[i for i,r in enumerate(doc['paths']) if r['role']=='long_structure' or length(raw[i])>=12.]
    chosen_raw=[raw[i] for i in chosen]
    records,duplicate=deduplicate(chosen_raw);records,joins=join_existing(records)
    if set(edges(chosen_raw))!=set(edges([r['points'] for r in records])):
        raise AssertionError('Selected source geometry changed during junction routing')
    records=order_paths(records)
    for row in records:
        row['points']=simplified(row['points'])
        row['source_ids']=[chosen[i] for i in row['source_ids']]
    image=Image.open(source).convert('RGB');w,h=image.size
    with session_speed():
        errors=validate(need_stations=True)
        if errors:raise ValueError(errors)
        station=PigmentStation(**read_measurements()['black']).validate()
        pw,ph=CanvasMap((w,h)).canvas_size_mm()
        image_size,mult,offset,mm_scale=padded_mapping(w,h,pw,ph)
        mapper=CanvasMap(image_size)
        policy=InkPolicy(max_strokes_per_load=1,initial_dip_s=station.dwell_s,
            refresh_dip_s=station.dwell_s,drip_s=station.drip_s,fresh_press_factor=1.,
            return_allowance_s=5.,paper_travel_mm_s=50.,draw_mm_s=5.)
        strokes=[]
        for i,row in enumerate(records):
            points=(row['points']*mult+offset).tolist()
            strokes.append(dict(id=f'{SAMPLE}-simple-{i:03d}',points=points,press=[.6]*len(points),
                speed=[20.]*len(points),horizontal_velocity_percent=60,layer=1,pal_index=0,
                tail_mm=0.,lift_mm=4.,source_ids=row['source_ids'],
                stroke_identity_verified=False,direction_verified=False))
        base=split_long_strokes(strokes,mapper.mm_per_px,policy)
        snapshot=dict(user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,rpy=list(ArmConfig.RPY),
            speed_factor=ArmConfig.SPEED_FACTOR,palette=PALETTE)
        common=dict(mode='one-hour-simple-composition',profile='mountain',ink_policy='balanced',
            image=str(source),image_size=list(image_size),canvas_corners=[list(p) for p in CanvasConfig.ROBOT_CORNERS],
            paper_z=ArmConfig.PAPER_Z,air_transfer_velocity_percent=60,pass_=1,
            source_to_padded_scale=mult.tolist(),source_to_padded_offset=offset.tolist(),
            image_fit='contain',margin_mm=10.,simplified_drawing=True,
            physical_brush=dict(press_mm=.6,draw_velocity_percent=20,global_speed_factor=40,
                dip_mm=station.dip_mm,dwell_s=station.dwell_s),
            speed_trial_verified=False,actually_drawn=False,
            ink_execution=dict(mm_per_px=mapper.mm_per_px,policy=asdict(policy),calibration=snapshot))

        def output_job(name,rows):
            steps,stats=build_plan(rows,0,mapper.mm_per_px,policy)
            wp,_=job.steps_to_waypoints(steps,mapper);wp=safe_transfers(wp)
            if not check_waypoints(wp)[0]:raise ValueError('Task outside existing reach')
            if stats['dips']!=len(rows):raise AssertionError('Per-stroke dip missing')
            meta=deepcopy(common);meta.update(strokes=len(rows),length_mm=stats['length_mm'],dips=stats['dips'],ink_plan=stats)
            meta['ink_execution']['base_strokes']=rows
            target=OUT/name;job.save_job(str(target),wp,meta);load_reviewed(str(target))
            return target,stats
        full,stats=output_job('G05_03_simple_full_job.json',base)
        pilot,pilot_stats=output_job('G05_03_first_two_job.json',base[:2])
    selection=dict(sample=SAMPLE,source=str(source),source_paths=str(pathfile),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original_segments=len(raw),selected_source_segments=len(chosen),
        dropped_source_ids=[i for i in range(len(raw)) if i not in chosen],
        selection_rule='All existing long_structure records, plus records with arc length >=12 source pixels',
        all_existing_structure_records_retained=True,full_composition_retained=True,
        fine_texture_reduced=True,exact_duplicate_edges_removed=duplicate,junction_pairs=joins,
        planned_strokes=len(base),planned_dips=stats['dips'],drawing_length_mm=stats['length_mm'],
        estimated_minutes=stats['estimated_total_s']/60,estimate_is_not_measured=True,
        target_minutes=60,shared_speed_factor_unchanged=ArmConfig.SPEED_FACTOR,
        run_speed_factor=40,air_velocity_percent=60,draw_velocity_percent=20,
        pilot_job=str(pilot),full_job=str(full),speed_trial_verified=False)
    save(OUT/'selection_plan.json',selection)
    save(OUT/'job_manifest.json',dict(full_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
        pilot_sha256=hashlib.sha256(pilot.read_bytes()).hexdigest()))
    save(OUT/'reference_timing.json',estimate_from_reference())
    save(OUT/'selected_image_paths.json',dict(width=w,height=h,paths=[dict(points=r['points'].tolist(),source_ids=r['source_ids']) for r in records]))
    image.save(OUT/'source.png')
    planned=draw_paths((h,w),records);planned.save(OUT/'simple_lines.png')
    comparison=Image.new('RGB',(1500,800),'#f5f3eb');draw=ImageDraw.Draw(comparison)
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',24)
    draw.text((20,12),f'G05_03 简稿试画 · {len(base)} 段 / 逐段蘸墨 · 规划约 {stats["estimated_total_s"]/60:.0f} 分钟（待实测）',font=font,fill='#252925')
    draw.text((20,55),'原图：旧批次完整小景',font=font,fill='#252925')
    draw.text((770,55),'计划：保留主线，减少细碎纹理',font=font,fill='#252925')
    comparison.paste(image.resize((710,710)),(20,88));comparison.paste(planned.resize((710,710)),(770,88))
    comparison.save(OUT/'selected_preview.jpg',quality=94)
    (OUT/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>一小时山石简稿试画</title><style>body{{font:18px system-ui;max-width:1300px;margin:24px auto;padding:0 20px;background:#f5f3eb;color:#252925}}img{{width:100%}}li{{margin:12px 0}}a{{color:#17694c}}</style><h1>G05_03 山石简稿 · 一小时目标</h1><p>从前一批30个小景中选择，保留整幅构图；875条原始记录筛选为{len(base)}条实际轨迹段，省去细碎纹理。不是原图全部线条的等量复刻。</p><img src="selected_preview.jpg"><p>已验证下压0.60mm，逐段蘸墨1.20mm、停留3秒。计划将全局倍率20调至40，空行程v=60，行笔v=20。现场速度效果尚需前两笔验证；现有标定、可达范围和转移高度保持。</p><p>规划估时{stats['estimated_total_s']/60:.1f}分钟，是目标估算，不是保证。前两笔结束后应根据实测耗时、纸上墨迹判断是否继续。</p><ul><li><a href="selection_plan.json">筛选规则与规划统计</a></li><li><a href="reference_timing.json">上一轮真实耗时参考</a></li><li><a href="G05_03_simple_full_job.json">完整任务</a></li><li><a href="G05_03_first_two_job.json">前两笔速度验证</a></li><li><a href="prior_30_contact.jpg">此前30个小景总览</a></li></ul><p>整幅尚未开始。使用专用入口运行；共享配置里的默认倍率保持20。</p>''',encoding='utf-8')
    return selection


if __name__=='__main__':
    result=prepare()
    print(json.dumps({k:result[k] for k in ['sample','original_segments','planned_strokes',
        'planned_dips','drawing_length_mm','estimated_minutes','run_speed_factor',
        'air_velocity_percent','draw_velocity_percent','pilot_job','full_job']},ensure_ascii=False,indent=2))
