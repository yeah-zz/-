# -*- coding: utf-8 -*-
"""Prepare the user's exact G03_01 complete extraction, retaining all source edges."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from config import ArmConfig,CanvasConfig,PALETTE,validate
from geometry import CanvasMap,check_waypoints
from pigment_station import read_measurements,PigmentStation
from ink_control import InkPolicy,split_long_strokes,build_plan
from run_mountain_balanced import load_reviewed
from run_mountain_v4_robot import safe_transfers
from prepare_one_hour_ink_20261102 import session_speed
from bind_full_ink_legacy_20261102 import padded_mapping
from prepare_full_ink_trial_20261102 import deduplicate,join_existing,order_paths,collinear,draw_paths,edges
import job

LAB=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022')
SOURCE=LAB/'mountain_complete_lines_20261101'/'outputs'/'G03_01'
OUT=LAB/'selected_complete_G03_01_20261102'


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def prepare(output=OUT):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    doc=json.loads((SOURCE/'complete_paths.json').read_text('utf-8'))
    raw=[np.array(r['points'] if isinstance(r,dict) else r,float) for r in doc['paths']]
    records,removed=deduplicate(raw);records,joins=join_existing(records)
    if set(edges(raw))!=set(edges([r['points'] for r in records])):
        raise AssertionError('Complete source edge coverage must be unchanged')
    records=order_paths(records)
    for r in records:r['points']=collinear(r['points'])
    source=Image.open(SOURCE/'input.png').convert('RGB');w,h=source.size
    source.save(output/'source.png')
    line_image=draw_paths((h,w),records);line_image.save(output/'complete_lines.png')
    raw_mask=np.array(draw_paths((h,w),[dict(points=p) for p in raw]))<128
    final_mask=np.array(line_image)<128
    # Antialias darkness at joined endpoints can differ. Source edge union is
    # checked above; use a separate integer raster for exact geometry coverage.
    import cv2
    def raster(ps):
        a=np.zeros((h,w),np.uint8)
        for p in ps:cv2.polylines(a,[np.rint(p).astype(np.int32)],False,1,1,cv2.LINE_8)
        return a
    if not np.array_equal(raster(raw),raster([r['points'] for r in records])):
        raise AssertionError('Complete source raster changed')
    with session_speed(20):
        errors=validate(need_stations=True)
        if errors:raise ValueError(errors)
        station=PigmentStation(**read_measurements()['black']).validate()
        pw,ph=CanvasMap((w,h)).canvas_size_mm()
        image_size,mult,offset,scale=padded_mapping(w,h,pw,ph)
        mapper=CanvasMap(image_size)
        policy=InkPolicy(max_strokes_per_load=station.strokes_per_load,initial_dip_s=station.dwell_s,
            refresh_dip_s=station.dwell_s,drip_s=station.drip_s,fresh_press_factor=1.,
            return_allowance_s=8.,paper_travel_mm_s=35.,draw_mm_s=3.5)
        base=[]
        for i,r in enumerate(records):
            p=(r['points']*mult+offset).tolist()
            base.append(dict(id=f'G03_01-complete-{i:04d}',points=p,press=[.6]*len(p),
                speed=[20.]*len(p),horizontal_velocity_percent=60,layer=1,pal_index=0,
                tail_mm=0.,lift_mm=4.,source_ids=r['source_ids'],stroke_identity_verified=False))
        base=split_long_strokes(base,mapper.mm_per_px,policy)
        common=dict(mode='user-selected-complete-composition',profile='mountain',ink_policy='balanced',
            image=str(output/'source.png'),image_size=list(image_size),paper_z=ArmConfig.PAPER_Z,
            canvas_corners=[list(p) for p in CanvasConfig.ROBOT_CORNERS],air_transfer_velocity_percent=60,
            complete_extraction_retained=True,source_paths_sha256=hashlib.sha256((SOURCE/'complete_paths.json').read_bytes()).hexdigest(),
            actually_drawn=False,speed_trial_verified=False,speed_trial_required=False,
            contact_profile_verified=True,contact_profile=dict(press_mm=.6,draw_velocity_percent=20,global_speed_factor=20),
            ink_execution=dict(mm_per_px=mapper.mm_per_px,policy=asdict(policy),
                calibration=dict(user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,
                    rpy=list(ArmConfig.RPY),speed_factor=20,palette=PALETTE)))
        def export(name,strokes):
            steps,stats=build_plan(strokes,0,mapper.mm_per_px,policy)
            wp,_=job.steps_to_waypoints(steps,mapper);wp=safe_transfers(wp)
            if not check_waypoints(wp)[0]:raise ValueError('Task exceeds existing reach')
            if policy.max_strokes_per_load==1 and stats['dips']!=len(strokes):
                raise AssertionError('Per-segment replenishment missing')
            meta=deepcopy(common);meta.update(strokes=len(strokes),length_mm=stats['length_mm'],dips=stats['dips'],ink_plan=stats)
            meta['ink_execution']['base_strokes']=strokes
            p=output/name;job.save_job(str(p),wp,meta);load_reviewed(str(p))
            return p,stats
        full,stats=export('G03_01_complete_full_job.json',base)
        pilot,_=export('G03_01_complete_first_two_job.json',base[:2])
    save(output/'job_manifest.json',dict(full_file=full.name,pilot_file=pilot.name,speed_factor=20,
        full_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),pilot_sha256=hashlib.sha256(pilot.read_bytes()).hexdigest()))
    summary=dict(sample='G03_01',source=str(SOURCE/'input.png'),source_segments=len(raw),
        planned_segments=len(base),planned_dips=stats['dips'],source_edges_preserved=True,
        max_strokes_per_load=policy.max_strokes_per_load,
        exact_raster_preserved=True,deleted_source_paths=0,duplicate_edges_removed=removed,
        drawing_length_mm=stats['length_mm'],estimated_minutes=stats['estimated_total_s']/60,
        dip_dwell_only_minutes=stats['dips']*station.dwell_s/60,
        dwell_plus_drain_only_minutes=stats['dips']*(station.dwell_s+station.drip_s)/60,
        time_estimate_is_not_measured=True,pressure_mm=.6,dip_mm=station.dip_mm,dwell_s=station.dwell_s,
        global_speed_factor=20,air_velocity_percent=60,draw_velocity_percent=20,shared_speed_factor_unchanged=ArmConfig.SPEED_FACTOR,
        full_job=str(full),pilot_job=str(pilot),robot_connected=False,actually_drawn=False)
    save(output/'plan_summary.json',summary)
    save(output/'complete_image_paths.json',dict(width=w,height=h,paths=[dict(points=r['points'].tolist(),source_ids=r['source_ids']) for r in records]))
    preview=Image.new('RGB',(1500,800),'#f5f3eb');draw=ImageDraw.Draw(preview)
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',24)
    draw.text((20,12),'G03_01 · 你指定的松树、亭子、临水山石 · 完整提取线稿',font=font,fill='#252925')
    draw.text((20,55),'原图',font=font,fill='#252925');draw.text((770,55),'完整线稿：保留全部已有提取线条',font=font,fill='#252925')
    preview.paste(source.resize((710,710)),(20,88));preview.paste(line_image.resize((710,710)),(770,88));preview.save(output/'complete_preview.jpg',quality=94)
    (output/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>G03_01 完整构图</title><style>body{{font:18px system-ui;max-width:1350px;margin:24px auto;padding:0 20px;background:#f5f3eb;color:#252925}}img{{width:100%}}a{{color:#17694c}}</style><h1>G03_01 · 指定完整构图</h1><img src="complete_preview.jpg"><p>对应你截图第三列的完整提取。2,103条输入记录整理为{len(base)}段，全部原有唯一边和整数像素栅格均保留，没有筛掉短线。完整保留提取结果不代表模型已准确找出原画全部笔触。</p><p>下压0.60mm、蘸入{station.dip_mm:.2f}mm、停留{station.dwell_s:.1f}秒，每{policy.max_strokes_per_load}段补墨一次（耗墨或时长达到门槛时提前补墨）。沿用已验证的接触设置：全局20、行笔v=20；空行程v=60。</p><p>仅蘸墨停留就需要{summary['dip_dwell_only_minutes']:.1f}分钟；当前时间模型估计约{summary['estimated_minutes']:.0f}分钟，非实测。用户已明确完整优先、不限制时间。当前执行状态以任务的 execution_log.json 为准。</p><p><a href="plan_summary.json">规划统计</a> · <a href="complete_image_paths.json">完整图像轨迹</a></p>''',encoding='utf-8')
    return summary


if __name__=='__main__':
    print(json.dumps(prepare(),ensure_ascii=False,indent=2))
