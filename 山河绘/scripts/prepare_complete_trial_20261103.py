# -*- coding: utf-8 -*-
"""Offline full G03_01 trial with task-scoped measured contacts and ink limits."""
import sys,json,hashlib,math
from pathlib import Path
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from config import ArmConfig,CanvasConfig,PALETTE
from geometry import CanvasMap,check_waypoints
from pigment_station import PigmentStation,read_measurements,set_measurements_override
import pigment_station
from ink_control import InkPolicy,refresh_stats,build_plan
from shuimo import press_to_width_mm
from run_mountain_balanced import load_reviewed
from run_mountain_v4_robot import safe_transfers
from bind_full_ink_legacy_20261102 import padded_mapping
from run_brush_pilot_contact_20261103 import contact_scope,CONTACT,ROOT as CONTACT_ROOT
import job

ROOT=LAB/'complete_G03_01_trial_20261103'
CURVES=LAB/'stroke_unit_model_20261102'/'outputs'/'G03_01'/'curves.json'
SOURCE=CURVES.with_name('source.png')
def read(p):return json.loads(Path(p).read_text('utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@contextmanager
def full_scope():
    with contact_scope():
        old=deepcopy(pigment_station._measurements_override);palette=deepcopy(PALETTE)
        m=read_measurements();m['black'].update(dwell_s=2.,strokes_per_load=3,level_compensation_enabled=False)
        set_measurements_override(m);PALETTE[0]['dip_time_s']=2.
        try:yield
        finally:PALETTE[:]=palette;pigment_station._measurements_override=old

def split_length(points,maximum):
    """Partition a polyline by arclength, with shared endpoints only, no tail."""
    p=np.asarray(points,float)
    if len(p)<2 or maximum<=0:raise ValueError('Invalid split request')
    pieces=[];current=[p[0].copy()];remaining=maximum
    for endpoint in p[1:]:
        endpoint=endpoint.copy()
        while True:
            distance=float(np.linalg.norm(endpoint-current[-1]))
            if distance<1e-9:break
            if distance<=remaining+1e-9:
                current.append(endpoint);remaining-=distance;break
            cut=current[-1]+(endpoint-current[-1])*(remaining/distance)
            if remaining>1e-8:current.append(cut)
            if len(current)>=2:pieces.append(np.array(current))
            current=[cut.copy()];remaining=maximum
        if remaining<1e-8:
            if len(current)>=2:pieces.append(np.array(current))
            current=[endpoint.copy()];remaining=maximum
    if len(current)>=2:pieces.append(np.array(current))
    return pieces

def prepare():
    ROOT.mkdir(exist_ok=True)
    if list(ROOT.glob('*_job_runs/*execution_log.json')):raise ValueError('Execution exists; never regenerate over a used sheet')
    previous=read(CONTACT_ROOT/'manifest.json')
    assert all(sha(r['path'])==r['sha256'] for r in previous['shared_files'])
    frozen=read(LAB/'stroke_unit_model_20261102'/'frozen_experiment.json')
    assert all(sha(r['path'])==r['sha256'] for r in frozen['files'])
    curves=read(CURVES);rgb=Image.open(SOURCE).convert('RGB');width,height=rgb.size
    with full_scope():
        station=PigmentStation(**read_measurements()['black']).validate()
        pw,ph=CanvasMap((width,height)).canvas_size_mm()
        size,mult,offset,scale=padded_mapping(width,height,pw,ph)
        mapper=CanvasMap(size);base=[];joins=[]
        for i,r in enumerate(curves['paths']):
            p=np.asarray(r['points'],float)*mult+offset
            # Collinear pixels may be omitted without geometric fitting or smoothing.
            if len(p)>2:
                keep=[0]
                for k in range(1,len(p)-1):
                    a=p[k]-p[keep[-1]];b=p[k+1]-p[k]
                    if abs(np.cross(a,b))>1e-7 or a@b<0:keep.append(k)
                keep.append(len(p)-1);p=p[keep]
            original_length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
            parts=split_length(p,19.8/mapper.mm_per_px)
            assert abs(sum(float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()) for q in parts)-original_length)<1e-6
            for j,q in enumerate(parts):
                s=dict(id=f'G03-learned-{i:04d}-{j:02d}',name=f'G03-{i:04d}-{j:02d}',points=q.tolist(),
                    press=[.6]*len(q),speed=[20.]*len(q),horizontal_velocity_percent=60,
                    layer=1,pal_index=0,tail_mm=0.,lift_mm=4.,source_curve_id=i,
                    source_ids=r['source_ids'],short_unit_pending=r['short_unit_pending'],
                    direction_verified=False,stroke_identity_verified=False,segments=len(parts),segment=j)
                base.append(refresh_stats(s,mapper.mm_per_px))
            joins.append(dict(source_curve_id=i,parts=len(parts),source_length_mm=original_length*mapper.mm_per_px))
        per_mm=press_to_width_mm(.6)/.9
        policy=InkPolicy(capacity_eq_mm=25+20*per_mm,reserve_eq_mm=25,max_segment_eq_mm=20*per_mm,
            max_strokes_per_load=3,initial_dip_s=2.,refresh_dip_s=2.,drip_s=station.drip_s,fresh_press_factor=1.)
        steps,stats=build_plan(base,0,mapper.mm_per_px,policy)
        waypoints,_=job.steps_to_waypoints(steps,mapper);waypoints=safe_transfers(waypoints)
        if not check_waypoints(waypoints)[0]:raise ValueError('Static reach failure')
        polygon=np.asarray(CanvasConfig.ROBOT_CORNERS,np.float32)
        for s in base:
            if any(cv2.pointPolygonTest(polygon,tuple(p),True)<9 for p in mapper.to_robot(s['points'])):raise ValueError('Paper edge clearance')
        meta=dict(mode='complete-learned-curves-trial',profile='mountain',ink_policy='balanced',
            image_size=list(size),paper_z=ArmConfig.PAPER_Z,canvas_corners=[list(p) for p in CanvasConfig.ROBOT_CORNERS],
            strokes=len(base),length_mm=stats['length_mm'],dips=stats['dips'],air_transfer_velocity_percent=60,
            actually_drawn=False,contact_measurements=CONTACT,site_confirmed_for_new_sheet=False,
            physical_parameters_unvalidated_after_dwell_change=True,quality_gate_after_strokes=6,
            source_curves_sha256=sha(CURVES),source_image_sha256=sha(SOURCE),
            preview_is_centerline_plan_only=True,all_remaining_short_details_preserved=True,
            ink_execution=dict(base_strokes=base,mm_per_px=mapper.mm_per_px,policy=asdict(policy),
                calibration=dict(user=2,tool=0,rpy=CONTACT['rpy'],speed_factor=20,palette=deepcopy(PALETTE))))
        target=ROOT/'G03_01_complete_job.json';job.save_job(str(target),waypoints,meta);load_reviewed(str(target))
        save(ROOT/'task_calibration.json',dict(contacts=CONTACT,black=read_measurements()['black'],
            paper_z=ArmConfig.PAPER_Z,canvas_corners=CanvasConfig.ROBOT_CORNERS,
            single_point_contact_only=True,other_color_stations_not_validated=True))
    # Real comparison timing includes all observed transfers and three dip cycles.
    comparison=CONTACT_ROOT.parent/'reload_each_stroke_comparison'/'completed.json'
    comp=read(read(comparison)['execution_log']);refill=comp['elapsed_seconds']/3
    estimate_by_actual_overhead=(stats['dips']*refill+stats['strokes']*2)/60
    summary=dict(image='G03_01 松树、亭子与临水山石完整构图',source_curve_units=len(curves['paths']),
        job_stroke_segments=len(base),added_length_limit_splits=len(base)-len(curves['paths']),
        length_mm=stats['length_mm'],planned_dips=stats['dips'],
        source_short_units_pending=sum(r['short_unit_pending'] for r in curves['paths']),
        estimated_minutes_heuristic=stats['estimated_total_s']/60,
        approximate_minutes_using_recent_dip_cycle=estimate_by_actual_overhead,
        runtime_can_refill_earlier=True,time_estimates_not_guaranteed=True,
        draw_z=-22.6686,dip_z=-20.9935,dip_mm=1.2,dwell_s=2.,press_mm=.6,draw_v_percent=20,global_speed_factor=20,
        max_segments_per_load=3,heuristic_length_per_load_mm=20,
        liquid_level_compensation=False,reason='No fresh measurement of liquid depletion; pause for remeasurement instead of automatic deeper dipping',
        quality_gate_after_strokes=6,no_stroke_dropped_after_relation_stage=True,
        supplied_photo_upper_group='per-line-reload still visibly dry; no improvement claim',
        ready_for_robot=False,requires_new_blank_sheet=True,robot_connected=False,job=str(target))
    save(ROOT/'plan_summary.json',summary);save(ROOT/'split_records.json',joins)
    mask=Image.new('RGB',size,'white');d=ImageDraw.Draw(mask)
    for s in base:d.line([tuple(p) for p in s['points']],fill='black',width=1)
    mask.save(ROOT/'complete_centerlines.png');rgb.save(ROOT/'source.png')
    board=Image.new('RGB',(1560,850),'#f5f3eb');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    d.text((20,12),'完整 G03_01 · 修改后的中心线计划（不是纸面实画）',font=font,fill='black')
    d.text((20,50),'原图',font=font,fill='black');d.text((800,50),'完整轨迹：已分组、长线优先，保留细节',font=font,fill='black')
    rgb.thumbnail((740,740));board.paste(rgb,(20,85));mask.thumbnail((740,740));board.paste(mask,(800,170));board.save(ROOT/'preview.jpg',quality=94)
    save(ROOT/'manifest.json',dict(job=str(target),job_sha256=sha(target),curves=str(CURVES),curves_sha256=sha(CURVES),
        contact_wrapper_sha256=sha(Path(__file__).with_name('run_brush_pilot_contact_20261103.py')),
        source_files=[dict(path=str(p),sha256=sha(p)) for p in [Path(__file__),Path(__file__).with_name('run_complete_trial_20261103.py')]],
        shared_files=previous['shared_files'],contacts=CONTACT,total_strokes=len(base),quality_gate_after_strokes=6))
    (ROOT/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>完整G03试画准备</title><style>body{{font:18px system-ui;max-width:1450px;margin:25px auto;padding:20px;background:#f5f3eb}}img{{width:100%}}p{{line-height:1.6}}</style><h1>完整G03_01 · 待换空白纸后启动</h1><img src="preview.jpg"><p>保留1345条关系分组候选，包括714条短单元；按约20mm最大行笔长度分为{len(base)}段。不是新一次像素模型训练，也没有把假设墨带重叠自动当错线删除。1345条仍非画家真实笔数，实画积墨和漏线未解决。</p><p>本次仅将蘸墨停留1秒改2秒；蘸入1.20mm、下压0.60mm、行笔v20/全局20不变。每最多3段或约20mm启发式用墨预算补墨，时间门槛可提前补墨。约{stats['dips']}次补墨；时间模型约{stats['estimated_total_s']/3600:.1f}小时，按最近实测蘸墨循环粗估约{estimate_by_actual_overhead/60:.1f}小时，均不保证。</p><p>前6段后自动抬笔暂停检查，续画从第7段开始，不重画前6段。关闭未经本次实测的液面自动加深补偿；后续液面变化时需暂停重测。长时间绘画需现场看护。当前没有启动、没有连接机械臂。</p><p>纸张必须换成空白纸，保持原位置和垫层厚度；新纸未确认前不执行。单点触纸高度不能保证整张纸平整。当前结果是整幅试画实验，不能保证每条细线连续。</p><a href="plan_summary.json">完整参数与估时</a> · <a href="manifest.json">冻结文件</a>''',encoding='utf-8')
    assert all(sha(r['path'])==r['sha256'] for r in previous['shared_files'])
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':prepare()
