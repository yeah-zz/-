# -*- coding: utf-8 -*-
"""Offline builder only: four separately reviewed three-stroke pilot groups."""
import sys,json,hashlib
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from config import ArmConfig,CanvasConfig,PALETTE,validate
from geometry import CanvasMap,check_waypoints
from ink_control import InkPolicy,refresh_stats,build_plan
from pigment_station import PigmentStation,read_measurements
from run_mountain_v4_robot import safe_transfers
from run_mountain_balanced import load_reviewed
from PIL import Image,ImageDraw,ImageFont
import job

ROOT=LAB/'brush_execution_model_20261103'/'pilot12'
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare():
    ROOT.mkdir(exist_ok=True)
    config=Path(__file__).resolve().parents[1]/'config.py'
    pigments=config.with_name('pigment_measurements.json')
    initial=[sha(p) for p in [config,pigments]]
    if validate(need_stations=True):raise ValueError(validate(need_stations=True))
    if (ArmConfig.USER_INDEX,ArmConfig.TOOL_INDEX)!=(2,0) or not np.allclose(ArmConfig.RPY,[-2.2793,-.7103,.2786],atol=1e-8):raise ValueError('Frame changed')
    if abs(ArmConfig.PAPER_Z+24.7818)>1e-8 or ArmConfig.SPEED_FACTOR!=20:raise ValueError('Contact/global speed changed')
    station=PigmentStation(**read_measurements()['black']).validate()
    if not np.allclose([station.x,station.y,station.surface_z,station.dip_mm,station.dwell_s],[-55.1854,85.0436,-23.5006,1.2,1.]):raise ValueError('Ink station changed')
    mapper=CanvasMap((1400,1000));inverse=np.linalg.inv(mapper.matrix)
    definitions=[('A_baseline',.6,20,110,75,'straight'),('B_lighter',.4,20,160,75,'straight'),
                 ('C_faster',.6,30,110,120,'straight'),('D_curve',.6,20,160,120,'bend')]
    preview=Image.new('RGB',(1200,1000),'#f5f3eb');d=ImageDraw.Draw(preview)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);rows=[]
    d.text((25,15),'12笔实测计划 · 4组分别执行 · 尚未运动',font=font,fill='black')
    for group,press,speed,x,y,shape in definitions:
        base=[];physical=[]
        for i in range(3):
            if shape=='straight':local=np.array([[0.,0.],[5,0],[10,0],[15,0],[20,0]])
            else:
                t=np.linspace(0,np.pi/2,21);local=np.column_stack([np.sin(t),1-np.cos(t)])*(20/(np.pi/2))
            xy=local+[x,y+i*10]
            if any(cv2.pointPolygonTest(np.array(CanvasConfig.ROBOT_CORNERS,np.float32),tuple(p),True)<20 for p in xy):raise ValueError('Paper edge clearance')
            pp=cv2.perspectiveTransform(xy.astype(np.float32).reshape(-1,1,2),inverse).reshape(-1,2)
            if not np.allclose(mapper.to_robot(pp),xy,atol=.001):raise ValueError('Mapping mismatch')
            s=dict(id=f'{group}_{i+1}',name=f'{group}_{i+1}',points=pp.tolist(),press=[press]*len(pp),speed=[float(speed)]*len(pp),
                horizontal_velocity_percent=60,layer=1,pal_index=0,tail_mm=0.,lift_mm=4.,direction_verified=True)
            base.append(refresh_stats(s,mapper.mm_per_px));physical.append(dict(id=s['id'],xy_mm=xy.tolist(),length_mm=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())))
        policy=InkPolicy(max_strokes_per_load=3,initial_dip_s=1.,refresh_dip_s=1.,drip_s=station.drip_s,fresh_press_factor=1.)
        steps,stats=build_plan(base,0,mapper.mm_per_px,policy)
        if stats['dips']!=1:raise ValueError('Pilot needs exactly one planned load per group')
        waypoints,_=job.steps_to_waypoints(steps,mapper);waypoints=safe_transfers(waypoints)
        if not check_waypoints(waypoints)[0]:raise ValueError('Outside static reach')
        meta=dict(mode='three-stroke-brush-measurement-pilot',profile='mountain',ink_policy='balanced',
            image_size=[1400,1000],paper_z=ArmConfig.PAPER_Z,canvas_corners=[list(p) for p in CanvasConfig.ROBOT_CORNERS],
            strokes=3,air_transfer_velocity_percent=60,group=group,actual_result=None,actually_drawn=False,
            operator_site_confirmation_required=True,robot_state_and_pose_not_read=True,
            full_artwork_resumed=False,controller_velocity_is_percent_not_mm_s=True,
            ink_execution=dict(base_strokes=base,mm_per_px=mapper.mm_per_px,policy=asdict(policy),
                calibration=dict(user=2,tool=0,rpy=list(ArmConfig.RPY),speed_factor=20,palette=PALETTE)))
        target=ROOT/(group+'_job.json');job.save_job(str(target),waypoints,meta);load_reviewed(str(target))
        rows.append(dict(group=group,job=str(target),sha256=sha(target),press_mm=press,
            draw_velocity_percent=speed,global_speed_factor=20,planned_dips=1,physical_paths=physical,
            planned_length_total_mm=sum(p['length_mm'] for p in physical),requires_separate_observation=True))
        j=len(rows)-1;ox=20+(j%2)*590;oy=65+(j//2)*405
        d.text((ox,oy),f'{group} · 下压{press:.2f}mm · v={speed}',font=font,fill='black')
        d.text((ox,oy+30),'蘸一次墨 → 第1/2/3笔（间距10mm）',font=font,fill='black')
        for i,p in enumerate(physical):
            q=(np.array(p['xy_mm'])-[x,y])*11+[ox+90,oy+90]
            d.line([tuple(a) for a in q],fill='#236d63',width=2)
            d.text((ox+15,oy+80+i*110),str(i+1),font=font,fill='black')
    preview.save(ROOT/'layout.png')
    assert initial==[sha(p) for p in [config,pigments]]
    save(ROOT/'plan.json',dict(groups=rows,calibration_reused_from='2026-11-02',
        current_state_not_verified=True,contact_z=-24.7818,ink_dip_z=-22.3006,ink_dwell_s=1.,
        group_order='A then B then C then D; observe and explicitly approve each next group',
        start_first_group_only=True,ready_for_robot=False,robot_connected=False,
        reason_not_ready='Await current site/paper/liquid-level confirmation and read-only robot preflight',
        configuration_unchanged=True,config_sha256=initial[0],pigments_sha256=initial[1]))
    (ROOT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>12笔小样计划</title><style>body{font:18px system-ui;max-width:1200px;margin:25px auto;padding:20px;background:#f5f3eb}img{width:100%}</style><h1>12笔小样 · 当前未启动</h1><p>第一组下压0.60mm、v=20，沿用昨日基准；第二组只改为0.40mm；第三组只改v=30；第四组只换弯线。v为控制器百分比，不是mm/s。全局倍率20、蘸入1.20mm、停留1秒。每组独立开始，先仅批准并执行第一组三笔，现场观察后再决定下一组。</p><p>纸面坐标范围约X110–180、Y75–153mm，User2/Tool0。需要空白纸、毛笔夹持与垫层不变；墨水液面可能隔夜变化，位置相同不代表液面高度相同。当前没有读取机械臂状态，不代表姿态或可达性已现场通过。</p><img src="layout.png"><p>绘画开始/结束实际时间与路径长度另作记录，不能将v=20填成20mm/s。3笔组的实时补墨保护仍保留；如果发生提前补墨，应记录并把该组标为偏离计划，不当作连续三笔耗墨样本。</p><a href="plan.json">详细参数与校验</a>''',encoding='utf-8')
    print(json.dumps(dict(groups=4,strokes=12,first_group_only=rows[0],robot_connected=False,ready_for_robot=False),ensure_ascii=False,indent=2))

if __name__=='__main__':prepare()
