# -*- coding: utf-8 -*-
"""Two measured-contact pressure probes near the actual incomplete drawing.

Offline builder; uses the existing guarded executor for physical motion.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from config import ArmConfig,CanvasConfig,ShuimoConfig,PALETTE,validate
from geometry import CanvasMap,check_waypoints
from pigment_station import read_measurements,PigmentStation
from ink_control import InkPolicy,refresh_stats,build_plan
from run_mountain_v4_robot import safe_transfers
from run_mountain_balanced import load_reviewed
import job

ROOT=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\full_ink_trial_20261102\contact_recalibration')
OUT=ROOT/'ink_pressure_probe_r2'


def prepare(pressures=(.3,.4),output=OUT,row_offset_mm=0.,max_strokes_per_load=0):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    errors=validate(need_stations=True)
    if errors:raise ValueError(errors)
    station=PigmentStation(**read_measurements()['black']).validate()
    measured=json.loads((ROOT/'measured_contacts.json').read_text('utf-8'))
    if abs(ArmConfig.PAPER_Z-measured['paper_touch_xyz'][2])>1e-8:
        raise ValueError('Paper-contact reference changed')
    pause=json.loads((ROOT/'USER_PAUSED_AFTER_INK_TEST.json').read_text('utf-8'))
    if not pause['stop_acknowledged'] or not pause['idle_observed']:
        raise ValueError('Full drawing stop must be acknowledged')
    location=np.array(pause['observations'][-1]['pose'][:2],float)
    mapper=CanvasMap((1400,1000));inverse=np.linalg.inv(mapper.matrix)
    polygon=np.array(CanvasConfig.ROBOT_CORNERS,np.float32)
    base=[];rows=[]
    for i,press in enumerate(pressures):
        if not 0<press<=ShuimoConfig.PRESS_MAX_MM:raise ValueError('Pressure outside existing limit')
        start=location+np.array([-20.,row_offset_mm+i*5.])
        xy=np.array([start,start+[5.,0.],start+[10.,0.]])
        if any(cv2.pointPolygonTest(polygon,tuple(p),True)<10 for p in xy):
            raise ValueError('Insufficient paper-edge clearance')
        points=cv2.perspectiveTransform(xy.astype(np.float32).reshape(-1,1,2),inverse).reshape(-1,2)
        if not np.allclose(mapper.to_robot(points),xy,atol=.001,rtol=0):
            raise ValueError('Mapping mismatch')
        stroke=dict(id='pressure-probe-%02d-%03d'%(i+1,round(press*100)),points=points.tolist(),
            press=[press]*len(points),speed=[20.]*len(points),layer=1,pal_index=0,
            tail_mm=0.,lift_mm=4.,direction_verified=True)
        base.append(refresh_stats(stroke,mapper.mm_per_px))
        rows.append(dict(number=i+1,pressure_mm=press,xy=xy.tolist(),z=ArmConfig.PAPER_Z+press))
    policy=InkPolicy(initial_dip_s=station.dwell_s,refresh_dip_s=station.dwell_s,
        drip_s=station.drip_s,fresh_press_factor=1.,max_strokes_per_load=max_strokes_per_load)
    steps,stats=build_plan(base,0,mapper.mm_per_px,policy)
    waypoints,_=job.steps_to_waypoints(steps,mapper);waypoints=safe_transfers(waypoints)
    ok,errors=check_waypoints(waypoints)
    if not ok:raise ValueError(errors)
    meta=dict(mode='two-line-pressure-probe',profile='mountain',ink_policy='balanced',
        image_size=[1400,1000],paper_z=ArmConfig.PAPER_Z,
        canvas_corners=[list(p) for p in CanvasConfig.ROBOT_CORNERS],strokes=len(base),
        length_mm=stats['length_mm'],dips=stats['dips'],ink_plan=stats,
        probe_rows=rows,full_artwork_resumed=False,physically_observed_result=None,
        ink_execution=dict(base_strokes=base,mm_per_px=mapper.mm_per_px,policy=asdict(policy),
            calibration=dict(user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,
                rpy=list(ArmConfig.RPY),speed_factor=ArmConfig.SPEED_FACTOR,palette=PALETTE)))
    target=output/'two_line_pressure_probe_job.json';job.save_job(str(target),waypoints,meta)
    load_reviewed(str(target))
    summary=dict(job=str(target),rows=rows,dip_mm=station.dip_mm,dwell_s=station.dwell_s,
        dip_z=station.surface_z+station.dip_mm,paper_z=ArmConfig.PAPER_Z,
        max_strokes_per_load=max_strokes_per_load,planned_dips=stats['dips'],
        robot_connected=False,full_artwork_resumed=False)
    (output/'plan_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    print(json.dumps(prepare(),ensure_ascii=False,indent=2))
