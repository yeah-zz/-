# -*- coding: utf-8 -*-
"""Offline single-line probe after the user's measured paper/ink contacts.

Uses the existing guarded balanced executor; no networking or motion here.
"""
from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from config import ArmConfig,CanvasConfig,ShuimoConfig,PALETTE,validate
from geometry import CanvasMap,check_waypoints
from pigment_station import read_measurements,PigmentStation
from ink_control import InkPolicy,refresh_stats,build_plan,split_long_strokes
from run_mountain_balanced import load_reviewed
from run_mountain_v4_robot import safe_transfers
import job

OUT=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\full_ink_trial_20261102\contact_recalibration')


def prepare(output=OUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    errors=validate(need_stations=True)
    if errors:raise ValueError(errors)
    measured=json.loads((OUT/'measured_contacts.json').read_text('utf-8'))
    if abs(ArmConfig.PAPER_Z-measured['paper_touch_xyz'][2])>1e-8:
        raise ValueError('Measured paper contact does not match configuration')
    station=PigmentStation(**read_measurements()['black']).validate()
    if not np.allclose([station.x,station.y,station.surface_z],measured['ink_touch_xyz'],rtol=0,atol=1e-8):
        raise ValueError('Measured ink contact does not match configuration')
    press=measured['drawing_press_mm']
    if not 0<press<=ShuimoConfig.PRESS_MAX_MM:raise ValueError('Invalid pressure')
    mapper=CanvasMap((1400,1000))
    # The full drawing is centered in an approximately 187-mm-wide square.
    # X=20..30 lies in the spare side margin; it is near the measured paper point.
    xy=np.array([[20.,59.8733],[25.,59.8733],[30.,59.8733]])
    polygon=np.array(CanvasConfig.ROBOT_CORNERS,np.float32)
    if any(cv2.pointPolygonTest(polygon,tuple(p),True)<10 for p in xy):
        raise ValueError('Probe lacks paper-edge clearance')
    inverse=np.linalg.inv(mapper.matrix)
    points=cv2.perspectiveTransform(xy.astype(np.float32).reshape(-1,1,2),inverse).reshape(-1,2)
    if not np.allclose(mapper.to_robot(points),xy,rtol=0,atol=.001):
        raise ValueError('Probe mapping inconsistency')
    stroke=dict(id='measured-contact-10mm-020',points=points.tolist(),press=[press]*len(points),
        speed=[20.]*len(points),layer=1,pal_index=0,tail_mm=0.,lift_mm=4.,
        source='2026-11-02 user measured contact verification',direction_verified=True)
    stroke=refresh_stats(stroke,mapper.mm_per_px)
    policy=InkPolicy(initial_dip_s=station.dwell_s,refresh_dip_s=station.dwell_s,
                     drip_s=station.drip_s,fresh_press_factor=1.,
                     max_strokes_per_load=measured.get('max_strokes_per_load',0))
    steps,stats=build_plan([stroke],0,mapper.mm_per_px,policy)
    waypoints,_=job.steps_to_waypoints(steps,mapper)
    waypoints=safe_transfers(waypoints)
    ok,errors=check_waypoints(waypoints)
    if not ok:raise ValueError(errors)
    meta=dict(mode='measured-contact-probe',profile='mountain',ink_policy='balanced',
        image_size=[1400,1000],paper_z=ArmConfig.PAPER_Z,
        canvas_corners=[list(p) for p in CanvasConfig.ROBOT_CORNERS],strokes=1,
        length_mm=stats['length_mm'],dips=stats['dips'],ink_plan=stats,
        probe_robot_xy=xy.tolist(),physically_observed_result=None,actually_drawn=False,
        full_artwork_resume_authorized_by_probe=False,
        ink_execution=dict(base_strokes=[stroke],mm_per_px=mapper.mm_per_px,policy=asdict(policy),
            calibration=dict(user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,rpy=list(ArmConfig.RPY),
                             speed_factor=ArmConfig.SPEED_FACTOR,palette=PALETTE)))
    target=output/'contact_probe_10mm_job.json'
    job.save_job(str(target),waypoints,meta)
    load_reviewed(str(target))
    # This does not execute or clear the user's paused full artwork.
    summary=dict(probe=str(target),paper_touch_z=ArmConfig.PAPER_Z,
        target_paper_z=ArmConfig.PAPER_Z+press,ink_touch_z=station.surface_z,
        target_dip_z=station.surface_z+station.dip_mm,dwell_s=station.dwell_s,
        xy=xy.tolist(),pressure_mm=press,robot_connected=False,full_artwork_resumed=False)
    (output/'probe_plan.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def rebuild_full(output=OUT/'full_artwork'):
    """Regenerate the same full composition only after the user's ink-line check."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    measured=json.loads((OUT/'measured_contacts.json').read_text('utf-8'))
    if measured.get('physical_trial_verified') is not True:
        raise ValueError('Full artwork requires the operator to verify real ink and a continuous probe line')
    original=OUT.parent/'legacy_calibration'/'N02_full_black_legacy_job.json'
    old=json.loads(original.read_text('utf-8'))
    meta=deepcopy(old['meta'])
    mapper=CanvasMap(meta['image_size'])
    if not np.allclose(meta['canvas_corners'],CanvasConfig.ROBOT_CORNERS,rtol=0,atol=1e-8):
        raise ValueError('Paper XY geometry changed; regenerate image layout')
    station=PigmentStation(**read_measurements()['black']).validate()
    if abs(ArmConfig.PAPER_Z-measured['paper_touch_xyz'][2])>1e-8:
        raise ValueError('Paper Z differs from the measured probe')
    press=measured['drawing_press_mm']
    base=deepcopy(meta['ink_execution']['base_strokes'])
    for stroke in base:
        stroke['press']=[press]*len(stroke['points'])
        stroke['speed']=[20.]*len(stroke['points'])
        stroke['tail_mm']=0.
    policy=InkPolicy(initial_dip_s=station.dwell_s,refresh_dip_s=station.dwell_s,
                     drip_s=station.drip_s,fresh_press_factor=1.,
                     max_strokes_per_load=measured.get('max_strokes_per_load',0))
    base=split_long_strokes(base,mapper.mm_per_px,policy)
    steps,stats=build_plan(base,0,mapper.mm_per_px,policy)
    waypoints,_=job.steps_to_waypoints(steps,mapper)
    waypoints=safe_transfers(waypoints)
    if not check_waypoints(waypoints)[0]:raise ValueError('Full task outside reach')
    meta.update(mode='full-black-measured-contact-trial',paper_z=ArmConfig.PAPER_Z,
        strokes=len(base),length_mm=stats['length_mm'],dips=stats['dips'],ink_plan=stats,
        calibration_reuse=dict(user_requested=True,mode='updated_user_measured_contacts',
            paper_xy_reused=True,paper_z_new=True,black_xy_and_surface_z_new=True,
            cup_rim_remeasured=False,source_record=str(OUT/'measured_contacts.json')),
        physical_brush=dict(profile='user_verified_continuous_probe',press_mm=press,
            velocity_percent=20,current_paper_test=True,width_mm=None),
        verification=dict(real_ink_observed_by_user=True,continuous_probe_observed_by_user=True,
            trial_length_mm=10.,whole_paper_planarity_verified=False),
        actually_drawn=False)
    meta['ink_execution']=dict(base_strokes=base,mm_per_px=mapper.mm_per_px,
        policy=asdict(policy),calibration=dict(user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,
            rpy=list(ArmConfig.RPY),speed_factor=ArmConfig.SPEED_FACTOR,palette=PALETTE))
    target=output/'N02_full_black_measured_job.json'
    job.save_job(str(target),waypoints,meta)
    load_reviewed(str(target))
    summary=dict(job=str(target),strokes=len(base),waypoints=len(waypoints),
        pressure_mm=press,paper_z=ArmConfig.PAPER_Z,dip_mm=station.dip_mm,
        dip_dwell_s=station.dwell_s,planned_dips=stats['dips'],
        estimated_minutes=stats['estimated_total_s']/60,source_geometry_unchanged=True,
        operator_probe_verified=True,actually_drawn=False)
    (output/'plan_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    print(json.dumps(prepare(),ensure_ascii=False,indent=2))
