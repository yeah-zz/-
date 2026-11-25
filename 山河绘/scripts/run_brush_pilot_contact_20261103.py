# -*- coding: utf-8 -*-
"""Task-scoped contact calibration and existing guarded three-stroke runner.

Default is offline binding. Execution requires an explicit phase and RUN.
Shared calibration files and all previous paused artwork remain unchanged.
"""
import sys,json,hashlib,argparse
from pathlib import Path
from contextlib import contextmanager
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from config import ArmConfig,PALETTE,CanvasConfig
import pigment_station
from geometry import CanvasMap,check_waypoints
from ink_control import InkPolicy,build_plan
from run_mountain_v4_robot import safe_transfers
from run_mountain_balanced import load_reviewed,check_robot_readonly,execute_reviewed_job
import job

ROOT=LAB/'brush_execution_model_20261103'/'pilot12'/'new_contacts'
SOURCE=ROOT.parent/'A_baseline_job.json'
CONTACT=dict(user=2,tool=0,rpy=[-2.3515,-.9198,.2575],
    ink_touch_xyz=[-76.3250,114.1259,-22.1935],paper_touch_xyz=[-5.6110,114.1259,-23.2686],
    press_mm=.6,dip_mm=1.2,dwell_s=1.,draw_velocity_percent=20,global_speed_factor=20)
PHOTOS=[LAB.parent/'CLIPascene_20261021'/'temp'/f for f in [
    'codex-clipboard-da6c26bb-74d0-464c-bbf8-10afd6452f3c.png',
    'codex-clipboard-ac637228-3447-47bd-a64f-1c17882050ce.png']]
SHARED=[Path(__file__).resolve().parents[1]/p for p in ['config.py','pigment_measurements.json']]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

@contextmanager
def contact_scope():
    old=(ArmConfig.USER_INDEX,ArmConfig.TOOL_INDEX,ArmConfig.RPY,ArmConfig.PAPER_Z,ArmConfig.SPEED_FACTOR)
    palette=deepcopy(PALETTE);override=deepcopy(pigment_station._measurements_override)
    measurement=pigment_station.read_measurements()
    ink=measurement['black'];ink.update(x=CONTACT['ink_touch_xyz'][0],y=CONTACT['ink_touch_xyz'][1],
        surface_z=CONTACT['ink_touch_xyz'][2],dip_mm=1.2,dwell_s=1.,strokes_per_load=3)
    measurement.update(user=2,tool=0,rpy=CONTACT['rpy'])
    ArmConfig.USER_INDEX=2;ArmConfig.TOOL_INDEX=0;ArmConfig.RPY=tuple(CONTACT['rpy'])
    ArmConfig.PAPER_Z=CONTACT['paper_touch_xyz'][2];ArmConfig.SPEED_FACTOR=20
    PALETTE[0].update(dip=(ink['x'],ink['y']),surface_z=ink['surface_z'],dip_mm=1.2,dip_time_s=1.)
    pigment_station.set_measurements_override(measurement)
    try:yield
    finally:
        ArmConfig.USER_INDEX,ArmConfig.TOOL_INDEX,ArmConfig.RPY,ArmConfig.PAPER_Z,ArmConfig.SPEED_FACTOR=old
        PALETTE[:]=palette;pigment_station._measurements_override=override

def bind():
    ROOT.mkdir(exist_ok=True)
    shared=[sha(p) for p in SHARED]
    source=json.loads(SOURCE.read_text('utf-8'));data=deepcopy(source);meta=data['meta']
    strokes=meta['ink_execution']['base_strokes']
    if len(strokes)!=3 or len({s['id'] for s in strokes})!=3:raise ValueError('Only the approved first three strokes are allowed')
    for s in strokes:
        if set(s['press'])!={.6} or set(s['speed'])!={20.} or s['tail_mm']!=0:raise ValueError('Approved baseline changed')
    with contact_scope():
        mapper=CanvasMap(meta['image_size']);policy=InkPolicy(**meta['ink_execution']['policy'])
        if policy.max_strokes_per_load!=3 or policy.initial_dip_s!=1 or policy.refresh_dip_s!=1 or policy.fresh_press_factor!=1:raise ValueError('Loading policy changed')
        station=pigment_station.PigmentStation(**pigment_station.read_measurements()['black']).validate()
        steps,stats=build_plan(strokes,0,mapper.mm_per_px,policy)
        if stats['dips']!=1:raise ValueError('Expected one planned dip')
        waypoints,_=job.steps_to_waypoints(steps,mapper);waypoints=safe_transfers(waypoints)
        if not check_waypoints(waypoints)[0]:raise ValueError('Static reach check failed')
        meta.update(paper_z=ArmConfig.PAPER_Z,contact_measurements=CONTACT,mode='new-contact-first-group-only',
            current_pose_checked=False,actually_drawn=False,full_artwork_resumed=False)
        meta['ink_execution']['calibration']=dict(user=2,tool=0,rpy=CONTACT['rpy'],speed_factor=20,palette=deepcopy(PALETTE))
        target=ROOT/'A_baseline_new_contacts_job.json'
        if list(ROOT.glob('*_job_runs/*execution_log.json')):raise ValueError('Run records exist. Do not automatically redraw the group.')
        job.save_job(str(target),waypoints,meta);load_reviewed(str(target))
        save(ROOT/'calibration_snapshot.json',dict(contacts=CONTACT,pigment_measurements=pigment_station.read_measurements(),
            canvas_corners=CanvasConfig.ROBOT_CORNERS,single_point_paper_height=True,paper_plane_not_measured=True,
            paper_touch_xy_not_used_to_change_canvas=True,black_rim_reference_not_remeasured=True,
            physical_speed_mm_s_not_assumed=True,photos=[dict(path=str(p),sha256=sha(p)) for p in PHOTOS]))
    save(ROOT/'manifest.json',dict(job=str(target),job_sha256=sha(target),source_job=str(SOURCE),source_sha256=sha(SOURCE),
        contacts=CONTACT,script_sha256=sha(__file__),shared_files=[dict(path=str(p),sha256=h) for p,h in zip(SHARED,shared)],
        paper_z=-23.2686,draw_z=-22.6686,ink_surface_z=-22.1935,dip_z=-20.9935,
        strokes=3,planned_dips=1,robot_connected=False,requires_readonly_preflight=True))
    assert shared==[sha(p) for p in SHARED]
    print('OFFLINE BOUND: 3 strokes; draw Z=-22.6686; dip Z=-20.9935; dwell=1s; shared configuration unchanged.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['bind','check','check-robot','execute'],default='bind')
    a=p.parse_args()
    if a.phase=='bind':bind();return 0
    manifest=json.loads((ROOT/'manifest.json').read_text('utf-8'));target=Path(manifest['job'])
    if manifest['job_sha256']!=sha(target) or manifest['script_sha256']!=sha(__file__) or manifest['contacts']!=CONTACT:raise ValueError('Bound task changed')
    if any(sha(r['path'])!=r['sha256'] for r in manifest['shared_files']):raise ValueError('Shared settings changed; rebind explicitly')
    with contact_scope():
        load_reviewed(str(target))
        if a.phase=='check':print('Offline check passed; no robot connection.');return 0
        if a.phase=='check-robot':return check_robot_readonly(str(target))
        if list(ROOT.glob('*_job_runs/*execution_log.json')):raise ValueError('Run record exists; no automatic restart')
        if input('Only approved first group: 3 strokes, new measured contacts. Type RUN to execute: ').strip()!='RUN':return 0
        # Existing executor repeats idle, alarm, orientation and initial-lift checks.
        return execute_reviewed_job(str(target))

if __name__=='__main__':raise SystemExit(main())
