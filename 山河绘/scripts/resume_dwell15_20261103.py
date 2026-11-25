# -*- coding: utf-8 -*-
"""Continuation-only reduction of ink dwell2 ->1.5s; no stroke changes."""
import sys,argparse
from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace,asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure
configure()
from resume_pavilion_deferred_20261103 import ROOT as PRIOR,checked as prior_check,scope as previous_scope
from prepare_complete_trial_20261103 import read,save,sha
from pigment_station import read_measurements,set_measurements_override
import pigment_station
from config import PALETTE
from ink_control import InkPolicy,build_plan
from geometry import CanvasMap,check_waypoints
from run_mountain_v4_robot import safe_transfers
from run_mountain_balanced import load_reviewed,check_robot_readonly,execute_reviewed_job
from painting_session import Session,StageControl,Console,session_lock,PAUSED
import job
ROOT=PRIOR.parent/'dwell15_speed70_continuation'

@contextmanager
def scope():
    with previous_scope():
        old=deepcopy(pigment_station._measurements_override);palette=deepcopy(PALETTE)
        values=read_measurements();values['black']['dwell_s']=1.5
        set_measurements_override(values);PALETTE[0]['dip_time_s']=1.5
        try:yield
        finally:PALETTE[:]=palette;pigment_station._measurements_override=old

def prepare():
    parent=prior_check();oldpath=Path(parent['job']);data=read(oldpath);meta=deepcopy(data['meta']);strokes=meta['ink_execution']['base_strokes']
    sp=next((PRIOR/'session').glob('painting_session_*.json'));state=read(sp);n=state['completed']['black']
    lp=sorted(PRIOR.glob('*_job_runs/*execution_log.json'))[-1];log=read(lp)
    if state['status']!='paused' or not log.get('paused') or log['completed_strokes']!=n or log['active_action'] is not None:raise ValueError('Requires confirmed boundary pause')
    if len(strokes)!=1184 or n!=467:raise ValueError('Unexpected continuation checkpoint')
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'checkpoint.json').exists():raise ValueError('Do not overwrite used continuation')
    with scope():
        mapper=CanvasMap(meta['image_size']);policy=replace(InkPolicy(**meta['ink_execution']['policy']),initial_dip_s=1.5,refresh_dip_s=1.5)
        if policy.max_strokes_per_load!=4:raise ValueError('Reload count changed')
        meta['ink_execution']['policy']=asdict(policy);meta['ink_execution']['calibration']['palette']=deepcopy(PALETTE)
        meta.update(mode='dwell15-speed70-continuation',required_start_index=n,strokes=len(strokes),ink_dwell_s=1.5,
            actual_paper_effect_after_shorter_dwell_unknown=True)
        steps,_=build_plan(strokes,0,mapper.mm_per_px,policy);wp,_=job.steps_to_waypoints(steps,mapper);wp=safe_transfers(wp)
        if not check_waypoints(wp)[0]:raise ValueError('Static reach failed')
        target=ROOT/'G03_01_dwell15_job.json';job.save_job(str(target),wp,meta);load_reviewed(str(target))
    assert read(target)['meta']['ink_execution']['base_strokes']==data['meta']['ink_execution']['base_strokes']
    cp=dict(job=str(target),job_sha256=sha(target),prior_job=str(oldpath),prior_job_sha256=sha(oldpath),
        prior_log=str(lp),prior_log_sha256=sha(lp),prior_session=str(sp),prior_session_sha256=sha(sp),
        completed=n,total=len(strokes),remaining=len(strokes)-n,dwell_s=1.5,global_speed_factor=70,
        draw_velocity_percent=20,press_mm=.6,dip_mm=1.2,reload_every_segments=4,deferred_pavilion_candidates=136,
        source_files=[dict(path=str(p),sha256=sha(p)) for p in [Path(__file__),Path(__file__).with_name('resume_pavilion_deferred_20261103.py'),Path(__file__).with_name('reload4_full_trial_20261103.py')]],
        calibration_files=parent['calibration_files'],old_source_files=parent['old_source_files'])
    save(ROOT/'checkpoint.json',cp)
    manifest=dict(registration=dict(job_sha256=cp['job_sha256'],contacts=meta['contact_measurements']),stages=dict(black=strokes))
    s=Session(ROOT/'session',target,manifest);s.data['completed']['black']=n;s.parked('paused')
    save(ROOT/'operator_request.json',dict(request='Shorten ink dipping time and continue remaining',
        selected_change='dwell2s to1.5s only',all_strokes_geometry_and_order_unchanged=True,approved_continue=True))
    print('Prepared: completed467 unchanged; resume468; remaining717; dwell1.5s; global70/drawv20/reload4 unchanged.')

def checked():
    cp=read(ROOT/'checkpoint.json')
    for k in ['job','prior_job','prior_log','prior_session']:
        if sha(cp[k])!=cp[k+'_sha256']:raise ValueError('Frozen checkpoint changed')
    for x in cp['source_files']+cp['calibration_files']+cp['old_source_files']:
        if sha(x['path'])!=x['sha256']:raise ValueError('Code/calibration changed')
    if read(cp['job'])['meta']['ink_execution']['base_strokes']!=read(cp['prior_job'])['meta']['ink_execution']['base_strokes']:raise ValueError('Stroke changes not authorized')
    return cp

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','check','check-robot','resume','pause'],default='check');a=p.parse_args()
    if a.phase=='prepare':prepare();return 0
    cp=checked();target=Path(cp['job'])
    if a.phase=='pause':save(ROOT/'pause_requested.json',dict(pause=True));print('Boundary pause requested.');return 0
    with scope():
        strokes,mapper,policy,meta=load_reviewed(str(target))
        if policy.initial_dip_s!=1.5 or policy.refresh_dip_s!=1.5 or policy.max_strokes_per_load!=4:raise ValueError('Policy mismatch')
        if a.phase=='check':print('PASS:1184 identical strokes;467 completed; dwell1.5s; shared configuration untouched.');return 0
        if a.phase=='check-robot':return check_robot_readonly(str(target))
        if input('Approved continuation only: dwell1.5s, global70, reload4. Type RUN: ').strip()!='RUN':return 0
        manifest=dict(registration=dict(job_sha256=cp['job_sha256'],contacts=meta['contact_measurements']),stages=dict(black=strokes))
        with session_lock(ROOT/'session'):
            session=Session(ROOT/'session',target,manifest);n=session.count('black')
            if session.data['status']!='paused' or n<cp['completed']:raise ValueError('Invalid paused checkpoint')
            pause=ROOT/'pause_requested.json'
            if pause.exists():pause.unlink()
            control=StageControl(session,'black','其他区域·蘸墨1.5秒',Console(ask=lambda _: 'q',poll=pause.exists))
            session.begin('black')
            try:code=execute_reviewed_job(str(target),start_index=n,control=control)
            except BaseException:session.uncertain();raise
            if code==0:session.parked('complete');print('\nSELECTED REGIONS COMPLETE;136 PAVILION CANDIDATES REMAIN DEFERRED.')
            elif code==PAUSED:print('\nPAUSED at%d/%d.'%(session.count('black'),len(strokes)))
            else:session.uncertain()
            return code

if __name__=='__main__':raise SystemExit(main())
