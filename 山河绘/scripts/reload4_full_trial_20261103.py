# -*- coding: utf-8 -*-
"""Continuation only: preserve all strokes and change count loading to four."""
import sys,json,argparse
from pathlib import Path
from dataclasses import replace,asdict
from copy import deepcopy
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure
configure()
from prepare_complete_trial_20261103 import ROOT as OLD,full_scope,read,save,sha
from run_complete_trial_20261103 import verify_manifest
from pigment_station import read_measurements,set_measurements_override
import pigment_station
from ink_control import InkPolicy,build_plan
from geometry import CanvasMap,check_waypoints
from run_mountain_v4_robot import safe_transfers
from run_mountain_balanced import load_reviewed,check_robot_readonly,execute_reviewed_job
from painting_session import Session,StageControl,Console,session_lock,PAUSED
import job
ROOT=OLD/'reload4_continuation'

@contextmanager
def scope():
    with full_scope():
        old=deepcopy(pigment_station._measurements_override)
        m=read_measurements();m['black']['strokes_per_load']=4
        set_measurements_override(m)
        try:yield
        finally:pigment_station._measurements_override=old

def new_policy(old):
    # Four maximum-length segments cost <180 equivalent mm at the fixed .60mm
    # pressure. This is a bookkeeping allowance, not sensed ink volume.
    return replace(old,max_strokes_per_load=4,capacity_eq_mm=205.,reserve_eq_mm=25.,max_segment_eq_mm=120.)

def prepare():
    m=verify_manifest();oldpath=Path(m['job']);data=read(oldpath)
    sessions=list((OLD/'session').glob('painting_session_*.json'));assert len(sessions)==1
    oldstate=read(sessions[0]);count=oldstate['completed']['black']
    if oldstate['status']!='paused' or count!=22:raise ValueError('Expected verified paused prefix of22; no automatic repair')
    logs=sorted(OLD.glob('*_job_runs/*execution_log.json'));prior=logs[-1];log=read(prior)
    if not log.get('paused') or log['completed_strokes']!=count or log.get('active_action') is not None:raise ValueError('Uncertain prior action')
    if Path(log['job_path']).resolve()!=oldpath.resolve():raise ValueError('Prior task mismatch')
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'checkpoint.json').exists():raise ValueError('Continuation already prepared; no overwrite')
    target=ROOT/'G03_01_reload4_job.json';meta=deepcopy(data['meta']);strokes=meta['ink_execution']['base_strokes']
    with scope():
        mapper=CanvasMap(meta['image_size']);policy=new_policy(InkPolicy(**meta['ink_execution']['policy']))
        steps,stats=build_plan(strokes,0,mapper.mm_per_px,policy)
        wp,_=job.steps_to_waypoints(steps,mapper);wp=safe_transfers(wp)
        if not check_waypoints(wp)[0]:raise ValueError('Static reach failure')
        meta['ink_execution']['policy']=asdict(policy)
        meta.update(mode='complete-trial-reload4-continuation',strokes=len(strokes),dips=stats['dips'],
            resume_only=True,required_start_index=count,ink_frequency_changed_by_user=True)
        job.save_job(str(target),wp,meta);load_reviewed(str(target))
        _,remaining=build_plan(strokes[count:],0,mapper.mm_per_px,policy)
    assert read(target)['meta']['ink_execution']['base_strokes']==data['meta']['ink_execution']['base_strokes']
    cp=dict(prior_job=str(oldpath),prior_job_sha256=sha(oldpath),prior_log=str(prior),prior_log_sha256=sha(prior),
        prior_session=str(sessions[0]),prior_session_sha256=sha(sessions[0]),completed=count,
        job=str(target),job_sha256=sha(target),script_sha256=sha(__file__),shared_files=m['shared_files'],
        old_source_files=m['source_files'],contact_wrapper_sha256=m['contact_wrapper_sha256'],
        total=len(strokes),policy=asdict(policy),remaining_segments=len(strokes)-count,
        planned_remaining_dips=remaining['dips'],estimated_remaining_minutes=remaining['estimated_total_s']/60,
        physical_settings_unchanged=True,ink_allowance_is_heuristic=True,no_extra_quality_pauses=True)
    save(ROOT/'checkpoint.json',cp)
    manifest=dict(registration=dict(job_sha256=cp['job_sha256'],contacts=m['contacts']),stages=dict(black=strokes))
    s=Session(ROOT/'session',target,manifest);s.data['completed']['black']=count;s.parked('paused')
    save(ROOT/'operator_request.json',dict(request='修改为画4笔蘸取一次墨',continue_remaining_previously_authorized=True,
        start_stroke=count+1,ordinary_reload_every_segments=4,keep_time_idle_protection=True,
        press_mm=.6,velocity_percent=20,global_speed_factor=20,dip_mm=1.2,dwell_s=2.))
    print(json.dumps({k:cp[k] for k in ['completed','remaining_segments','planned_remaining_dips','estimated_remaining_minutes']},ensure_ascii=False))

def checked():
    cp=read(ROOT/'checkpoint.json')
    if sha(cp['job'])!=cp['job_sha256'] or sha(__file__)!=cp['script_sha256']:raise ValueError('Task/runner changed')
    if any(sha(x['path'])!=x['sha256'] for x in cp['shared_files']+cp['old_source_files']):raise ValueError('Calibration/frozen code changed')
    if sha(cp['prior_job'])!=cp['prior_job_sha256'] or sha(cp['prior_log'])!=cp['prior_log_sha256'] or sha(cp['prior_session'])!=cp['prior_session_sha256']:raise ValueError('Prior checkpoint changed')
    old=read(cp['prior_job'])['meta']['ink_execution']['base_strokes']
    current=read(cp['job'])['meta']['ink_execution']['base_strokes']
    if old!=current:raise ValueError('Stroke geometry or order changed')
    return cp

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','check','check-robot','resume','pause'],default='check');a=p.parse_args()
    if a.phase=='prepare':prepare();return 0
    cp=checked();target=Path(cp['job'])
    if a.phase=='pause':save(ROOT/'pause_requested.json',dict(pause=True));print('Boundary pause requested.');return 0
    with scope():
        strokes,mapper,policy,meta=load_reviewed(str(target))
        if policy.max_strokes_per_load!=4 or len(strokes)!=1359:raise ValueError('Reload4 task mismatch')
        if a.phase=='check':print('PASS:1359 unchanged strokes;22 completed; resume23; reload4 policy checked offline.');return 0
        if a.phase=='check-robot':return check_robot_readonly(str(target))
        if input('User requested reload4: resume remaining only. Type RUN: ').strip()!='RUN':return 0
        manifest=dict(registration=dict(job_sha256=cp['job_sha256'],contacts=read(OLD/'manifest.json')['contacts']),stages=dict(black=strokes))
        with session_lock(ROOT/'session'):
            session=Session(ROOT/'session',target,manifest);count=session.count('black')
            if session.data['status']!='paused' or count<cp['completed']:raise ValueError('No verified continuation prefix')
            pause=ROOT/'pause_requested.json'
            if pause.exists():pause.unlink()
            control=StageControl(session,'black','整幅黑线·每4段补墨',Console(ask=lambda _: 'q',poll=pause.exists))
            session.begin('black')
            try:result=execute_reviewed_job(str(target),start_index=count,control=control)
            except BaseException:session.uncertain();raise
            if result==0:session.parked('complete');print('\nFULL DRAWING COMPLETE.')
            elif result==PAUSED:print('\nPARKED at%d/%d.'%(session.count('black'),len(strokes)))
            else:session.uncertain()
            return result

if __name__=='__main__':raise SystemExit(main())
