# -*- coding: utf-8 -*-
"""User-approved pavilion deferral and global speed70, continuation only."""
import sys,argparse
from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure
configure()
from resume_joined_remaining_20261103 import ROOT as PRIOR,check as prior_check,scope as prior_scope
from prepare_complete_trial_20261103 import read,save,sha
from config import ArmConfig,PALETTE
from geometry import CanvasMap,check_waypoints
from ink_control import InkPolicy,build_plan
from run_mountain_v4_robot import safe_transfers
from run_mountain_balanced import load_reviewed,check_robot_readonly,execute_reviewed_job
from painting_session import Session,StageControl,Console,session_lock,PAUSED
import job
ROOT=PRIOR.parent/'pavilion_deferred_speed70'

@contextmanager
def scope():
    with prior_scope():
        old=ArmConfig.SPEED_FACTOR;ArmConfig.SPEED_FACTOR=70
        try:yield
        finally:ArmConfig.SPEED_FACTOR=old

def select_strokes(strokes,count,deferred):
    indices=list(deferred)
    if len(set(indices))!=len(indices) or any(type(i) is not int or i<count or i>=len(strokes) for i in indices):raise ValueError('Invalid deferred indices or painted prefix selected')
    keep=[i for i in range(len(strokes)) if i not in set(indices)]
    result=[deepcopy(strokes[i]) for i in keep]
    if result[:count]!=strokes[:count]:raise ValueError('Completed prefix changed')
    assert sorted(keep+indices)==list(range(len(strokes)))
    return result,keep

def prepare():
    prior=prior_check();oldpath=Path(prior['job']);data=read(oldpath);oldss=data['meta']['ink_execution']['base_strokes']
    sp=next((PRIOR/'session').glob('painting_session_*.json'));state=read(sp);count=state['completed']['black']
    if state['status']!='paused' or count!=272:raise ValueError('Requires confirmed paused272 checkpoint')
    logpath=sorted(PRIOR.glob('*_job_runs/*execution_log.json'))[-1];log=read(logpath)
    if not log.get('paused') or log['completed_strokes']!=count or log['active_action'] is not None:raise ValueError('Prior stroke uncertain')
    proposal=PRIOR.parent/'pavilion_ink_review'/'pending_scope_proposal.json';plan=read(proposal)
    deferred=plan['proposed_deferred_indices']
    if len(deferred)!=136 or plan['completed_prefix']!=count:raise ValueError('Approved selection mismatch')
    if (ROOT/'checkpoint.json').exists():raise ValueError('Continuation already prepared; do not overwrite')
    ROOT.mkdir(exist_ok=True)
    base,kept=select_strokes(oldss,count,deferred);assert len(base)==1184 and len(base)-count==912
    with scope():
        meta=deepcopy(data['meta']);mapper=CanvasMap(meta['image_size']);policy=InkPolicy(**meta['ink_execution']['policy'])
        if policy.max_strokes_per_load!=4:raise ValueError('Reload policy changed')
        steps,stats=build_plan(base,0,mapper.mm_per_px,policy)
        wp,_=job.steps_to_waypoints(steps,mapper);wp=safe_transfers(wp)
        if not check_waypoints(wp)[0]:raise ValueError('Static reach failure')
        meta['ink_execution']['base_strokes']=base
        meta['ink_execution']['calibration']['speed_factor']=70
        meta.update(mode='user-approved-pavilion-deferral-speed70',strokes=len(base),dips=stats['dips'],
            required_start_index=count,resume_only=True,deferred_pavilion_count=136,
            complete_original_artwork=False,all_original_curves_kept_for_later=True,
            global_speed_factor=70,draw_velocity_percent_unchanged=20,
            user_approved_global_speed70=True,speed70_physical_quality_not_yet_verified=True)
        target=ROOT/'G03_01_other_regions_speed70_job.json';job.save_job(str(target),wp,meta);load_reviewed(str(target))
        for s in base:
            if set(s['press'])!={.6} or set(s['speed'])!={20.} or s['tail_mm']!=0:raise ValueError('Unapproved drawing parameters changed')
    cp=dict(job=str(target),job_sha256=sha(target),prior_job=str(oldpath),prior_job_sha256=sha(oldpath),
        prior_log=str(logpath),prior_log_sha256=sha(logpath),prior_session=str(sp),prior_session_sha256=sha(sp),
        proposal=str(proposal),proposal_sha256=sha(proposal),completed=count,prior_total=len(oldss),total=len(base),
        deferred_indices=deferred,kept_old_indices=kept,remaining=912,
        global_speed_factor=70,draw_velocity_percent=20,physical_speed_mm_s_unknown=True,
        source_files=[dict(path=str(p),sha256=sha(p)) for p in [Path(__file__),Path(__file__).with_name('resume_joined_remaining_20261103.py'),Path(__file__).with_name('reload4_full_trial_20261103.py')]],
        calibration_files=prior['calibration_files'],old_source_files=prior['old_source_files'],
        completed_strokes_unchanged=True,other_regions_geometry_and_order_unchanged=True)
    save(ROOT/'checkpoint.json',cp)
    save(ROOT/'deferred_pavilion_strokes.json',dict(original_job=str(oldpath),indices=deferred,
        strokes=[oldss[i] for i in deferred],pending=True,permanently_deleted=False,
        boundary_ambiguous_indices=plan['boundary_ambiguous_indices']))
    manifest=dict(registration=dict(job_sha256=cp['job_sha256'],contacts=meta['contact_measurements']),stages=dict(black=base))
    session=Session(ROOT/'session',target,manifest);session.data['completed']['black']=count;session.parked('paused')
    save(ROOT/'operator_authorization.json',dict(defer_pavilion_candidates=136,continue_other_strokes=912,
        source='Explicit user approval to defer pavilion candidates and subsequent request to change TCP global speed to70',
        global_speed_factor=70,draw_velocity_percent_kept=20,press_mm=.6,dip_mm=1.2,dwell_s=2.,reload_every_segments=4,
        no_new_quality_pause=True,allow_continue=True))
    print('PREPARED: painted272 unchanged;136 deferred;912 remaining;1184 total; global70, draw v20 unchanged.')

def checked():
    cp=read(ROOT/'checkpoint.json')
    for k in ['job','prior_job','prior_log','prior_session','proposal']:
        if sha(cp[k])!=cp[k+'_sha256']:raise ValueError('Frozen artifact changed: '+k)
    for x in cp['source_files']+cp['calibration_files']+cp['old_source_files']:
        if sha(x['path'])!=x['sha256']:raise ValueError('Source or calibration changed')
    old=read(cp['prior_job'])['meta']['ink_execution']['base_strokes'];new=read(cp['job'])['meta']['ink_execution']['base_strokes']
    expected,kept=select_strokes(old,cp['completed'],cp['deferred_indices'])
    if expected!=new or kept!=cp['kept_old_indices']:raise ValueError('Unexpected stroke edit')
    return cp

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','check','check-robot','resume','pause'],default='check');a=p.parse_args()
    if a.phase=='prepare':prepare();return 0
    cp=checked();target=Path(cp['job'])
    if a.phase=='pause':save(ROOT/'pause_requested.json',dict(pause=True));print('Boundary pause requested.');return 0
    with scope():
        strokes,mapper,policy,meta=load_reviewed(str(target))
        if a.phase=='check':print('PASS: prefix272 intact;136 exact candidates deferred;912 unchanged remaining; global70 only.');return 0
        if a.phase=='check-robot':return check_robot_readonly(str(target))
        if input('Approved: defer pavilion136; continue other912; global70/drawv20. Type RUN: ').strip()!='RUN':return 0
        manifest=dict(registration=dict(job_sha256=cp['job_sha256'],contacts=meta['contact_measurements']),stages=dict(black=strokes))
        with session_lock(ROOT/'session'):
            session=Session(ROOT/'session',target,manifest);n=session.count('black')
            if session.data['status']!='paused' or n<cp['completed']:raise ValueError('Not a confirmed paused prefix')
            pause=ROOT/'pause_requested.json'
            if pause.exists():pause.unlink()
            control=StageControl(session,'black','其他区域续画·全局70',Console(ask=lambda _: 'q',poll=pause.exists))
            session.begin('black')
            try:code=execute_reviewed_job(str(target),start_index=n,control=control)
            except BaseException:session.uncertain();raise
            if code==0:session.parked('complete');print('\nSELECTED REGIONS COMPLETE;136 PAVILION CANDIDATES STILL DEFERRED.')
            elif code==PAUSED:print('\nPAUSED at%d/%d.'%(session.count('black'),len(strokes)))
            else:session.uncertain()
            return code

if __name__=='__main__':raise SystemExit(main())
