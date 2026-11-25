# -*- coding: utf-8 -*-
"""Explicit full black trial with a six-stroke quality gate and boundary pause."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
from prepare_complete_trial_20261103 import ROOT,full_scope,read,save,sha
from run_mountain_balanced import load_reviewed,check_robot_readonly,execute_reviewed_job
from painting_session import Session,StageControl,Console,session_lock,PAUSED

def verify_manifest():
    m=read(ROOT/'manifest.json')
    for r in m['source_files']+m['shared_files']:
        if sha(r['path'])!=r['sha256']:raise ValueError('Frozen source/shared settings changed: '+r['path'])
    if sha(m['job'])!=m['job_sha256'] or sha(m['curves'])!=m['curves_sha256']:raise ValueError('Job/curves changed')
    if sha(Path(__file__).with_name('run_brush_pilot_contact_20261103.py'))!=m['contact_wrapper_sha256']:raise ValueError('Contact scope changed')
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['check','check-robot','start','resume','pause'],default='check');a=p.parse_args()
    m=verify_manifest();target=Path(m['job'])
    if a.phase=='pause':
        save(ROOT/'pause_requested.json',dict(pause=True));print('Boundary pause requested; no robot commands in this process.');return 0
    with full_scope():
        strokes,_,_,meta=load_reviewed(str(target))
        if len(strokes)!=m['total_strokes']:raise ValueError('Stroke count changed')
        if a.phase=='check':print('PASS: complete job and task calibration, offline only.');return 0
        if a.phase=='check-robot':return check_robot_readonly(str(target))
        if input('Confirm blank sheet/site watch for START, or unchanged sheet for RESUME. Type RUN: ').strip()!='RUN':return 0
        folder=ROOT/'session'
        manifest=dict(registration=dict(job_sha256=m['job_sha256'],contacts=m['contacts']),stages=dict(black=strokes))
        with session_lock(folder):
            session=Session(folder,target,manifest)
            count=session.count('black')
            if a.phase=='start' and (count or list(ROOT.glob('*_job_runs/*execution_log.json'))):raise ValueError('An attempt exists; inspect before resume, never start over silently')
            if a.phase=='resume' and (session.data.get('status')!='paused' or not count):raise ValueError('No verified boundary pause to resume')
            # Consume a preexisting pause request only after a new explicit RUN.
            pause_path=ROOT/'pause_requested.json'
            if pause_path.exists():pause_path.unlink()
            gate=m['quality_gate_after_strokes'];gate_pending=count<gate
            control=None
            def poll():
                hit_gate=gate_pending and session.count('black')>=gate
                return hit_gate or pause_path.exists()
            # Existing StageControl calls existing park(), waits, persists pause,
            # and Q exits without drawing another stroke. No Ctrl+C for pause.
            console=Console(ask=lambda _: 'q',poll=poll)
            control=StageControl(session,'black','整幅黑线',console)
            session.begin('black')
            try:
                result=execute_reviewed_job(str(target),start_index=count,control=control)
            except BaseException:
                session.uncertain();raise
            if result==PAUSED:
                print('\nPARKED: completed %d/%d; sheet must stay fixed. Resume requires another explicit RUN.'%(session.count('black'),len(strokes)))
            elif result==0:
                session.parked('complete');print('\nFULL BLACK TRIAL COMPLETE.')
            else:session.uncertain()
            return result

if __name__=='__main__':raise SystemExit(main())
