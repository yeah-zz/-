# -*- coding: utf-8 -*-
"""Session-local speed choice, then delegate to the existing guarded executor."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from prepare_one_hour_ink_20261102 import OUT,session_speed
from run_mountain_balanced import main,execute_reviewed_job


def validate_resume(checkpoint_path,full):
    state=json.loads(checkpoint_path.read_text('utf-8'))
    prior_path=Path(state['prior_execution_log'])
    prior=json.loads(prior_path.read_text('utf-8'))
    stop=json.loads(Path(state['stop_record']).read_text('utf-8'))
    withdraw=json.loads(Path(state['withdraw_record']).read_text('utf-8'))
    count=state['completed_strokes']
    if (type(count) is not int or count<=0 or prior.get('completed_strokes')!=count or
            prior.get('active_action',{}).get('kind')!='dip' or not prior.get('interrupted') or
            not stop.get('stop_acknowledged') or not stop.get('idle_observed') or
            not withdraw.get('success') or withdraw.get('new_paper_strokes')!=0):
        raise ValueError('A confirmed ink-loading boundary is required for automatic continuation')
    if state.get('paper_unchanged') is not True:
        raise ValueError('Continuation requires the same unmoved paper')
    old_file=Path(prior['job_path'])
    if hashlib.sha256(old_file.read_bytes()).hexdigest()!=state['prior_job_sha256']:
        raise ValueError('Prior drawing definition changed')
    old=json.loads(old_file.read_text('utf-8'));new=json.loads(full.read_text('utf-8'))
    a=old['meta']['ink_execution']['base_strokes'];b=new['meta']['ink_execution']['base_strokes']
    if len(a)!=len(b) or count>=len(b):raise ValueError('Continuation count/geometry mismatch')
    for before,after in zip(a,b):
        for key in ('points','press','speed','tail_mm'):
            if before.get(key)!=after.get(key):raise ValueError('Continuation must preserve all drawing geometry/contact settings')
    for key in ('paper_z','canvas_corners','image_size'):
        if old['meta'][key]!=new['meta'][key]:raise ValueError('Drawing calibration changed during pause')
    old_snap=old['meta']['ink_execution'];new_snap=new['meta']['ink_execution']
    before=deepcopy(old_snap['calibration']);after=deepcopy(new_snap['calibration'])
    for snap in (before,after):
        for pigment in snap['palette']:pigment.pop('dip_time_s',None)
    if before!=after or old_snap['mm_per_px']!=new_snap['mm_per_px']:
        raise ValueError('Only ink dwell may change at this continuation checkpoint')
    before=dict(old_snap['policy']);after=dict(new_snap['policy'])
    for value in (before,after):
        value.pop('initial_dip_s',None);value.pop('refresh_dip_s',None)
    if before!=after:raise ValueError('Ink policy changed beyond the requested dwell setting')
    return count


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['check','check-robot','pilot','finish','full','resume'],default='check')
    parser.add_argument('--pilot-log',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--task-dir',type=Path)
    args=parser.parse_args()
    if args.phase in ('pilot','finish','full','resume') and args.task_dir is None:
        parser.error('Physical execution requires --task-dir to identify the reviewed composition')
    task_dir=(args.task_dir or OUT).resolve()
    manifest=json.loads((task_dir/'job_manifest.json').read_text('utf-8'))
    full=task_dir/manifest.get('full_file','G05_03_simple_full_job.json')
    pilot=task_dir/manifest.get('pilot_file','G05_03_first_two_job.json')
    if full.resolve().parent!=task_dir or pilot.resolve().parent!=task_dir:
        raise ValueError('Task manifest must reference files inside the selected task folder')
    if (hashlib.sha256(full.read_bytes()).hexdigest()!=manifest['full_sha256'] or
            hashlib.sha256(pilot.read_bytes()).hexdigest()!=manifest['pilot_sha256']):
        raise ValueError('Reviewed full/pilot task changed; regenerate and review before motion')
    if manifest.get('resume_only') and args.phase in ('pilot','finish','full'):
        raise ValueError('This task is continuation-only; first-stroke replay is refused')
    with session_speed(manifest.get('speed_factor',40)):
        if args.phase=='check':return main(['--job',str(full),'--check'])
        if args.phase=='check-robot':return main(['--job',str(full),'--check-robot'])
        if args.phase=='resume':
            if args.checkpoint is None:raise ValueError('Continuation checkpoint required')
            count=validate_resume(args.checkpoint,full)
            print('从第%d笔继续，已完成的%d笔保持不变。'%(count+1,count))
            if input('确认纸张未移动、纸笔墨仍准备好；输入 RUN 按新蘸墨时间继续：').strip().upper()!='RUN':
                return 0
            return execute_reviewed_job(str(full),start_index=count)
        if args.phase=='pilot':return main(['--job',str(pilot)])
        if args.phase=='full':
            data=json.loads(full.read_text('utf-8'))
            meta=data['meta']
            profile=meta.get('contact_profile',{})
            if (meta.get('speed_trial_required',True) or not meta.get('contact_profile_verified') or
                    profile!={'press_mm':.6,'draw_velocity_percent':20,'global_speed_factor':20} or
                    manifest.get('speed_factor')!=20):
                raise ValueError('Full start requires the unchanged operator-verified contact profile; use pilot for speed changes')
            return main(['--job',str(full)])
        if not args.pilot_log:raise ValueError('Completed pilot log is required; do not redraw its first two strokes')
        log=json.loads(args.pilot_log.read_text('utf-8'))
        if (not log.get('success') or log.get('completed_strokes')!=2 or
                Path(log.get('job_path','')).resolve()!=pilot.resolve()):
            raise ValueError('Pilot did not finish successfully for this task')
        full_data=json.loads(full.read_text('utf-8'))
        pilot_data=json.loads(pilot.read_text('utf-8'))
        if (full_data['meta']['ink_execution']['base_strokes'][:2] !=
                pilot_data['meta']['ink_execution']['base_strokes']):
            raise ValueError('Pilot does not match the first two full-task strokes')
        print('从第3笔继续，保持前两笔的纸张与现场设置。')
        if input('确认纸、笔、墨准备好，前两笔墨迹连续且速度可接受；输入 RUN 继续剩余部分：').strip().upper()!='RUN':
            return 0
        return execute_reviewed_job(str(full),start_index=2)


if __name__=='__main__':
    raise SystemExit(run())
