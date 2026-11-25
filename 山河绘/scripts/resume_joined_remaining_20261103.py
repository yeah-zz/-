# -*- coding: utf-8 -*-
"""Checkpoint-preserving update of unpainted curves; existing guarded motion."""
import sys,argparse,json
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure
configure()
import numpy as np,cv2
from PIL import Image
from reload4_full_trial_20261103 import ROOT as PRIOR,checked as prior_checked,scope
from prepare_complete_trial_20261103 import read,save,sha
from ink_control import InkPolicy,refresh_stats,build_plan
from geometry import CanvasMap,check_waypoints
from run_mountain_v4_robot import safe_transfers
from run_mountain_balanced import load_reviewed,check_robot_readonly,execute_reviewed_job
from remaining_stroke_join_20261103 import raster,length
from painting_session import Session,StageControl,Console,session_lock,PAUSED
import job
ROOT=PRIOR.parent/'remaining_replan'

def prepare():
    oldcp=prior_checked();oldpath=Path(oldcp['job']);old=read(oldpath)
    sp=next((PRIOR/'session').glob('painting_session_*.json'));state=read(sp)
    count=state['completed']['black'];assert state['status']=='paused' and count==207
    priorlog=sorted(PRIOR.glob('*_job_runs/*execution_log.json'))[-1];log=read(priorlog)
    if not log.get('paused') or log['completed_strokes']!=count or log['active_action'] is not None:raise ValueError('Prior action uncertain')
    if (ROOT/'checkpoint.json').exists():raise ValueError('Do not overwrite continuation checkpoint')
    doc=read(ROOT/'candidate_paths.json');report=read(ROOT/'join_report.json')
    if doc['completed_prefix']!=count or not report['coverage_within_one_pixel'] or report['exact_raster_tie_difference_pixels']!=0:raise ValueError('Candidate geometry check failed')
    previous=old['meta']['ink_execution']['base_strokes'];prefix=deepcopy(previous[:count]);records=doc['paths']
    represented=sorted(k for r in records for k in r['old_remaining_indices'])
    assert represented==list(range(len(previous)-count))
    with scope():
        meta=deepcopy(old['meta']);mapper=CanvasMap(meta['image_size']);mult=np.array(doc['mult']);offset=np.array(doc['offset'])
        current=np.array(prefix[-1]['points'][-1]);pending=[]
        for i,r in enumerate(records):
            p=np.array(r['points'])*mult+offset
            ids=[k+count for k in r['old_remaining_indices']]
            s=dict(id=f'continuation-chain-{i:04d}',name=f'chain-{i:04d}',points=p.tolist(),
                press=[.6]*len(p),speed=[20.]*len(p),horizontal_velocity_percent=60,layer=1,pal_index=0,
                tail_mm=0.,lift_mm=4.,old_job_indices=ids,source_curve_id=previous[ids[0]]['source_curve_id'],
                source_ids=sorted(set(k for ix in ids for k in previous[ix].get('source_ids',[]))),
                stroke_identity_verified=False,direction_verified=False,join_is_rule_based=True)
            pending.append(refresh_stats(s,mapper.mm_per_px))
        remaining=[]
        # Whole-curve selection balances useful length with pen-up travel.
        # This is a deterministic planning prototype, not semantic recognition.
        while pending:
            def score(s):
                p=np.array(s['points']);travel=min(np.linalg.norm(p[0]-current),np.linalg.norm(p[-1]-current))*mapper.mm_per_px
                return s['length_mm']/(1+travel/10)
            k=max(range(len(pending)),key=lambda k:score(pending[k]));s=pending.pop(k);p=np.array(s['points'])
            if np.linalg.norm(p[-1]-current)<np.linalg.norm(p[0]-current):s['points']=s['points'][::-1]
            current=np.array(s['points'][-1]);remaining.append(s)
        # Exact coverage of the old remaining raster must be preserved after reorder.
        shape=(496,496)
        oldps=[(np.array(s['points'])-offset)/mult for s in previous[count:]]
        newps=[(np.array(s['points'])-offset)/mult for s in remaining]
        before=raster(oldps,shape);after=raster(newps,shape)
        if np.any((before>0)&(after==0)):raise ValueError('An original source pixel was lost')
        done=raster([(np.array(s['points'])-offset)/mult for s in prefix],shape)
        if np.any((after>0)&(before==0)&(cv2.dilate(done,np.ones((5,5),np.uint8))>0)):raise ValueError('New bridge enters painted protection band')
        base=prefix+remaining
        assert base[:count]==previous[:count]
        policy=InkPolicy(**meta['ink_execution']['policy'])
        for s in base:
            if s['length_mm']>19.8001:raise ValueError('Continuous stroke outside existing validated length limit')
        steps,stats=build_plan(base,0,mapper.mm_per_px,policy)
        wp,_=job.steps_to_waypoints(steps,mapper);wp=safe_transfers(wp)
        if not check_waypoints(wp)[0]:raise ValueError('Static reach failure')
        meta['ink_execution']['base_strokes']=base
        meta.update(mode='protected-remaining-source-backed-replan',strokes=len(base),resume_only=True,
            required_start_index=count,no_original_curve_deleted=True,complete_stroke_neural_policy_trained=False,
            join_report=str(ROOT/'join_report.json'),current_pose_checked=False,actually_drawn=False)
        target=ROOT/'G03_01_remaining_joined_job.json';job.save_job(str(target),wp,meta);load_reviewed(str(target))
        _,left=build_plan(remaining,0,mapper.mm_per_px,policy)
    cp=dict(job=str(target),job_sha256=sha(target),prior_job=str(oldpath),prior_job_sha256=sha(oldpath),
        prior_log=str(priorlog),prior_log_sha256=sha(priorlog),prior_session=str(sp),prior_session_sha256=sha(sp),
        completed=count,old_total=len(previous),total=len(base),old_remaining=len(previous)-count,new_remaining=len(remaining),
        saved_pen_lifts=len(previous)-len(base),planned_remaining_dips=left['dips'],estimated_remaining_minutes=left['estimated_total_s']/60,
        calibration_files=oldcp['shared_files'],old_source_files=oldcp['old_source_files'],
        source_files=[dict(path=str(p),sha256=sha(p)) for p in [Path(__file__),Path(__file__).with_name('remaining_stroke_join_20261103.py'),Path(__file__).with_name('reload4_full_trial_20261103.py')]],
        source_coverage_exact=True,painted_prefix_unchanged=True,added_bridges_avoid_painted_band=True,
        physical_parameters_and_reload4_unchanged=True,semantic_next_stroke_model_trained=False)
    save(ROOT/'checkpoint.json',cp)
    manifest=dict(registration=dict(job_sha256=cp['job_sha256'],contacts=old['meta']['contact_measurements']),stages=dict(black=base))
    session=Session(ROOT/'session',target,manifest);session.data['completed']['black']=count;session.parked('paused')
    (ROOT/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><style>body{{font:18px system-ui;max-width:1250px;margin:auto;padding:24px;background:#f5f3eb}}img{{width:100%}}</style><h1>当前纸张：只修改未画部分</h1><p>前207段原样保留。剩余1152段→1113笔，减少39次起落笔，合并成35条较长曲线。所有原有线条保留，新增接续避开已画区域。预估仍需{cp['estimated_remaining_minutes']:.0f}分钟，不保证。</p><p>当前是原图墨迹与几何约束的续接加局部顺序优化，没有新训练完整笔画识别或自主选笔模型，不能声称根本问题已经解决。纸上墨迹未配准，已画保护来自命令日志。松针、平行线和歧义交汇保留分笔。</p><img src="joined_examples.jpg"><a href="checkpoint.json">进度与一致性检查</a> · <a href="join_report.json">接受及拒绝记录</a></p>''',encoding='utf-8')
    print(json.dumps({k:cp[k] for k in ['completed','old_total','total','old_remaining','new_remaining','saved_pen_lifts','planned_remaining_dips','estimated_remaining_minutes']},ensure_ascii=False))

def check():
    cp=read(ROOT/'checkpoint.json')
    for key in ['job','prior_job','prior_log','prior_session']:
        if sha(cp[key])!=cp[key+'_sha256']:raise ValueError('Frozen checkpoint changed: '+key)
    for r in cp['source_files']+cp['calibration_files']+cp['old_source_files']:
        if sha(r['path'])!=r['sha256']:raise ValueError('Frozen source/calibration changed')
    old=read(cp['prior_job'])['meta']['ink_execution']['base_strokes'];new=read(cp['job'])['meta']['ink_execution']['base_strokes']
    if old[:cp['completed']]!=new[:cp['completed']]:raise ValueError('Painted prefix changed')
    covered=sorted(i for s in new[cp['completed']:] for i in s['old_job_indices'])
    if covered!=list(range(cp['completed'],len(old))):raise ValueError('Old remaining stroke missing or repeated')
    return cp

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','check','check-robot','resume','pause'],default='check');a=p.parse_args()
    if a.phase=='prepare':prepare();return 0
    cp=check();target=Path(cp['job'])
    if a.phase=='pause':save(ROOT/'pause_requested.json',dict(pause=True));print('Boundary pause requested.');return 0
    with scope():
        strokes,mapper,policy,meta=load_reviewed(str(target))
        if a.phase=='check':print('PASS: painted207 unchanged; all1152 prior remaining records represented once;1320 total; same physical settings.');return 0
        if a.phase=='check-robot':return check_robot_readonly(str(target))
        if input('Approved remaining-only replan; preserve sheet. Type RUN: ').strip()!='RUN':return 0
        manifest=dict(registration=dict(job_sha256=cp['job_sha256'],contacts=meta['contact_measurements']),stages=dict(black=strokes))
        with session_lock(ROOT/'session'):
            session=Session(ROOT/'session',target,manifest);n=session.count('black')
            if session.data['status']!='paused' or n<cp['completed']:raise ValueError('Unverified resume state')
            pause=ROOT/'pause_requested.json'
            if pause.exists():pause.unlink()
            control=StageControl(session,'black','完整曲线候选续画',Console(ask=lambda _: 'q',poll=pause.exists))
            session.begin('black')
            try:code=execute_reviewed_job(str(target),start_index=n,control=control)
            except BaseException:session.uncertain();raise
            if code==0:session.parked('complete');print('\nFULL DRAWING COMPLETE.')
            elif code==PAUSED:print('\nPAUSED at%d/%d.'%(session.count('black'),len(strokes)))
            else:session.uncertain()
            return code

if __name__=='__main__':raise SystemExit(main())
