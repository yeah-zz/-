"""Supervised v2 color runner. Default read-only; explicit hover/paint actions.

No enable, clear-error, station dip, or automatic recovery. Every contact intent
is fsynced BEFORE movement; an interrupted contact blocks subsequent painting.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
import color_pass
from config import ArmConfig, CanvasConfig, station_approach_z, z_for_lift, z_for_press
from geometry import CanvasMap, check_waypoints
from park_color_loading import target_pose as loading_pose
from plan_blue_wash import inputs, clearance_for, dense_points, GAP, EDGE
from prepare_robot import close_to, move_to, read_idle_pose
from robot import E6Robot, RobotError
from run_mountain_v4_robot import connect_readonly

DEFAULT_PLAN='artifacts/blue/mountain_v6_blue_wash_plan.json'
# Onsite user explicitly selected 50% during the continuous pass, 2026-09-24.
# Individual MovL v=10 remains in prepare_robot.move_to; recovery tools unchanged.
GLOBAL_SPEED_FACTOR=50

def load_plan(path):
    plan=json.loads(Path(path).read_text(encoding='utf8'))
    if plan.get('format')!='blue-wash-v2': raise ValueError('Requires v2 wash plan')
    for key in ('ink_job','ink_image'):
        if plan[key+'_sha256']!=color_pass._digest(plan[key]): raise ValueError(key+' changed')
    if (plan['paper_z_snapshot']!=ArmConfig.PAPER_Z or plan['rpy_snapshot']!=list(ArmConfig.RPY)
            or plan['user']!=ArmConfig.USER_INDEX or plan['tool']!=ArmConfig.TOOL_INDEX
            or plan['canvas_corners']!=[list(p) for p in CanvasConfig.ROBOT_CORNERS]):
        raise ValueError('Calibration changed')
    meta,_,mapper,clearance,_=inputs(plan['ink_job'],plan['ink_image'])
    if plan['image_corners']!=[list(p) for p in mapper.image_corners] or plan['image_size']!=meta['image_size']:
        raise ValueError('Image registration changed')
    # Fixed exclusions cannot be removed by modifying plan metadata.
    clearance[240:530,440:820]=0
    clearance[350:510,1430:1659]=0
    targets=[]
    for s in plan['strokes']:
        p=np.asarray(s['points'],float)
        if p.shape!=(2,2) or not np.isfinite(p).all(): raise ValueError('Expected finite straight stroke')
        if s['color']!='blue' or s['brush_width_mm_estimate']!=1.: raise ValueError('Unreviewed brush')
        gap,edge=clearance_for(p,mapper,clearance)
        if gap<GAP or edge<EDGE: raise ValueError('Recomputed path clearance insufficient')
        xy=mapper.to_robot(p)
        length=float(np.linalg.norm(xy[1]-xy[0]))
        if not 5<=length<=32: raise ValueError('Stroke length outside limits')
        a,b=xy
        pose=lambda xy,z: (float(xy[0]),float(xy[1]),z,*ArmConfig.RPY)
        seq=[('high',pose(a,station_approach_z())),('hover',pose(a,z_for_lift(15))),
             ('touch',pose(a,z_for_press(.30))),('draw',pose(b,z_for_press(.30))),
             ('lift',pose(b,z_for_lift(15))),('raise',pose(b,station_approach_z()))]
        if not check_waypoints([dict(x=v[0],y=v[1],z=v[2]) for _,v in seq]+[
            dict(x=loading_pose()[0],y=loading_pose()[1],z=loading_pose()[2])])[0]:
            raise ValueError('Outside configured reach')
        targets.append(dict(id=s['id'],xy=xy.tolist(),sequence=seq,length=length))
    if not targets or len({t['id'] for t in targets})!=len(targets): raise ValueError('Invalid stroke IDs')
    return plan,targets

def persist(path,record,compact=False):
    """原子写记录：写 .tmp 再改名。

    ★ 2026-09-26 事故：跑图期间外部进程（我读记录、DobotStudio、编辑器、杀毒软件）
      只要打开着那个 .json，Windows 就会让 os.replace 报 WinError 5（拒绝访问），
      而这里没兜住 -> 一个 30 分钟的任务直接崩掉 ✗。
      现在改成重试若干次 + 最后退化成"直接覆写 json"，绝不让记录写失败把绘制打断。
    """
    temp=path.with_suffix('.tmp')
    with temp.open('w',encoding='utf8') as f:
        json.dump(record,f,ensure_ascii=False,indent=None if compact else 2,
                  separators=(',', ':') if compact else None); f.flush(); os.fsync(f.fileno())
    last=None
    for wait in (0.0,0.05,0.15,0.4,1.0,2.0):
        if wait: time.sleep(wait)
        try:
            os.replace(temp,path)
            return
        except OSError as exc:            # WinError 5 / 32：文件被别的进程占着
            last=exc
    try:                                  # 退路：直接覆写目标文件（不再改名）
        with path.open('w',encoding='utf8') as f:
            json.dump(record,f,ensure_ascii=False,indent=2)
        try: os.unlink(temp)
        except OSError: pass
    except OSError:
        raise last

def history(folder,job_digest):
    contacts=[]
    for path in Path(folder).glob('blue_wash_*.json'):
        old=json.loads(path.read_text(encoding='utf8')) # corrupt history fails closed
        if old.get('ink_job_sha256')!=job_digest or old.get('action')!='paint': continue
        for c in old.get('contacts',[]):
            if not c.get('completed'): raise RobotError('Unresolved contact in '+str(path))
            contacts.append(c)
    # Previous full runner did not persist contact intent; failed real runs need review.
    for path in Path(folder).glob('blue_full_*.json'):
        old=json.loads(path.read_text(encoding='utf8'))
        if old.get('action')=='paint':
            raise RobotError('Legacy paint history needs manual review: '+str(path))
    return contacts

def reject_overlap(targets,contacts):
    # Geometric, not merely plan hash, so renaming/replanning cannot repaint.
    for t in targets:
        points=dense_points(t['xy'],step=.25)
        for old in contacts:
            a,b=np.asarray(old['xy'],float)
            delta=b-a
            u=np.clip((points-a)@delta/max(float(delta@delta),1e-12),0,1)
            if np.linalg.norm(points-(a+u[:,None]*delta),axis=1).min()<.65:
                raise RobotError('Selected stroke overlaps previously contacted paper')

def run(plan_path=DEFAULT_PLAN,start=0,count=2,action='check',confirm_site=False,
        registration_record=None,log_dir='robot_runs',resume_hover_record=None,remaining=False,
        resume_high_record=None):
    if action not in ('check','hover','paint','park'): raise ValueError('Unknown action')
    if action!='check' and not confirm_site: raise ValueError('Current site readiness required')
    plan,all_targets=load_plan(plan_path)
    if remaining:
        count=len(all_targets)-start
    if (count<1 or (not remaining and count>2) or start<0
            or start+count>len(all_targets)): raise ValueError('Invalid batch')
    targets=all_targets[start:start+count]
    digest=color_pass._digest(plan_path)
    folder=Path(log_dir); folder.mkdir(exist_ok=True)
    path=folder/('blue_wash_%s_%s.json'%(datetime.now().strftime('%Y%m%d_%H%M%S_%f'),uuid4().hex[:8]))
    record=dict(action=action,plan=str(Path(plan_path).resolve()),plan_sha256=digest,
        ink_job_sha256=plan['ink_job_sha256'],start=start,count=count,success=False,
        global_speed_factor=GLOBAL_SPEED_FACTOR,movl_velocity_percent=10,
        continuous_remaining=remaining,
        steps=[],contacts=[],started_at=datetime.now().astimezone().isoformat())
    robot=E6Robot(); lock=None; lock_path=folder/'blue_wash_motion.lock'
    try:
        if action!='check':
            lock=os.open(lock_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            os.write(lock,str(path.resolve()).encode('utf8')); os.fsync(lock)
        if action=='paint':
            if not registration_record: raise ValueError('Requires successful hover record and onsite registration confirmation')
            reg=json.loads(Path(registration_record).read_text(encoding='utf8'))
            if (reg.get('action')!='hover' or not reg.get('success') or reg.get('plan_sha256')!=digest
                    or reg.get('start')!=start): raise ValueError('Hover record mismatch')
            age=(datetime.now().astimezone()-datetime.fromisoformat(reg['finished_at'])).total_seconds()
            if not 0<=age<=1200: raise ValueError('Hover registration older than 20 minutes')
            reject_overlap(targets,history(folder,plan['ink_job_sha256']))
        persist(path,record)
        connect_readonly(robot)
        pose=read_idle_pose(robot,record)
        record['initial_pose']=pose
        expected=loading_pose()
        if resume_high_record:
            resume=json.loads(Path(resume_high_record).read_text(encoding='utf8'))
            contacts=resume.get('contacts',[])
            if (action!='hover' or resume_hover_record or resume.get('action')!='paint'
                    or not resume.get('paused_for_speed_change') or resume.get('pending_move')
                    or resume.get('plan_sha256')!=digest or not contacts
                    or not all(c.get('completed') for c in contacts)
                    or start!=resume.get('start',-999)+len(contacts)
                    or start<=0 or contacts[-1]['id']!=all_targets[start-1]['id']
                    or not close_to(resume.get('final_pose'),all_targets[start-1]['sequence'][-1][1])):
                raise ValueError('No reviewed completed-stroke high resume point')
            expected=all_targets[start-1]['sequence'][-1][1]
            record['resume_high_record']=str(resume_high_record)
        if resume_hover_record:
            resume=json.loads(Path(resume_hover_record).read_text(encoding='utf8'))
            if (action!='hover' or resume.get('action')!='hover' or resume.get('success')
                    or resume.get('plan_sha256')!=digest or resume.get('start')!=start
                    or resume.get('contacts') or resume.get('steps')
                    or resume.get('pending_move',{}).get('name')!='high'
                    or not close_to(resume['pending_move']['target'],targets[0]['sequence'][0][1])):
                raise ValueError('Only diagnosed completed high transfer can continue to hover')
            expected=targets[0]['sequence'][0][1]
            record['resume_hover_record']=str(resume_hover_record)
        if action in ('paint','park'): expected=targets[0]['sequence'][1][1]
        if not close_to(pose,expected): raise RobotError('Unexpected start pose; no automatic recovery')
        print('BATCH',start,count,'ACTION',action,'length',sum(t['length'] for t in targets),flush=True)
        if action=='check': record['success']=True; return 0
        if not close_to(read_idle_pose(robot),pose): raise RobotError('Start changed')
        if not robot._ok(robot.send('SpeedFactor(%d)'%GLOBAL_SPEED_FACTOR)): raise RobotError('Speed rejected')
        previous=pose
        def move_one(name,target):
            nonlocal previous
            if not close_to(read_idle_pose(robot),previous): raise RobotError('Pose changed before '+name)
            record['pending_move']=dict(name=name,target=target); persist(path,record)
            print('MOVING',name,flush=True)
            move_to(robot,target)
            previous=target
            record['steps'].append(dict(name=name,target=target,arrived=True))
            record.pop('pending_move',None); persist(path,record)
        def move(name,target):
            # Long high moves at v10/SpeedFactor10 can exceed the fixed 60 s
            # arrival deadline. Split only coplanar high travel into <=100 mm
            # legs; every leg retains the original arrival/alarm checks.
            distance=float(np.linalg.norm(np.asarray(target[:2])-previous[:2]))
            if abs(previous[2]-station_approach_z())<.01 and abs(target[2]-previous[2])<.01 and distance>100:
                origin=np.asarray(previous,float)
                n=int(np.ceil(distance/100))
                for j in range(1,n+1):
                    leg=target if j==n else tuple(float(v) for v in origin+(np.asarray(target)-origin)*j/n)
                    move_one(name+'-leg-'+str(j),leg)
            else: move_one(name,target)
        if action=='hover':
            for name,target in targets[0]['sequence'][1 if resume_hover_record else 0:2]: move(name,target)
        elif action=='park':
            move('raise-before-park',targets[0]['sequence'][0][1]); move('park',loading_pose())
        else:
            for n,t in enumerate(targets):
                for name,target in t['sequence'][2 if n==0 else 0:]:
                    if name=='touch':
                        record['contacts'].append(dict(id=t['id'],xy=t['xy'],completed=False))
                        persist(path,record) # BEFORE first potential contact, including ambiguous send failure
                    move(t['id']+':'+name,target)
                record['contacts'][-1]['completed']=True; persist(path,record)
            move('park',loading_pose())
        final=read_idle_pose(robot)
        if not close_to(final,previous): raise RobotError('Final arrival mismatch')
        record.update(success=True,final_pose=final)
        return 0
    except Exception as exc:
        record['error']=str(exc); print('STOPPED:',exc,flush=True); return 4
    finally:
        record['finished_at']=datetime.now().astimezone().isoformat()
        # Commit record before releasing motion lock. Failure intentionally leaves lock.
        try:
            persist(path,record)
            if lock is not None: os.close(lock); lock_path.unlink()
        finally:
            robot.close(disable=False)
        print('RECORD:',path,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',default=DEFAULT_PLAN)
    p.add_argument('--start',type=int,default=0); p.add_argument('--count',type=int,default=2)
    p.add_argument('--action',choices=['check','hover','paint','park'],default='check')
    p.add_argument('--confirm-site',action='store_true')
    p.add_argument('--registration-record')
    p.add_argument('--resume-hover-record')
    p.add_argument('--remaining',action='store_true',help='Explicitly run all remaining reviewed strokes with current manual pigment load')
    p.add_argument('--resume-high-record',help='Reviewed speed-change interruption after a completed stroke at high level')
    a=p.parse_args()
    raise SystemExit(run(a.plan,a.start,a.count,a.action,a.confirm_site,a.registration_record,
                        resume_hover_record=a.resume_hover_record,remaining=a.remaining,
                        resume_high_record=a.resume_high_record))
