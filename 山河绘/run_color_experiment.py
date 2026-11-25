"""Interactive real-robot sparse color experiment; one loaded color, 1–2 lines."""
from datetime import datetime
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
from color_profile import ColorProfile
from config import (ArmConfig, CanvasConfig, ShuimoConfig, station_approach_z,
                    z_for_lift, z_for_press)
from geometry import check_waypoints, CanvasMap
from park_color_loading import target_pose
from prepare_robot import close_to, read_idle_pose, wait_arrival
from robot import E6Robot, RobotError
from run_mountain_v4_robot import connect_readonly, no_errors
from run_blue_wash import persist as durable_persist
from project_paths import PROJECT_DIR, ROBOT_LOG_DIR
from experiment_resume import resume_suffix, unresolved_records
from experiment_start import validate_start
from painting_session import PauseRequested, PAUSED


def persist(path, record):
    durable_persist(path, record, compact=True)


def reload_stroke_limit(module, default=2):
    value = (module or {}).get('reload_strokes_per_load', default)
    # 2026-09-30 8 -> 40: 现场实测一次蘸色可画约 2000mm, 不必每 4 笔就补蘸
    if type(value) is not int or not 1 <= value <= 40:
        raise ValueError('Task reload stroke limit must be an integer from 1 to 40')
    return value


RELOAD_LENGTH_MM = 100.0   # 一次蘸色的累计画程上限(mm)


def long_stroke_reload_policy(module):
    """Optional artwork-specific long-contact batches; other jobs stay unchanged."""
    policy = (module or {}).get('long_stroke_reload')
    if policy is None:
        return None
    if isinstance(policy, dict) and policy.get('format') == 'three-length-classes-v1':
        expected = dict(format='three-length-classes-v1', short_below_mm=10.,
                        long_above_mm=20., short_strokes=6, medium_strokes=4,
                        long_strokes=2, max_length_mm=160.)
        if tuple(policy.get(k) for k in ('short_strokes','medium_strokes','long_strokes')) == (8,6,4):
            expected.update(short_strokes=8,medium_strokes=6,long_strokes=4)
        if set(policy) != set(expected) or any(
                type(policy[k]) not in (int, float) or not np.isfinite(policy[k])
                or policy[k] != v for k, v in expected.items() if k != 'format'):
            raise ValueError('Invalid three-class pigment refill settings')
        return dict(policy)
    keys = {'min_length_mm', 'very_long_length_mm', 'strokes_per_load',
            'very_long_strokes_per_load', 'max_length_mm'}
    if not isinstance(policy, dict) or set(policy) not in (keys, keys | {'short_max_length_mm'}):
        raise ValueError('Long-stroke refill settings are incomplete')
    for key in ('min_length_mm', 'very_long_length_mm', 'max_length_mm'):
        if type(policy[key]) not in (int, float) or not np.isfinite(policy[key]):
            raise ValueError('Long-stroke refill lengths must be finite numbers')
    if not 0 < policy['min_length_mm'] < policy['very_long_length_mm'] <= policy['max_length_mm'] <= 300:
        raise ValueError('Long-stroke refill length thresholds are invalid')
    for key in ('strokes_per_load', 'very_long_strokes_per_load'):
        if type(policy[key]) is not int or not 2 <= policy[key] <= 3:
            raise ValueError('Long strokes require 2 or 3 contacts per load')
    if policy['very_long_strokes_per_load'] > policy['strokes_per_load']:
        raise ValueError('Longer contacts cannot increase the batch count')
    if 'short_max_length_mm' in policy:
        value = policy['short_max_length_mm']
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 < value <= 300:
            raise ValueError('Short-stroke refill length limit is invalid')
    return dict(policy)


def color_reload_limits(policy, next_length, used_max_length, default_limit):
    # Legacy jobs keep their original limits. A new six/two policy classifies
    # a mixed batch by its longest contact, so short strokes after two long
    # ones cannot silently extend that load to six contacts.
    if policy is None:
        return default_limit, RELOAD_LENGTH_MM
    if policy.get('format') == 'three-length-classes-v1':
        longest = max(next_length, used_max_length)
        if longest < policy['short_below_mm']:
            limit = policy['short_strokes']
        elif longest <= policy['long_above_mm']:
            limit = policy['medium_strokes']
        else:
            limit = policy['long_strokes']
        return min(default_limit, limit), policy['max_length_mm']
    batch_length = max(next_length, used_max_length) if 'short_max_length_mm' in policy else next_length
    if batch_length < policy['min_length_mm']:
        return default_limit, policy.get('short_max_length_mm', RELOAD_LENGTH_MM)
    very_long = max(next_length, used_max_length) >= policy['very_long_length_mm']
    limit = policy['very_long_strokes_per_load'] if very_long else policy['strokes_per_load']
    return min(default_limit, limit), policy['max_length_mm']


def needs_color_reload(loaded_at, used_strokes, used_length, next_length, now, limit,
                       *, stroke_count_only=False, length_limit_mm=RELOAD_LENGTH_MM):
    if type(stroke_count_only) is not bool:
        raise ValueError('stroke_count_only must be a boolean')
    if stroke_count_only:
        return loaded_at is None or used_strokes >= limit
    return (loaded_at is None or used_strokes >= limit
            or used_length + next_length > length_limit_mm or now - loaded_at > 60)


def plan_lines(x, y, count=2, length=20., spacing=8., press=.22):
    """Robot User/Tool coordinates, NOT image coordinates. No implicit site choice."""
    if not np.isfinite([x,y,length,spacing,press]).all(): raise ValueError('Coordinates must be finite')
    if count not in (1,2) or not 5<=length<=30 or not 5<=spacing<=20:
        raise ValueError('Requires 1–2 lines, length 5–30 mm, spacing 5–20 mm')
    ColorProfile(press_mm=press).to_dict()
    polygon=np.asarray(CanvasConfig.ROBOT_CORNERS,np.float32)
    lines=[]
    for i in range(count):
        a=(float(x),float(y+i*spacing)); b=(float(x+length),float(y+i*spacing))
        for q in np.linspace(a,b,61):
            if cv2.pointPolygonTest(polygon,tuple(map(float,q)),True)<20:
                raise ValueError('Trial must remain at least 20 mm inside the calibrated canvas')
        pose=lambda xy,z: (*xy,z,*ArmConfig.RPY)
        sequence=[('high',pose(a,station_approach_z())),('hover',pose(a,z_for_lift(15))),
                  ('touch',pose(a,z_for_press(press))),('draw',pose(b,z_for_press(press))),
                  ('lift',pose(b,z_for_lift(15))),('raise',pose(b,station_approach_z()))]
        lines.append(dict(id='trial-%d'%i,xy=[a,b],sequence=sequence))
    poses=[p for line in lines for _,p in line['sequence']]+[target_pose()]
    ok,problems=check_waypoints([dict(x=p[0],y=p[1],z=p[2]) for p in poses])
    if not ok: raise ValueError('Trial outside reach: '+str(problems))
    return lines


def module_lines(module, color, profile):
    """Convert an independently planned module to bounded, physical pen motions."""
    if module.get('format') not in ('sparse-color-design-v1','outline-design-v1','pressure-test-design-v1'):
        raise ValueError('Unsupported experiment module')
    mapper=CanvasMap(module['image_size'],module['image_corners'])
    lines=[]; skipped=[]
    for s in module['strokes']:
        if color!='ink' and s.get('color')!=color: continue
        points=np.asarray(s['points'],float)
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
            raise ValueError('Invalid path')
        xy=mapper.to_robot(points)
        polygon=np.asarray(CanvasConfig.ROBOT_CORNERS,np.float32)
        # Skip edge paths explicitly instead of clipping or shifting composition.
        if any(cv2.pointPolygonTest(polygon,tuple(map(float,q)),True)<20 for q in xy):
            skipped.append(s['id']); continue
        presses=s.get('press'); speeds=s.get('speed')
        if presses is None: presses=[profile['press_mm']]*len(xy)
        if speeds is None: speeds=[profile['draw_velocity_percent']]*len(xy)
        presses=list(map(float,presses)); speeds=list(map(float,speeds))
        if len(presses)!=len(xy) or len(speeds)!=len(xy):
            raise ValueError('Invalid contour profile')
        if color=='ink':
            # 黑笔自带逐点速度，按档案系数放大（想更快/更慢只改 ColorProfile）
            speeds=[float(v)*float(profile['ink_draw_speed_scale']) for v in speeds]
            # 上限跟 ShuimoConfig.PRESS_MAX_MM 走（以前写死 .95，改笔压后会误报）
            if (not np.isfinite(presses).all() or min(presses)<0
                    or max(presses)>float(ShuimoConfig.PRESS_MAX_MM)+1e-9):
                raise ValueError('Invalid ink pressure')
        else:
            # 红花瓣的压力由计划给出（毫米），不允许超过档案声明的峰值
            if not np.isfinite(presses).all() or min(presses)<0 \
                    or max(presses)>float(profile['press_mm'])+1e-9:
                raise ValueError('Red petal pressure must stay within the profile ceiling')
        if not np.isfinite(speeds).all() or min(speeds)<1:
            raise ValueError('Invalid speed')
        pose=lambda q,z: (*map(float,q),z,*ArmConfig.RPY)
        lift = profile.get('interstroke_lift_mm', 5.) if color != 'ink' or module.get('format') == 'pressure-test-design-v1' else 15.
        seq=[('high',pose(xy[0],station_approach_z())),('hover',pose(xy[0],z_for_lift(lift))),
             ('touch',pose(xy[0],z_for_press(float(presses[0]))))]
        seq += [('draw',pose(q,z_for_press(float(p)))) for q,p in zip(xy[1:],presses[1:])]
        seq += [('lift',pose(xy[-1],z_for_lift(lift))),('raise',pose(xy[-1],station_approach_z()))]
        lines.append(dict(id=s['id'],xy=xy.tolist(),sequence=seq,
                          draw_speeds=[int(np.clip(v,5,profile['draw_velocity_max'])) for v in speeds[1:]]))
    poses=[p for line in lines for _,p in line['sequence']]+[target_pose()]
    reachable, problems = check_waypoints([dict(x=p[0],y=p[1],z=p[2]) for p in poses])
    if not reachable:
        raise ValueError('Module exceeds reach: ' + '; '.join(problems))
    return lines,skipped


def run(x=None, y=None, count=2, length=20., spacing=8., profile=None, confirm_site=False,
        log_dir=ROBOT_LOG_DIR, module=None, color=None, station=None, resume_record=None,
        start_index=0, control=None, save_progress=None):
    # One fresh site confirmation covers the whole small trial, never each line.
    if not confirm_site: raise ValueError('Current site and blank trial area must be confirmed')
    if station is not None: station.validate()
    if save_progress is None:
        save_progress = getattr(getattr(control, 'session', None), 'save_progress', True)
    if type(save_progress) is not bool:
        raise ValueError('save_progress must be a boolean')
    if resume_record and not save_progress:
        raise ValueError('Memory-only color runs cannot resume a motion record')
    reload_limit = reload_stroke_limit(module, station.strokes_per_load if station is not None else 2)
    reload_policy = (module or {}).get('reload_policy', 'standard')
    if reload_policy not in ('standard', 'every_n_strokes'):
        raise ValueError('Unknown reload policy')
    if reload_policy == 'every_n_strokes' and (module or {}).get('format') != 'pressure-test-design-v1':
        raise ValueError('Fixed-count loading is reserved for pressure calibration')
    long_reload = long_stroke_reload_policy(module)
    if long_reload is not None and reload_policy != 'standard':
        raise ValueError('Long-stroke batching requires the standard refill policy')
    profile=(profile or ColorProfile()).to_dict()
    lines=plan_lines(x,y,count,length,spacing,profile['press_mm']) if module is None else module_lines(module,color,profile)[0]
    if not lines: raise ValueError('No paths within the experimental paper margin')
    if type(start_index) is not int or not 0 <= start_index <= len(lines):
        raise ValueError('Invalid color continuation index')
    if resume_record and start_index:
        raise ValueError('Cannot combine two continuation mechanisms')
    total_lines = len(lines)
    lines = lines[start_index:]
    folder=Path(log_dir); folder.mkdir(parents=True,exist_ok=True)
    audit=None
    if resume_record:
        source=Path(resume_record).resolve()
        if source.parent!=folder.resolve(): raise ValueError('续画记录必须来自同一个 robot_runs 目录')
        for prior in folder.glob('color_experiment_*.json'):
            attempted=json.loads(prior.read_text(encoding='utf8'))
            if (attempted.get('resume_audit',{}).get('source')==str(source)
                    and (attempted.get('steps') or attempted.get('pending_move') or attempted.get('contacts'))):
                raise ValueError('此中断记录已有续画运动，请核查较新的续画记录，不能重复开始')
        lines,audit=resume_suffix(source,lines)
    for prior,old in unresolved_records(folder):
        if not audit or prior.resolve()!=Path(audit['source']):
            # ★ 2026-09-26 改成中文 + 给出下一步：原来只有一句英文
            #   "Unresolved previous movement/contact"，现场看不出该怎么办。
            #   这是一道【故意的】闸门：上一条记录里还有"指令发出去了但没确认到位"
            #   或者"有落笔点没标完成"，软件不知道机械臂到底停在哪、纸上多画了没有，
            #   所以拒绝开新画 —— 绝不允许靠猜继续。
            raise RobotError(
                '上一条中断记录还没了结，不能开新画：\n'
                '    %s\n'
                '  先看它到底断在哪（只读，不动文件）：\n'
                '    uv run python scripts/resolve_interrupt.py\n'
                '  (A) 接着画同一张纸：\n'
                '    uv run python scripts/resolve_interrupt.py --confirm-reviewed\n'
                '    uv run python scripts/run_plum_stage.py --stage ink '
                '--resume-record robot_runs\\%s --confirm-site\n'
                '  (B) 不画了、改参数换新纸：\n'
                '    .\\.venv\\Scripts\\python.exe abandon_old_task.py\n'
                '    或双击“放弃旧任务.bat”，按提示确认 NEW'
                % (prior, prior.name))
    path=folder/('color_experiment_%s_%s.json'%(datetime.now().strftime('%Y%m%d_%H%M%S_%f'),uuid4().hex[:8]))
    record=dict(action='real-color-experiment',success=False,profile=profile,lines=lines,
                pigment_station=asdict(station) if station else None,dips=[],
                long_stroke_reload=long_reload,short_reload_stroke_limit=reload_limit,
                paper_z=ArmConfig.PAPER_Z,user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,
                canvas_corners=CanvasConfig.ROBOT_CORNERS,rpy=ArmConfig.RPY,
                contacts=[],steps=[],started_at=datetime.now().astimezone().isoformat())
    motion_attempted = False
    if audit: record['resume_audit']=audit
    def write_record(force=False):
        if save_progress:
            persist(path, record)
        elif force:
            # No geometry, completed counts or per-stroke disk writes. Retain
            # only a start/end marker so a crashed controller queue is guarded.
            # Keep scalar setup evidence even when continuation is disabled.
            marker = {k: record[k] for k in ('action', 'success', 'started_at',
                       'finished_at', 'paused', 'interrupted', 'error', 'diagnostic_path',
                       'profile', 'pigment_station', 'paper_z', 'user', 'tool', 'rpy',
                       'long_stroke_reload', 'short_reload_stroke_limit') if k in record}
            marker['resumable'] = False
            marker['motion_attempted'] = motion_attempted
            if motion_attempted and not record.get('success') and not record.get('paused'):
                marker['pending_move'] = dict(name='memory-only-unconfirmed-run')
            persist(path, marker)
    def prepare_motion():
        nonlocal motion_attempted
        if not motion_attempted:
            motion_attempted = True
            if not save_progress:
                # Persist once before the FIRST movement command, not at
                # startup. Failed read-only checks must not block the next run.
                write_record(force=True)
    robot=E6Robot(); lock=None; journal=None
    # Share lock with the existing real blue-wash runner.
    lock_path=folder/'blue_wash_motion.lock'
    try:
        lock=os.open(lock_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(lock,str(path.resolve()).encode('utf8')); os.fsync(lock)
        if (module or {}).get('motion_diagnostics'):
            from motion_diagnostics import MotionJournal
            diagnostic_path = path.with_name(path.stem + '_diagnostic.jsonl')
            journal = MotionJournal(diagnostic_path)
            journal.poll_interval = .15
            robot._diagnostic = journal
            record['diagnostic_path'] = str(diagnostic_path.resolve())
        write_record(force=True)
        connect_readonly(robot)
        previous=read_idle_pose(robot,record)
        # 绘图起点放宽：只要"纸外 + 姿态是标定值 + 可达"就允许，不再要求必须站在
        # 装载点 10 mm 内（现场反馈每次都在墨盘附近起步却总被拒）。
        # 原地整组转姿态那种恢复动作仍走严格模式（restore_drawing_pose.py）。
        initial_high=validate_start(previous, require_park=False)
        response=robot.send('SpeedFactor(%d)'%profile['global_speed_factor'])
        if not robot._ok(response): raise RobotError('SpeedFactor rejected: '+str(response))
        # Explicit profiles require acknowledgement. Zero ratios mean inherit
        # controller state; they do not reset acceleration or send Acc*(0).
        if profile.get('acceleration_required'):
            from motion_settings import apply_acceleration
            apply_acceleration(robot,dict(joint=profile['accel_joint_percent'],linear=profile['accel_linear_percent']))
        for cmd,key in (() if profile.get('acceleration_required') else (('AccJ','accel_joint_percent'),('AccL','accel_linear_percent'))):
            ratio=int(profile.get(key,0) or 0)
            if not ratio: continue
            reply=robot.send('%s(%d)'%(cmd,ratio))
            if robot._ok(reply):
                print('ACCEL %s=%d%%'%(cmd,ratio),flush=True)
            else:
                print('WARN: %s(%d) 被控制器拒绝，沿用默认加速度: %r'%(cmd,ratio,reply),flush=True)
        record['accel']=dict(joint=profile.get('accel_joint_percent'),linear=profile.get('accel_linear_percent'))
        inherited_acceleration = not any(record['accel'].values())
        if inherited_acceleration:
            print('ACCEL 沿用控制器当前加速度，未发送AccJ/AccL，实际比例未读回。', flush=True)
        if journal:
            journal.record('motion_settings', profile=profile,
                           acceleration=('inherited controller setting; not read back'
                                         if inherited_acceleration else 'see acceleration command replies'))
        write_record()

        contact_active = False

        def checkpoint():
            # The whole contact sequence is durable BEFORE touchdown. Avoid full
            # JSON writes/fsync while the brush is on paper. On any exception the
            # finally block still saves the exact in-memory pending move.
            if not contact_active:
                write_record()

        def move(name,target,drawing=False,velocity=None):
            nonlocal previous
            # During contact, the preceding wait_arrival already verified this
            # endpoint and alarms. Avoid querying the same idle pose twice.
            if not contact_active and not close_to(read_idle_pose(robot),previous):
                raise RobotError('Unexpected pose before '+name)
            if velocity is None:
                velocity=profile['draw_velocity_percent'] if drawing else profile['travel_velocity_percent']
                if not drawing and profile.get('approach_velocity_percent'):
                    vertical = abs(previous[0]-target[0]) < .01 and abs(previous[1]-target[1]) < .01
                    velocity = (profile['approach_velocity_percent'] if vertical
                                else profile.get('horizontal_velocity_percent') or velocity)
            velocity=int(np.clip(velocity,1,100))
            record['pending_move']=dict(name=name,target=target,velocity=velocity)
            checkpoint()
            command=('MovL(pose={%.4f,%.4f,%.4f,%.4f,%.4f,%.4f},user=%d,tool=%d,v=%d,cp=0)'%
                     (*target,ArmConfig.USER_INDEX,ArmConfig.TOOL_INDEX,velocity))
            print('MOVING',name,'v=',velocity,flush=True)
            if journal:
                journal.context.update(action=name, waypoint=dict(tag=name, target=list(target), velocity=velocity))
                journal.record('action_start')
            prepare_motion()
            reply=robot.send(command)
            if not robot._ok(reply): raise RobotError('Move rejected: '+str(reply))
            wait_arrival(robot,target)
            previous=target
            if journal: journal.record('action_complete')
            record['steps'].append(record.pop('pending_move')); checkpoint()

        def draw_path(line, draws):
            nonlocal previous
            if not contact_active and not close_to(read_idle_pose(robot),previous):
                raise RobotError('Pose changed before drawing')
            record['pending_move']=dict(name=line['id']+':draw-path',targets=[p for _,p in draws])
            checkpoint()
            # Queue one stroke without artificial per-vertex sleeps or idle waits.
            # ★ 行笔段用档案里的 cp（默认 70）圆滑过渡：原来写死 cp=0，控制器在每个
            #   1.5mm 轨迹点都要减速到 0 再起步，现场看着就是"一抖一抖"。
            #   抬笔/落笔/横移仍是 cp=0（落点要准、不能切角），只有这里放平滑。
            speeds=line.get('draw_speeds',[profile['draw_velocity_percent']]*len(draws))
            cp=int(profile.get('continuous_path_ratio',0))
            prepare_motion()
            for (_,target),velocity in zip(draws,speeds):
                if robot.get_mode() not in (5,7,8) or not no_errors(robot.get_error_id()):
                    raise RobotError('Drawing interrupted or alarmed')
                command=('MovL(pose={%.4f,%.4f,%.4f,%.4f,%.4f,%.4f},user=%d,tool=%d,v=%d,cp=%d)'%
                         (*target,ArmConfig.USER_INDEX,ArmConfig.TOOL_INDEX,velocity,cp))
                if not robot._ok(robot.send(command)): raise RobotError('Drawing segment rejected')
            wait_arrival(robot,draws[-1][1])
            previous=draws[-1][1]
            record['steps'].append(record.pop('pending_move')); checkpoint()

        def dip_batch(high, dip, velocity):
            """Keep vertical/cup clearance points; wait only at the dip endpoint."""
            nonlocal previous
            from queued_motion import send_batch
            targets = [(*previous[:2], station_approach_z(), *ArmConfig.RPY), high, dip]
            record['pending_move'] = dict(name='queued-dip', targets=targets, velocity=velocity)
            checkpoint()
            motions = [(targets[0], profile.get('approach_velocity_percent') or profile['travel_velocity_percent'], 0),
                       (high, profile.get('horizontal_velocity_percent') or profile['travel_velocity_percent'], 0),
                       (dip, velocity, 0)]
            prepare_motion()
            if journal:
                journal.context.update(action='cup-approach')
                metadata=[dict(tag=tag, target=list(target), velocity=v) for tag,(target,v,_) in
                          zip(('cup-raise','cup-high','cup-dip'),motions)]
                send_batch(robot, motions, wait_arrival, metadata=metadata)
            else:
                send_batch(robot, motions, wait_arrival)
            previous = dip
            record.pop('pending_move')
            record['steps'].append(dict(name='queued-dip', target=dip))
            checkpoint()

        # Normalize height vertically before any horizontal transfer; no rotation.
        if abs(previous[2]-initial_high[2])>.01:
            move('normalize-start-height',initial_high)
        loaded_at=None; used_length=0.; used_strokes=0; used_max_length=0.
        completed_drawing_length_mm=0.0
        progress_label = {'blue2': '淡蓝', 'blue': '蓝色', 'yellow': '黄色', 'red': '红色', 'green': '绿色'}.get(color, '上色')

        def show_progress(completed):
            if control:
                control.progress(start_index + completed)
                return
            ratio = (start_index + completed) / total_lines
            filled = min(24, int(ratio * 24))
            print('%s [%s%s] 已完成 %d/%d 笔（%.1f%%）| 蘸色 %d 次' % (
                progress_label, '#' * filled, '-' * (24-filled),
                start_index + completed, total_lines, ratio * 100, len(record['dips'])), flush=True)

        show_progress(0)
        low_transfers = module is not None and color in ('red','yellow','blue','blue2','green','ink')
        from run_painting_experiment import calibration_snapshot
        initial_calibration = calibration_snapshot()

        def park():
            move('pause-raise', (*previous[:2], station_approach_z(), *ArmConfig.RPY))
            move('pause-park', target_pose())

        def verify():
            if calibration_snapshot() != initial_calibration or not close_to(read_idle_pose(robot), target_pose()):
                raise RobotError('暂停期间姿态或标定改变，不能直接继续。')

        def boundary():
            nonlocal loaded_at
            if control and control.boundary(park, verify):
                loaded_at = None  # Refill after a pause, regardless of its duration.
                return True
            return False

        for completed, line in enumerate(lines, 1):
            if journal:
                journal.context = dict(color=color, stroke_number=start_index+completed,
                                       stroke_id=line['id'], dip_number=len(record['dips']))
            paused = boundary()
            from_high = completed == 1 or paused
            length=float(np.linalg.norm(np.diff(np.asarray(line['xy']),axis=0),axis=1).sum())
            effective_limit, length_limit = color_reload_limits(long_reload, length, used_max_length, reload_limit)
            if station is not None and needs_color_reload(
                    loaded_at, used_strokes, used_length, length, time.monotonic(), effective_limit,
                    stroke_count_only=reload_policy == 'every_n_strokes', length_limit_mm=length_limit):
                offset = station.level_offset(len(record['dips']), completed_drawing_length_mm)
                high,dip=station.poses(len(record['dips']), completed_drawing_length_mm)
                from_high = True
                if journal: journal.context.update(dip_number=len(record['dips'])+1)
                if station.level_compensation_enabled:
                    print('液面下降补偿 %.3f mm（累计行笔 %.1f mm，第 %d 次蘸色）' % (offset,completed_drawing_length_mm,len(record['dips'])+1),flush=True)
                dip_v=int(profile['dip_velocity_percent'])   # 进出杯子保持慢速
                dip_batch(high, dip, dip_v)
                time.sleep(station.dwell_s)
                move('cup-withdraw',high,velocity=dip_v)
                time.sleep(station.drip_s)
                loaded_at=time.monotonic(); used_length=0.; used_strokes=0; used_max_length=0.
                record['dips'].append(dict(before_stroke=line['id'],at=datetime.now().astimezone().isoformat(),
                    level_offset_mm=offset, completed_drawing_length_mm=completed_drawing_length_mm, dip_z=dip[2]))
                write_record()
            drawn=False
            if low_transfers:
                from queued_motion import send_batch
                # Previous batch endpoint was already checked by wait_arrival.
                sequence=[(n,p) for n,p in line['sequence']
                          if n != 'raise' and not (n == 'high' and not from_high)]
                record['contacts'].append(dict(id=line['id'],xy=line['xy'],completed=False))
                record['pending_move']=dict(name=line['id']+':queued-stroke',targets=[p for _,p in sequence])
                write_record()
                speeds=iter(line.get('draw_speeds',[profile['draw_velocity_percent']]*sum(n=='draw' for n,_ in sequence)))
                motions=[]
                for i,(name,target) in enumerate(sequence):
                    velocity=next(speeds) if name=='draw' else profile['travel_velocity_percent']
                    if name == 'touch':
                        velocity = profile.get('touchdown_velocity_percent') or velocity
                    elif name == 'lift':
                        velocity = profile.get('lift_velocity_percent') or velocity
                    elif name in ('hover', 'high'):
                        start = previous if i == 0 else sequence[i-1][1]
                        if abs(start[2]-target[2]) < .01 and np.linalg.norm(np.array(start[:2])-target[:2]) > .01:
                            velocity = profile.get('horizontal_velocity_percent') or velocity
                        elif abs(start[0]-target[0]) < .01 and abs(start[1]-target[1]) < .01:
                            velocity = profile.get('approach_velocity_percent') or velocity
                    cp=profile['continuous_path_ratio'] if name=='draw' and i+1<len(sequence) and sequence[i+1][0]=='draw' else 0
                    motions.append((target,int(velocity),int(cp)))
                prepare_motion()
                metadata = [dict(tag=name,target=list(target),velocity=motion[1])
                            for (name,target),motion in zip(sequence,motions)]
                # Confirm the high transfer and vertical approach before placing
                # any contact commands in the controller queue.
                contact = next(i for i,(name,_) in enumerate(sequence) if name == 'touch')
                parts = ((0,contact,'paper-approach'),(contact,len(motions),'paper-stroke')) if from_high and profile.get('approach_velocity_percent') else ((0,len(motions),'paper-stroke'),)
                for start,end,action in parts:
                    if start == end: continue
                    options = {'continuous': True} if action == 'paper-stroke' else {}
                    if journal:
                        journal.context.update(action=action)
                        send_batch(robot,motions[start:end],wait_arrival,metadata=metadata[start:end],sequence_offset=start,**options)
                    else:
                        if start:
                            send_batch(robot,motions[start:end],wait_arrival,sequence_offset=start,**options)
                        else:
                            send_batch(robot,motions[start:end],wait_arrival,**options)
                previous=sequence[-1][1]
                record.pop('pending_move')
                record['steps'].append(dict(name=line['id']+':queued-stroke'))
                for name,target in sequence:
                    if name != 'draw':
                        record['steps'].append(dict(name=line['id']+':'+name,target=target))
                record['contacts'][-1]['completed']=True
                write_record()
                used_length+=length; used_strokes+=1; used_max_length=max(used_max_length,length); completed_drawing_length_mm+=length
                show_progress(completed)
                continue
            for name,target in line['sequence']:
                if low_transfers and (name == 'raise' or (name == 'high' and not from_high)):
                    continue
                if name=='touch':
                    if not close_to(read_idle_pose(robot),previous):
                        raise RobotError('Pose changed before contact')
                    record['contacts'].append(dict(id=line['id'],xy=line['xy'],completed=False))
                    record['pending_move']=dict(name=line['id']+':contact-sequence',
                        targets=[p for n,p in line['sequence'] if n in ('touch','draw','lift')])
                    write_record()
                    contact_active = True
                if name=='draw':
                    if not drawn: draw_path(line,[(n,p) for n,p in line['sequence'] if n=='draw'])
                    drawn=True
                else: move(line['id']+':'+name,target)
                if name=='lift':
                    contact_active = False
                    write_record()
            record['contacts'][-1]['completed']=True; write_record()
            used_length+=length; used_strokes+=1; used_max_length=max(used_max_length,length); completed_drawing_length_mm+=length
            show_progress(completed)
        boundary()
        if low_transfers:
            move('raise-before-return',(*previous[:2],station_approach_z(),*ArmConfig.RPY))
        move('return-loading-high',target_pose())
        record.update(success=True,final_pose=read_idle_pose(robot))
        if not close_to(record['final_pose'],target_pose()): raise RobotError('Final pose mismatch')
        print('实验完成，已回纸外高位。',flush=True)
        return 0
    except PauseRequested:
        record.update(success=False, paused=True, final_pose=read_idle_pose(robot))
        return PAUSED
    except KeyboardInterrupt:
        record.update(success=False, interrupted=True, error='KeyboardInterrupt')
        print('\n已中断上色程序；机械臂停止状态未确认，请用控制器停止并取消旧队列。', flush=True)
        print('换新纸、新任务：结束本程序后运行 .\\.venv\\Scripts\\python.exe abandon_old_task.py', flush=True)
        print('旧记录已保留；同纸续画不要直接从头重跑。' if save_progress else
              '本次未保存续画进度；确认控制器停止后再重新开始。', flush=True)
        return 130
    except Exception as exc:
        if journal: journal.fault(robot, exc)
        record.update(success=False,error=str(exc)); print('STOPPED:',exc,flush=True)
        return 4
    finally:
        record['finished_at']=datetime.now().astimezone().isoformat()
        try:
            write_record(force=True)
            if lock is not None: os.close(lock); lock_path.unlink()
        finally:
            robot.close(disable=False)
            if journal: journal.close()
        print('RECORD:',path,flush=True)


def main():
    os.chdir(PROJECT_DIR)
    prof=ColorProfile().to_dict()
    print('真机上色实验：一次画 1–2 条线；确认后会实际运动。')
    print('User %d / Tool %d，纸面 Z=%.3f，下压 %.2f mm，行笔 v=%d，移动 v=%d，蘸色 v=%d，全局 %d%%。'%
          (ArmConfig.USER_INDEX,ArmConfig.TOOL_INDEX,ArmConfig.PAPER_Z,prof['press_mm'],
           prof['draw_velocity_percent'],prof['travel_velocity_percent'],
           prof['dip_velocity_percent'],prof['global_speed_factor']))
    print('起始位置必须是纸外补色高位：',target_pose())
    try:
        x,y=map(float,input('试色区第一笔起点 X Y（用户坐标 mm，用空格隔开）: ').split())
        count=int(input('笔数 1 或 2 [2]: ').strip() or '2')
        lines=plan_lines(x,y,count)
        for line in lines: print(line['id'],line['xy'],'沿用户坐标 +X 方向画 20 mm')
        print('确认：纸/笔/垫板与当前标定一致；上述整个试色区空白、干燥；')
        print('已补单色且不滴液；高位往返路径无障碍；人员已移开并在旁看护。')
        if input('全部满足，输入 RUN 开始整个实验；其他输入取消: ').strip()!='RUN': return 0
        return run(x,y,count,confirm_site=True)
    except (ValueError,EOFError,KeyboardInterrupt) as exc:
        print('已取消或输入无效:',exc); return 2


if __name__=='__main__': raise SystemExit(main())
