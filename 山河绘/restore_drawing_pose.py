"""Supervised small orientation restoration at the outside-paper high station."""
from datetime import datetime
import os
from uuid import uuid4
import numpy as np
from config import ArmConfig
from experiment_start import validate_start
from prepare_robot import read_idle_pose, close_to, move_to
from project_paths import ROBOT_LOG_DIR
from robot import E6Robot, RobotError
from run_mountain_v4_robot import connect_readonly
from run_blue_wash import persist


def target_for(pose):
    if pose is None or len(pose)!=6 or not np.isfinite(pose).all():
        raise ValueError('无法读取有效姿态')
    target=(*pose[:3],*ArmConfig.RPY)
    # Reuse the bounded outside-paper high region; never rotate near paper.
    validate_start(target)
    if max(abs(a-b) for a,b in zip(pose[3:],target[3:]))>3:
        raise ValueError('姿态偏差超过 3°，不做自动恢复')
    return target


def main():
    folder=ROBOT_LOG_DIR; folder.mkdir(exist_ok=True)
    record=dict(action='restore-high-orientation',success=False,steps=[])
    path=folder/('orientation_%s_%s.json'%(datetime.now().strftime('%Y%m%d_%H%M%S'),uuid4().hex[:8]))
    lock_path=folder/'blue_wash_motion.lock'
    lock=None; robot=E6Robot()
    try:
        lock=os.open(lock_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(lock,str(path).encode('utf8')); os.fsync(lock)
        connect_readonly(robot)
        pose=read_idle_pose(robot,record)
        target=target_for(pose)
        print('当前:',pose,'\n目标:',target)
        print('保持工具原点 XYZ，仅恢复三个姿态角；笔尖会随笔杆转动摆动。')
        print('确认笔在纸外高位，笔杆/笔尖周围无杯沿、夹具或人员，现场有人看护。')
        if input('输入 ALIGN 执行小角度恢复；其他输入取消: ').strip()!='ALIGN': return 0
        if not close_to(read_idle_pose(robot),pose): raise RobotError('确认期间位置改变，请重新检查')
        if not robot._ok(robot.send('SpeedFactor(10)')): raise RobotError('速度设置失败')
        record['pending_move']=dict(target=target); persist(path,record)
        move_to(robot,target)
        record['final_pose']=read_idle_pose(robot)
        if not close_to(record['final_pose'],target): raise RobotError('最终姿态未确认')
        record.pop('pending_move'); record['success']=True
        print('姿态恢复完成。未蘸墨、未画线；现在可重新运行 run_plum_experiment.py 续画。')
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        record['error']=str(exc); print('停止:',exc); return 4
    finally:
        record['finished_at']=datetime.now().astimezone().isoformat()
        try:
            persist(path,record)
            if lock is not None:
                os.close(lock)
                if not record.get('pending_move'): lock_path.unlink()
                else: print('运动未确认，保留运动锁；请核查现场及本次日志，勿直接删除。')
        finally: robot.close(disable=False)
        print('记录:',path)


if __name__=='__main__': raise SystemExit(main())
