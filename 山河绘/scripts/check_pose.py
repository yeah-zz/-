"""只读诊断：为什么程序不肯动？

打印当前位姿 vs 标定值、逐轴差多少（含 360 度回绕识别）、
以及"下一步该跑哪条命令"。不发送任何运动指令，不写 robot_runs 记录。
"""
# --- path bootstrap: this script lives in scripts/, put the project root on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import cv2
import numpy as np

from config import ArmConfig, CanvasConfig, PALETTE, station_approach_z
from robot import E6Robot
from run_mountain_v4_robot import connect_readonly, no_errors

TOL_DEG = 0.15          # check_orientation / validate_start 的容差
RESTORE_MAX_DEG = 3.0   # restore_drawing_pose.py 的上限
PREPARE_MAX_DEG = 20.0  # prepare_robot.plan_recovery 的上限


def wrap180(delta):
    """折算到 (-180, 180]，用来识别"其实是同一个姿态、只是差了 360 度"。"""
    return (delta + 180.0) % 360.0 - 180.0


def main():
    robot = E6Robot()
    try:
        connect_readonly(robot)
    except Exception as exc:
        print('[!!] 连不上 %s:%d -> %s' % (ArmConfig.IP, ArmConfig.PORT, exc))
        print('     检查：网线插控制器 LAN1、电脑 IP 设成 192.168.5.x、ping 192.168.5.1 通')
        return 3

    print('=' * 72)
    print(' 只读诊断：为什么程序不肯动机械臂')
    print('=' * 72)

    try:
        mode, alarms, pose = robot.get_mode(), robot.get_error_id(), robot.get_pose()
        print(' RobotMode = %s     （程序要求 5 = 已使能且空闲）' % mode)
        print(' Alarms    = %s' % alarms)
        blocking = []
        if 'not tcp' in str(alarms).lower():
            blocking.append('当前是软件在线控制模式 -> 在 DobotStudio 里切回 TCP/IP 控制模式，'
                            '并断开它的 29999 连接')
        try:
            clear = no_errors(alarms)
        except Exception:
            clear = False
        if not clear:
            blocking.append('有报警未排除 -> 在控制器上查明原因后人工排除（脚本不自动清报警）')
        if mode != 5:
            blocking.append('RobotMode=%s 不是空闲使能态 -> 现场确认停机/急停/碰撞/占用原因'
                            '（脚本不自动复位、不自动使能）' % mode)
        if pose is None or len(pose) != 6 or not np.isfinite(pose).all():
            blocking.append('读不到有效的 6 位姿 -> 检查 User %d / Tool %d 是否真的定义过'
                            % (ArmConfig.USER_INDEX, ArmConfig.TOOL_INDEX))

        if pose is not None and len(pose) == 6 and np.isfinite(pose).all():
            pose = tuple(float(v) for v in pose)
            print('')
            print(' 当前 XYZ    = %9.4f %9.4f %9.4f' % pose[:3])
            print(' 当前 RxRyRz = %9.4f %9.4f %9.4f' % pose[3:])
            print(' 标定 RxRyRz = %9.4f %9.4f %9.4f' % ArmConfig.RPY)
            print('')
            print(' 逐轴差值（raw = 直接相减；wrapped = 折算 360 度后）')
            print('  轴    raw          wrapped      判断')
            worst = 0.0
            wrapped_worst = 0.0
            for name, cur, want in zip(('Rx', 'Ry', 'Rz'), pose[3:], ArmConfig.RPY):
                raw = cur - want
                wrapped = wrap180(raw)
                worst = max(worst, abs(raw))
                wrapped_worst = max(wrapped_worst, abs(wrapped))
                if abs(raw) <= TOL_DEG:
                    verdict = 'OK（在 ±%.2f 内）' % TOL_DEG
                elif abs(wrapped) <= TOL_DEG:
                    verdict = '其实同姿态，只是差了 360 度的表示'
                else:
                    verdict = '超差'
                print('  %-4s %+11.4f  %+11.4f   %s' % (name, raw, wrapped, verdict))
            print('')
            if worst <= TOL_DEG:
                print(' [结论] 姿态合格，姿态不是拦路的原因。看上面 Alarms / RobotMode。')
            elif wrapped_worst <= TOL_DEG:
                print(' [结论] 姿态其实是对的，只是控制器返回的角度差了 360 度。')
                print('        这种情况需要在比较时折算角度；先别动机械臂，把这个结果发我。')
            else:
                eff = wrapped_worst
                print(' [结论] 姿态偏差 %.4f 度/轴（折算后），处于：' % eff)
                if eff <= RESTORE_MAX_DEG:
                    print('        <= %.0f 度  -> 用 restore_drawing_pose.py' % RESTORE_MAX_DEG)
                elif eff <= PREPARE_MAX_DEG:
                    print('        <= %.0f 度  -> 用 prepare_robot.py --execute --confirm-clearance'
                          % PREPARE_MAX_DEG)
                else:
                    print('        > %.0f 度   -> 两个脚本都会拒绝，只能人工/示教器处理'
                          % PREPARE_MAX_DEG)

            # 位置判断
            x, y, z = pose[:3]
            poly = np.asarray(CanvasConfig.ROBOT_CORNERS, np.float32)
            edge = float(cv2.pointPolygonTest(poly, (x, y), True))
            high = station_approach_z()
            park = PALETTE[0].get('dip')
            print('')
            print(' 位置判断')
            print('   离画布边界 %+.1f mm  （正=纸内，负=纸外）' % edge)
            print('   相对纸面高度 %+.1f mm  （纸面 %.2f，当前 %.2f）'
                  % (ArmConfig.PAPER_Z - z, ArmConfig.PAPER_Z, z))
            print('   接近高位 station_approach_z = %.2f' % high)
            if park:
                dist = float(np.hypot(x - park[0], y - park[1]))
                print('   距停车点 (%s) = %.1f mm' % (park, dist))
            print('   restore_drawing_pose.py 要求：纸外 >=20mm、距停车点 <=10mm、'
                  'Z 在 %.2f~%.2f' % (high - 20.0, high + 0.5))
            print('   prepare_robot.py 要求：纸内且离边 >=20mm、参考高度 >=40mm（需转姿态时）')

            print('')
            print(' 兜底命令（保持当前 XYZ，只把姿态整组设成标定值）')
            print('   MovL(pose={%.4f,%.4f,%.4f,%.4f,%.4f,%.4f},user=%d,tool=%d,v=10,cp=0)'
                  % (*pose[:3], *ArmConfig.RPY, ArmConfig.USER_INDEX, ArmConfig.TOOL_INDEX))
            if park:
                print(' 兜底命令（先回纸外高位停车点，含姿态）')
                print('   MovL(pose={%.4f,%.4f,%.4f,%.4f,%.4f,%.4f},user=%d,tool=%d,v=10,cp=0)'
                      % (park[0], park[1], high, *ArmConfig.RPY,
                         ArmConfig.USER_INDEX, ArmConfig.TOOL_INDEX))

        print('')
        print('=' * 72)
        if blocking:
            print(' 阻塞项（按优先级）：')
            for i, item in enumerate(blocking, 1):
                print('  %d) %s' % (i, item))
        else:
            print(' 模式/报警/位姿读取都正常。若程序仍不肯动，把上面完整输出发我。')
        print('=' * 72)
        return 0
    finally:
        robot.close(disable=False)


if __name__ == '__main__':
    raise SystemExit(main())