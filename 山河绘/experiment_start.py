"""Drawing may start over paper; strict parking remains for pose recovery."""
import numpy as np
import cv2
from config import ArmConfig, CanvasConfig, ShuimoConfig, station_approach_z
from geometry import check_waypoints
from park_color_loading import target_pose
from robot import RobotError

PARK_RADIUS_MM = 10.0        # 严格模式（原地转姿态）允许距装载点的水平偏差
PARK_WARN_MM = 150.0         # 放宽模式：离装载点超过这么多只提示
START_Z_ABOVE_MM = 20.0      # 允许起点比装载高度更高（更负）多少
START_Z_BELOW_MM = 0.5       # 严格模式：允许比装载高度低多少
DRAW_Z_BELOW_MM = 60.0       # 放宽模式（绘图起点）：允许低这么多


def validate_start(pose, require_park=True):
    if pose is None or len(pose)!=6 or not np.isfinite(pose).all():
        raise RobotError('无法读取有效起始位姿')
    if not require_park:
        # No XY parking requirement or arbitrary height window. Before any
        # horizontal transfer, the caller must move to this vertical endpoint.
        sign = 1 if ArmConfig.Z_INCREASES_DOWN else -1
        if sign * (pose[2] - ArmConfig.PAPER_Z) > ShuimoConfig.PRESS_MAX_MM + 1e-6:
            raise RobotError('起点低于当前允许的下压深度，请先抬笔。')
        for current, expected in zip(pose[3:], ArmConfig.RPY):
            if abs(current - expected) > .15:
                raise RobotError('姿态与标定不符；为避免起步时转笔，未发送运动。')
        high = station_approach_z()
        # Never lower an already higher pen during start normalization.
        high = min(high, pose[2]) if sign == 1 else max(high, pose[2])
        normalized = (*pose[:2], high, *pose[3:])
        if not check_waypoints([dict(x=p[0], y=p[1], z=p[2]) for p in (pose, normalized)])[0]:
            raise RobotError('起点或原地抬笔终点超出可达范围。')
        return normalized
    target=target_pose()
    errors=[]; warnings=[]
    distance=float(np.linalg.norm(np.asarray(pose[:2])-target[:2]))
    if require_park:
        if distance>PARK_RADIUS_MM:
            errors.append('请移到纸外停车点附近 %.0f mm 内；当前水平距离 %.2f mm'
                          %(PARK_RADIUS_MM,distance))
        below_limit=START_Z_BELOW_MM
    else:
        if distance>PARK_WARN_MM:
            warnings.append('起点距装载点 %.0f mm（>%.0f mm）：请确认第一段高位横移路径上没有障碍'
                            %(distance,PARK_WARN_MM))
        below_limit=DRAW_Z_BELOW_MM
    if cv2.pointPolygonTest(np.asarray(CanvasConfig.ROBOT_CORNERS,np.float32),tuple(pose[:2]),True)>-20:
        errors.append('起点必须在纸外至少 20 mm')
    sign=1 if ArmConfig.Z_INCREASES_DOWN else -1
    higher=sign*(station_approach_z()-pose[2])
    if not -below_limit<=higher<=START_Z_ABOVE_MM:
        errors.append('Z 必须在装载高度 %.4f 的 %+.1f ~ -%.1f mm 之间；当前 %.4f'
                      %(station_approach_z(),START_Z_ABOVE_MM,below_limit,pose[2]))
    for axis,current,expected in zip(('Rx','Ry','Rz'),pose[3:],ArmConfig.RPY):
        if abs(current-expected)>.15:
            errors.append('%s 姿态不符：当前 %.4f°，要求 %.4f°（±0.15°）'%(axis,current,expected))
    normalized=(*pose[:2],station_approach_z(),*pose[3:])
    if not check_waypoints([dict(x=p[0],y=p[1],z=p[2]) for p in (pose,normalized)])[0]:
        errors.append('起点或竖直调整位置超出配置的可达范围')
    if errors: raise RobotError('起始检查未通过，未发送运动：\n'+'\n'.join(errors))
    for text in warnings:
        print('[提示] '+text)
    return normalized
