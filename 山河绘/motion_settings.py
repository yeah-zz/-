"""Explicit acceleration setup before the first movement of each web stage."""
from robot import RobotError


def apply_acceleration(robot, values):
    if not isinstance(values,dict) or set(values)!={'joint','linear'}:
        raise ValueError('加速度设置不完整。')
    if any(type(v) is not int or not 1<=v<=100 for v in values.values()):
        raise ValueError('加速度必须是1到100的整数百分比。')
    for command,key in (('AccJ','joint'),('AccL','linear')):
        reply=robot.send('%s(%d)'%(command,values[key]))
        if not robot._ok(reply) or not robot.wait_idle(120):
            raise RobotError('加速度设置未被控制器确认，本阶段未开始移动，请核查控制器状态。')
