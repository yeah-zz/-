"""Apply saved speed settings inside the worker/check process only."""
import json
import math
from pathlib import Path

_base_motion = None
# 2026-10-10: user requested twice the previous 1.2x paper-drawing speed.
DRAW_SPEED_SCALE = 2.4
# Process-local approach/loading limits requested on 2026-10-10. Drawing
# velocity and the saved artwork geometry/pressure/order are independent.
INK_CONTACT_LIMITS = dict(approach=30, touchdown=15, dip=5, withdraw=10)


def paper_draw_scale(plan):
    """Opt-in per-artwork boost; preserve the established default elsewhere."""
    multiplier = ((plan or {}).get('speed_revision') or {}).get('paper_draw_multiplier', 1.0)
    if type(multiplier) not in (int, float) or not math.isfinite(multiplier) or not 1 <= multiplier <= 2:
        raise ValueError('画稿行笔倍率必须在 1 到 2 之间。')
    return DRAW_SPEED_SCALE * multiplier


def prepare(path):
    global _base_motion
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    draw_scale = paper_draw_scale(data.get('workflow', data))
    from config import ArmConfig
    import job
    if _base_motion is None:
        _base_motion = (ArmConfig.SPEED_FACTOR, job.TRANSIT_SPEED, job.dip_waypoints,
                        job.stroke_waypoints)
    # Each artwork starts from the process defaults, including unprofiled jobs.
    # Keep the original function so repeated preparation cannot stack wrappers.
    (ArmConfig.SPEED_FACTOR, job.TRANSIT_SPEED, job.dip_waypoints,
     job.stroke_waypoints) = _base_motion
    original_stroke = job.stroke_waypoints

    def stroke(*args, **kwargs):
        points = original_stroke(*args, **kwargs)
        for point in points:
            if not point['pen_up'] and point['tag'].startswith(('draw:', 'tail:')):
                point['velocity'] = min(100, point['velocity'] * draw_scale)
            elif point['tag'].startswith('down:'):
                point['velocity'] = min(point['velocity'], INK_CONTACT_LIMITS['touchdown'])
        return points

    # Only the new web worker/check process receives this overlay. Reset above
    # prevents repeated preparation from multiplying the boost again.
    job.stroke_waypoints = stroke
    profile = data.get('workflow', data).get('speed_revision')
    values = None
    if profile and profile.get('process_local'):
        values = {key: profile[key] for key in ('global_speed_factor', 'travel', 'dip', 'withdraw')}
        if any(type(value) is not int or not 1 <= value <= 80 for value in values.values()):
            raise ValueError('画稿速度设置无效，请重新核对画稿。')
        ArmConfig.SPEED_FACTOR = values['global_speed_factor']
        job.TRANSIT_SPEED = values['travel']
    job.TRANSIT_SPEED = min(job.TRANSIT_SPEED, INK_CONTACT_LIMITS['approach'])
    original = job.dip_waypoints

    def dip(*args, **kwargs):
        points = original(*args, **kwargs)
        for point in points:
            if point['tag'] == 'dip':
                point['velocity'] = min(values['dip'] if values else point['velocity'], INK_CONTACT_LIMITS['dip'])
            elif point['tag'] == 'dip-out':
                point['velocity'] = min(values['withdraw'] if values else point['velocity'], INK_CONTACT_LIMITS['withdraw'])
        return points

    job.dip_waypoints = dip


def public_error(error):
    """Keep machine diagnostics on disk, show actionable operator language."""
    text = str(error)
    if any(word in text.lower() for word in ('winerror 5', 'winerror 32', 'winerror 33', 'permissionerror', '拒绝访问')):
        return '绘画进度或任务记录保存失败，执行已退出。请核查最后一笔后再处理续画。'
    if 'mode=11' in text or 'mode 11' in text:
        return '机械臂碰撞检测触发，执行已退出。请保持停止，核查接触、毛笔固定和工具负载；不要直接重画。'
    if any(word in text.lower() for word in ('calibration', 'canvas scale', '标定', '画布、标定')):
        return '画稿与当前现场设置不一致，请核对纸张位置、笔尖高度和颜料杯标定后重新准备画稿。'
    if any(word in text.lower() for word in ('connection', 'timed out', 'timeout', '10061', '10060', 'socket')):
        return '机械臂连接失败，请检查控制器电源和本机网络连接。'
    if any(word in text for word in ('Orientation', '姿态', 'Robot mode', 'Alarm', '报警', 'enabled')):
        return '机械臂状态未就绪，请在控制器检查使能、报警和毛笔姿态。'
    if any(word in text for word in ('未完成画作', '上次未正常', '半笔', '会话锁', 'File exists')):
        return '存在未结束的绘画任务，请先核查机械臂和纸面，再续画或处理旧任务。'
    if '\\' in text or '.py' in text or 'Traceback' in text or any('a' <= c.lower() <= 'z' for c in text):
        return '任务未能继续，请检查机械臂状态与画稿。详细原因已保存在本机记录中。'
    return text
