"""Offline site settings and execution rebasing; never connects to hardware."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CUPS = ('black', 'blue', 'green')
BASE_FIELDS = ('x', 'y', 'surface_z', 'rim_z', 'dip_mm', 'dwell_s', 'drip_s', 'strokes_per_load')
COMPENSATION_DEFAULTS = dict(level_compensation_enabled=True, level_every_length_mm=500.0,
                             level_step_mm=0.1, level_max_mm=0.5)
FIELDS = BASE_FIELDS + tuple(COMPENSATION_DEFAULTS)
LABELS = {'black': '黑墨', 'blue': '蓝色', 'green': '绿色'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def frame():
    from config import ArmConfig
    return dict(user=ArmConfig.USER_INDEX, tool=ArmConfig.TOOL_INDEX, rpy=list(ArmConfig.RPY))


def defaults():
    from config import ArmConfig
    from pigment_station import read_measurements, PigmentStation
    measurements = read_measurements()
    values = {'paper_z': ArmConfig.PAPER_Z, 'stations': {}}
    for name in CUPS:
        station = PigmentStation(**measurements[name])
        values['stations'][name] = {key: getattr(station, key) for key in FIELDS}
    return dict(format='shanhe-site-settings-v1', values=values,
                measurements=measurements, frame=frame())


def normalize(values):
    if not isinstance(values, dict) or not {'paper_z','stations'}<=set(values) or set(values)-{'paper_z','stations','speeds','acceleration'}:
        raise ValueError('参数项目不完整，请重新打开修改参数窗口。')
    def number(value, label):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(label + '需要填写有效数值。')
        return float(value)
    result = {'paper_z': number(values['paper_z'], '纸面高度'), 'stations': {}}
    # Legacy web speed/acceleration fields are accepted only to migrate old files.
    # They are discarded and never affect validation or execution.
    if not isinstance(values['stations'], dict) or set(values['stations']) != set(CUPS):
        raise ValueError('请填写黑墨、蓝色、绿色三个杯子的参数。')
    for name in CUPS:
        row = values['stations'][name]
        if not isinstance(row, dict) or set(row) not in (set(BASE_FIELDS), set(FIELDS)):
            raise ValueError(LABELS[name] + '杯参数不完整。')
        row = dict(COMPENSATION_DEFAULTS) | row
        enabled = row['level_compensation_enabled']
        if type(enabled) is not bool:
            raise ValueError(LABELS[name] + '液面补偿开关必须为开或关。')
        result['stations'][name] = {key: number(row[key], LABELS[name] + '杯参数')
                                   for key in FIELDS if key != 'level_compensation_enabled'}
        result['stations'][name]['level_compensation_enabled'] = enabled
        interval, step, maximum = (row[key] for key in ('level_every_length_mm', 'level_step_mm', 'level_max_mm'))
        if not 0 < interval <= 100000 or not 0 < step <= maximum <= 0.5:
            raise ValueError(LABELS[name] + '液面补偿：累计长度须大于零且不超过100000毫米，单次下降须大于零且不超过上限，累计上限不得超过0.5毫米。')
        count = row['strokes_per_load']
        if type(count) is not int or not 1 <= count <= 4:
            raise ValueError(LABELS[name] + '每次蘸取笔数必须是一到四的整数。')
        result['stations'][name]['strokes_per_load'] = count
    return result


def make_record(values, base):
    values = normalize(values)
    if base['frame'] != frame():
        raise ValueError('现场坐标系或毛笔姿态已改变，请先重新标定。')
    record = copy.deepcopy(base)
    record['values'] = values
    from web.motion_profile import DEFAULTS, normalize as normalize_motion
    record['motion_profile'] = normalize_motion(record.get('motion_profile', DEFAULTS))
    for name in CUPS:
        record['measurements'][name].update(values['stations'][name])
    return record


def load_record(path=None):
    record = read(path) if path else defaults()
    if record.get('format') != 'shanhe-site-settings-v1':
        raise ValueError('现场参数记录无效，请重新设置。')
    # Reconstruct the editable portion so a stale duplicate cannot override it.
    return make_record(record['values'], record)


def revision(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def apply(record):
    from config import ArmConfig, PALETTE, validate
    from pigment_station import PigmentStation, set_measurements_override
    record = make_record(record['values'], record)
    ArmConfig.PAPER_Z = record['values']['paper_z']
    black = record['measurements']['black']
    if len(PALETTE) != 1:
        raise ValueError('当前画稿需要单黑墨绘画配置，请先核对现场设置。')
    PALETTE[0].update(dip=(black['x'], black['y']), surface_z=black['surface_z'],
                      dip_mm=black['dip_mm'], dip_time_s=black['dwell_s'])
    set_measurements_override(record['measurements'])
    if validate(need_stations=True):
        raise ValueError('纸面高度超出当前允许范围，请核对纸面高度与运动设置。')
    for name in CUPS:
        try:
            PigmentStation(**record['measurements'][name]).validate()
        except ValueError as exc:
            text = str(exc)
            if 'rim' in text:
                reason = '液面应低于杯沿，杯沿上方应留足安全移动空间。'
            elif 'outside paper' in text:
                reason = '杯子需要放在纸张范围外，并留足间距。'
            elif 'reach' in text:
                reason = '杯位或蘸取高度超出当前允许运动范围。'
            elif 'depth/time' in text:
                reason = '蘸入深度应大于零且不超过两毫米，停留零至三秒，沥液零点五至五秒。'
            else:
                reason = '杯子参数不符合当前运动要求，请重新核对。'
            raise ValueError(LABELS[name] + '：' + reason) from exc
    return record


def prepared_artwork(path, settings_path=None):
    """Keep the original source identity and exact relative strokes/pressures."""
    from pending_artwork import apply_pending
    apply_pending(path)
    apply_pending(path, '待应用_右峰黑线')
    from web.execution import prepare
    prepare(path)
    from run_json_painting import load_artwork, read_json
    if not settings_path:
        from web.motion_profile import DEFAULTS, apply_to_artwork
        return apply_to_artwork(load_artwork(path), DEFAULTS)
    record=apply(load_record(settings_path))
    data = read_json(path)
    if data.get('format') != 'arm-painting-bundle-v1':
        raise ValueError('修改现场参数需要完整绘画画稿。')
    ink, plan = data['ink_job'], data['workflow']
    from config import ArmConfig, PALETTE
    from run_mountain_balanced import drawing_calibration, load_reviewed
    expected = dict(rpy=list(ArmConfig.RPY), user=ArmConfig.USER_INDEX,
                    tool=ArmConfig.TOOL_INDEX, speed_factor=ArmConfig.SPEED_FACTOR, palette=PALETTE)
    snapshot = ink['meta']['ink_execution']
    if drawing_calibration(snapshot['calibration']) != drawing_calibration(expected):
        raise ValueError('画稿坐标系、毛笔姿态或速度设置不一致，请先核对现场标定。')
    # Paper/cup rebasing must preserve every original motion setting.
    # Never replay a former web-injected black/title acceleration field.
    ink['meta'].pop('runtime_acceleration', None)
    ink['meta']['paper_z'] = ArmConfig.PAPER_Z
    snapshot['calibration'] = copy.deepcopy(expected)
    from ink_control import InkPolicy, build_plan
    from geometry import CanvasMap
    from shuimo import ink_palette_index
    from run_mountain_v4_robot import safe_transfers
    import job
    mapper = CanvasMap(ink['meta']['image_size'])
    steps, _ = build_plan(snapshot['base_strokes'], ink_palette_index(), mapper.mm_per_px,
                          InkPolicy(**snapshot['policy']))
    points, _ = job.steps_to_waypoints(steps, mapper)
    ink['waypoints'] = safe_transfers(points)
    strokes, _, _, meta = load_reviewed(ink)
    ink['meta'] = meta
    from run_color_from_job import validate_plan
    validate_plan(plan, ink)
    from unified_painting import color_station
    for color in set(stroke['color'] for stroke in plan['modules']['color']['strokes']):
        color_station(color, ask=lambda _: '')
    from web.motion_profile import apply_to_artwork
    return apply_to_artwork((ink, plan, strokes, meta), record['motion_profile'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', action='store_true')
    parser.add_argument('--saved')
    parser.add_argument('--validate')
    parser.add_argument('--output')
    parser.add_argument('--jobs', nargs='*', default=[])
    args = parser.parse_args()
    try:
        record = load_record(args.saved)
        checks = []
        with contextlib.redirect_stdout(io.StringIO()):
            if args.validate:
                record = make_record(read(args.validate)['values'], record)
                apply(record)
                # Use the normalized snapshot for all checks in this subprocess.
                from painting_session import save
                save(args.output, record)
                for source in args.jobs:
                    artwork = prepared_artwork(source, args.output)
                    from run_json_painting import manifest_for
                    manifest_for(artwork[2], artwork[1], artwork[3])
                    checks.append(dict(strokes=len(artwork[2]), passed=True))
        print(json.dumps({'record': record, 'revision': revision(record), 'checks': checks}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
