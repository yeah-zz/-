"""Operator chooses each color after manually preparing the brush."""
import contextlib
import io
from color_profile import ColorProfile
from run_painting_experiment import calibration_snapshot
from project_paths import ROBOT_LOG_DIR
from painting_session import Console, PauseRequested, PAUSED


class ColorProgressControl:
    """Live stroke progress and pause controls, kept only in memory."""
    def __init__(self, console, label, total):
        self.console, self.label, self.total = console, label, total
        self.completed = 0

    def progress(self, completed):
        self.completed = completed
        self.console.progress(self.label, completed, self.total)

    def boundary(self, park, verify):
        if not self.console.poll():
            return False
        park()
        self.console.say('已停在纸外。C继续；Q退出。本次暂停位置仅保留在内存。')
        while True:
            reply = self.console.ask('输入 C 或 Q：').lower()
            if reply == 'q':
                raise PauseRequested()
            if reply == 'c':
                verify()
                return True

LABELS = {'blue2': '淡蓝', 'blue': '蓝色', 'yellow': '黄色', 'red': '红色', 'green': '绿色'}
ALIASES = {'淡蓝': 'blue2', '淡': 'blue2', '蓝': 'blue', '蓝色': 'blue', '黄': 'yellow', '黄色': 'yellow',
           '红': 'red', '红色': 'red', '绿': 'green', '绿色': 'green'}


def paint_selected_colors(plan, stations=None, ask=input, runner=None, station_reader=None,
                          console=None):
    from run_color_experiment import run, module_lines
    from unified_painting import color_station
    runner = runner or run
    station_reader = station_reader or color_station
    stations = dict(stations or {})
    console = console or Console(ask=ask)
    module = plan['modules']['color']
    profile = ColorProfile(**module['motion_candidate'])
    remaining = list(dict.fromkeys(s['color'] for s in module['strokes']))
    if any(c not in LABELS for c in remaining):
        raise ValueError('任务含当前未支持的颜色。')
    initial = calibration_snapshot()
    completed = []
    while remaining:
        print('\n已停住，等待手动换色。剩余颜色：' + '、'.join(LABELS[c] for c in remaining))
        print('保持纸张不动；首次上色先等黑墨干。洗净擦笔或换笔，核对笔尖高度与颜料杯。')
        raw = ask('准备好后输入要画的颜色（红/黄/蓝/绿或red/yellow/blue/green）；回车或0结束：').strip().lower()
        if raw in ('', '0', 'q', 'exit'):
            return 0
        color = ALIASES.get(raw, raw)
        if color in completed:
            print('本轮已完成此颜色，不重复绘制。')
            continue
        if color not in remaining:
            print('本任务没有待画的这种颜色，请从剩余颜色中选择。')
            continue
        if calibration_snapshot() != initial:
            raise ValueError('标定已改变，请重新生成任务。')
        lines, skipped = module_lines(module, color, profile.to_dict())
        if skipped or not lines:
            raise ValueError('上色路径与当前画布边界不匹配，需重新生成任务。')
        if color not in stations:
            stations[color] = station_reader(color)
        print('开始%s，共%d笔。完成后会再次停住。' % (LABELS[color], len(lines)))
        console.say('绘画中按 P：完成当前笔后抬笔暂停；C继续，Q退出。')
        control = ColorProgressControl(console, LABELS[color], len(lines))
        # Keep command chatter in memory; the terminal displays one live bar.
        # On failure show the final diagnostics without saving a progress file.
        diagnostics = io.StringIO()
        with contextlib.redirect_stdout(diagnostics):
            result = runner(profile=profile, confirm_site=True, module=module,
                            color=color, station=stations[color], log_dir=ROBOT_LOG_DIR,
                            save_progress=plan.get('save_progress', True), control=control)
        console.say('')
        if result == PAUSED:
            console.say('已退出。本次位置未保存；重开并选择同一颜色会从该颜色第一笔开始。')
            return result
        if result != 0:
            console.say('\n'.join(diagnostics.getvalue().splitlines()[-12:]))
            return result
        remaining.remove(color)
        completed.append(color)
    print('本次计划中的颜色全部完成。')
    return 0
