"""One JSON -> black -> explicitly selected colors, with durable safe pauses."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path

from project_paths import ROBOT_LOG_DIR
from painting_session import (Console, Session, StageControl, PAUSED, normalized,
                              save, session_lock, unfinished_sessions)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def make_bundle(plan, ink_job):
    return dict(format='arm-painting-bundle-v1', version=1, workflow=plan, ink_job=ink_job)


def load_artwork(path):
    from run_mountain_balanced import load_reviewed
    from run_color_from_job import validate_plan
    path = Path(path)
    data = read_json(path)
    if data.get('format') == 'arm-painting-bundle-v1':
        ink, plan = data['ink_job'], data['workflow']
    elif data.get('format') == 'painting-workflow-design-v1':
        if not path.stem.endswith('_计划'):
            raise ValueError('旧计划需要同名“_黑线任务.json”；也可先打包成完整绘画 JSON。')
        ink = read_json(path.with_name(path.stem[:-3] + '_黑线任务.json'))
        plan = data
    elif data.get('format') == 'arm-painting job':
        ink = data
        companion = path.with_name(path.stem.removesuffix('_黑线任务') + '_计划.json')
        plan = read_json(companion) if companion.exists() else None
    else:
        raise ValueError('不是本项目的绘画 JSON。需要完整绘画包、计划或黑线任务；普通 JSON 不能直接运动。')
    strokes, mapper, policy, meta = load_reviewed(ink)
    if plan is not None:
        if plan.get('format') != 'painting-workflow-design-v1':
            raise ValueError('完整绘画包的 workflow 格式错误。')
        validate_plan(plan, ink)
    return ink, plan, strokes, meta


def manifest_for(strokes, plan, meta):
    from run_painting_experiment import calibration_snapshot
    from run_mountain_balanced import drawing_calibration
    from artwork_order import ink_stages
    ink_parts = ink_stages({'meta':meta}, plan)
    stages = {'black': ink_parts['black']}
    color_geometry = None
    if plan:
        module = plan['modules']['color']
        for stroke in module['strokes']:
            stages.setdefault(stroke['color'], []).append(stroke)
        color_geometry = {k: module[k] for k in
                          ('image_size', 'image_corners', 'mm_per_px', 'motion_candidate')}
    if 'title' in ink_parts:
        stages['title'] = ink_parts['title']
    for name, values in stages.items():
        ids = [s.get('id', '%s-%d' % (name, i)) for i, s in enumerate(values)]
        if len(ids) != len(set(ids)):
            raise ValueError('笔触ID重复：' + name)
    return normalized(dict(stages=stages, registration=dict(
        site=calibration_snapshot(), image_size=meta['image_size'],
        paper_z=meta['paper_z'], canvas=meta['canvas_corners'],
        ink_calibration=drawing_calibration(meta['ink_execution']['calibration']),
        color_geometry=color_geometry)))


def execute_stage(session, stage, label, console, callback):
    control = StageControl(session, stage, label, console)
    session.begin(stage)
    trace = session.path.with_suffix('.log')
    try:
        # Keep motion diagnostics on disk. The console shows one live progress bar.
        with trace.open('a', encoding='utf-8') as log, contextlib.redirect_stdout(log):
            result = callback(control)
    except BaseException:
        session.uncertain()
        raise
    if result == PAUSED:
        if session.data['status'] != 'paused' or session.data['active']:
            session.uncertain()
            raise ValueError('暂停位置未确认，不能自动续画。')
        if session.save_progress:
            console.say('进度已保存；重新打开同一个 JSON 可从下一笔继续。')
        else:
            console.say('已退出，未保存续画进度；重新打开会从第一笔开始。')
    elif result == 0:
        if session.count(stage) != control.total:
            session.uncertain()
            raise ValueError('执行器没有确认全部笔触，不能进入下一阶段。')
        session.parked('ready')
    else:
        session.uncertain()
        console.say('运动未正常完成，请用控制器确认机械臂已停止。已保留记录，不能自动续画。详情：' + str(trace))
    return result


def run_file(path, console=None, folder=ROBOT_LOG_DIR, ink_runner=None, color_runner=None,
             station_reader=None, confirmed=False, save_progress=None, prepared_artwork=None, fresh_run=False,
             free_stages=False, initial_stage='black'):
    from run_mountain_balanced import execute_reviewed_job
    from run_color_experiment import run
    from unified_painting import color_station
    from color_profile import ColorProfile
    from color_selection import ALIASES, LABELS
    console = console or Console()
    ink_runner = ink_runner or execute_reviewed_job
    color_runner = color_runner or run
    station_reader = station_reader or color_station
    with session_lock(folder):
        ink, plan, strokes, meta = prepared_artwork if prepared_artwork is not None else load_artwork(path)
        if save_progress is None:
            save_progress = (plan or {}).get('save_progress', True)
        manifest = manifest_for(strokes, plan, meta)
        if free_stages and (initial_stage not in manifest['stages'] or not manifest['stages'][initial_stage]):
            raise ValueError('画稿不包含所选绘画阶段。')
        session = Session(folder, path, manifest,
                          save_progress=save_progress, fresh_run=fresh_run)
        if hasattr(console, 'bind_session'):
            console.bind_session(session)
        for other, record in unfinished_sessions(folder):
            if other != session.path:
                raise ValueError('另有未完成画作，请先续画或放弃旧任务：' + record['source'])
        stages = session.data['manifest']['stages']
        remaining = [k for k in stages if session.count(k) < len(stages[k])]
        if not remaining:
            session.parked('complete')
            console.say('这幅画已完成，不会重复绘制。换新纸重画可复制绘画 JSON 为新文件名。')
            return 0
        console.say('画稿：' + str(Path(path).resolve()) +
                    '\n绘画中按 P：完成当前笔后抬笔暂停（不是急停）。纸张位置和笔尖标定需保持一致。')
        if not session.save_progress:
            console.say('已关闭续画进度保存；本次 P/C 可暂停继续，退出后重开从第一笔开始。')
        if not confirmed and console.ask('确认画稿、纸张及笔已准备好，输入 RUN；其他输入返回：') != 'RUN':
            return 0
        session.persist()
        def draw_ink(stage, label):
            from artwork_order import ink_payload
            stage_ink = ink if len(stages[stage]) == len(strokes) else ink_payload(ink, stages[stage])
            # Immutable input for this attempt; source edits take effect only on reopen.
            payload = json.dumps(stage_ink, ensure_ascii=False, sort_keys=True).encode('utf-8')
            cache = Path(folder) / 'session_inputs' / (hashlib.sha256(payload).hexdigest() + '.json')
            if not cache.exists():
                save(cache, stage_ink)
            return execute_stage(session, stage, label, console,
                lambda control: ink_runner(cache, start_index=session.count(stage), control=control))
        if free_stages:
            from artwork_order import available_stages
            counts = {key: len(values) for key, values in stages.items()}
            labels = dict(LABELS, black='勾勒', title='题字')
            chosen = initial_stage
            while True:
                remaining = available_stages(session.data['completed'], counts)
                if not remaining:
                    session.parked('complete')
                    console.say('绘画完成。')
                    return 0
                if chosen is None:
                    console.say('本阶段已完成。请等墨干，按下一阶段需要洗笔并准备墨色；可自由选择下一阶段，也可结束本次作画。')
                    raw = console.ask('选择绘画阶段（' + '、'.join(labels[k] for k in remaining) + '）；回车退出：').lower()
                    if raw in ('', '0', 'q', 'exit'):
                        session.parked()
                        return 0
                    chosen = {'勾勒': 'black', '题字': 'title'}.get(raw, ALIASES.get(raw, raw))
                if chosen not in remaining:
                    console.say('请选择尚未完成的绘画阶段。')
                    chosen = None
                    continue
                stage = chosen
                if stage in ('black', 'title'):
                    result = draw_ink(stage, '黑线' if stage == 'black' else '题字')
                else:
                    module = plan['modules']['color']
                    station = station_reader(stage)
                    result = execute_stage(session, stage, LABELS[stage], console,
                        lambda control: color_runner(profile=ColorProfile(**module['motion_candidate']),
                            confirm_site=True, module=module, color=stage, station=station,
                            log_dir=folder, start_index=session.count(stage), control=control))
                if result:
                    return result
                chosen = None
        if 'black' in remaining:
            result = draw_ink('black', '黑线')
            if result:
                return result
        if plan:
            module = plan['modules']['color']
            while True:
                remaining = [k for k in stages if k not in ('black', 'title') and session.count(k) < len(stages[k])]
                if not remaining:
                    break
                console.say('已停在纸外，等墨干、洗笔/换笔，准备好后再输入颜色。\n待画：' +
                            '、'.join(LABELS[k] for k in remaining))
                prompt = '输入颜色（如红/黄/蓝/淡蓝/绿）；回车%s退出：' % ('保存' if session.save_progress else '')
                raw = console.ask(prompt).lower()
                if raw in ('', '0', 'q', 'exit'):
                    session.parked()
                    return 0
                color = ALIASES.get(raw, raw)
                if color not in remaining:
                    console.say('请选择待画颜色；已完成的颜色不会重复画。')
                    continue
                station = station_reader(color)
                result = execute_stage(session, color, LABELS[color], console,
                    lambda control: color_runner(profile=ColorProfile(**module['motion_candidate']),
                        confirm_site=True, module=module, color=color, station=station,
                        log_dir=folder, start_index=session.count(color), control=control))
                if result:
                    return result
        if 'title' in stages and session.count('title') < len(stages['title']):
            console.say('山水勾勒和全部上色已完成。请等墨干、洗净并擦拭毛笔，准备黑墨最后写题字。')
            while True:
                raw = console.ask('输入颜色（黑/black/title）开始最后题字；回车退出：').lower()
                if raw in ('', '0', 'q', 'exit'):
                    session.parked()
                    return 0
                if raw in ('黑', '黑色', '黑墨', 'black', 'title', '题字'):
                    break
                console.say('请准备黑墨，输入黑或 black。')
            result = draw_ink('title', '题字')
            if result:
                return result
        session.parked('complete')
        console.say('绘画完成。')
        return 0


def main(argv=None):
    from launcher import setup_console
    setup_console()
    parser = argparse.ArgumentParser(description='读取绘画JSON，按颜色确认，支持按笔暂停和续画。')
    parser.add_argument('--job')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--no-progress',dest='save_progress',action='store_const',const=False,default=None,
                        help='关闭磁盘续画进度；P/C 仍可在本次运行内暂停继续')
    parser.add_argument('--pack', help='只将旧计划及黑线任务合成这个单文件，不连接机械臂')
    args = parser.parse_args(argv)
    console = Console()
    try:
        path = args.job or console.ask('拖入绘画 JSON（留空返回）：').strip('"').strip("'")
        if not path:
            return 0
        if args.check or args.pack:
            ink, plan, strokes, meta = load_artwork(path)
            manifest_for(strokes, plan, meta)
            if args.pack:
                if plan is None:
                    raise ValueError('打包整幅画需要配套上色计划。')
                destination = Path(args.pack)
                if destination.exists():
                    raise ValueError('输出文件已存在，请使用新名字。')
                save(destination, make_bundle(plan, ink))
                console.say('完整绘画 JSON：' + str(destination.resolve()))
            else:
                console.say('JSON 离线检查通过；没有连接机械臂。')
            return 0
        return run_file(path, console,save_progress=args.save_progress)
    except (ValueError, KeyError, TypeError, OSError, EOFError, KeyboardInterrupt) as exc:
        console.say('已停止：' + str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
