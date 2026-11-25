"""Subprocess bridge to the existing executor, preserving all session safeguards."""
import json
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from painting_session import Console, save
from run_json_painting import run_file
from web.execution import public_error


def calibrated_station(name):
    from unified_painting import color_station
    def missing_measurement(prompt):
        raise ValueError(prompt + '请先在原项目入口完成颜料杯实测标定，再继续此画稿。')
    return color_station(name, ask=missing_measurement)


class WebConsole(Console):
    def __init__(self, control, channel=None):
        self.control = Path(control)
        self.channel = channel
        self.session = None
        self.notice = '执行器正在核查任务与现场状态'
        self.stage_order = []
        super().__init__(ask=self.read_reply, poll=self.pause_requested)

    def publish(self, **kwargs):
        snapshot = {'notice': self.notice, 'prompt': None, 'stageOrder': self.stage_order}
        if self.session:
            data = self.session.data
            snapshot.update(status=data['status'], active=data['active'], completed=dict(data['completed']),
                            stageOrder=list(data.get('stage_order', self.stage_order)))
        snapshot.update(kwargs)
        if self.channel:
            self.channel.publish(snapshot)
        else:
            save(self.control/'status.json', snapshot)

    def bind_session(self, session):
        self.session = session

    def say(self, text):
        if text.startswith('画稿：'):
            self.notice = '画稿已准备，正在检查机械臂状态。'
        elif '等墨干' in text:
            self.notice = '本阶段已完成，可自由选择下一阶段或结束作画。请等墨干，并准备下一阶段的墨色。' if '自由选择' in text else '山水勾勒和上色已完成，请等墨干、洗笔并准备黑墨最后写题字。' if '最后写题字' in text else '本阶段已完成。请等墨干、洗净并擦拭毛笔，准备下一种颜色。'
        elif '已抬笔' in text:
            self.notice = '毛笔已抬起并停靠，可以继续本次作画或结束。本次不保存续画进度。'
        elif '绘画完成' in text or '这幅画已完成' in text:
            self.notice = '整幅画已完成，请取下作品并清洗毛笔。'
        elif '运动未正常完成' in text:
            self.notice = '绘画执行异常退出，请核查机械臂和最后一笔。本次不保存续画进度，不能直接重画。'
        else:
            self.notice = public_error(text)
        self.publish()
        super().say(text)

    def ask(self, text):
        token = uuid.uuid4().hex
        kind = 'pause' if 'C 或 Q' in text else 'stage' if '选择绘画阶段' in text else 'color' if '输入颜色' in text else 'input'
        # Unexpected executor prompts must never receive an invented answer.
        self.publish(prompt={'id': token, 'kind': kind, 'text': '请选择下一阶段，或结束本次作画' if kind == 'stage' else '请洗笔并准备下一种颜色' if kind == 'color' else '已暂停，可以继续或结束作画' if kind == 'pause' else '请检查现场准备情况'})
        while True:
            if self.channel:
                value = self.channel.poll()
                if value and value.get('id') == token:
                    self.publish()
                    return value['value']
                time.sleep(.15)
                continue
            reply = self.control/'reply.json'
            if reply.exists():
                value = json.loads(reply.read_text(encoding='utf-8'))
                reply.unlink()
                if value.get('id') == token:
                    self.publish()
                    return value['value']
            time.sleep(.15)

    def read_reply(self, _):
        raise RuntimeError('Use ask() for the web bridge')

    def pause_requested(self):
        if self.channel:
            command = self.channel.poll()
            return bool(command and command.get('pause'))
        path = self.control/'pause.json'
        if path.exists():
            path.unlink()
            return True
        return False

    def progress(self, label, completed, total):
        # Called only after confirmed physical completion, independent of disk.
        from web.server import LABELS
        stage = next((key for key, value in LABELS.items() if value == label), None)
        if stage and stage not in self.stage_order:
            self.stage_order.append(stage)
        self.notice = '%s中 · 已完成 %d / %d 笔' % ('勾勒' if stage == 'black' else '题字' if stage == 'title' else label + '上色', completed, total)
        if self.session and stage in self.session.data['manifest']['stages']:
            strokes = self.session.data['manifest']['stages'][stage]
            stroke = strokes[min(max(completed-1, 0), len(strokes)-1)] if strokes else {}
            subject = stroke.get('drawing_region')
            if subject:
                self.notice = self.notice.replace('中 ·', '中 · ' + (stroke.get('title_character', '题字') if stage == 'title' else subject) + ' ·')
        self.publish()


if __name__ == '__main__':
    settings_path = sys.argv[sys.argv.index('--settings') + 1] if '--settings' in sys.argv else None
    initial_stage = sys.argv[sys.argv.index('--start-stage') + 1] if '--start-stage' in sys.argv else 'black'
    from web.site_settings import prepared_artwork
    if sys.argv[2] == '--check':
        artwork = prepared_artwork(sys.argv[1], settings_path)
        from run_json_painting import manifest_for
        manifest = manifest_for(artwork[2], artwork[1], artwork[3])
        if not manifest['stages'].get(initial_stage):
            raise ValueError('画稿不包含所选绘画阶段。')
        print('画稿检查通过，没有连接机械臂。')
        raise SystemExit(0)
    from web.bridge import WorkerChannel
    channel = WorkerChannel(os.environ['SHH_BRIDGE_PORT'], os.environ['SHH_BRIDGE_KEY']) if 'SHH_BRIDGE_PORT' in os.environ else None
    console = WebConsole(sys.argv[2], channel)
    try:
        artwork = prepared_artwork(sys.argv[1], settings_path)
        code = run_file(sys.argv[1], console=console, confirmed=True, save_progress=False, fresh_run=True,
                        station_reader=calibrated_station, prepared_artwork=artwork,
                        # Older running servers do not pass --start-stage and
                        # still understand only the original color prompts.
                        free_stages='--start-stage' in sys.argv, initial_stage=initial_stage)
        console.publish(done=True)
        raise SystemExit(code)
    except Exception as exc:
        console.publish(error=public_error(exc), done=True)
        raise
    finally:
        if channel:
            channel.close()
