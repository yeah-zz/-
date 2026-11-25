"""Durable stroke-boundary sessions and console controls; no robot commands here."""
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager

PAUSED = 75


class PauseRequested(Exception):
    """Raised only after verified parking, never while touching paper."""


def normalized(value):
    return json.loads(json.dumps(value))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Encode once and issue one write. json.dump(..., indent=2) performs many
    # tiny Python writes for the multi-MB stroke manifest at every boundary.
    # Keep the same fsync-before-replace ordering and complete snapshot.
    payload = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    # Each writer owns its temporary file. On Windows a reader of the old
    # snapshot can briefly deny replacement; retain the synced new snapshot
    # and retry the commit, never rerun the completed robot stroke.
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
            temp = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(25):
            try:
                os.replace(temp, path)
                break
            except OSError as exc:
                if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 24:
                    raise
                time.sleep(min(.01 * 2**attempt, .1))
    finally:
        if temp is not None:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


@contextmanager
def session_lock(folder):
    """Separate shared guard for the entire black + color session."""
    path = Path(folder) / 'painting_session.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode('ascii'))
        os.fsync(fd)
        yield
    finally:
        os.close(fd)
        path.unlink()


def pause_key():
    if os.name != 'nt' or not sys.stdin.isatty():
        return False
    import msvcrt
    requested = False
    while msvcrt.kbhit():
        requested = msvcrt.getwch().lower() == 'p' or requested
    return requested


class Console:
    def __init__(self, stream=None, ask=None, poll=None):
        self.stream = stream or sys.stdout
        self.reader = ask or input
        self.poll = poll or pause_key

    def say(self, text):
        print('\n' + text, file=self.stream, flush=True)

    def ask(self, text):
        print('\n' + text, end='', file=self.stream, flush=True)
        return self.reader('').strip()

    def progress(self, label, completed, total):
        ratio = completed / total if total else 1.
        filled = min(30, int(ratio * 30))
        print('\r%s [%s%s] %d/%d %.1f%%   ' %
              (label, '#' * filled, '-' * (30-filled), completed, total, ratio*100),
              end='', file=self.stream, flush=True)


class Session:
    def __init__(self, folder, source, manifest, *, save_progress=True, fresh_run=False):
        if type(save_progress) is not bool:
            raise ValueError('save_progress must be a boolean')
        self.save_progress = save_progress
        self.folder = Path(folder)
        self.source = str(Path(source).resolve())
        key = hashlib.sha256(self.source.casefold().encode('utf-8')).hexdigest()[:24]
        self.path = self.folder / ('painting_session_' + key + '.json')
        manifest = normalized(manifest)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
            if self.data.get('abandoned'):
                # Keep the retired session, and start a new sheet only on a later RUN.
                self.path.replace(self.path.with_name(self.path.stem + '_retired_' +
                                  __import__('uuid').uuid4().hex[:8] + '.json'))
                self.data = self.fresh(manifest)
            elif self.data.get('resumable') is False:
                if self.data.get('status') in ('running', 'uncertain') or self.data.get('active'):
                    raise ValueError('上次未正常结束，请确认机械臂停止后使用重新开始入口；此模式没有保存续画进度。')
                self.data = self.fresh(manifest)
            elif fresh_run:
                if self.data.get('status') in ('running', 'uncertain') or self.data.get('active'):
                    raise ValueError('上次任务未正常结束，请先核查现场并清空旧绘画记录。')
                self.data = self.fresh(manifest)
            else:
                self.check_update(manifest)
        else:
            self.data = self.fresh(manifest)

    def fresh(self, manifest):
        return dict(format='painting-session-v1', source=self.source, status='ready',
                    manifest=manifest, completed={k: 0 for k in manifest['stages']}, active=None)

    def check_update(self, manifest):
        if self.data.get('status') in ('running', 'uncertain') or self.data.get('active'):
            raise ValueError('上次未正常暂停或结束，可能存在半笔/未确认运动。请核查记录，不自动续画。')
        old = self.data['manifest']
        if old['registration'] != manifest['registration']:
            raise ValueError('画布、标定或运动档案改变，不能沿用原纸的进度。')
        for stage, count in self.data['completed'].items():
            previous = old['stages'][stage]
            current = manifest['stages'].get(stage, [])
            if type(count) is not int or not 0 <= count <= len(previous):
                raise ValueError('续画进度记录无效。')
            if previous[:count] != current[:count]:
                raise ValueError('已画笔触被修改、删除或重排；只可调整未画部分。')
        self.data['manifest'] = manifest
        self.data['completed'] = {k: self.data['completed'].get(k, 0) for k in manifest['stages']}

    def persist(self):
        if self.save_progress:
            save(self.path, self.data)
        else:
            # Only lifecycle state remains on disk to reject overlapping or
            # uncertain runs. No stroke counts or resumable manifest is saved.
            save(self.path, dict(format='painting-session-v1', source=self.source,
                                 status=self.data['status'], active=self.data['active'],
                                 resumable=False))

    def count(self, stage):
        return self.data['completed'][stage]

    def begin(self, stage):
        self.data.update(status='running', active=stage)
        # Keep the chosen color order so a reopened canvas deposits layers
        # in the same order as the physical painting, including after pauses.
        order = self.data.setdefault('stage_order', [])
        if stage not in order:
            order.append(stage)
        self.persist()  # Before connecting or issuing any motion.

    def complete_stroke(self, stage, count):
        if count != self.count(stage) + 1:
            raise ValueError('笔触进度不是连续前缀。')
        self.data['completed'][stage] = count
        if self.save_progress:
            self.persist()

    def parked(self, status='paused'):
        self.data.update(status=status, active=None)
        self.persist()

    def uncertain(self):
        self.data['status'] = 'uncertain'
        self.persist()


class StageControl:
    def __init__(self, session, stage, label, console):
        self.session, self.stage, self.label, self.console = session, stage, label, console
        self.total = len(session.data['manifest']['stages'][stage])

    def progress(self, completed):
        if completed > self.session.count(self.stage):
            self.session.complete_stroke(self.stage, completed)
        self.console.progress(self.label, completed, self.total)

    def boundary(self, park, verify):
        if not self.console.poll():
            return False
        park()  # Must wait for verified arrival before announcing a pause.
        self.session.parked()
        if self.session.save_progress:
            self.console.say('已抬笔停在纸外。C 继续当前计划；Q 保存退出（调整 JSON 后重新打开可续画）。')
        else:
            self.console.say('已抬笔停在纸外。C 继续当前计划；Q 退出（不保存进度，重开从第一笔开始）。')
        while True:
            reply = self.console.ask('输入 C 或 Q：').lower()
            if reply == 'q':
                raise PauseRequested()
            if reply == 'c':
                verify()  # Pose, alarms and site settings must still match.
                self.session.begin(self.stage)
                return True


def unfinished_sessions(folder):
    out = []
    for path in Path(folder).glob('painting_session_*.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        if not record.get('abandoned') and record.get('status') != 'complete':
            out.append((path, record))
    return out
