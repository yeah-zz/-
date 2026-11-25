"""山河汇 local workbench. Simulation never imports or connects to robot hardware."""
import argparse
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web.catalog import ARTWORKS
from web.execution import public_error
from painting_session import session_lock
from web.bridge import MemoryBridge

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'
STORE = WEB / 'data'
LABELS = {'black': '黑线', 'title': '题字', 'blue': '蓝色', 'blue2': '淡蓝', 'green': '绿色', 'yellow': '黄色', 'red': '红色'}


class WorkbenchHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can permit two workbenches on the same port.
    allow_reuse_address = False

    def server_bind(self):
        if os.name == 'nt' and hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def artwork_data(path, session=None):
    """Use execution strokes, rather than the different outline preview module."""
    data = read(path)
    if data.get('format') == 'arm-painting-bundle-v1':
        ink, plan = data['ink_job'], data['workflow']
    elif data.get('format') == 'painting-workflow-design-v1':
        ink = read(path.with_name(path.stem.removesuffix('_计划') + '_黑线任务.json'))
        plan = data
    else:
        raise ValueError('请导入本项目的完整绘画 JSON，或上传 PNG / JPG 参考画。')
    meta = ink['meta']
    from artwork_order import ink_stages
    stages = ink_stages(ink, plan)
    for stroke in plan.get('modules', {}).get('color', {}).get('strokes', []):
        stages.setdefault(stroke['color'], []).append(stroke)
    size = meta['image_size']
    if session and session.get('manifest'):
        stages = session['manifest']['stages']
        size = session['manifest']['registration']['image_size']
    if len(size) != 2 or min(size) <= 0 or not stages['black']:
        raise ValueError('绘画数据缺少有效画布或黑线笔触。')
    scale = meta['ink_execution'].get('mm_per_px', .2)
    compact = {}
    for key, values in stages.items():
        if key not in LABELS:
            raise ValueError('任务中含不支持的颜色：' + key)
        compact[key] = []
        for s in values:
            points = s['points']
            if not points or any(len(p) != 2 or any(not isinstance(n, (float, int)) or not __import__('math').isfinite(n) for n in p) for p in points):
                raise ValueError('笔触坐标无效。')
            width = s.get('brush_width_mm_estimate', s.get('width_mm', .32 if key in ('black', 'title') else 1.3)) / scale
            compact[key].append({'points': points, 'width': width, 'opacity': s.get('target_opacity', .6)})
    preferred = (plan or {}).get('execution_order', {}).get('preferred_stage_order', [])
    compact = {key: compact[key] for key in dict.fromkeys([*preferred, *compact]) if key in compact}
    return {'size': size, 'stages': compact, 'counts': {k: len(v) for k, v in compact.items()}, 'mm': [round(size[0]*scale), round(size[1]*scale)],
            'colorLead': (plan or {}).get('execution_order', {}).get('color_lead', '')}


class AttachedWorker:
    """Read-only process check when reconnecting a still-running web worker."""
    def __init__(self, pid):
        self.pid = int(pid)

    def poll(self):
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, self.pid)
            if not handle:
                return 1
            try:
                code = wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return 1
                return None if code.value == 259 else code.value
            finally:
                kernel.CloseHandle(handle)
        try:
            os.kill(self.pid, 0)
            return None
        except OSError:
            return 1


class Workbench:
    def __init__(self):
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.items = {}
        self.geometry = {}
        self.runtime = None
        self.worker = None
        self.bridge = None
        self.operations = {}
        self.speed = 4
        self.restarting = False
        deleted_path = STORE/'deleted_artworks.json'
        self.deleted_artworks = read(deleted_path) if deleted_path.exists() else {}
        self.load_library()
        from web.studio import studio_for
        studio_for(self)
        self.reconnect_worker()

    def reconnect_worker(self):
        record = STORE/'control'/'runtime.json'
        guard = ROOT/'robot_runs'/'painting_session.lock'
        try:
            if not record.exists() or not guard.exists():
                return
            rt = read(record)
            pid = int(rt['workerPid'])
            if guard.read_text().strip() != str(pid):
                return
            worker = AttachedWorker(pid)
            if worker.poll() is not None:
                return
            ident = next(k for k,v in self.items.items() if v['path'].resolve() == Path(rt['source']).resolve())
            self.worker = worker
            self.runtime = {k:v for k,v in rt.items() if k not in ('workerPid','source')}
            self.runtime['id'] = ident
            self.event('本地服务已重新连接原执行器，逐笔进度保留')
        except (OSError, ValueError, KeyError, StopIteration):
            pass

    def add(self, path, title, image=None, ident=None, **info):
        ident = ident or hashlib.sha256(str(path).encode()).hexdigest()[:12]
        if ident in self.deleted_artworks:
            return ident
        geom = artwork_data(path)
        item = {'id': ident, 'title': title, 'path': Path(path), 'image': image, 'counts': geom['counts'], 'size': geom['size'], 'mm': geom['mm'], 'colorLead': geom.get('colorLead', ''), **info}
        self.items[ident] = item
        self.geometry[ident] = geom
        return ident

    def load_library(self):
        self.library_errors = []
        for number, (directory, title, description, source) in enumerate(ARTWORKS, 1):
            try:
                # The material library is now the authoritative execution source.
                image = ROOT/'待画作品'/directory/'参考原图.png'
                self.add(ROOT/source, title, image,
                         ident='painting-%02d' % number, featured=True, description=description)
            except (ValueError, KeyError, OSError) as exc:
                self.library_errors.append({'title': title, 'message': '此作品暂时无法读取，请检查本机画稿资料。'})
                print('画库读取失败：%s: %s' % (title, exc), file=sys.stderr)

    def display_image(self, item):
        """A photographed result can replace the cover without replacing the reference."""
        reference = item['image']
        if reference:
            for suffix in ('.png', '.jpg', '.jpeg', '.webp'):
                result = reference.parent / ('机械臂实绘' + suffix)
                if result.is_file():
                    return result
        return reference

    def public_items(self):
        items = []
        for key, item in self.items.items():
            display = self.display_image(item)
            items.append({k: v for k, v in item.items() if k not in ('path', 'image')} | {
                'image': '/api/image?id=' + key if item['image'] and item['image'].exists() else None,
                'displayImage': '/api/image?id=%s&view=display&v=%s' % (key, display.stat().st_mtime_ns)
                    if display and display.is_file() else None,
            })
        return items

    def delete_artwork(self, body):
        with self.lock:
            ident = body.get('id')
            if ident not in self.items:
                raise ValueError('画作已移除，请刷新画库。')
            if self.settings_locked() or (self.runtime and self.runtime.get('id') == ident
                    and self.runtime.get('status') not in ('idle', 'complete', 'stopped', 'saved', 'error')):
                raise ValueError('任务尚未结束，不能删除画作。请先正常结束作画。')
            item = self.items[ident]
            deleted = dict(self.deleted_artworks)
            deleted[ident] = dict(title=item['title'], path=str(item['path']), deletedAt=time.time())
            write(STORE/'deleted_artworks.json', deleted)
            self.deleted_artworks = deleted
            del self.items[ident]
            self.geometry.pop(ident, None)
            if self.runtime and self.runtime.get('id') == ident:
                self.runtime = None
            return {'deleted': ident, 'message': '画作已从画库移除，本地绘画材料保留。'}

    def settings_locked(self):
        return bool((self.worker and self.worker.poll() is None)
                    or (ROOT/'robot_runs'/'painting_session.lock').exists()
                    or (ROOT/'robot_runs'/'blue_wash_motion.lock').exists())

    def request_restart(self):
        with self.lock:
            if self.settings_locked() or self.state()['status'] not in ('idle','complete','stopped','saved','error'):
                raise ValueError('当前任务尚未退出，本次只打开网页，不重启后台。')
            if any(r.get('status') == 'building' for r in self.operations.values()):
                raise ValueError('图片正在处理，本次只打开网页，不重启后台。')
            studio = getattr(self, 'studio', None)
            if studio:
                with studio.lock:
                    if any(r.get('busy') for r in studio.jobs.values()):
                        raise ValueError('图片正在处理，本次只打开网页，不重启后台。')
            self.restarting = True

    def settings_snapshot(self):
        args = [sys.executable, str(WEB/'site_settings.py'), '--read']
        saved = STORE/'site_settings.json'
        if saved.exists():
            args += ['--saved', str(saved)]
        result = subprocess.run(args, cwd=ROOT, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=30, env=os.environ | {'PYTHONIOENCODING': 'utf-8'})
        try:
            data = json.loads(result.stdout)
        except ValueError:
            raise ValueError('现场参数暂时无法读取，请检查本机设置。')
        if result.returncode:
            raise ValueError(public_error(data.get('error', '现场参数读取失败。')))
        return data

    def settings(self):
        with self.lock:
            snapshot = self.settings_snapshot()
            locked = self.settings_locked()
            return dict(values=snapshot['record']['values'], revision=snapshot['revision'],
                        saved=(STORE/'site_settings.json').exists(), locked=locked,
                        message='机械臂任务尚未退出，当前只能查看参数。请正常退出任务后再保存。' if locked else
                                '保存后用于下一次网页绘画，原画稿的笔触和下压量保持不变。')

    def save_settings(self, body):
        with self.lock:
            if self.settings_locked():
                raise ValueError('机械臂任务尚未退出，不能修改参数。请正常退出任务后再保存。')
            jobs = [ROOT/source for _, _, _, source in ARTWORKS]
            if any(not path.exists() for path in jobs):
                raise ValueError('五幅画稿未全部就绪，请先检查画库后再保存参数。')
            # Hold the same guard as every real executor while validating and
            # committing. No external terminal can start during this save.
            with session_lock(ROOT/'robot_runs'):
                current = self.settings_snapshot()
                if body.get('revision') != current['revision']:
                    raise ValueError('参数已更新，请重新打开修改参数窗口后再保存。')
                folder = STORE/'settings_checks'/uuid.uuid4().hex
                folder.mkdir(parents=True)
                request, checked = folder/'request.json', folder/'checked.json'
                write(request, {'values': body.get('values')})
                args = [sys.executable, str(WEB/'site_settings.py'), '--validate', str(request), '--output', str(checked)]
                if (STORE/'site_settings.json').exists():
                    args += ['--saved', str(STORE/'site_settings.json')]
                args += ['--jobs', *(str(path) for path in jobs)]
                result = subprocess.run(args, cwd=ROOT, capture_output=True, encoding='utf-8', errors='replace',
                                        timeout=120, env=os.environ | {'PYTHONIOENCODING': 'utf-8'})
                write(folder/'result.json', {'output': result.stdout, 'diagnostics': result.stderr})
                try:
                    data = json.loads(result.stdout)
                except ValueError:
                    raise ValueError('参数检查未能完成，原参数保留，请重新检查后再保存。')
                if result.returncode:
                    raise ValueError(public_error(data.get('error', '参数检查未通过，原参数保留。')))
                if not checked.exists() or len(data.get('checks', [])) != len(jobs) or not all(check.get('passed') for check in data['checks']):
                    raise ValueError('参数检查未能完成，原参数保留。')
                saved = STORE/'site_settings.json'
                if saved.exists():
                    write(folder/'previous.json', read(saved))
                os.replace(checked, saved)
            return {'values': data['record']['values'], 'revision': data['revision'], 'saved': True,
                    'locked': False, 'checkedArtworks': len(data.get('checks', [])),
                    'message': '参数已保存并通过画稿检查，下一次网页绘画将使用这些参数。'}

    def event(self, message):
        runtime = self.runtime
        runtime['events'] = (runtime.get('events', []) + [{'at': time.time(), 'message': message}])[-30:]

    def session(self, item):
        key = hashlib.sha256(str(item['path'].resolve()).casefold().encode('utf-8')).hexdigest()[:24]
        path = ROOT / 'robot_runs' / ('painting_session_' + key + '.json')
        return read(path) if path.exists() else None

    def state(self):
        with self.lock:
            if not self.runtime:
                return {'status': 'idle', 'speed': self.speed, 'parametersLocked': self.settings_locked()}
            rt = self.runtime
            if rt['mode'] in ('live', 'monitor'):
                memory = rt.get('progressMode') == 'memory'
                bridge = self.bridge.snapshot() if memory and self.bridge else {}
                session = bridge if memory else self.session(self.items[rt['id']])
                if session:
                    rt.update(completed=session.get('completed', {}), status=session.get('status', 'uncertain'), active=session.get('active'))
                    if session.get('stage_order'):
                        rt['stageOrder'] = session['stage_order']
                    for stage, total in self.items[rt['id']]['counts'].items():
                        if total and rt['completed'].get(stage, 0) == total and stage not in rt.setdefault('finishedStages', []):
                            rt['finishedStages'].append(stage)
                            self.event(('勾勒' if stage == 'black' else '题字' if stage == 'title' else LABELS[stage] + '上色') + '完成')
                    if session.get('resumable') is False:
                        rt.update(status='uncertain', error='当前终端任务关闭了逐笔进度保存，无法可靠同步大屏。')
                if rt['mode'] == 'live':
                    control = STORE / 'control'
                    if memory or (control / 'status.json').exists():
                        if not memory:
                            bridge = read(control / 'status.json')
                        rt.update(prompt=bridge.get('prompt'), error=bridge.get('error'), notice=bridge.get('notice'))
                        if bridge.get('stageOrder') and not (session or {}).get('stage_order'):
                            rt['stageOrder'] = bridge['stageOrder']
                        prompt = rt.get('prompt')
                        if prompt and prompt['id'] != rt.get('lastPrompt'):
                            rt['lastPrompt'] = prompt['id']
                            self.event('本阶段完成，请选择下一阶段或结束作画' if prompt['kind'] == 'stage' else '请等墨干、洗净毛笔，准备下一种颜色' if prompt['kind'] == 'color' else '已抬笔停靠，可以继续或结束本次作画')
                        if bridge.get('error'):
                            rt['status'] = 'error'
                        elif bridge.get('prompt') and rt['status'] != 'uncertain':
                            rt['status'] = 'waiting'
                        if memory and bridge.get('done') and rt['status'] in ('paused', 'ready', 'starting'):
                            rt.update(status='stopped', active=None)
                    if self.worker and self.worker.poll() is not None and rt['status'] in ('running', 'starting'):
                        rt.update(status='uncertain', error='执行进程已退出，请核查日志与机械臂状态。')
                    elif self.worker and self.worker.poll() is not None and rt['status'] in ('paused', 'ready') and not rt.get('prompt'):
                        rt['status'] = 'stopped' if memory else 'saved'
                    if rt.get('prompt') or rt['status'] != 'running':
                        rt['pauseRequested'] = False
            return json.loads(json.dumps(rt)) | {'speed': self.speed, 'parametersLocked': self.settings_locked()}

    def start_demo(self, ident, initial_stage='black'):
        with self.lock:
            if self.worker and self.worker.poll() is None:
                raise ValueError('真实任务仍在运行，请先按笔暂停并保存退出。')
            # An external terminal may keep drawing while this isolated demo
            # replays paths. It never acquires the robot lock or issues motion.
            if self.runtime and self.runtime['status'] in ('running', 'paused', 'waiting'):
                raise ValueError('请先结束当前演示任务。')
            geometry = artwork_data(self.items[ident]['path'])
            if not geometry['counts'].get(initial_stage):
                raise ValueError('画稿不包含所选绘画阶段。')
            self.geometry[ident] = geometry
            self.items[ident].update(counts=geometry['counts'], size=geometry['size'], mm=geometry['mm'])
            self.runtime = {'id': ident, 'mode': 'demo', 'freeStages': True, 'status': 'running', 'active': initial_stage, 'stageOrder': [initial_stage], 'completed': {k: 0 for k in self.items[ident]['counts']}, 'started': time.time(), 'events': []}
            self.event('开始预演 · ' + ('勾勒' if initial_stage == 'black' else '题字' if initial_stage == 'title' else LABELS[initial_stage] + '上色'))

    def clear_records(self, body):
        if self.settings_locked():
            raise ValueError('机械臂任务尚未退出，不能清空记录。请先暂停并结束作画。')
        if body.get('confirmation') != 'CLEAR' or body.get('controllerStopped') is not True:
            raise ValueError('请先确认机械臂已停止、旧指令队列已取消且毛笔已离纸，再清空记录。')
        with session_lock(ROOT/'robot_runs'):
            if (ROOT/'robot_runs'/'blue_wash_motion.lock').exists():
                raise ValueError('洗笔或上色任务尚未退出，不能清空记录。')
            archive = ROOT/'robot_runs'/'cleared_records'/uuid.uuid4().hex
            archive.mkdir(parents=True)
            records = [*(ROOT/'robot_runs').glob('painting_session_*.json'),
                       *(ROOT/'robot_runs').glob('color_experiment_*.json')]
            for name in ('runtime.json', 'status.json', 'reply.json', 'pause.json'):
                path = STORE/'control'/name
                if path.exists():
                    records.append(path)
            allowed = [(ROOT/'robot_runs').resolve(), (STORE/'control').resolve()]
            for path in records:
                if path.resolve().parent not in allowed:
                    raise ValueError('记录路径不符合当前项目范围，未继续清空。')
            moved = []
            try:
                for index, path in enumerate(records):
                    target = archive/('%03d_%s' % (index, path.name))
                    os.replace(path, target)
                    moved.append((path, target))
            except OSError:
                for path, target in reversed(moved):
                    os.replace(target, path)
                raise
        if self.bridge:
            self.bridge.close()
        self.bridge, self.worker, self.runtime = None, None, None
        return self.state() | {'clearedRecords': len(records), 'message': '旧绘画记录已清空，画布已重置。下次将从第一笔开始，请使用新纸。'}

    def tick(self):
        with self.lock:
            rt = self.runtime
            if not rt or rt['mode'] != 'demo' or rt['status'] != 'running':
                return
            key = rt['active']
            rt['completed'][key] += 1
            if rt['completed'][key] >= self.items[rt['id']]['counts'][key]:
                self.event(('勾勒' if key == 'black' else '题字' if key == 'title' else LABELS[key] + '上色') + '完成')
                remaining = [k for k, total in self.items[rt['id']]['counts'].items() if rt['completed'][k] < total]
                rt.update(status='waiting' if remaining else 'complete', active=None)
                self.event('本阶段完成，按需洗笔后选择下一阶段，或结束作画' if remaining else '山河成画 · 所有笔触已完成')

    def action(self, body):
        command = body['command']
        with self.lock:
            if command == 'clear':
                return self.clear_records(body)
            elif command == 'demo':
                self.start_demo(body['id'], body.get('stage', 'black'))
            elif command == 'speed':
                if body['value'] not in (1, 4, 12):
                    raise ValueError('无效演示速度。')
                self.speed = body['value']
            elif command == 'monitor':
                if self.worker and self.worker.poll() is None:
                    raise ValueError('已有网页真实任务在运行。')
                record = self.session(self.items[body['id']])
                if not record or record.get('abandoned'):
                    raise ValueError('此画稿没有可同步的执行记录。请先在终端启动同一份画稿。')
                geom = artwork_data(self.items[body['id']]['path'], record)
                self.geometry[body['id']] = geom
                self.items[body['id']].update(counts=geom['counts'], size=geom['size'], mm=geom['mm'])
                self.runtime = {'id': body['id'], 'mode': 'monitor', 'status': record['status'], 'completed': record.get('completed', {}), 'events': [], 'started': time.time()}
                self.event('已同步终端的真实逐笔执行记录')
            elif command == 'unmonitor':
                if self.runtime and self.runtime['mode'] == 'monitor':
                    self.runtime = None
                else:
                    raise ValueError('当前没有终端同步任务。')
            elif command == 'live':
                if body.get('confirmation') != 'RUN':
                    raise ValueError('现场确认后输入 RUN 才能启动真实绘画。')
                if (self.worker and self.worker.poll() is None) or (ROOT / 'robot_runs' / 'painting_session.lock').exists():
                    raise ValueError('已有真实会话或未解除的会话锁，请使用原入口处理。')
                item = self.items[body['id']]
                initial_stage = body.get('stage', 'black')
                if not isinstance(initial_stage, str) or not item['counts'].get(initial_stage):
                    raise ValueError('画稿不包含所选绘画阶段。')
                settings_args = []
                saved = STORE/'site_settings.json'
                if saved.exists():
                    # Both validation and execution get the identical immutable
                    # snapshot, even across a server restart or later edit.
                    payload = saved.read_bytes()
                    digest = hashlib.sha256(payload).hexdigest()
                    snapshot = STORE/'settings_snapshots'/(digest + '.json')
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    if snapshot.exists():
                        if snapshot.read_bytes() != payload:
                            raise ValueError('执行参数记录不一致，请重新保存参数。')
                    else:
                        snapshot.write_bytes(payload)
                    settings_args = ['--settings', str(snapshot)]
                # Run the existing offline validation before any worker can move.
                stage_args = ['--start-stage', initial_stage]
                result = subprocess.run([sys.executable, str(WEB/'worker.py'), str(item['path']), '--check', *stage_args, *settings_args], cwd=ROOT, capture_output=True, encoding='utf-8', errors='replace', timeout=90, env=os.environ | {'PYTHONIOENCODING': 'utf-8'})
                if result.returncode:
                    write(STORE/'validation_error.json', {'detail': result.stderr or result.stdout})
                    raise ValueError('画稿校验未通过：' + public_error(result.stderr or result.stdout))
                geom = artwork_data(item['path'])
                self.geometry[body['id']] = geom
                item.update(counts=geom['counts'], size=geom['size'], mm=geom['mm'])
                control = STORE / 'control'
                control.mkdir(parents=True, exist_ok=True)
                for name in ('reply.json', 'pause.json', 'status.json'):
                    (control / name).unlink(missing_ok=True)
                if self.bridge:
                    self.bridge.close()
                self.bridge = MemoryBridge()
                self.runtime = {'id': body['id'], 'mode': 'live', 'freeStages': True, 'progressMode': 'memory', 'status': 'starting', 'active': None, 'completed': {}, 'started': time.time(), 'events': []}
                self.event('画稿检查通过 · 正在连接机械臂')
                log = (control / 'worker.log').open('w', encoding='utf-8')
                try:
                    self.worker = subprocess.Popen([sys.executable, str(WEB/'worker.py'), str(item['path']), str(control), *stage_args, *settings_args], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), env=os.environ | {'PYTHONIOENCODING': 'utf-8'} | self.bridge.environment())
                finally:
                    log.close()
            else:
                rt = self.runtime
                if not rt:
                    raise ValueError('请先启动任务。')
                if rt['mode'] == 'monitor':
                    raise ValueError('同步模式请在原绘画终端控制任务。')
                if rt['mode'] == 'live':
                    self.state()
                    if command == 'pause' and rt['status'] == 'running':
                        if rt.get('pauseRequested'):
                            raise ValueError('已经请求暂停，请等待当前笔完成并抬笔停靠。')
                        if rt.get('progressMode') == 'memory':
                            self.bridge.command({'pause': True})
                            rt['pauseRequested'] = True
                        else:
                            write(STORE/'control'/'pause.json', {'pause': True})
                        self.event('已请求按笔暂停 · 等待机械臂抬笔停靠')
                    elif command in ('resume', 'color', 'stage', 'finish'):
                        prompt = rt.get('prompt')
                        if not prompt or body.get('promptId') != prompt['id']:
                            raise ValueError('执行器尚未停靠并等待此操作，请刷新状态。')
                        kind = prompt['kind']
                        value = 'c' if command == 'resume' else 'q' if command == 'finish' else body.get('stage') if command == 'stage' else body.get('color')
                        selecting = command in ('color', 'stage')
                        if (command == 'resume' and kind != 'pause') or (selecting and kind != ('stage' if rt.get('freeStages') else 'color')) or (command == 'stage' and not rt.get('freeStages')):
                            raise ValueError('当前执行器没有等待此操作。')
                        if selecting:
                            from artwork_order import available_colors, available_stages
                            available = available_stages if rt.get('freeStages') else available_colors
                            if value not in available(rt['completed'], self.items[rt['id']]['counts']):
                                raise ValueError('请选择尚未完成的绘画阶段。')
                        if rt.get('progressMode') == 'memory':
                            if rt.get('replyPending') == prompt['id']:
                                raise ValueError('上一次操作正在处理，请稍候。')
                            self.bridge.command({'id': prompt['id'], 'value': value})
                            rt['replyPending'] = prompt['id']
                        else:
                            if (STORE/'control'/'reply.json').exists():
                                raise ValueError('上一次操作正在处理，请稍候。')
                            write(STORE/'control'/'reply.json', {'id': prompt['id'], 'value': value})
                        self.event('准备绘制' + LABELS[value] if selecting else '继续作画' if command == 'resume' else '结束本次作画，不保存续画进度')
                    else:
                        raise ValueError('当前阶段不支持此操作。')
                elif command == 'pause' and rt['status'] == 'running':
                    rt['status'] = 'paused'
                    self.event('演示已暂停，笔触进度保留')
                elif command == 'resume' and rt['status'] == 'paused':
                    rt['status'] = 'running'
                    self.event('继续逐笔演示')
                elif command in ('color', 'stage') and rt['status'] == 'waiting':
                    key = body.get('stage') if command == 'stage' else body.get('color')
                    counts = self.items[rt['id']]['counts']
                    from artwork_order import available_colors, available_stages
                    available = available_stages if rt.get('freeStages') else available_colors
                    if key not in available(rt['completed'], counts):
                        raise ValueError('请选择尚未完成的绘画阶段。')
                    rt.update(status='running', active=key)
                    rt.setdefault('stageOrder', ['black']).append(key)
                    self.event('开始' + ('勾勒' if key == 'black' else '题字' if key == 'title' else LABELS[key] + '上色'))
                elif command == 'finish':
                    rt.update(status='stopped', active=None)
                    self.event('演示已结束')
                else:
                    raise ValueError('当前阶段不支持此操作。')
        return self.state()

    def upload(self, body):
        ident = uuid.uuid4().hex[:12]
        folder = STORE / ident
        suffix = Path(body['name']).suffix.lower()
        if suffix not in ('.json', '.png', '.jpg', '.jpeg', '.webp'):
            raise ValueError('支持 PNG、JPG、WEBP 图片或完整绘画 JSON。')
        data = base64.b64decode(body['data'], validate=True)
        if len(data) > 25 * 1024 * 1024:
            raise ValueError('文件不能超过 25 MB。')
        folder.mkdir(parents=True)
        path = folder / ('shh_' + ident + suffix)
        path.write_bytes(data)
        title = Path(body['name']).stem[:60]
        if suffix == '.json':
            with self.lock:
                self.add(path, title, ident=ident)
                write(folder/'metadata.json', {'path': str(path), 'title': title})
            return {'id': ident, 'status': 'ready'}
        import cv2
        import numpy as np
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or max(image.shape[:2]) > 8000:
            raise ValueError('无法读取此图片，或尺寸超过 8000 像素。')
        self.operations[ident] = {'id': ident, 'title': title, 'status': 'building', 'message': '后台正在提取黑线、规划上色并校验画稿'}
        threading.Thread(target=self.build, args=(ident, path, title), daemon=True).start()
        return self.operations[ident]

    def build(self, ident, image, title):
        folder = image.parent
        try:
            with (folder/'build.log').open('w', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, str(ROOT/'scripts'/'一张图出任务.py'), '--image', str(image), '--name', 'shh_'+ident, '--check'], cwd=ROOT, stdout=log, stderr=log, timeout=1800, env=os.environ | {'PYTHONIOENCODING': 'utf-8'})
            if result.returncode:
                raise ValueError('画稿生成失败，详情：' + str(folder/'build.log'))
            directories = sorted((ROOT/'输出').glob('shh_'+ident+'_铺色版_*'), key=lambda p: p.stat().st_mtime)
            generated = directories[-1]
            plan = read(next(generated.glob('*_计划.json')))
            ink = read(next(generated.glob('*_黑线任务.json')))
            # Pack final dense-color plan, not the pipeline's earlier intermediate bundle.
            bundle = folder/'painting.json'
            write(bundle, {'format': 'arm-painting-bundle-v1', 'version': 1, 'workflow': plan, 'ink_job': ink})
            check = subprocess.run([sys.executable, str(ROOT/'run_json_painting.py'), '--job', str(bundle), '--check'], cwd=ROOT, capture_output=True, encoding='utf-8', errors='replace', timeout=90, env=os.environ | {'PYTHONIOENCODING': 'utf-8'})
            if check.returncode:
                raise ValueError('已生成画稿，但最终校验失败：' + (check.stderr or check.stdout)[-800:])
            with self.lock:
                self.add(bundle, title, image, ident)
                write(folder/'metadata.json', {'path': str(bundle), 'title': title, 'image': str(image)})
                self.operations[ident].update(status='ready', message='黑线与上色任务已生成，可逐笔演示或现场启动')
        except Exception as exc:
            self.operations[ident].update(status='error', message=public_error(exc))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send(self, data, status=200, content_type='application/json; charset=utf-8'):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        app = self.server.app
        try:
            from web.studio import route
            if route(self, 'GET'):
                return
            if parsed.path == '/api/config':
                self.send({'token': app.token, 'labels': LABELS, 'root': str(ROOT), 'serviceRestart': True, 'freeStages': True})
            elif parsed.path == '/api/library':
                self.send({'items': app.public_items(), 'operations': list(app.operations.values()), 'unavailable': getattr(app, 'library_errors', [])})
            elif parsed.path == '/api/state':
                self.send(app.state())
            elif parsed.path == '/api/settings':
                self.send(app.settings())
            elif parsed.path == '/api/artwork':
                self.send(app.geometry[q['id'][0]])
            elif parsed.path == '/api/image':
                item = app.items[q['id'][0]]
                path = app.display_image(item) if q.get('view') == ['display'] else item['image']
                self.send(path.read_bytes(), content_type=mimetypes.guess_type(path)[0] or 'image/png')
            elif parsed.path in ('/', '/screen', '/app.js', '/style.css'):
                path = WEB / ('index.html' if parsed.path in ('/', '/screen') else parsed.path[1:])
                self.send(path.read_bytes(), content_type=mimetypes.guess_type(path)[0] or 'text/plain')
            else:
                self.send({'error': '页面不存在'}, 404)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            self.send({'error': public_error(exc)}, 400)

    def do_POST(self):
        app = self.server.app
        if self.headers.get('X-Workbench-Token') != app.token:
            self.send({'error': '请从本地山河汇页面操作。'}, 403)
            return
        try:
            with app.lock:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 36 * 1024 * 1024:
                    raise ValueError('请求文件过大。')
                body = json.loads(self.rfile.read(length))
                with app.lock:
                    if app.restarting:
                        raise ValueError('网页后台正在重启，请稍后刷新。')
                    if self.path == '/api/service/restart':
                        app.request_restart()
                        app.restarting = True
                        self.send({'restarting': True})
                        threading.Thread(target=self.server.shutdown, daemon=True).start()
                        return
                from web.studio import route
                if route(self, 'POST', body):
                    return
                if self.path == '/api/action':
                    self.send(app.action(body))
                elif self.path == '/api/upload':
                    self.send(app.upload(body))
                elif self.path == '/api/library/delete':
                    self.send(app.delete_artwork(body))
                elif self.path == '/api/settings':
                    self.send(app.save_settings(body))
                else:
                    self.send({'error': '接口不存在'}, 404)
        except Exception as exc:
            self.send({'error': public_error(exc)}, 400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    app = Workbench()
    try:
        server = WorkbenchHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError:
        if args.open_browser:
            from urllib.request import urlopen
            try:
                config = json.load(urlopen('http://127.0.0.1:%d/api/config'%args.port, timeout=2))
                if 'token' in config and 'labels' in config:
                    import webbrowser
                    webbrowser.open('http://127.0.0.1:%d'%args.port)
                    print('山河汇已在运行，已打开工作台。', flush=True)
                    return
            except Exception:
                pass
        raise
    server.app = app
    def clock():
        while True:
            time.sleep(.36 / app.speed)
            app.tick()
    threading.Thread(target=clock, daemon=True).start()
    print('山河汇工作台：http://127.0.0.1:%d' % args.port, flush=True)
    print('大屏：http://127.0.0.1:%d/screen' % args.port, flush=True)
    if args.open_browser:
        import webbrowser
        threading.Timer(.3, lambda: webbrowser.open('http://127.0.0.1:%d'%args.port)).start()
    server.serve_forever()


if __name__ == '__main__':
    main()
