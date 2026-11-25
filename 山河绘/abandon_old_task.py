"""Retire interrupted black-line and color tasks before starting on new paper."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from experiment_resume import digest, unresolved_records
from painting_session import unfinished_sessions
from project_paths import ROBOT_LOG_DIR, PROJECT_DIR
from run_blue_wash import persist


def unfinished_ink_records(roots):
    """Only live per-job run logs; never plans, actual trajectories or backups."""
    records = []
    seen = set()
    for root in roots:
        for path in Path(root).rglob('*_execution_log.json'):
            if not path.parent.name.endswith('_runs'):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            record = json.loads(path.read_text(encoding='utf-8-sig'))
            if (record.get('success') is False and not record.get('abandoned')
                    and 'completed_strokes' in record and 'total_strokes' in record):
                records.append((path, record))
    return records


def abandon(folder, ink_roots=()):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / 'blue_wash_motion.lock'
    if os.name != 'nt':
        raise RuntimeError('此清理入口目前仅支持 Windows。')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # Exclusive access also rejects legacy runners holding an ordinary os.open fd.
    # OPEN_ALWAYS creates the same guard when there is no stale lock.
    handle = create(str(lock), 0x80000000 | 0x40000000, 0, None, 4, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise OSError('运行锁仍被占用或无法访问；请先结束旧绘画进程，未清理任何记录。')
    session_path = folder / 'painting_session.lock'
    session_handle = create(str(session_path), 0x80000000 | 0x40000000, 0, None, 4, 0x80, None)
    if session_handle == ctypes.c_void_p(-1).value:
        kernel.CloseHandle(handle)
        raise OSError('JSON 绘画会话仍在运行；请先暂停并保存退出，不修改任何记录。')
    try:
        blocked = unresolved_records(folder) + unfinished_ink_records(ink_roots) + unfinished_sessions(folder)
        for path, record in blocked:
            record['abandoned'] = dict(
                abandoned_at=datetime.now().astimezone().isoformat(),
                reason='Operator confirmed controller stopped, pen clear, and new paper for a new task',
                previous_sha256=digest(path))
            # Keep pending moves and incomplete contacts intact for later inspection.
            persist(path, record)
    finally:
        kernel.CloseHandle(session_handle)
        kernel.CloseHandle(handle)
    archive = folder / 'retired_locks'
    archive.mkdir(exist_ok=True)
    lock.replace(archive / ('%s_%s.lock' % (
        datetime.now().strftime('%Y%m%d_%H%M%S'), uuid4().hex[:8])))
    session_path.replace(archive / ('session_%s.lock' % uuid4().hex))
    return len(blocked)


def main(argv=()):
    from launcher import setup_console
    setup_console()
    parser = argparse.ArgumentParser(description='放弃旧任务；只处理记录，不发送机械臂指令。')
    parser.add_argument('--check', action='store_true', help='只查看待处理记录，不修改文件、不清理锁')
    args = parser.parse_args(argv)
    print('统一放弃旧黑线和上色任务，保留日志；不会停止或移动机械臂，也不会自动开始新任务。')
    print('先在控制器停止运动并取消旧队列，抬笔离纸，结束旧绘画程序。')
    print('仅用于放弃旧画、换新纸；同一张纸接着画不要使用此入口。')
    print('范围：输出、展示选画、robot_runs 的黑线记录，以及上色中断记录和 JSON 绘画续画记录。')
    print('黑线程序没有运行锁，请务必先关闭旧黑线程序；此入口不能替你停止它。')
    try:
        ink_roots = [PROJECT_DIR / '输出', ROBOT_LOG_DIR, *PROJECT_DIR.glob('展示选画_*')]
        ink_count = len(unfinished_ink_records(ink_roots))
        color_count = len(unresolved_records(ROBOT_LOG_DIR))
        session_count = len(unfinished_sessions(ROBOT_LOG_DIR))
        print('待放弃：黑线 %d 条，上色 %d 条，完整绘画续画记录 %d 条。画稿JSON和成功记录不修改。' % (ink_count, color_count, session_count))
        if args.check:
            print('只读检查完成，未修改记录、未清理锁、未连接机械臂。')
            return 0
        if input('确认以上已完成，并决定换新任务，输入 NEW；其他输入取消：').strip() != 'NEW':
            return 0
        count = abandon(ROBOT_LOG_DIR, ink_roots)
        print('已统一放弃 %d 条黑线/上色未解决记录，上色运行锁已归档。现在可以启动新任务。' % count)
        return 0
    except (OSError, ValueError, RuntimeError, EOFError, KeyboardInterrupt) as exc:
        print('未完成清理：', exc)
        return 2


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
