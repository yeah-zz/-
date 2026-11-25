"""Refresh only black liquid height in a reviewed job, preserving drawing geometry."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from uuid import uuid4

from config import PALETTE
import job
from run_mountain_balanced import load_reviewed, normalized


def synchronize(path):
    path = Path(path).resolve()
    original = path.read_bytes()
    data = json.loads(original.decode('utf-8'))
    old = data['meta']['ink_execution']['calibration']['palette']
    current = normalized(PALETTE)
    without_surface = lambda rows: [{k:v for k,v in row.items() if k != 'surface_z'} for row in rows]
    if without_surface(old) != without_surface(current):
        raise ValueError('仅支持液面变化；杯位、蘸墨深度或其他参数改变，请重新生成任务。')
    if old == current:
        load_reviewed(path)
        return None
    # This project's balanced mountain job has a single black palette entry.
    if len(current) != 1:
        raise ValueError('仅支持单墨盘山水任务。')
    from pigment_station import PigmentStation
    measurements = json.loads(Path(__file__).with_name('pigment_measurements.json').read_text(encoding='utf-8-sig'))
    PigmentStation(**measurements['black']).validate()
    data['meta']['ink_execution']['calibration']['palette'] = copy.deepcopy(current)
    candidate = path.with_name(path.stem + '.liquid-sync-' + uuid4().hex + '.json')
    try:
        # Validate all remaining calibration and stroke constraints before rebuilding.
        candidate.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        strokes, mapper, policy, meta = load_reviewed(candidate)
        from ink_control import build_plan
        from shuimo import ink_palette_index
        from run_mountain_v4_robot import safe_transfers
        steps, _ = build_plan(strokes, ink_palette_index(), mapper.mm_per_px, policy)
        waypoints, _ = job.steps_to_waypoints(steps, mapper)
        job.save_job(candidate, safe_transfers(waypoints), meta)
        load_reviewed(candidate)
        if path.read_bytes() != original:
            raise ValueError('任务文件同步期间被修改，已取消。')
        backup = path.with_name(path.name + '.before-liquid-' + uuid4().hex[:8] + '.bak')
        shutil.copy2(path, backup)
        candidate.replace(path)
        return backup
    finally:
        if candidate.exists():
            candidate.unlink()


def main():
    from launcher import setup_console
    setup_console()
    parser = argparse.ArgumentParser(description='仅同步黑墨液面；保留备份，不连接机械臂。运行前先结束绘画程序。')
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    try:
        backup = synchronize(args.job)
        print('液面已同步并通过离线校验。备份：' + str(backup) if backup else '液面一致，离线校验通过，无需修改。')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('同步取消：', exc)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
