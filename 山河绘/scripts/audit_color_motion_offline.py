"""Replay catalog color commands with hardware connections forbidden.

This checks software intent, not dynamics, collision clearance or real run time.
Source replay reconstructs the pre-overlay settings; it is not measured history.
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import socket
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from color_profile import ColorProfile
from web.catalog import ARTWORKS
from web.site_settings import apply, load_record, prepared_artwork
from unified_painting import color_station
from run_json_painting import load_artwork
import run_color_experiment as runner


class ReplayRobot:
    def __init__(self):
        self.pose = runner.target_pose()
        self.commands = []
        self.connected = False

    def send(self, command):
        self.commands.append(command)
        return '0,{1},' + command

    def _ok(self, response):
        return response.startswith('0,')

    def wait_idle(self, timeout):
        return True

    def close(self, **kwargs):
        pass


def replay(module, color):
    module = copy.deepcopy(module)
    module['motion_diagnostics'] = False  # Journaling is covered by separate fault tests.
    robot = ReplayRobot()
    start = tuple(robot.pose)
    arrival_count = 0
    dwell_seconds = 0.

    def arrive(_, target):
        nonlocal arrival_count
        arrival_count += 1
        robot.pose = tuple(target)

    def dwell(seconds):
        nonlocal dwell_seconds
        dwell_seconds += seconds

    station = color_station(color, ask=lambda _: (_ for _ in ()).throw(AssertionError('Missing station')))
    with tempfile.TemporaryDirectory() as folder, \
            patch.object(runner, 'E6Robot', return_value=robot), \
            patch.object(runner, 'connect_readonly'), \
            patch.object(runner, 'read_idle_pose', side_effect=lambda *args: robot.pose), \
            patch.object(runner, 'wait_arrival', side_effect=arrive), \
            patch.object(runner.time, 'sleep', side_effect=dwell), \
            patch.object(runner.time, 'monotonic', return_value=0.), \
            contextlib.redirect_stdout(io.StringIO()):
        code = runner.run(profile=ColorProfile(**module['motion_candidate']), confirm_site=True,
                          module=module, color=color, station=station, log_dir=folder,
                          save_progress=False)
        marker = json.loads(next(Path(folder).glob('color_experiment_*.json')).read_text(encoding='utf-8'))
    if code or not marker['success']:
        raise AssertionError(marker)
    moves = []
    geometry = []
    previous = start
    for command in robot.commands:
        if not command.startswith('MovL('):
            continue
        target = tuple(map(float, re.search(r'pose=\{([^}]+)\}', command)[1].split(',')))
        velocity = int(re.search(r',v=(\d+)', command)[1])
        cp = int(re.search(r',cp=(\d+)', command)[1])
        delta = [target[i]-previous[i] for i in range(3)]
        distance = sum(v*v for v in delta)**.5
        kind = ('stationary' if distance < .001 else 'vertical' if abs(delta[0])+abs(delta[1]) < .01
                else 'horizontal' if abs(delta[2]) < .01 else 'mixed')
        moves.append(dict(kind=kind, length_mm=distance, velocity=velocity,
                          start=list(previous), target=list(target), cp=cp))
        geometry.append((target, cp))
        previous = target
    groups = {}
    for kind in ('stationary','vertical','horizontal','mixed'):
        rows = [m for m in moves if m['kind']==kind]
        groups[kind] = dict(count=len(rows), length_mm=round(sum(m['length_mm'] for m in rows),2),
                            velocities=sorted(set(m['velocity'] for m in rows)))
    long_vertical = [m for m in moves if m['kind']=='vertical' and m['length_mm'] > 20]
    return dict(profile=marker['profile'], commands=len(moves), arrival_checks=arrival_count,
                dwell_seconds=dwell_seconds, segments=groups,
                long_vertical_velocities=sorted(set(m['velocity'] for m in long_vertical)),
                longest_horizontal_mm=round(max((m['length_mm'] for m in moves if m['kind']=='horizontal'),default=0),2),
                geometry_sha256=hashlib.sha256(repr(geometry).encode()).hexdigest()), geometry


def main():
    settings = ROOT/'web/data/site_settings.json'
    protected = [settings, ROOT/'config.py', ROOT/'pigment_measurements.json',
                 *(ROOT/row[3] for row in ARTWORKS)]
    original_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    report = dict(hardware_connected=False,
                  caveat='Intent only. No kinematics, torque or dynamics validation. Timed reload disabled for deterministic replay; counts use length/strokes only.',
                  artworks=[], protected_sha256=original_hashes)
    # Block accidental robot access, including helpers added in future versions.
    with patch.object(socket, 'create_connection', side_effect=AssertionError('Hardware forbidden')), \
            patch.object(socket, 'socket', side_effect=AssertionError('Hardware forbidden')), \
            patch('pending_artwork.apply_pending', return_value=False), \
            contextlib.redirect_stdout(io.StringIO()):
        for ident, title, _, name in ARTWORKS:
            path = ROOT/name
            apply(load_record(settings))
            from web.execution import prepare
            prepare(path)
            # Compare the same current paper/cup rebasing on both sides. Raw
            # source files can contain older absolute Z values and metadata.
            with patch('web.motion_profile.apply_to_artwork', side_effect=lambda artwork, _: artwork):
                source = prepared_artwork(path, settings)
            prepared = prepared_artwork(path, settings)
            old_module = source[1]['modules']['color']
            new_module = prepared[1]['modules']['color']
            assert source[0] == prepared[0], 'Black payload changed: '+ident
            assert source[2:] == prepared[2:], 'Black strokes/metadata changed: '+ident
            assert old_module['strokes'] == new_module['strokes'], 'Color geometry changed: '+ident
            row = dict(artwork=ident, black_payload_unchanged=True, colors=[])
            for color in sorted({s['color'] for s in old_module['strokes']}):
                old, old_geometry = replay(old_module, color)
                new, new_geometry = replay(new_module, color)
                assert [g[0] for g in old_geometry] == [g[0] for g in new_geometry], 'Command geometry changed: '+ident+'/'+color
                cp_changes = sum(a[1]!=b[1] for a,b in zip(old_geometry,new_geometry))
                assert cp_changes == 0, 'Batch split changed draw stop boundaries: '+ident+'/'+color
                row['colors'].append(dict(color=color, source_reconstruction=old, revised=new,
                                          command_geometry_unchanged=True,
                                          batch_end_cp_changes=cp_changes))
            report['artworks'].append(row)
    for p in protected:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == original_hashes[str(p.relative_to(ROOT))]
    destination = ROOT/'artifacts/motion_audit_20261009/replay.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for artwork in report['artworks']:
        for color in artwork['colors']:
            old,new = color['source_reconstruction'],color['revised']
            print(artwork['artwork'],color['color'],'moves',new['commands'],
                  'vertical',old['long_vertical_velocities'],'->',new['long_vertical_velocities'],
                  'max XY mm',new['longest_horizontal_mm'],'geometry equal; CP boundaries',color['batch_end_cp_changes'])
    print('Verified',len(report['artworks']),'artworks; no hardware. Report:',destination)


if __name__ == '__main__':
    main()
