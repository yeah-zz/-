"""Offline settings persistence, frozen launches and real five-artwork rebasing."""
import copy
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web import server
from web.catalog import ARTWORKS
from web.site_settings import defaults, normalize, load_record


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for _,_,_,source in ARTWORKS:
            target=self.root/source
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text('{}',encoding='utf-8')
        for change in (patch.object(server, 'ROOT', self.root),
                       patch.object(server, 'STORE', self.root/'data'),
                       patch.object(server.Workbench, 'load_library')):
            change.start()
            self.addCleanup(change.stop)
        self.app = server.Workbench()
        self.app.items = {str(i): {'path': ROOT/source, 'featured': True}
                          for i, (_, _, _, source) in enumerate(ARTWORKS)}
        self.current = {'record': {'values': {'paper_z': -36}}, 'revision': 'old'}
        self.app.settings_snapshot = Mock(return_value=self.current)
        self.body = {'values': {'paper_z': -35}, 'revision': 'old'}

    def test_active_robot_or_wash_lock_blocks_before_any_validation(self):
        folder = self.root/'robot_runs'
        folder.mkdir()
        for name in ('painting_session.lock', 'blue_wash_motion.lock'):
            lock = folder/name
            lock.write_text('external-task')
            with patch.object(server.subprocess, 'run') as check, self.assertRaisesRegex(ValueError, '尚未退出'):
                self.app.save_settings(self.body)
            check.assert_not_called()
            self.assertEqual(lock.read_text(), 'external-task')
            lock.unlink()

    def test_stale_editor_and_missing_artwork_never_commit(self):
        with self.assertRaisesRegex(ValueError, '参数已更新'):
            self.app.save_settings(self.body | {'revision': 'stale'})
        (self.root/ARTWORKS[0][3]).unlink()
        with self.assertRaisesRegex(ValueError, '五幅'):
            self.app.save_settings(self.body)
        self.assertFalse((server.STORE/'site_settings.json').exists())

    def test_failed_validation_preserves_saved_bytes_and_releases_guard(self):
        saved = server.STORE/'site_settings.json'
        server.write(saved, {'original': True})
        previous = saved.read_bytes()
        with patch.object(server.subprocess, 'run', return_value=Mock(returncode=2, stdout=json.dumps({'error': '纸面高度无效'}), stderr='')):
            with self.assertRaises(ValueError):
                self.app.save_settings(self.body)
        self.assertEqual(saved.read_bytes(), previous)
        self.assertFalse((self.root/'robot_runs'/'painting_session.lock').exists())

    def test_success_checks_all_five_then_atomically_saves_and_keeps_backup(self):
        saved = server.STORE/'site_settings.json'
        server.write(saved, {'original': True})
        def validated(args, **kwargs):
            self.assertTrue((self.root/'robot_runs'/'painting_session.lock').exists())
            self.assertEqual(len(args[args.index('--jobs')+1:]), len(ARTWORKS))
            record = {'values': self.body['values']}
            server.write(args[args.index('--output')+1], record)
            return Mock(returncode=0, stdout=json.dumps({'record': record, 'revision': 'new', 'checks': [{'passed': True}]*len(ARTWORKS)}), stderr='')
        with patch.object(server.subprocess, 'run', side_effect=validated), patch.object(server.subprocess, 'Popen') as motion:
            result = self.app.save_settings(self.body)
        motion.assert_not_called()
        self.assertEqual(result['checkedArtworks'], len(ARTWORKS))
        self.assertEqual(server.read(saved)['values']['paper_z'], -35)
        backup = next((server.STORE/'settings_checks').glob('*/previous.json'))
        self.assertEqual(server.read(backup), {'original': True})
        self.assertFalse((self.root/'robot_runs'/'painting_session.lock').exists())

    def test_launch_check_and_execution_share_frozen_snapshot(self):
        from tests.test_web_workbench import WorkbenchTests
        fixture = WorkbenchTests()
        fixture.setUp()
        try:
            saved = server.STORE/'site_settings.json'
            server.write(saved, {'values': {'paper_z': -36}})
            original = saved.read_bytes()
            process = Mock(pid=3456)
            process.poll.return_value = None
            def check(args, **kwargs):
                server.write(saved, {'values': {'paper_z': -30}})
                return Mock(returncode=0, stdout='', stderr='')
            with patch.object(server.subprocess, 'run', side_effect=check) as validation, patch.object(server.subprocess, 'Popen', return_value=process) as launch:
                fixture.app.action({'command': 'live', 'id': fixture.ident, 'confirmation': 'RUN'})
            check_args, run_args = validation.call_args.args[0], launch.call_args.args[0]
            self.assertEqual(check_args[-2:], run_args[-2:])
            self.assertEqual(Path(run_args[-1]).read_bytes(), original)
            self.assertEqual(run_args[2], str(fixture.source))
        finally:
            fixture.tearDown()

    def test_bad_numbers_and_count_rejected(self):
        with self.assertRaises(FileNotFoundError):
            load_record(self.root/'missing-execution-snapshot.json')
        for value in (True, float('nan'), float('inf'), '1'):
            values = defaults()['values']
            values['paper_z'] = value
            with self.assertRaises(ValueError):
                normalize(values)
        values = defaults()['values']
        values['stations']['blue']['strokes_per_load'] = 2.5
        with self.assertRaises(ValueError):
            normalize(values)
        legacy = defaults()['values']
        legacy['speeds'] = dict(draw=25, travel=50, dip=15, global_speed_factor=60)
        legacy['acceleration'] = dict(joint=25, linear=30)
        self.assertEqual(normalize(legacy), defaults()['values'])
        self.assertNotIn('speeds', normalize(legacy))
        self.assertNotIn('acceleration', normalize(legacy))

    def test_real_five_artworks_in_isolated_process_without_hardware(self):
        result = subprocess.run([sys.executable, __file__, '--integration'], cwd=ROOT,
                                capture_output=True, text=True, encoding='utf-8', timeout=120,
                                env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_legacy_settings_use_distance_compensation_and_roundtrip(self):
        from web.site_settings import BASE_FIELDS, make_record
        from unified_painting import color_station
        from pigment_station import PigmentStation
        record = defaults()
        legacy = copy.deepcopy(record['values'])
        for row in legacy['stations'].values():
            for key in tuple(row):
                if key not in BASE_FIELDS:
                    del row[key]
        migrated = make_record(legacy, record)
        for name in ('black', 'blue', 'green'):
            row = migrated['values']['stations'][name]
            self.assertTrue(row['level_compensation_enabled'])
            self.assertEqual(row['level_every_length_mm'], 500)
            with patch('pigment_station.read_measurements', return_value=migrated['measurements']):
                station = (PigmentStation(**migrated['measurements'][name]).validate() if name == 'black'
                           else color_station(name, ask=Mock(side_effect=AssertionError('No prompt'))))
            self.assertEqual(station.level_offset(999, 499), 0)
            self.assertAlmostEqual(station.level_offset(0, 500), .1)
            self.assertAlmostEqual(station.level_offset(0, 1000), .2)
            self.assertAlmostEqual(station.level_offset(0, 5000), .5)
        values = copy.deepcopy(migrated['values'])
        values['stations']['blue'].update(level_every_length_mm=750, level_step_mm=.05, level_max_mm=.3,
                                        level_compensation_enabled=False)
        saved = make_record(values, migrated)
        path = self.root/'compensation-settings.json'
        server.write(path, saved)
        self.assertEqual(load_record(path)['values'], values)

    def test_compensation_invalid_values_rejected(self):
        for key, bad in [('level_compensation_enabled', 1), ('level_every_length_mm', 0),
                         ('level_every_length_mm', float('nan')), ('level_every_length_mm', True),
                         ('level_step_mm', 0), ('level_step_mm', .6), ('level_max_mm', .6),
                         ('level_max_mm', .05)]:
            values = defaults()['values']
            values['stations']['blue'][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                normalize(values)


def integration():
    import contextlib
    import io
    from config import ArmConfig
    import job
    from geometry import CanvasMap
    from web.site_settings import apply, make_record, prepared_artwork
    from web.execution import prepare
    from run_json_painting import load_artwork, manifest_for, run_file
    from painting_session import Session, Console
    from unified_painting import color_station
    baseline = defaults()
    changed = copy.deepcopy(baseline['values'])
    changed['paper_z'] += .25
    changed['speeds']=dict(draw=25,travel=50,dip=15,global_speed_factor=60)
    changed['acceleration']=dict(joint=25,linear=30)
    for row in changed['stations'].values():
        row.update(surface_z=row['surface_z']+.2, x=row['x']-.2, dip_mm=.8, dwell_s=.7, drip_s=.6)
    record = make_record(changed, baseline)
    protected = [ROOT/'config.py', ROOT/'pigment_measurements.json', *(ROOT/source for _, _, _, source in ARTWORKS)]
    originals = {path: path.read_bytes() for path in protected}
    with tempfile.TemporaryDirectory() as folder, patch('robot.E6Robot.__init__', side_effect=AssertionError('Hardware forbidden')), patch('pending_artwork.apply_pending',return_value=False), contextlib.redirect_stdout(io.StringIO()):
        settings = Path(folder)/'settings.json'
        server.write(settings, record)
        for _, _, _, source in ARTWORKS:
            path = ROOT/source
            apply(baseline)
            prepare(path)
            old = load_artwork(path)
            old_points = job.stroke_waypoints(CanvasMap(old[3]['image_size']), old[2][0])
            old_manifest = manifest_for(old[2], old[1], old[3])
            changed_ink_speed = next(p for p in job.dip_waypoints(0) if p['tag']=='dip')['velocity']
            changed_withdraw_speed = next(p for p in job.dip_waypoints(0) if p['tag']=='dip-out')['velocity']
            new = prepared_artwork(path, settings)
            new_points = job.stroke_waypoints(CanvasMap(new[3]['image_size']), new[2][0])
            for before,after in zip(old[2],new[2]):
                assert before['points']==after['points'] and before['press']==after['press']
                assert before.get('speed') == after.get('speed')
                assert before.get('horizontal_velocity_percent') == after.get('horizontal_velocity_percent')
            for before,after in zip(old[1]['modules']['color']['strokes'],new[1]['modules']['color']['strokes']):
                assert before['points']==after['points'] and before['press']==after['press'] and before['color']==after['color']
                from web.execution import DRAW_SPEED_SCALE
                old_cap = min(45, old[1]['modules']['color']['motion_candidate']['draw_velocity_max'])
                assert after.get('speed') == [min(100, max(5, min(v, old_cap))*DRAW_SPEED_SCALE) for v in before['speed']]
            assert ArmConfig.SPEED_FACTOR == old[3]['ink_execution']['calibration']['speed_factor']
            assert new[3]['ink_execution']['calibration']['speed_factor'] == old[3]['ink_execution']['calibration']['speed_factor']
            assert new[3].get('air_transfer_velocity_percent') == min(40,old[3].get('air_transfer_velocity_percent',40))
            assert 'runtime_acceleration' not in new[3]
            assert 'web_motion_profile' not in new[3]
            from artwork_order import ink_payload
            title=[s for s in new[2] if s.get('role')=='title']
            if title:
                assert 'runtime_acceleration' not in ink_payload(new[0],title)['meta']
            module=new[1]['modules']['color']; profile=module['motion_candidate']
            from run_color_experiment import module_lines
            old_module = old[1]['modules']['color']
            for color in {s['color'] for s in module['strokes']}:
                old_lines, old_skipped = module_lines(old_module, color, old_module['motion_candidate'])
                new_lines, new_skipped = module_lines(module, color, profile)
                assert new_skipped == old_skipped
                assert len(new_lines) == len(old_lines)
                for before, after in zip(old_lines, new_lines):
                    assert before['xy'] == after['xy']
                    assert after['draw_speeds'] == [min(100, int(v*DRAW_SPEED_SCALE)) for v in before['draw_speeds']]
            assert profile['global_speed_factor'] == old[1]['modules']['color']['motion_candidate']['global_speed_factor']
            assert profile['approach_velocity_percent'] == min(30,old[1]['modules']['color']['motion_candidate']['travel_velocity_percent'])
            assert profile['touchdown_velocity_percent'] <= 15
            assert profile['dip_velocity_percent'] <= 10
            assert profile['acceleration_required'] is False
            assert profile['accel_joint_percent'] == profile['accel_linear_percent'] == 0
            assert new[3]['paper_z'] == changed['paper_z']
            for before, after in zip(old_points, new_points):
                assert before['x'] == after['x'] and before['y'] == after['y']
                assert abs(after['z']-before['z']-.25) < 1e-8
            dip = next(point for point in job.dip_waypoints(0) if point.get('tag') == 'dip')
            black = changed['stations']['black']
            assert dip['x'] == black['x'] and dip['y'] == black['y']
            assert abs(dip['z']-(black['surface_z']+black['dip_mm'])) < 1e-8
            assert dip['velocity'] == changed_ink_speed
            assert next(p for p in job.dip_waypoints(0) if p['tag']=='dip-out')['velocity'] == changed_withdraw_speed
            assert [p['velocity'] for p in new_points] == [p['velocity'] for p in old_points]
            for color in ('blue', 'green'):
                station = color_station(color, ask=Mock(side_effect=AssertionError('No fabricated measurements')))
                for key in ('x', 'surface_z', 'dip_mm', 'dwell_s', 'drip_s'):
                    assert getattr(station, key) == changed['stations'][color][key]
            # Original source identity ensures changed paper cannot silently resume.
            logs = Path(folder)/__import__('hashlib').sha256(str(path).encode()).hexdigest()[:16]
            session = Session(logs, path, old_manifest)
            session.begin('black')
            session.complete_stroke('black', 1)
            session.parked()
            runner = Mock(side_effect=AssertionError('Changed registration must block motion'))
            try:
                run_file(path, console=Console(io.StringIO()), folder=logs,
                         prepared_artwork=new, confirmed=True, ink_runner=runner)
            except ValueError as exc:
                assert '标定' in str(exc)
            else:
                raise AssertionError('Changed paper resumed previous progress')
            runner.assert_not_called()
        assert ArmConfig.PAPER_Z == changed['paper_z']
        # Other calibration mismatches remain blocked even with the overlay.
        invalid = json.loads(protected[-1].read_text(encoding='utf-8-sig'))
        invalid['ink_job']['meta']['canvas_corners'][0][0] += 1
        bad = Path(folder)/'invalid.json'
        server.write(bad, invalid)
        try:
            prepared_artwork(bad, settings)
        except ValueError:
            pass
        else:
            raise AssertionError('Canvas mismatch accepted')
        invalid['ink_job']['meta']['ink_execution']['calibration']['tool'] += 1
        server.write(bad, invalid)
        try:
            prepared_artwork(bad, settings)
        except ValueError:
            pass
        else:
            raise AssertionError('Tool mismatch accepted')
    for path, data in originals.items():
        assert path.read_bytes() == data, 'Original file changed: ' + str(path)
    print('All catalog artworks rebased; strokes and source files preserved; no hardware.')


if __name__ == '__main__':
    if '--integration' in sys.argv:
        integration()
    else:
        unittest.main()
