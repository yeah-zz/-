"""No hardware: JSON input, stroke-prefix edits, pause ordering and durable continuation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

import cv2
import numpy as np
import unified_painting
import run_json_painting as entry
from painting_session import Session, Console, StageControl, PauseRequested, PAUSED, session_lock


class SessionTests(unittest.TestCase):
    def manifest(self):
        return dict(registration={'paper': 1}, stages={'black': [{'id': 'a'}, {'id': 'b'}]})

    def test_completed_prefix_frozen_remaining_editable(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'painting.json'
            manifest = self.manifest()
            session = Session(folder, source, manifest)
            session.begin('black')
            session.complete_stroke('black', 1)
            session.parked()
            changed = copy.deepcopy(manifest)
            changed['stages']['black'][1]['points'] = [[1, 2], [3, 4]]
            resumed = Session(folder, source, changed)
            self.assertEqual(resumed.count('black'), 1)
            changed['stages']['black'][0]['id'] = 'wrong'
            with self.assertRaisesRegex(ValueError, '已画'):
                Session(folder, source, changed)

    def test_crash_and_registration_change_block_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'painting.json'
            session = Session(folder, source, self.manifest())
            session.begin('black')
            with self.assertRaisesRegex(ValueError, '未正常'):
                Session(folder, source, self.manifest())
            session.parked()
            changed = self.manifest()
            changed['registration']['paper'] = 2
            with self.assertRaisesRegex(ValueError, '标定'):
                Session(folder, source, changed)

    def test_pause_parks_before_prompt_and_continue_verifies(self):
        with tempfile.TemporaryDirectory() as folder:
            session = Session(folder, Path(folder)/'a.json', self.manifest())
            session.begin('black')
            events = []
            console = Console(io.StringIO(), ask=lambda _: events.append('ask') or 'c', poll=lambda: True)
            control = StageControl(session, 'black', '黑线', console)
            self.assertTrue(control.boundary(lambda: events.append('park'), lambda: events.append('verify')))
            self.assertEqual(events, ['park', 'ask', 'verify'])
            self.assertEqual(session.data['status'], 'running')

    def test_pause_failed_park_never_prompts_or_marks_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            session = Session(folder, Path(folder)/'a.json', self.manifest())
            session.begin('black')
            ask = Mock()
            control = StageControl(session, 'black', '黑线', Console(io.StringIO(), ask, lambda: True))
            with self.assertRaises(RuntimeError):
                control.boundary(Mock(side_effect=RuntimeError('arrival failed')), Mock())
            ask.assert_not_called()
            self.assertEqual(session.data['status'], 'running')

    def test_abandon_includes_session_and_rejects_live_session(self):
        from abandon_old_task import abandon
        with tempfile.TemporaryDirectory() as folder:
            session = Session(folder, Path(folder)/'a.json', self.manifest())
            session.parked()
            with session_lock(folder), self.assertRaises(OSError):
                abandon(folder)
            self.assertFalse(json.loads(session.path.read_text(encoding='utf-8')).get('abandoned'))
            self.assertEqual(abandon(folder), 1)
            self.assertTrue(json.loads(session.path.read_text(encoding='utf-8'))['abandoned'])
            fresh = Session(folder, Path(folder)/'a.json', self.manifest())
            self.assertEqual(fresh.count('black'), 0)

    def test_progress_is_single_line(self):
        output = io.StringIO()
        console = Console(output)
        console.progress('红色', 1, 4)
        console.progress('红色', 2, 4)
        self.assertEqual(output.getvalue().count('\r'), 2)
        self.assertNotIn('\n', output.getvalue())
        self.assertNotIn('MOVING', output.getvalue())


class JsonFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root/'logs'
        self.path = self.root/'painting.json'
        source = self.root/'reference.png'
        image = np.full((180, 240, 3), 255, np.uint8)
        cv2.line(image, (35, 135), (170, 40), (0, 0, 0), 3)
        cv2.circle(image, (145, 105), 26, (160, 180, 240), -1)
        unified_painting.imgio.imwrite(str(source), image)
        plan, _, _, points, meta = unified_painting.build_plan(source)
        self.bundle = entry.make_bundle(plan, dict(format='arm-painting job', version=1,
                                                   waypoints=points, meta=meta))
        self.path.write_text(json.dumps(self.bundle), encoding='utf-8')
        self.guard = patch('robot.E6Robot.__init__', side_effect=AssertionError('Hardware forbidden'))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def console(self, replies):
        answers = iter(replies)
        return Console(io.StringIO(), ask=lambda _: next(answers), poll=lambda: False)

    def runner(self, events):
        def run(*args, control, start_index, **kwargs):
            events.append((control.stage, start_index))
            for index in range(start_index + 1, control.total + 1):
                control.progress(index)
            print('MOVING hidden diagnostic')
            return 0
        return run

    def test_single_bundle_independent_of_companion_files(self):
        ink, plan, strokes, meta = entry.load_artwork(self.path)
        self.assertTrue(strokes)
        self.assertTrue(plan['modules']['color']['strokes'])
        self.assertEqual(meta['image'], plan['reference_image'])

    def test_wait_for_color_and_persist_across_reopen(self):
        events = []
        first = self.console(['RUN', ''])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.run_file(self.path, first, self.folder,
                ink_runner=self.runner(events), color_runner=self.runner(events)), 0)
        self.assertEqual([e[0] for e in events], ['black'])
        colors = list(dict.fromkeys(s['color'] for s in self.bundle['workflow']['modules']['color']['strokes']))
        second = self.console(['RUN', 'not-a-color', *colors])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.run_file(self.path, second, self.folder,
                ink_runner=self.runner(events), color_runner=self.runner(events),
                station_reader=lambda _: object()), 0)
        self.assertEqual([e[0] for e in events], ['black', *colors])
        self.assertNotIn('MOVING', second.stream.getvalue())
        self.assertIn('MOVING', next(self.folder.glob('*.log')).read_text(encoding='utf-8'))
        no_input = Mock(side_effect=AssertionError('Completed painting must not restart'))
        self.assertEqual(entry.run_file(self.path, Console(io.StringIO(), no_input), self.folder,
            ink_runner=no_input, color_runner=no_input), 0)

    def test_pause_quit_then_resume_at_next_black_stroke(self):
        events = []
        def pause_runner(*args, control, start_index):
            control.progress(1)
            control.console.poll = lambda: True
            try:
                control.boundary(lambda: events.append('park'), lambda: None)
            except PauseRequested:
                return PAUSED
        first = self.console(['RUN', 'Q'])
        self.assertEqual(entry.run_file(self.path, first, self.folder, ink_runner=pause_runner), PAUSED)
        second = self.console(['RUN', ''])
        self.assertEqual(entry.run_file(self.path, second, self.folder, ink_runner=self.runner(events)), 0)
        self.assertEqual(events, ['park', ('black', 1)])

    def test_failures_do_not_start_color_or_guess_resume(self):
        color = Mock(side_effect=AssertionError('Color must not start'))
        self.assertEqual(entry.run_file(self.path, self.console(['RUN']), self.folder,
            ink_runner=lambda *a, **kw: 4, color_runner=color), 4)
        with self.assertRaisesRegex(ValueError, '未正常'):
            entry.run_file(self.path, self.console([]), self.folder, color_runner=color)
        color.assert_not_called()


class ExecutorHookTests(unittest.TestCase):
    def color_run(self, folder, control, start_index=0):
        import run_color_experiment as trial
        from tests.test_color_experiment import ExperimentTests
        from pigment_station import PigmentStation
        lines = []
        for index in range(4):
            line = trial.plan_lines(100, 100+index*8, count=1, length=5)[0]
            line['id'] = 'short-%d' % index
            lines.append(line)
        station = PigmentStation(x=-89.7096, y=53.2723, surface_z=-18.581,
                                 rim_z=-31.118, strokes_per_load=4)
        real = trial.run
        with patch.object(trial, 'module_lines', return_value=(lines, [])), \
             patch.object(trial.time, 'sleep'), \
             patch.object(trial, 'run', side_effect=lambda *a, **kw:
                          real(*a, **kw, control=control, start_index=start_index)):
            return ExperimentTests().execute(folder, station=station, module={})

    def test_color_pause_continue_refills_and_skips_finished_prefix(self):
        calls, progress = [], []
        class Control:
            def progress(self, count): progress.append(count)
            def boundary(self, park, verify):
                calls.append('boundary')
                if len(calls) == 2:
                    park()
                    verify()
                    return True
                return False
        with tempfile.TemporaryDirectory() as folder:
            code, _ = self.color_run(folder, Control(), start_index=1)
            self.assertEqual(code, 0)
            record = json.loads(next(Path(folder).glob('color_experiment_*.json')).read_text())
            self.assertEqual([c['id'] for c in record['contacts']], ['short-1', 'short-2', 'short-3'])
            self.assertEqual(len(record['dips']), 2)
            self.assertEqual(progress, [1, 2, 3, 4])

    def test_color_pause_quit_leaves_no_pending_contact(self):
        class Control:
            count = 0
            def progress(self, count): self.count = count
            def boundary(self, park, verify):
                if self.count == 1:
                    park()
                    raise PauseRequested()
                return False
        with tempfile.TemporaryDirectory() as folder:
            code, _ = self.color_run(folder, Control())
            self.assertEqual(code, PAUSED)
            record = json.loads(next(Path(folder).glob('color_experiment_*.json')).read_text())
            self.assertTrue(record['paused'])
            self.assertNotIn('pending_move', record)
            self.assertEqual(len(record['contacts']), 1)
            self.assertTrue(record['contacts'][0]['completed'])

    def test_black_pause_quit_and_start_index_use_acknowledged_strokes(self):
        import run_mountain_balanced as black
        from tests.test_ink_control import stroke
        from ink_control import InkPolicy
        from park_color_loading import target_pose
        from types import SimpleNamespace
        strokes = [dict(stroke(8), id='b%d' % i) for i in range(3)]
        drawn = []
        pose = [target_pose()]
        robot = MagicMock()
        robot.get_mode.return_value = 5
        robot.get_error_id.return_value = '0,{[]},GetErrorID();'
        robot.get_pose.side_effect = lambda: pose[0]
        robot._ok.return_value = True
        class Motion:
            def __init__(self, *a):
                self.waypoints = []
                self.at_station = False
            def raise_at_current(self, z, **kw): pose[0] = (*pose[0][:2], z, *pose[0][3:])
            def send(self, points): pose[0] = (*[points[-1][k] for k in ('x', 'y', 'z')], *pose[0][3:])
            def dip(self, step): pass
            def stroke(self, value): drawn.append(value['id'])
            def rest(self, seconds): pass
        class Control:
            count = 1
            def progress(self, count): self.count = count
            def boundary(self, park, verify):
                if self.count == 2:
                    park()
                    raise PauseRequested()
                return False
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(black, 'load_reviewed', return_value=(strokes, SimpleNamespace(mm_per_px=1), InkPolicy(), {})), \
             patch.object(black, 'E6Robot', return_value=robot), \
             patch.object(black, 'Motion', Motion), patch.object(black, 'connect_readonly'), \
             contextlib.redirect_stdout(io.StringIO()):
            code = black.execute_reviewed_job(Path(folder)/'ink.json', start_index=1, control=Control())
            self.assertEqual(code, PAUSED)
            record = json.loads(next(Path(folder).rglob('*_execution_log.json')).read_text(encoding='utf-8'))
            self.assertEqual(record['completed_strokes'], 2)
            self.assertIsNone(record['active_action'])
            self.assertTrue(record['paused'])
            self.assertEqual(drawn, ['b1'])
            self.assertEqual(pose[0], target_pose())


if __name__ == '__main__':
    unittest.main()
