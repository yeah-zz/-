"""Real IPC and executor controls, with all hardware calls forbidden."""
import copy
import io
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from web import server
from web.bridge import MemoryBridge, WorkerChannel
from web.worker import WebConsole
from painting_session import Session
from tests import test_web_workbench as fixtures


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkbenchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.app = self.fixture.app
        self.bridge = MemoryBridge()
        self.addCleanup(self.bridge.close)
        self.channel = WorkerChannel(self.bridge.address[1], self.bridge.key.hex())
        self.addCleanup(self.channel.close)
        self.app.bridge = self.bridge
        self.app.worker = Mock()
        self.app.worker.poll.return_value = None
        self.app.runtime = {'id':self.fixture.ident,'mode':'live','progressMode':'memory',
                            'status':'starting','completed':{},'events':[]}

    def wait(self, condition):
        deadline = time.monotonic()+4
        while time.monotonic() < deadline:
            result = condition()
            if result:
                return result
            time.sleep(.01)
        self.fail('Expected worker state did not arrive')

    def test_pause_continue_color_and_canvas_counts_without_progress_files(self):
        import run_json_painting as entry
        source = self.fixture.source
        logs = self.fixture.root/'robot_runs'
        manifest = {'registration':{},'stages':{'black':[{'id':'a'},{'id':'b'}], 'blue':[{'id':'c'}]}}
        plan = {'modules':{'color':{'strokes':[{'color':'blue'}], 'motion_candidate':{}}}}
        console = WebConsole(self.fixture.root/'control', self.channel)
        console.stream = io.StringIO()
        reached = threading.Event()
        boundary = threading.Event()
        events, errors = [], []
        def ink_runner(*args, control, start_index):
            self.assertEqual(start_index, 0)
            control.progress(1)
            reached.set()
            self.assertTrue(boundary.wait(4))
            control.boundary(lambda: events.append('park'), lambda: events.append('verify'))
            control.progress(2)
            return 0
        def color_runner(*args, control, start_index, **kwargs):
            control.progress(1)
            return 0
        def execute():
            try:
                entry.run_file(source, console, logs, confirmed=True, save_progress=False, fresh_run=True,
                               prepared_artwork=({}, plan, manifest['stages']['black'], {}), ink_runner=ink_runner,
                               color_runner=color_runner, station_reader=lambda _:object())
                console.publish(done=True)
            except Exception as exc:
                errors.append(exc)
        with patch.object(entry,'manifest_for',return_value=manifest), patch('web.worker.save',side_effect=AssertionError('No status-file writes')), patch('robot.E6Robot.__init__',side_effect=AssertionError('No hardware')):
            thread = threading.Thread(target=execute,daemon=True)
            thread.start()
            self.assertTrue(reached.wait(4))
            self.assertEqual(self.app.state()['completed']['black'],1)
            progress_file = next(logs.glob('painting_session_*.json'))
            before = progress_file.read_bytes()
            self.assertNotIn('completed',server.read(progress_file))
            self.app.action({'command':'pause'})
            with self.assertRaisesRegex(ValueError,'已经请求'):
                self.app.action({'command':'pause'})
            boundary.set()
            paused = self.wait(lambda:self.app.state().get('prompt'))
            self.assertEqual(paused['kind'],'pause')
            self.assertEqual(self.app.state()['completed']['black'],1)
            self.assertNotIn('completed',server.read(progress_file))
            with self.assertRaisesRegex(ValueError,'尚未退出'):
                self.app.clear_records({'confirmation':'CLEAR','controllerStopped':True})
            with self.assertRaises(ValueError):
                self.app.action({'command':'resume','promptId':'stale'})
            self.app.action({'command':'resume','promptId':paused['id']})
            color_prompt = self.wait(lambda:(self.app.state().get('prompt') or {}).get('kind')=='color')
            current = self.app.state()
            self.assertEqual(current['completed']['black'],2)
            self.assertEqual(events,['park','verify'])
            self.app.action({'command':'color','color':'blue','promptId':current['prompt']['id']})
            thread.join(4)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors,errors)
        self.app.worker.poll.return_value = 0
        state = self.app.state()
        self.assertEqual(state['status'],'complete')
        self.assertEqual(state['completed'],{'black':2,'blue':1})
        self.assertEqual(state['stageOrder'],['black','blue'])
        self.assertFalse(server.read(progress_file)['resumable'])
        self.assertNotIn('manifest',server.read(progress_file))
        self.assertFalse((self.fixture.root/'control'/'status.json').exists())

    def test_fresh_memory_run_does_not_inherit_old_saved_counts(self):
        logs = self.fixture.root/'records'
        manifest = {'registration':{},'stages':{'black':[{'id':'a'},{'id':'b'}]}}
        old = Session(logs,self.fixture.source,manifest)
        old.begin('black');old.complete_stroke('black',1);old.parked()
        fresh = Session(logs,self.fixture.source,manifest,save_progress=False,fresh_run=True)
        self.assertEqual(fresh.count('black'),0)
        old.begin('black');old.uncertain()
        with self.assertRaises(ValueError):
            Session(logs,self.fixture.source,manifest,save_progress=False,fresh_run=True)

    def test_clear_requires_confirmation_and_archives_only_records(self):
        self.app.worker.poll.return_value = 0
        root = self.fixture.root/'robot_runs'
        original = self.fixture.source.read_bytes()
        server.write(root/'painting_session_test.json',{'status':'uncertain','completed':{'black':23}})
        server.write(root/'color_experiment_test.json',{'pending_move':{'id':1}})
        server.write(server.STORE/'site_settings.json',{'paper_z':-36})
        with self.assertRaises(ValueError):
            self.app.clear_records({'confirmation':'CLEAR'})
        with patch.object(server.subprocess,'Popen') as motion:
            result = self.app.clear_records({'confirmation':'CLEAR','controllerStopped':True})
        motion.assert_not_called()
        self.assertEqual(result['clearedRecords'],2)
        self.assertEqual(result['status'],'idle')
        self.assertFalse(list(root.glob('painting_session_*.json')))
        self.assertEqual(len(list((root/'cleared_records').glob('*/*.json'))),2)
        self.assertEqual(self.fixture.source.read_bytes(),original)
        self.assertEqual(server.read(server.STORE/'site_settings.json'),{'paper_z':-36})

    def test_live_start_has_no_durable_runtime_or_status(self):
        self.app.worker.poll.return_value = 0
        with patch.object(server.subprocess,'run',return_value=Mock(returncode=0)), patch.object(server.subprocess,'Popen',return_value=Mock(pid=1234,poll=lambda:None)):
            self.app.action({'command':'live','id':self.fixture.ident,'confirmation':'RUN'})
        self.addCleanup(self.app.bridge.close)
        self.assertEqual(self.app.runtime['progressMode'],'memory')
        self.assertFalse((server.STORE/'control'/'runtime.json').exists())


if __name__=='__main__':unittest.main()
