"""Independent web stages, with real session/IPC controls and no hardware."""
import copy
import io
import itertools
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import run_json_painting as entry
from painting_session import Console, PAUSED, PauseRequested
from web import server
from web.bridge import MemoryBridge, WorkerChannel
from web.worker import WebConsole
from tests import test_web_workbench as fixtures


class CapturingConsole(Console):
    def bind_session(self, session):
        self.session = session


class FreeStageExecutionTests(unittest.TestCase):
    def execute(self, order, replies=None, fail=None, pause=False):
        stages = {'black': [{'id':'body'}], 'blue': [{'id':'wash-blue','color':'blue'}],
                  'green': [{'id':'wash-green','color':'green'}], 'title': [{'id':'title','role':'title'}]}
        manifest = {'registration': {}, 'stages': stages}
        ink = {'meta': {'ink_execution': {'base_strokes': stages['black'] + stages['title']}}}
        plan = {'modules': {'color': {'motion_candidate': {}, 'strokes': stages['blue'] + stages['green']}}}
        answers = iter(replies if replies is not None else [*order[1:], 'q'])
        console = CapturingConsole(io.StringIO(), ask=lambda _: next(answers), poll=lambda: pause)
        calls, boundaries = [], []
        def runner(path=None, *, control, start_index, color=None, **kwargs):
            calls.append(control.stage)
            self.assertEqual(start_index, 0)
            if path:
                self.assertEqual(json.loads(Path(path).read_text())['strokes'], stages[control.stage])
            else:
                self.assertEqual(color, control.stage)
            if fail == control.stage:
                return 4
            if pause:
                try:
                    control.boundary(lambda: boundaries.append('park'), lambda: boundaries.append('verify'))
                except PauseRequested:
                    return PAUSED
            control.progress(1)
            return 0
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(entry, 'manifest_for', return_value=copy.deepcopy(manifest)), \
                patch('artwork_order.ink_payload', side_effect=lambda ink, strokes: {'strokes':strokes}), \
                patch('robot.E6Robot.__init__', side_effect=AssertionError('No hardware')):
            result = entry.run_file(Path(folder)/'art.json', console, Path(folder)/'runs',
                confirmed=True, save_progress=False, fresh_run=True, prepared_artwork=(ink,plan,stages['black']+stages['title'],{}),
                ink_runner=runner, color_runner=runner, station_reader=lambda _:object(),
                free_stages=True, initial_stage=order[0])
            return result, calls, console.session.data, boundaries

    def test_all_24_orders_execute_only_chosen_stages(self):
        for order in itertools.permutations(['black','blue','green','title']):
            with self.subTest(order=order):
                result,calls,data,_ = self.execute(order)
                self.assertEqual(result, 0)
                self.assertEqual(calls, list(order))
                self.assertEqual(data['stage_order'], list(order))
                self.assertEqual(data['status'], 'complete')

    def test_each_stage_can_run_alone_without_marking_others_complete(self):
        for stage in ['black','blue','green','title']:
            with self.subTest(stage=stage):
                result,calls,data,_ = self.execute([stage])
                self.assertEqual(calls, [stage])
                self.assertEqual(data['completed'], {k:int(k==stage) for k in data['completed']})
                self.assertEqual(data['status'], 'paused')

    def test_body_then_title_skips_all_colors(self):
        _,calls,data,_ = self.execute(['black','title'])
        self.assertEqual(calls, ['black','title'])
        self.assertEqual(data['completed']['blue'], 0)
        self.assertEqual(data['completed']['green'], 0)

    def test_invalid_or_completed_choice_never_moves(self):
        _,calls,_,_ = self.execute(['title'], replies=['title','unknown','blue','q'])
        self.assertEqual(calls, ['title','blue'])

    def test_failed_stage_never_runs_following_stage(self):
        result,calls,data,_ = self.execute(['title','black'], fail='title')
        self.assertEqual(result, 4)
        self.assertEqual(calls, ['title'])
        self.assertEqual(data['status'], 'uncertain')

    def test_pause_continue_still_verifies_before_drawing(self):
        result,calls,data,boundaries = self.execute(['title'], replies=['c','q'], pause=True)
        self.assertEqual(result, 0)
        self.assertEqual(boundaries, ['park','verify'])
        self.assertEqual(data['completed']['title'], 1)

    def test_pause_exit_leaves_unfinished_stage_uncompleted(self):
        result,_,data,boundaries = self.execute(['blue'], replies=['q'], pause=True)
        self.assertEqual(result, PAUSED)
        self.assertEqual(boundaries, ['park'])
        self.assertEqual(data['completed']['blue'], 0)


class FreeStageWebTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkbenchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.app = self.fixture.app
        bundle = self.fixture.bundle
        bundle['workflow']['execution_order'] = {'title_last':True}
        for i,s in enumerate(bundle['ink_job']['meta']['ink_execution']['base_strokes']):
            s['role'] = 'title' if i else 'coarse'
        server.write(self.fixture.source,bundle)
        self.app.add(self.fixture.source,'test',ident=self.fixture.ident)

    def test_demo_title_first_then_body_then_color(self):
        self.app.action({'command':'demo','id':self.fixture.ident,'stage':'title'})
        self.app.tick()
        self.assertEqual(self.app.state()['completed'], {'black':0,'blue':0,'title':1})
        self.app.action({'command':'stage','stage':'black'})
        self.app.tick()
        self.app.action({'command':'stage','stage':'blue'})
        self.app.tick()
        self.assertEqual(self.app.state()['stageOrder'], ['title','black','blue'])
        self.assertEqual(self.app.state()['status'], 'complete')

    def test_demo_stage_selection_requires_boundary_and_rejects_repeat(self):
        self.app.start_demo(self.fixture.ident, 'blue')
        with self.assertRaises(ValueError):
            self.app.action({'command':'stage','stage':'title'})
        self.app.tick()
        with self.assertRaises(ValueError):
            self.app.action({'command':'stage','stage':'blue'})
        self.app.action({'command':'finish'})
        self.assertEqual(self.app.state()['completed']['black'], 0)

    def test_launch_passes_same_initial_stage_to_check_and_worker(self):
        with patch.object(server.subprocess,'run',return_value=Mock(returncode=0)) as check, \
                patch.object(server.subprocess,'Popen',return_value=Mock(poll=lambda:None)) as launch:
            self.app.action({'command':'live','id':self.fixture.ident,'confirmation':'RUN','stage':'title'})
        self.addCleanup(self.app.bridge.close)
        for call in (check.call_args, launch.call_args):
            args=call.args[0]
            self.assertEqual(args[args.index('--start-stage')+1], 'title')
        self.assertTrue(self.app.runtime['freeStages'])

    def test_missing_stage_rejected_before_check_or_launch(self):
        with patch.object(server.subprocess,'run') as check, patch.object(server.subprocess,'Popen') as launch:
            with self.assertRaises(ValueError):
                self.app.action({'command':'live','id':self.fixture.ident,'confirmation':'RUN','stage':'red'})
            check.assert_not_called()
            launch.assert_not_called()

    def test_real_bridge_stage_prompt_token_and_pending_guard(self):
        bridge = MemoryBridge()
        self.addCleanup(bridge.close)
        channel = WorkerChannel(bridge.address[1], bridge.key.hex())
        self.addCleanup(channel.close)
        self.app.bridge = bridge
        self.app.worker = Mock(poll=lambda:None)
        self.app.runtime = {'id':self.fixture.ident,'mode':'live','freeStages':True,'progressMode':'memory',
                            'status':'ready','completed':{'black':0,'blue':1,'title':0},'events':[]}
        console = WebConsole(self.fixture.root/'control', channel)
        from types import SimpleNamespace
        console.session = SimpleNamespace(data={'status':'ready','active':None,
            'completed':{'black':0,'blue':1,'title':0},'stage_order':['blue']})
        received=[]
        thread=threading.Thread(target=lambda:received.append(console.ask('选择绘画阶段：')),daemon=True)
        with patch('web.worker.save',side_effect=AssertionError('No progress-file writes')):
            thread.start()
            deadline=time.monotonic()+4
            while time.monotonic()<deadline:
                state=self.app.state()
                if state.get('prompt'):break
                time.sleep(.01)
            token=state['prompt']['id']
            self.assertEqual(state['prompt']['kind'],'stage')
            with self.assertRaises(ValueError):
                self.app.action({'command':'stage','stage':'title','promptId':'stale'})
            with self.assertRaises(ValueError):
                self.app.action({'command':'stage','stage':'blue','promptId':token})
            self.app.action({'command':'stage','stage':'title','promptId':token})
            with self.assertRaises(ValueError):
                self.app.action({'command':'stage','stage':'black','promptId':token})
            thread.join(4)
            self.assertFalse(thread.is_alive())
            self.assertEqual(received,['title'])


if __name__ == '__main__':
    unittest.main()
