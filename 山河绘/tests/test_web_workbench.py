"""No hardware: stroke truth, stage gating, live launch safeguards and HTTP auth."""
import base64
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock
from urllib.error import HTTPError
from urllib.request import urlopen, Request

from web import server
from web.catalog import ARTWORKS
from web.execution import prepare, public_error
from web.worker import WebConsole

load_catalog = server.Workbench.load_library


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(server, 'ROOT', self.root), patch.object(server, 'STORE', self.root/'data'), patch.object(server.Workbench, 'load_library')]
        for p in self.patches:
            p.start()
        self.app = server.Workbench()
        self.source = self.root/'painting.json'
        self.bundle = {'format':'arm-painting-bundle-v1', 'workflow':{'modules':{'outline':{'strokes':[{'points':[[999,999]]}]}, 'color':{'strokes':[{'color':'blue','points':[[3,4],[5,6]]}]}}}, 'ink_job':{'meta':{'image_size':[100,80], 'ink_execution':{'mm_per_px':.2, 'base_strokes':[{'points':[[1,2],[2,3]]},{'points':[[2,3],[3,4]]}]}}}}
        server.write(self.source, self.bundle)
        self.ident = self.app.add(self.source, '试画')

    def tearDown(self):
        for p in self.patches[::-1]:
            p.stop()
        self.temp.cleanup()

    def test_canvas_uses_execution_not_preview(self):
        geometry = self.app.geometry[self.ident]
        self.assertEqual(geometry['counts'], {'black':2,'blue':1})
        self.assertEqual(geometry['stages']['black'][0]['points'][0], [1,2])

    def test_focal_color_preference_preserves_free_stage_selection(self):
        self.bundle['workflow']['execution_order'] = {
            'title_last': True, 'preferred_stage_order': ['black','green','blue','title'],
            'color_lead': '主松树'}
        self.bundle['ink_job']['meta']['ink_execution']['base_strokes'].append(
            {'role':'title', 'points':[[10,10],[11,11]]})
        self.bundle['workflow']['modules']['color']['strokes'].append(
            {'color':'green', 'points':[[7,8],[9,10]]})
        server.write(self.source, self.bundle)
        self.app.add(self.source, '试画', ident=self.ident)
        geometry = self.app.geometry[self.ident]
        self.assertEqual(list(geometry['counts']), ['black','green','blue','title'])
        self.assertEqual(self.app.items[self.ident]['colorLead'], '主松树')
        self.assertEqual(geometry['stages']['blue'][0]['points'], [[3,4],[5,6]])
        self.app.start_demo(self.ident, 'title')
        self.assertEqual(self.app.state()['active'], 'title')

    def test_idle_service_can_restart(self):
        self.app.request_restart()
        self.assertTrue(self.app.restarting)

    def test_restart_rejects_active_painting(self):
        self.app.start_demo(self.ident)
        with self.assertRaises(ValueError):
            self.app.request_restart()
        self.assertFalse(self.app.restarting)

    def test_restart_rejects_draft_and_import_processing(self):
        self.app.studio.jobs['draft'] = {'busy': True}
        with self.assertRaises(ValueError):
            self.app.request_restart()
        self.app.studio.jobs.clear()
        self.app.operations['import'] = {'status': 'building'}
        with self.assertRaises(ValueError):
            self.app.request_restart()
        self.assertFalse(self.app.restarting)

    def test_delete_preserves_materials_and_stays_deleted_after_reload(self):
        original = self.source.read_bytes()
        self.app.delete_artwork({'id': self.ident})
        self.assertNotIn(self.ident, self.app.items)
        self.assertNotIn(self.ident, self.app.geometry)
        self.assertEqual(self.source.read_bytes(), original)
        reloaded = server.Workbench()
        reloaded.add(self.source, '试画', ident=self.ident)
        self.assertNotIn(self.ident, reloaded.items)

    def test_delete_rejects_active_demo_and_live_worker(self):
        self.app.start_demo(self.ident)
        with self.assertRaises(ValueError):
            self.app.delete_artwork({'id': self.ident})
        self.app.runtime = None
        self.app.worker = Mock()
        self.app.worker.poll.return_value = None
        with self.assertRaises(ValueError):
            self.app.delete_artwork({'id': self.ident})
        self.assertIn(self.ident, self.app.items)

    def test_delete_completed_selection_clears_runtime(self):
        self.app.runtime = {'id': self.ident, 'status': 'complete', 'mode': 'demo'}
        self.app.delete_artwork({'id': self.ident})
        self.assertIsNone(self.app.runtime)

    def test_canvas_keeps_saved_ink_width(self):
        self.bundle['ink_job']['meta']['ink_execution']['base_strokes'][0]['width_mm'] = .98
        server.write(self.source, self.bundle)
        geometry = server.artwork_data(self.source)
        self.assertAlmostEqual(geometry['stages']['black'][0]['width'], 4.9)

    def test_black_then_explicit_color_and_completion(self):
        self.app.start_demo(self.ident)
        self.app.tick()
        self.assertEqual(self.app.state()['completed'], {'black':1,'blue':0})
        self.app.tick()
        self.assertEqual(self.app.state()['status'], 'waiting')
        self.app.tick()
        self.assertEqual(self.app.state()['completed']['blue'], 0)
        self.app.action({'command':'color','color':'blue'})
        self.app.tick()
        self.assertEqual(self.app.state()['status'], 'complete')
        self.assertEqual(self.app.state()['completed'], {'black':2,'blue':1})

    def test_demo_remains_isolated_from_an_external_robot_session(self):
        guard = self.root/'robot_runs'/'painting_session.lock'
        guard.parent.mkdir()
        guard.write_text('external-worker')
        with patch.object(server.subprocess, 'Popen') as motion:
            self.app.start_demo(self.ident)
            self.app.tick()
            self.assertEqual(guard.read_text(), 'external-worker')
            self.assertEqual(self.app.state()['mode'], 'demo')
            motion.assert_not_called()

    def test_two_colors_wait_for_preparation_and_keep_chosen_order(self):
        self.bundle['workflow']['modules']['color']['strokes'].append({'color':'green', 'points':[[7,8],[8,9]]})
        server.write(self.source, self.bundle)
        self.app.add(self.source, '试画', ident=self.ident)
        self.app.start_demo(self.ident)
        self.app.tick(); self.app.tick()
        self.app.action({'command':'color', 'color':'green'})
        self.app.tick()
        self.assertEqual(self.app.state()['status'], 'waiting')
        for _ in range(8): self.app.tick()
        self.assertEqual(self.app.state()['completed']['blue'], 0)
        self.app.action({'command':'color', 'color':'blue'})
        self.app.tick()
        self.assertEqual(self.app.state()['stageOrder'], ['black','green','blue'])
        self.assertEqual(self.app.state()['status'], 'complete')
        self.assertTrue(any('洗笔' in e['message'] for e in self.app.state()['events']))

    def test_catalog_only_lists_selected_paintings_from_material_library(self):
        for directory, _, _, source in ARTWORKS:
            path = self.root/source
            server.write(path, self.bundle)
            image = self.root/'待画作品'/directory/'参考原图.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b'image')
        self.app.items.clear()
        load_catalog(self.app)
        self.assertEqual([i['title'] for i in self.app.public_items()], [i[1] for i in ARTWORKS])
        self.assertEqual(len(self.app.items), len(ARTWORKS))
        for item, (_, _, _, source) in zip(self.app.items.values(), ARTWORKS):
            self.assertEqual(item['path'], self.root/source)
            self.assertTrue(item['featured'])
        self.assertTrue(all('path' not in i for i in self.app.public_items()))

    def test_missing_selected_artwork_has_operator_notice(self):
        with patch.object(server.sys, 'stderr'):
            load_catalog(self.app)
        self.assertEqual(len(self.app.library_errors), len(ARTWORKS))
        self.assertNotIn('.json', self.app.library_errors[0]['message'])

    def test_pause_resume_retains_prefix(self):
        self.app.start_demo(self.ident)
        self.app.tick()
        self.app.action({'command':'pause'})
        self.app.tick()
        self.assertEqual(self.app.state()['completed']['black'], 1)
        self.app.action({'command':'resume'})
        self.app.tick()
        self.assertEqual(self.app.state()['completed']['black'], 2)

    def test_cannot_color_before_black_or_restart_running(self):
        self.app.start_demo(self.ident)
        with self.assertRaises(ValueError):
            self.app.action({'command':'color','color':'blue'})
        with self.assertRaises(ValueError):
            self.app.start_demo(self.ident)

    def test_live_requires_explicit_confirmation_before_any_process(self):
        with patch.object(server.subprocess, 'run') as validation, patch.object(server.subprocess, 'Popen') as motion:
            with self.assertRaises(ValueError):
                self.app.action({'command':'live','id':self.ident})
            validation.assert_not_called()
            motion.assert_not_called()

    def test_failed_offline_check_never_launches_worker(self):
        with patch.object(server.subprocess, 'run', return_value=Mock(returncode=1, stderr='Calibration changed',stdout='')), patch.object(server.subprocess, 'Popen') as motion:
            with self.assertRaisesRegex(ValueError,'校验未通过'):
                self.app.action({'command':'live','id':self.ident,'confirmation':'RUN'})
            motion.assert_not_called()

    def test_valid_live_launch_uses_same_prepared_worker_for_check_and_execution(self):
        process = Mock(pid=2345)
        process.poll.return_value = None
        with patch.object(server.subprocess, 'run', return_value=Mock(returncode=0, stderr='', stdout='')) as validation, patch.object(server.subprocess, 'Popen', return_value=process) as launch:
            state = self.app.action({'command':'live','id':self.ident,'confirmation':'RUN'})
        check_args = validation.call_args.args[0]
        execute_args = launch.call_args.args[0]
        self.assertEqual(check_args[1], execute_args[1])
        self.assertEqual(check_args[2], str(self.source))
        self.assertEqual(check_args[3], '--check')
        self.assertEqual(check_args[check_args.index('--start-stage')+1], 'black')
        self.assertEqual(execute_args[2], str(self.source))
        self.assertEqual(state['status'], 'starting')
        self.assertEqual(state['progressMode'], 'memory')
        self.assertIn('SHH_BRIDGE_PORT', launch.call_args.kwargs['env'])
        self.assertFalse((server.STORE/'control'/'runtime.json').exists())
        self.app.bridge.close()

    def test_existing_session_lock_never_launches_motion(self):
        lock=self.root/'robot_runs'/'painting_session.lock'
        lock.parent.mkdir()
        lock.write_text('1234')
        with patch.object(server.subprocess,'Popen') as motion:
            with self.assertRaises(ValueError):
                self.app.action({'command':'live','id':self.ident,'confirmation':'RUN'})
            motion.assert_not_called()

    def test_monitor_only_confirms_durable_progress_no_timer(self):
        record = {'status':'running','active':'black','completed':{'black':1,'blue':0}}
        with patch.object(self.app,'session',return_value=record):
            self.app.action({'command':'monitor','id':self.ident})
            for _ in range(10):
                self.app.tick()
            self.assertEqual(self.app.state()['completed']['black'],1)
            with self.assertRaises(ValueError):
                self.app.action({'command':'pause'})
            record['completed']['black']=2
            self.assertEqual(self.app.state()['completed']['black'],2)
            self.app.action({'command':'unmonitor'})
            self.assertEqual(self.app.state()['status'],'idle')
            self.assertEqual(record['status'],'running')

    def test_live_stale_prompt_cannot_be_replayed(self):
        self.app.runtime={'id':self.ident,'mode':'live','status':'waiting','completed':{'black':2,'blue':0},'events':[]}
        server.write(server.STORE/'control'/'status.json', {'prompt':{'id':'current','kind':'color'}})
        with patch.object(self.app,'session',return_value=None):
            with self.assertRaises(ValueError):
                self.app.action({'command':'color','color':'blue','promptId':'old'})
            self.assertFalse((server.STORE/'control'/'reply.json').exists())
            self.app.action({'command':'color','color':'blue','promptId':'current'})
            self.assertEqual(server.read(server.STORE/'control'/'reply.json')['value'],'blue')

    def test_malformed_geometry_rejected(self):
        self.bundle['ink_job']['meta']['ink_execution']['base_strokes'][0]['points']=[[float('nan'),2]]
        server.write(self.source,self.bundle)
        with self.assertRaises(ValueError):
            server.artwork_data(self.source)

    def test_monitor_replays_saved_manifest_even_if_source_changed(self):
        manifest={'stages':{'black':[{'points':[[8,9],[9,10]]}], 'blue':[{'points':[[5,6],[6,7]]}]}, 'registration':{'image_size':[100,80]}}
        record={'status':'paused','active':None,'completed':{'black':1,'blue':0},'manifest':manifest}
        with patch.object(self.app,'session',return_value=record):
            self.app.action({'command':'monitor','id':self.ident})
            self.assertEqual(self.app.geometry[self.ident]['counts']['black'],1)
            self.assertEqual(self.app.geometry[self.ident]['stages']['black'][0]['points'][0],[8,9])

    def test_saved_live_worker_can_be_reopened_instead_of_fake_resume(self):
        self.app.runtime={'id':self.ident,'mode':'live','status':'paused','completed':{'black':1},'events':[]}
        self.app.worker=Mock()
        self.app.worker.poll.return_value=0
        with patch.object(self.app,'session',return_value={'status':'paused','active':None,'completed':{'black':1}}):
            self.assertEqual(self.app.state()['status'],'saved')

    def test_server_reconnects_original_worker_without_new_motion(self):
        guard=self.root/'robot_runs'/'painting_session.lock'
        guard.parent.mkdir()
        guard.write_text('2345')
        server.write(server.STORE/'control'/'runtime.json', {'id':'old-id','mode':'live','status':'running','completed':{},'events':[],'source':str(self.source),'workerPid':2345})
        with patch.object(server.AttachedWorker,'poll',return_value=None), patch.object(server.subprocess,'Popen') as launch:
            self.app.reconnect_worker()
            self.assertEqual(self.app.runtime['id'],self.ident)
            self.assertEqual(self.app.worker.pid,2345)
            launch.assert_not_called()

    def test_json_import_is_persisted_and_does_not_move(self):
        with patch.object(server.subprocess,'Popen') as motion:
            result=self.app.upload({'name':'参考作品.json','data':base64.b64encode(json.dumps(self.bundle).encode()).decode()})
            self.assertEqual(result['status'],'ready')
            self.assertEqual(self.app.items[result['id']]['title'],'参考作品')
            motion.assert_not_called()

    def test_mutations_require_same_workbench_token(self):
        http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        http.app=self.app
        thread=threading.Thread(target=http.serve_forever,daemon=True)
        thread.start()
        url='http://127.0.0.1:%d/api/action'%http.server_port
        payload=json.dumps({'command':'demo','id':self.ident}).encode()
        try:
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(url,data=payload,headers={'Content-Type':'application/json'}))
            self.assertEqual(error.exception.code,403)
            self.assertIsNone(self.app.runtime)
            result=json.load(urlopen(Request(url,data=payload,headers={'Content-Type':'application/json','X-Workbench-Token':self.app.token})))
            self.assertEqual(result['status'],'running')
        finally:
            http.shutdown()
            http.server_close()

    def test_live_waiting_and_stage_events_use_confirmed_counts_only(self):
        self.app.runtime = {'id':self.ident,'mode':'live','status':'starting','completed':{},'events':[]}
        self.app.worker = Mock()
        self.app.worker.poll.return_value = None
        record = {'status':'ready','active':None,'completed':{'black':2,'blue':0}}
        server.write(server.STORE/'control'/'status.json', {'prompt':{'id':'next','kind':'color'},'stageOrder':['black']})
        with patch.object(self.app, 'session', return_value=record):
            for _ in range(3): self.app.state(); self.app.tick()
        state = self.app.state()
        self.assertEqual(state['completed']['blue'], 0)
        self.assertEqual(state['status'], 'waiting')
        self.assertEqual(sum(e['message'] == '勾勒完成' for e in state['events']), 1)
        self.assertEqual(sum('洗净毛笔' in e['message'] for e in state['events']), 1)


class WorkerBridgeTests(unittest.TestCase):
    def test_worker_notice_hides_source_and_publishes_stage_order(self):
        with tempfile.TemporaryDirectory() as folder:
            console = WebConsole(folder)
            with patch('builtins.print'):
                console.say('画稿：E:\\private\\painting.json\n绘画中按 P')
            state = server.read(Path(folder)/'status.json')
            self.assertNotIn('private', state['notice'])
            console.progress('黑线', 1, 2)
            console.progress('绿色', 1, 3)
            self.assertEqual(server.read(Path(folder)/'status.json')['stageOrder'], ['black','green'])

    def test_bridge_waits_for_matching_prepared_color_reply(self):
        with tempfile.TemporaryDirectory() as folder:
            console = WebConsole(folder)
            publish = console.publish
            def reply(**kwargs):
                publish(**kwargs)
                if kwargs.get('prompt'):
                    prompt = kwargs['prompt']
                    self.assertEqual(prompt['kind'], 'color')
                    server.write(Path(folder)/'reply.json', {'id':prompt['id'],'value':'blue'})
            with patch.object(console, 'publish', side_effect=reply):
                self.assertEqual(console.ask('输入颜色：'), 'blue')
            self.assertIsNone(server.read(Path(folder)/'status.json')['prompt'])

    def test_speed_profile_is_process_local_and_preserves_registration(self):
        import types
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'painting.json'
            server.write(path, {'workflow':{'speed_revision':{'process_local':True,'global_speed_factor':70,'travel':80,'dip':20,'withdraw':20}}})
            arm = types.SimpleNamespace(SPEED_FACTOR=30, PAPER_Z=-36)
            job = types.SimpleNamespace(TRANSIT_SPEED=60, dip_waypoints=lambda:[{'tag':'dip','velocity':40},{'tag':'dip-out','velocity':40},{'tag':'dip-in','velocity':80}])
            job.stroke_waypoints = lambda *args, **kwargs: []
            original = path.read_bytes()
            with patch('web.execution._base_motion', None), patch.dict('sys.modules', {'config':types.SimpleNamespace(ArmConfig=arm), 'job':job}):
                prepare(path)
            self.assertEqual(arm.SPEED_FACTOR, 70)
            self.assertEqual(arm.PAPER_Z, -36)
            self.assertEqual([p['velocity'] for p in job.dip_waypoints()], [20,20,80])
            self.assertEqual(path.read_bytes(), original)

    def test_error_language_hides_machine_details(self):
        message = public_error('Calibration/settings changed; E:\\private\\task.json')
        self.assertIn('现场', message)
        self.assertNotIn('private', message)
        self.assertIn('网络', public_error('Connection refused: 192.168.5.1'))
        message = public_error("[WinError 5] 拒绝访问: E:\\private\\session.tmp -> session.json")
        self.assertIn('保存失败', message)
        self.assertNotIn('private', message)

    def test_color_order_survives_saving_and_reopening_session(self):
        from painting_session import Session
        with tempfile.TemporaryDirectory() as folder:
            manifest = {'registration':{}, 'stages':{'black':[{}], 'blue':[{}], 'green':[{}]}}
            session = Session(folder, Path(folder)/'painting.json', manifest)
            for stage in ('black', 'green'):
                session.begin(stage)
                session.complete_stroke(stage, 1)
                session.parked('ready')
            reopened = Session(folder, Path(folder)/'painting.json', manifest)
            reopened.begin('blue')
            self.assertEqual(server.read(reopened.path)['stage_order'], ['black','green','blue'])


if __name__=='__main__':
    unittest.main()
