import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import run_color_experiment as trial
import run_painting_experiment as entry
from color_profile import ColorProfile

PROFILE = ColorProfile().to_dict()
TEST_DRAW_V = 11          # 测试里给行笔一个与众不同的速度，才能精确拒绝"某一条行笔指令"
                          # ★ 必须与 ColorProfile 的 travel/dip 默认值都不同（原来用 15，
                          #   降速后 travel 也变成 15，会把"横移指令"误当成"行笔指令"拒掉）


class ExperimentTests(unittest.TestCase):
    def test_coordinates_and_parameters(self):
        self.assertEqual(len(trial.plan_lines(100,100)),2)
        for x,y in [(10000,100), (float('nan'),100),(-35,35)]:
            with self.assertRaises(ValueError): trial.plan_lines(x,y)
        with self.assertRaises(ValueError): trial.plan_lines(100,100,press=3.0)
        with self.assertRaises(ValueError): trial.run(100,100)

    def execute(self, folder, reject=False, bad_pose=False, station=None, interrupt=False,
                events=None, module=None, profile=None, save_progress=None, control=None,
                color='yellow'):
        robot=MagicMock(); robot.get_mode.return_value=5; robot.get_error_id.return_value='0,{[]},GetErrorID();'
        robot._ok.side_effect=lambda response: response!='rejected'
        def send(command):
            if reject and 'MovL' in command and 'v=%d' % TEST_DRAW_V in command:
                return 'rejected'
            return 'ok'
        robot.send.side_effect=send
        current=[trial.target_pose()]
        if bad_pose: current[0]=(0,0,0,0,0,0)
        def arrival(r,target):
            if interrupt: raise KeyboardInterrupt()
            current[0]=target
            if events is not None:
                events.append(('arrived', target[2]))
        original_persist = trial.persist
        def save(path, record):
            if events is not None:
                events.append(('persist', record.get('pending_move', {}).get('name')))
            original_persist(path, record)
        with patch.object(trial,'E6Robot',return_value=robot), \
             patch.object(trial,'persist',side_effect=save), \
             patch.object(trial,'connect_readonly'), \
             patch.object(trial,'read_idle_pose',side_effect=lambda *a:current[0]), \
             patch.object(trial,'wait_arrival',side_effect=arrival),contextlib.redirect_stdout(io.StringIO()):
            extra={}
            if control is not None: extra['control']=control
            if save_progress is not None: extra['save_progress']=save_progress
            code=trial.run(100,100,confirm_site=True,log_dir=folder,station=station,
                           profile=profile or ColorProfile(press_mm=.22, draw_velocity_percent=TEST_DRAW_V),
                           module=module,color=color if module is not None else None,
                           **extra)
        return code,robot

    def test_four_short_color_strokes_share_load_and_low_transfers(self):
        from pigment_station import PigmentStation
        lines=[]
        for i in range(4):
            line=trial.plan_lines(100,100+i*8,count=1,length=5)[0]
            line['id']='short-%d'%i
            line['sequence']=[(name,(*pose[:2],trial.z_for_lift(5),*pose[3:])
                              if name in ('hover','lift') else pose) for name,pose in line['sequence']]
            lines.append(line)
        station=PigmentStation(x=-89.7096,y=53.2723,surface_z=-18.581,rim_z=-31.118,strokes_per_load=4)
        with tempfile.TemporaryDirectory() as d, patch.object(trial,'module_lines',return_value=(lines,[])), \
                patch.object(trial.time,'sleep'):
            code,_=self.execute(d,station=station,module={})
            self.assertEqual(code,0)
            record=json.loads(next(Path(d).glob('color_experiment_*.json')).read_text())
            self.assertEqual(len(record['dips']),1)
            names=[s['name'] for s in record['steps']]
            self.assertEqual([n for n in names if n.endswith(':high')],['short-0:high'])
            self.assertFalse(any(n.endswith(':raise') for n in names))
            self.assertEqual(names[-2:],['raise-before-return','return-loading-high'])
            self.assertTrue(all(c['completed'] for c in record['contacts']))
            self.assertEqual(sum(n == 'queued-dip' for n in names), 1)
            queued = [s for s in record['steps'] if s['name'].endswith(':queued-stroke')]
            self.assertTrue(all('targets' not in s for s in queued))
            self.assertEqual(len(record['lines']), 4)  # Full geometry retained once for recovery.

    def test_failed_dip_keeps_pending_batch(self):
        from pigment_station import PigmentStation
        from queued_motion import send_batch as real_send_batch
        station=PigmentStation(x=-89.7096,y=53.2723,surface_z=-18.581,rim_z=-31.118)
        def reject_dip(robot, motions, wait):
            if len(motions) == 3:
                raise trial.RobotError('Rejected dip batch')
            return real_send_batch(robot, motions, wait)
        with tempfile.TemporaryDirectory() as d, patch('queued_motion.send_batch', side_effect=reject_dip):
            code, _ = self.execute(d, station=station)
            self.assertEqual(code, 4)
            record=json.loads(next(Path(d).glob('color_experiment_*.json')).read_text())
            self.assertEqual(record['pending_move']['name'], 'queued-dip')
            self.assertEqual(len(record['pending_move']['targets']), 3)
            self.assertFalse(record['contacts'])

    def test_contact_is_prelogged_and_no_disk_writes_until_lift(self):
        events = []
        with tempfile.TemporaryDirectory() as d:
            code, _ = self.execute(d, events=events)
            self.assertEqual(code, 0)
        on_paper = False
        contact_count = 0
        for kind, value in events:
            if kind == 'persist':
                self.assertFalse(on_paper, 'Full log write while brush on paper')
                if value and value.endswith(':contact-sequence'):
                    contact_count += 1
            else:
                on_paper = abs(value-trial.z_for_press(.22)) < 1e-6
        self.assertEqual(contact_count, 2)

    def test_ctrl_c_preserves_pending_move_and_releases_owned_lock(self):
        with tempfile.TemporaryDirectory() as d:
            code, robot = self.execute(d, interrupt=True)
            self.assertEqual(code, 130)
            record = json.loads(next(Path(d).glob('color_experiment_*.json')).read_text())
            self.assertTrue(record['interrupted'])
            self.assertTrue(record['pending_move'])
            self.assertFalse(record['success'])
            self.assertFalse((Path(d)/'blue_wash_motion.lock').exists())
            robot.close.assert_called_once_with(disable=False)

    def test_actual_commands_light_fast_and_parked(self):
        with tempfile.TemporaryDirectory() as d:
            code,robot=self.execute(d)
            self.assertEqual(code,0)
            commands=[c.args[0] for c in robot.send.call_args_list]
            self.assertEqual(commands[0],'SpeedFactor(%d)' % PROFILE['global_speed_factor'])
            self.assertIn('AccJ(%d)' % PROFILE['accel_joint_percent'], commands)   # 降加速度治关节抖
            moves=[c for c in commands if c.startswith('MovL')]
            self.assertIn('v=%d' % PROFILE['travel_velocity_percent'], moves[0])
            self.assertFalse(any('EnableRobot' in c or 'ClearError' in c for c in commands))
            record=json.loads(next(Path(d).glob('color_experiment_*.json')).read_text())
            self.assertTrue(record['success'])
            self.assertTrue(all(c['completed'] for c in record['contacts']))
            self.assertEqual(record['final_pose'],list(trial.target_pose()))

    def test_error_records_contact_and_blocks_retry(self):
        with tempfile.TemporaryDirectory() as d:
            code,_=self.execute(d,reject=True)
            self.assertEqual(code,4)
            record=json.loads(next(Path(d).glob('color_experiment_*.json')).read_text())
            self.assertTrue(record['pending_move'])
            self.assertFalse(record['contacts'][0]['completed'])
            with self.assertRaises(trial.RobotError): trial.run(100,100,confirm_site=True,log_dir=d, profile=ColorProfile(press_mm=.22))

    def test_wrong_start_sends_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            code,robot=self.execute(d,bad_pose=True)
            self.assertEqual(code,4); robot.send.assert_not_called()

    def test_fast_color_has_low_lifts_and_only_lifecycle_disk_writes(self):
        from pigment_station import PigmentStation
        from types import SimpleNamespace
        profile=ColorProfile(press_mm=.22, draw_velocity_percent=40,
            travel_velocity_percent=80, horizontal_velocity_percent=80,
            lift_velocity_percent=60, touchdown_velocity_percent=30,
            interstroke_lift_mm=2.5)
        lines=[]
        for i in range(3):
            line=trial.plan_lines(100,100+i*8,count=1,length=5,press=.22)[0]
            line['id']='fast-%d'%i
            line['draw_speeds']=[40]
            line['sequence']=[(name,(*pose[:2],trial.z_for_lift(2.5),*pose[3:])
                if name in ('hover','lift') else pose) for name,pose in line['sequence']]
            lines.append(line)
        station=PigmentStation(x=-89.7096,y=53.2723,surface_z=-18.581,rim_z=-31.118,strokes_per_load=4)
        progress=[]
        control=SimpleNamespace(session=SimpleNamespace(save_progress=False),
            progress=progress.append,boundary=lambda *args:False)
        events=[]
        with tempfile.TemporaryDirectory() as d, patch.object(trial,'module_lines',return_value=(lines,[])), \
                patch.object(trial.time,'sleep'):
            code,robot=self.execute(d,station=station,module={},profile=profile,events=events,control=control)
            self.assertEqual(code,0)
            self.assertEqual(sum(kind=='persist' for kind,_ in events),3)
            record=json.loads(next(Path(d).glob('color_experiment_*.json')).read_text())
            self.assertTrue(record['success']); self.assertFalse(record['resumable'])
            self.assertTrue(record['motion_attempted'])
            for key in ('lines','contacts','steps','dips','pending_move'):
                self.assertNotIn(key,record)
            self.assertFalse(trial.unresolved_records(d))
            self.assertEqual(progress,[0,1,2,3])
            moves=[c.args[0] for c in robot.send.call_args_list if c.args[0].startswith('MovL')]
            def commanded(pose,v):
                return any('pose={%.4f,%.4f,%.4f,%.4f,%.4f,%.4f}'%pose in c
                    and ',v=%d,'%v in c for c in moves)
            high,dip=station.poses()
            self.assertTrue(commanded(high,80))
            self.assertTrue(commanded(high,20))  # Vertical withdrawal stays slow.
            self.assertTrue(commanded(dip,20))
            for name,pose in lines[1]['sequence']:
                if name in ('hover','touch','draw','lift'):
                    self.assertTrue(commanded(pose,{'hover':80,'touch':30,'draw':40,'lift':60}[name]))

    def test_memory_only_color_failure_still_blocks_unconfirmed_queue(self):
        with tempfile.TemporaryDirectory() as d:
            code,_=self.execute(d,interrupt=True,save_progress=False)
            self.assertEqual(code,130)
            record=json.loads(next(Path(d).glob('color_experiment_*.json')).read_text())
            self.assertFalse(record['resumable'])
            self.assertTrue(record['motion_attempted'])
            self.assertNotIn('lines',record)
            self.assertEqual(len(trial.unresolved_records(d)),1)
            with self.assertRaises(trial.RobotError):
                trial.run(100,100,confirm_site=True,log_dir=d,save_progress=False)

    def test_memory_only_failed_start_does_not_block_retry(self):
        with tempfile.TemporaryDirectory() as d:
            code,robot=self.execute(d,bad_pose=True,save_progress=False)
            self.assertEqual(code,4)
            robot.send.assert_not_called()
            record=json.loads(next(Path(d).glob('color_experiment_*.json')).read_text(encoding='utf-8'))
            self.assertFalse(record['motion_attempted'])
            self.assertNotIn('pending_move',record)
            self.assertFalse(trial.unresolved_records(d))
            code,_=self.execute(d,save_progress=False)
            self.assertEqual(code,0)

    def test_memory_only_color_pause_quit_has_no_saved_prefix(self):
        from types import SimpleNamespace
        class Control:
            session=SimpleNamespace(save_progress=False)
            count=0
            def progress(self,count):self.count=count
            def boundary(self,park,verify):
                if self.count:
                    park()
                    raise trial.PauseRequested()
                return False
        with tempfile.TemporaryDirectory() as d:
            code,_=self.execute(d,control=Control())
            self.assertEqual(code,trial.PAUSED)
            record=json.loads(next(Path(d).glob('color_experiment_*.json')).read_text())
            self.assertTrue(record['paused'])
            self.assertNotIn('contacts',record)
            self.assertFalse(trial.unresolved_records(d))

    def test_outline_decision_happens_after_color_and_can_skip(self):
        plan={'modules':{'color':{'type':'color'},'outline':{'type':'outline'}}}
        answers=iter(['','READY','RUN','n'])
        events=[]
        def ask(prompt):
            if '是否继续勾线' in prompt: self.assertEqual(events,['paint'])
            return next(answers)
        with patch.object(entry,'build_plan',return_value=(plan,None)),patch.object(entry,'save_outputs'), \
             patch.object(entry,'module_lines',side_effect=lambda module,color,p: ([{}],[]) if color=='blue' else ([],[])), \
             patch.object(entry,'run',side_effect=lambda **kw: events.append('paint') or 0), \
             patch('builtins.input',side_effect=ask),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(),0)
        self.assertEqual(events,['paint'])


if __name__=='__main__': unittest.main()
