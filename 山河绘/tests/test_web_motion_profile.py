"""Check actual commands, approach boundaries and failures without hardware."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from config import ArmConfig, station_approach_z, z_for_lift, z_for_press
from color_profile import ColorProfile
from web.motion_profile import DEFAULTS, apply_to_artwork, normalize
import run_color_experiment as color
from run_mountain_balanced import Motion


class MotionProfileTests(unittest.TestCase):
    def profile(self):
        ink = {'meta': {}}
        plan = {'modules': {'color': {'motion_candidate': ColorProfile(
            press_mm=.22, draw_velocity_percent=40, travel_velocity_percent=80,
            horizontal_velocity_percent=80, touchdown_velocity_percent=30,
            lift_velocity_percent=60).to_dict()}}}
        return apply_to_artwork((ink, plan, [], {}), DEFAULTS)[1]['modules']['color']

    def test_limits_reject_invalid_snapshots(self):
        for key, value in (('approach', 80), ('dip', True), ('linear_acceleration', -1),
                           ('linear_acceleration', True), ('joint_acceleration', 100)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize(DEFAULTS | {key: value})

    def test_old_saved_acceleration_caps_migrate_to_inherit(self):
        saved = DEFAULTS | {'joint_acceleration':20, 'linear_acceleration':15}
        self.assertEqual(normalize(saved), DEFAULTS)
        self.assertEqual(saved['joint_acceleration'],20)

    def test_ink_commands_keep_original_speeds(self):
        robot = MagicMock()
        robot._ok.return_value = True
        pose = (100,100,station_approach_z(),*ArmConfig.RPY)
        points = [dict(x=100,y=100,z=z,velocity=80,pen_up=up,tag=tag,pause=0)
                  for z,up,tag in ((z_for_lift(2.5),True,'hover:s'),
                                   (z_for_press(.22),False,'down:s'),
                                   (z_for_press(.22),False,'draw:s'),
                                   (z_for_lift(2.5),True,'lift:s'))]
        with patch('prepare_robot.wait_arrival'):
            motion = Motion(robot,None,pose)
            motion.send(points)
        commands = [c.args[0] for c in robot.send.call_args_list]
        self.assertEqual([int(c.split(',v=')[1].split(',')[0]) for c in commands], [80,80,80,80])
        self.assertEqual([p['z'] for p in motion.waypoints], [p['z'] for p in points])
        self.assertTrue(all(p['velocity']==80 for p in points))

    def test_web_overlay_does_not_inject_ink_settings_or_raise_color_speeds(self):
        import copy
        ink = {'meta': {'web_motion_profile': DEFAULTS, 'runtime_acceleration': {'joint':15,'linear':15}}}
        plan = {'modules': {'color': {'motion_candidate': ColorProfile(
            press_mm=.22, travel_velocity_percent=22, global_speed_factor=50,
            dip_velocity_percent=10, accel_joint_percent=10, accel_linear_percent=10).to_dict(),
            'strokes': [{'points': [[1,2],[3,4]], 'press': [.1,.2]}]}}}
        old = copy.deepcopy((ink,plan,[],dict(ink['meta'])))
        new = apply_to_artwork(old, DEFAULTS)
        self.assertEqual(old, (ink,plan,[],dict(ink['meta'])))
        for meta in (new[0]['meta'], new[3]):
            self.assertNotIn('web_motion_profile', meta)
            self.assertNotIn('runtime_acceleration', meta)
        candidate = new[1]['modules']['color']['motion_candidate']
        for key in ('travel_velocity_percent','horizontal_velocity_percent','approach_velocity_percent',
                    'lift_velocity_percent'):
            self.assertEqual(candidate[key],22)
        self.assertEqual(candidate['touchdown_velocity_percent'],15)
        self.assertEqual(candidate['global_speed_factor'],50)
        self.assertEqual(candidate['dip_velocity_percent'],10)
        self.assertEqual(candidate['accel_linear_percent'],0)
        self.assertEqual(candidate['accel_joint_percent'],0)
        self.assertFalse(candidate['acceleration_required'])
        self.assertEqual(new[1]['modules']['color']['strokes'], plan['modules']['color']['strokes'])

    def run_color(self, folder, fail_approach=False, reject_acceleration=False, station=None,
                  explicit_acceleration=False):
        module = self.profile()
        if explicit_acceleration:
            module['motion_candidate'].update(accel_joint_percent=20, accel_linear_percent=20,
                                              acceleration_required=True)
        profile = ColorProfile(**module['motion_candidate'])
        line = color.plan_lines(100,100,count=1,length=5,press=.22)[0]
        line['id'] = 'verified-test-stroke'
        line['sequence'] = [(tag,(*pose[:2],z_for_lift(2.5),*pose[3:]) if tag in ('hover','lift') else pose)
                            for tag,pose in line['sequence']]
        line['draw_speeds'] = [120]
        # module_lines normally performs this clamp from the profile ceiling.
        line['draw_speeds'] = [min(v,profile.draw_velocity_max) for v in line['draw_speeds']]
        robot = MagicMock()
        robot.get_mode.return_value = 5
        robot.get_error_id.return_value = '0,{[]},GetErrorID()'
        robot._ok.side_effect = lambda response: response != 'rejected'
        robot.send.side_effect = lambda command: 'rejected' if reject_acceleration and command.startswith('AccL(') else '0,{1},command()'
        robot.wait_idle.return_value = True
        current = [color.target_pose()]
        arrivals = []
        def arrival(r,target):
            arrivals.append(target)
            if fail_approach and abs(target[2]-z_for_lift(2.5)) < .001:
                raise color.RobotError('Recovery interrupted; mode=11; alarms=0,{[]},GetErrorID()')
            current[0] = target
        with patch.object(color,'E6Robot',return_value=robot), patch.object(color,'connect_readonly'), \
                patch.object(color,'module_lines',return_value=([line],[])), \
                patch.object(color,'read_idle_pose',side_effect=lambda *args:current[0]), \
                patch.object(color,'wait_arrival',side_effect=arrival), \
                patch.object(color.time,'sleep'), contextlib.redirect_stdout(io.StringIO()):
            code = color.run(profile=profile,confirm_site=True,module=module,color='green',
                             log_dir=folder,save_progress=False,station=station)
        return code, robot, arrivals

    def test_web_color_confirms_approach_and_sends_no_acceleration_override(self):
        with tempfile.TemporaryDirectory() as folder:
            code,robot,arrivals = self.run_color(folder)
            self.assertEqual(code,0)
            commands = [c.args[0] for c in robot.send.call_args_list]
            moves = [c for c in commands if c.startswith('MovL(')]
            self.assertFalse(any(c.startswith(('AccJ(', 'AccL(')) for c in commands))
            self.assertIn('SpeedFactor(70)',commands)
            self.assertIn(',v=40,', moves[0])
            self.assertIn(',v=30,', moves[1])
            self.assertIn(',v=15,', moves[2])
            self.assertIn(',v=100,', moves[3])
            self.assertIn(',v=60,', moves[4])
            self.assertAlmostEqual(arrivals[0][2], z_for_lift(2.5))
            marker = json.loads(next(Path(folder).glob('color_experiment_*.json')).read_text())
            self.assertTrue(Path(marker['diagnostic_path']).exists())
            self.assertEqual(marker['profile']['global_speed_factor'],70)
            self.assertEqual(marker['profile']['accel_linear_percent'],0)
            self.assertFalse(marker['profile']['acceleration_required'])
            rows = [json.loads(s) for s in Path(marker['diagnostic_path']).read_text().splitlines()]
            settings = next(row for row in rows if row['event']=='motion_settings')
            self.assertEqual(settings['acceleration'],'inherited controller setting; not read back')
            self.assertFalse(marker['resumable'])
            self.assertNotIn('lines',marker)

    def test_failed_approach_submits_no_touch_or_draw_and_preserves_fault_context(self):
        with tempfile.TemporaryDirectory() as folder:
            code,robot,_ = self.run_color(folder,fail_approach=True)
            self.assertEqual(code,4)
            moves = [c.args[0] for c in robot.send.call_args_list if c.args[0].startswith('MovL(')]
            self.assertEqual(len(moves),2)
            rows = [json.loads(s) for s in next(Path(folder).glob('*_diagnostic.jsonl')).read_text().splitlines()]
            fault = next(row for row in rows if row['event']=='fault')
            self.assertEqual(fault['action'],'paper-approach')
            self.assertEqual(fault['stroke_id'],'verified-test-stroke')
            self.assertIn('mode=11',fault['error'])

    def test_rejected_acceleration_submits_no_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            code,robot,_ = self.run_color(folder,reject_acceleration=True,explicit_acceleration=True)
            self.assertEqual(code,4)
            self.assertFalse(any(c.args[0].startswith('MovL(') for c in robot.send.call_args_list))

    def test_green_round_trip_keeps_xy_speed_but_separates_vertical_and_cup_speeds(self):
        from pigment_station import PigmentStation
        station=PigmentStation(x=-89.7096,y=53.2723,surface_z=-18.581,rim_z=-31.118)
        with tempfile.TemporaryDirectory() as folder:
            code,robot,_=self.run_color(folder,station=station)
            self.assertEqual(code,0)
            moves=[c.args[0] for c in robot.send.call_args_list if c.args[0].startswith('MovL(')]
            speeds=[int(c.split(',v=')[1].split(',')[0]) for c in moves]
            # Cup raise/XY/dip/withdraw, paper XY/descent/touch/draw/lift, return raise/XY.
            self.assertEqual(speeds,[30,40,10,10,40,30,15,100,60,30,40])
            self.assertTrue(all('cp=0)' in c for c in moves[:4]))


if __name__ == '__main__':
    unittest.main()
