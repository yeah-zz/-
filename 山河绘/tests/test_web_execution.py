"""An artwork must not inherit the previous artwork's web speed profile."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config import ArmConfig
import job
from web import execution


class SpeedProfileTests(unittest.TestCase):
    def test_ink_cup_return_and_touchdown_use_slower_speeds(self):
        from copy import deepcopy
        from geometry import CanvasMap
        from run_mountain_balanced import Motion
        from unittest.mock import MagicMock
        from config import station_approach_z
        stroke = dict(points=[[500,400],[520,400]], press=[.1,.1], speed=[40,40],
                      layer=0, pal_index=0, name='title', role='title', tail_mm=0,
                      lift_mm=2.5, horizontal_velocity_percent=80,
                      touchdown_velocity_percent=30, lift_velocity_percent=60)
        baseline = deepcopy(stroke)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(execution,'_base_motion',None), \
                patch.object(ArmConfig,'SPEED_FACTOR',20), \
                patch.object(job,'TRANSIT_SPEED',60), \
                patch.object(job,'stroke_waypoints',job.stroke_waypoints), \
                patch.object(job,'dip_waypoints',job.dip_waypoints):
            path = Path(folder)/'art.json'
            path.write_text(json.dumps({'workflow':{'speed_revision':dict(process_local=True,
                global_speed_factor=70,travel=80,dip=10,withdraw=20)}}))
            execution.prepare(path)
            motion=Motion(MagicMock(),CanvasMap((1505,1049)),(0,100,-100),40)
            sent=[]
            def capture(points,**kwargs):
                sent.extend(deepcopy(points))
                motion.current=points[-1]
            with patch.object(motion,'send',side_effect=capture):
                motion.dip({'pal_index':0})
                motion.stroke(stroke)
            speeds={p['tag']:p['velocity'] for p in sent}
            self.assertEqual([speeds[k] for k in ('dip-in','dip','dip-out','station-return-high')],[40,5,10,40])
            self.assertEqual([speeds[k] for k in ('hover:title','down:title','draw:title','lift:title')],[30,15,96,60])
            self.assertEqual(stroke,baseline)
            stroke['touchdown_velocity_percent']=7
            self.assertEqual(next(p['velocity'] for p in job.stroke_waypoints(motion.mapper,stroke)
                                  if p['tag']=='down:title'),7)

    def test_profile_then_unprofiled_restores_defaults_and_reapplies_contact_caps(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(execution, '_base_motion', None), \
                patch.object(ArmConfig, 'SPEED_FACTOR', 20), \
                patch.object(job, 'TRANSIT_SPEED', 60), \
                patch.object(job, 'stroke_waypoints', wraps=job.stroke_waypoints), \
                patch.object(job, 'dip_waypoints', wraps=job.dip_waypoints) as original:
            path = Path(folder) / 'artwork.json'
            profile = dict(global_speed_factor=70, travel=80, dip=10,
                           withdraw=20, process_local=True)
            path.write_text(json.dumps({'workflow': {'speed_revision': profile}}))
            execution.prepare(path)
            self.assertEqual(ArmConfig.SPEED_FACTOR, 70)
            self.assertEqual(job.TRANSIT_SPEED, 30)
            wrapper = job.dip_waypoints
            execution.prepare(path)
            points = job.dip_waypoints(0)
            self.assertEqual(original.call_count, 1)
            self.assertEqual(next(p['velocity'] for p in points if p['tag'] == 'dip'), 5)
            self.assertEqual(next(p['velocity'] for p in points if p['tag'] == 'dip-out'), 10)
            self.assertIsNot(job.dip_waypoints, wrapper)
            path.write_text(json.dumps({'workflow': {}}))
            execution.prepare(path)
            self.assertEqual(ArmConfig.SPEED_FACTOR, 20)
            self.assertEqual(job.TRANSIT_SPEED, 30)
            self.assertIsNot(job.dip_waypoints, original)
            points = job.dip_waypoints(0)
            self.assertLessEqual(next(p['velocity'] for p in points if p['tag']=='dip'), 5)
            self.assertLessEqual(next(p['velocity'] for p in points if p['tag']=='dip-out'), 10)

    def test_paper_draw_boost_reaches_commands_once_and_keeps_contact_moves(self):
        from geometry import CanvasMap
        from run_mountain_balanced import Motion
        from unittest.mock import MagicMock
        from config import z_for_lift
        original = job.stroke_waypoints
        stroke = dict(id='speed-check', points=[[100,100],[110,110],[120,120]],
                      press=[.12]*3, speed=[40]*3, tail_mm=0, pal_index=0)
        mapper = CanvasMap((500,500))
        before = original(mapper, stroke)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(execution, '_base_motion', None), \
                patch.object(job, 'TRANSIT_SPEED', job.TRANSIT_SPEED), \
                patch.object(job, 'dip_waypoints', job.dip_waypoints), \
                patch.object(job, 'stroke_waypoints', original):
            path = Path(folder)/'artwork.json'
            path.write_text('{"workflow":{}}')
            execution.prepare(path)
            execution.prepare(path)
            after = job.stroke_waypoints(mapper, stroke)
            for old, new in zip(before, after):
                limit = 15 if old['tag'].startswith('down:') else 30
                expected = old | {'velocity': 96.0 if old['tag'].startswith('draw:') else min(old['velocity'],limit)}
                self.assertEqual(new, expected)
            robot = MagicMock()
            robot._ok.return_value = True
            with patch('prepare_robot.wait_arrival'):
                Motion(robot, mapper, (100,100,z_for_lift(5),*ArmConfig.RPY)).send(after)
            moves = [c.args[0] for c in robot.send.call_args_list if c.args[0].startswith('MovL(')]
            self.assertEqual(sum(',v=96,' in c for c in moves), 2)
            self.assertEqual(stroke['speed'], [40]*3)


if __name__ == '__main__':
    unittest.main()
