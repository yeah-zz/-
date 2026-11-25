"""Fault injection tests; no socket and no real robot commands."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import run_blue_wash as r
from robot import RobotError

class FakeRobot:
    def __init__(self,pose): self.pose=pose; self.commands=[]; self.mode=5
    def get_mode(self): return self.mode
    def get_error_id(self): return '0,{[]},GetErrorID()'
    def get_pose(self): return self.pose
    def send(self,c): self.commands.append(c); return '0,{},ok()'
    def _ok(self,s): return s.startswith('0,')
    def close(self,disable=False): assert not disable

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.plan,cls.targets=r.load_plan(r.DEFAULT_PLAN)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.robot=FakeRobot(r.loading_pose()); self.moves=[]
    def motion(self,robot,target): self.moves.append(target); robot.pose=target
    def run_action(self,action='check',**kw):
        with patch.object(r,'E6Robot',return_value=self.robot),patch.object(r,'connect_readonly'),\
             patch.object(r,'move_to',side_effect=kw.pop('motion',self.motion)),\
             contextlib.redirect_stdout(io.StringIO()):
            return r.run(action=action,confirm_site=action!='check',log_dir=self.folder,**kw)
    def hover(self):
        self.assertEqual(self.run_action('hover'),0)
        return next(p for p in self.folder.glob('*.json') if json.loads(p.read_text(encoding='utf8'))['action']=='hover')
    def test_readonly_sends_no_settings_or_moves(self):
        self.assertEqual(self.run_action(),0); self.assertEqual(self.robot.commands,[]); self.assertEqual(self.moves,[])
    def test_hover_never_contacts_and_holds(self):
        self.hover(); self.assertGreaterEqual(len(self.moves),2)
        self.assertTrue(all(p[2]<=r.z_for_lift(15) for p in self.moves))
        self.assertEqual(self.robot.commands,['SpeedFactor(50)'])
    def test_paint_full_sequence_and_high_horizontal_transfers(self):
        reg=self.hover(); self.moves=[]
        self.assertEqual(self.run_action('paint',registration_record=reg),0)
        previous=self.targets[0]['sequence'][1][1]
        for target in self.moves:
            xy=np.linalg.norm(np.array(target[:2])-previous[:2])
            if xy>.01:
                self.assertAlmostEqual(target[2],previous[2])
                self.assertTrue(target[2]==r.station_approach_z() or target[2]==r.z_for_press(.3))
            previous=target
        self.assertEqual(self.robot.pose,r.loading_pose())
        contacts=r.history(self.folder,self.plan['ink_job_sha256']); self.assertEqual(len(contacts),2)
        with self.assertRaises(RobotError): r.reject_overlap(self.targets[:2],contacts)
    def test_touch_failure_persists_intent_and_blocks_all_later_contact(self):
        reg=self.hover(); self.moves=[]
        def fail(robot,target):
            # Must already have durable contact intent before an ambiguous command.
            if target[2]==r.z_for_press(.3):
                logs=[json.loads(p.read_text(encoding='utf8')) for p in self.folder.glob('*.json')]
                self.assertTrue(any(x.get('contacts') for x in logs))
                raise RobotError('simulated disconnect during contact')
            self.motion(robot,target)
        self.assertEqual(self.run_action('paint',registration_record=reg,motion=fail),4)
        self.assertEqual(self.moves,[])
        with self.assertRaisesRegex(RobotError,'Unresolved contact'): r.history(self.folder,self.plan['ink_job_sha256'])
    def test_alarm_after_touch_sends_no_draw_or_recovery(self):
        reg=self.hover(); self.moves=[]
        def alarm(robot,target): self.motion(robot,target); robot.mode=11
        self.assertEqual(self.run_action('paint',registration_record=reg,motion=alarm),4)
        self.assertEqual(len(self.moves),1)
    def test_no_registration_no_motion(self):
        self.assertEqual(self.run_action('paint'),4); self.assertEqual(self.moves,[])
    def test_bad_start_does_not_recover_or_set_speed(self):
        self.robot.pose=(*r.loading_pose()[:2],-55.08,*r.ArmConfig.RPY)
        self.assertEqual(self.run_action('hover'),4); self.assertEqual(self.robot.commands,[])
    def test_lock_prevents_second_process(self):
        (self.folder/'blue_wash_motion.lock').write_text('existing')
        self.assertEqual(self.run_action('hover'),4); self.assertEqual(self.moves,[])
        self.assertEqual((self.folder/'blue_wash_motion.lock').read_text(),'existing')
    def test_forged_metadata_cannot_put_path_on_ink(self):
        plan=json.loads(Path(r.DEFAULT_PLAN).read_text(encoding='utf8'))
        plan['strokes'][0]['points']=[[260.,266.],[280.,280.]]
        plan['strokes'][0]['min_ink_clearance_mm']=999
        p=self.folder/'forged.json'; p.write_text(json.dumps(plan),encoding='utf8')
        with self.assertRaises(ValueError): r.load_plan(p)
    def test_park_from_hover_only_raises_then_transfers(self):
        self.hover(); self.moves=[]
        self.assertEqual(self.run_action('park'),0)
        self.assertEqual(self.moves[0],self.targets[0]['sequence'][0][1])
        self.assertTrue(all(abs(p[2]-r.station_approach_z())<.001 for p in self.moves))
        self.assertTrue(r.close_to(self.moves[-1],r.loading_pose()))
    def test_resume_hover_needs_exact_diagnosed_high_target(self):
        high=self.targets[0]['sequence'][0][1]
        record=dict(action='hover',success=False,plan_sha256=r.color_pass._digest(r.DEFAULT_PLAN),
                    start=0,contacts=[],steps=[],pending_move=dict(name='high',target=high))
        p=self.folder/'failed_hover.json'; p.write_text(json.dumps(record),encoding='utf8')
        self.robot.pose=high
        self.assertEqual(self.run_action('hover',resume_hover_record=p),0)
        self.assertEqual(self.moves,[self.targets[0]['sequence'][1][1]])
        self.moves=[]; self.robot.pose=(high[0]+1,*high[1:])
        self.assertEqual(self.run_action('hover',resume_hover_record=p),4)
        self.assertEqual(self.moves,[])
    def test_all_planned_strokes_validate_and_are_not_old_trial(self):
        self.assertGreaterEqual(len(self.targets),40)
        old=[dict(xy=[[-6.3851,157.9793],[-9.5452,156.1148]])]
        r.reject_overlap(self.targets,old)
    def test_explicit_remaining_runs_every_stroke_once_and_parks(self):
        self.assertEqual(self.run_action('hover',start=6,remaining=True),0)
        reg=next(p for p in self.folder.glob('*.json') if json.loads(p.read_text(encoding='utf8'))['action']=='hover')
        self.moves=[]
        self.assertEqual(self.run_action('paint',start=6,remaining=True,registration_record=reg),0)
        contacts=r.history(self.folder,self.plan['ink_job_sha256'])
        self.assertEqual([c['id'] for c in contacts],[t['id'] for t in self.targets[6:]])
        self.assertEqual(self.robot.pose,r.loading_pose())
        # Every horizontal segment is either high travel or constant-depth paint.
        previous=self.targets[6]['sequence'][1][1]
        for target in self.moves:
            if np.linalg.norm(np.asarray(target[:2])-previous[:2])>.01:
                self.assertAlmostEqual(target[2],previous[2])
                self.assertIn(target[2],(r.station_approach_z(),r.z_for_press(.3)))
            previous=target
    def test_speed_change_resume_only_from_reviewed_completed_high_pose(self):
        high=self.targets[6]['sequence'][-1][1]
        record=dict(action='paint',paused_for_speed_change=True,plan_sha256=r.color_pass._digest(r.DEFAULT_PLAN),
                    start=6,contacts=[dict(id=self.targets[6]['id'],completed=True)],final_pose=high)
        p=self.folder/'speed_resume.json'; p.write_text(json.dumps(record),encoding='utf8')
        self.robot.pose=high
        self.assertEqual(self.run_action('hover',start=7,resume_high_record=p),0)
        self.assertEqual(self.robot.pose,self.targets[7]['sequence'][1][1])
        self.moves=[]; self.robot.pose=high
        self.assertEqual(self.run_action('hover',start=8,resume_high_record=p),4)
        self.assertEqual(self.moves,[])
        record['contacts'][0]['completed']=False; p.write_text(json.dumps(record),encoding='utf8')
        self.assertEqual(self.run_action('hover',start=7,resume_high_record=p),4)
        self.assertEqual(self.moves,[])

if __name__=='__main__': unittest.main()
