"""Planner/runtime agreement for explicit per-stroke ink loading, offline."""
from dataclasses import asdict
from unittest import TestCase
from unittest.mock import patch

from config import JobConfig
from ink_control import InkPolicy, InkState, build_plan, refresh_stats
from run_mountain_balanced import run_feedback


def stroke(i):
    return refresh_stats(dict(id=str(i),points=[[i*12.,0.],[i*12.+8.,0.]],
        press=[.6,.6],speed=[20.,20.],layer=1,pal_index=0,tail_mm=0),1.)


class PerStrokeInkTests(TestCase):
    def test_planner_dips_before_every_stroke(self):
        steps,stats=build_plan([stroke(i) for i in range(3)],0,1.,
                              InkPolicy(max_strokes_per_load=1))
        self.assertEqual([s['kind'] for s in steps],['dip','stroke']*3)
        self.assertEqual(stats['dips'],3)

    def test_runtime_enforces_count_with_real_callbacks_not_only_preview(self):
        events=[]
        with patch('socket.socket',side_effect=AssertionError('No networking')):
            rows,count=run_feedback([stroke(i) for i in range(3)],1.,
                InkPolicy(max_strokes_per_load=1,fresh_press_factor=1.),
                lambda _:events.append('dip'),lambda _:events.append('draw'),
                lambda _:events.append('rest'),clock=lambda:0.)
        self.assertEqual(events,['dip','draw']*3)
        self.assertEqual(count,3)
        self.assertIn('stroke_count_budget',rows[1]['refill_reason'])

    def test_zero_default_preserves_automatic_loading(self):
        steps,stats=build_plan([stroke(i) for i in range(3)],0,1.,InkPolicy())
        self.assertEqual(stats['dips'],1)
        self.assertEqual([s['kind'] for s in steps],['dip','stroke','stroke','stroke'])

    def test_two_strokes_per_load_and_reset(self):
        steps,stats=build_plan([stroke(i) for i in range(3)],0,1.,
                              InkPolicy(max_strokes_per_load=2))
        self.assertEqual(stats['dips'],2)
        self.assertEqual([s['kind'] for s in steps],['dip','stroke','stroke','dip','stroke'])
        state=InkState(InkPolicy(max_strokes_per_load=1));state.load(0)
        state.consume(stroke(0),1.,1.)
        self.assertIn('stroke_count_budget',state.reasons(1.,1.,1.))
        state.load(2.)
        self.assertEqual(state.completed_since_load,0)
        self.assertNotIn('stroke_count_budget',state.reasons(2.,1.,1.))

    def test_three_strokes_per_load_matches_planner_and_runtime(self):
        policy=InkPolicy(max_strokes_per_load=3,initial_dip_s=2.,refresh_dip_s=2.)
        strokes=[stroke(i) for i in range(4)]
        steps,stats=build_plan(strokes,0,1.,policy)
        self.assertEqual([s['kind'] for s in steps],['dip','stroke','stroke','stroke','dip','stroke'])
        self.assertEqual(stats['dips'],2)
        events=[]
        _,count=run_feedback(strokes,1.,policy,lambda _:events.append('dip'),
            lambda _:events.append('draw'),lambda _:None,clock=lambda:0.)
        self.assertEqual(events,['dip','draw','draw','draw','dip','draw'])
        self.assertEqual(count,2)

    def test_does_not_disable_pressure_length_limit(self):
        state=InkState(InkPolicy(max_strokes_per_load=1));state.load(0.)
        self.assertIn('pressure_length_budget',state.reasons(0.,1000.,1.))

    def test_invalid_count_and_debug_skip_rejected(self):
        for v in (True,-1,1.5,None,'1'):
            with self.assertRaises(ValueError):InkPolicy(max_strokes_per_load=v)
        with patch.object(JobConfig,'SKIP_PIGMENT_ACTIONS',True):
            with self.assertRaisesRegex(ValueError,'requires pigment actions'):
                build_plan([stroke(0)],0,1.,InkPolicy(max_strokes_per_load=1))

    def test_serialization_and_old_manifest_compatibility(self):
        data=asdict(InkPolicy(max_strokes_per_load=1))
        self.assertEqual(InkPolicy(**data).max_strokes_per_load,1)
        del data['max_strokes_per_load']
        self.assertEqual(InkPolicy(**data).max_strokes_per_load,0)
