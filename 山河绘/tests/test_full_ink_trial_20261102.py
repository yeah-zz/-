# -*- coding: utf-8 -*-
"""Offline invariants for the full-frame trial planner; no hardware fixtures."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

SPEC=importlib.util.spec_from_file_location('full_trial',Path(__file__).resolve().parents[1]/'scripts'/'prepare_full_ink_trial_20261102.py')
m=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class FullTrialTests(unittest.TestCase):
    def rows(self,*paths):
        return [dict(points=np.array(p,dtype=float),source_ids=[i]) for i,p in enumerate(paths)]

    def test_exact_duplicate_edge_removed_but_unique_coverage_retained(self):
        paths=[np.array([[0,0],[1,0],[2,0]],float),np.array([[2,0],[1,0],[1,1]],float)]
        rows,removed=m.deduplicate(paths)
        self.assertEqual(removed,1)
        self.assertEqual(set(m.edges(paths)),set(m.edges([r['points'] for r in rows])))

    def test_straight_t_junction_only_joins_opposite_pair(self):
        rows=self.rows([[0,0],[10,0]],[[10,0],[20,0]],[[10,0],[10,10]])
        out,audit=m.join_existing(rows)
        self.assertEqual(len(out),2)
        self.assertEqual(len(audit),1)
        self.assertEqual(m.edges([r['points'] for r in rows]),m.edges([r['points'] for r in out]))

    def test_ambiguous_y_junction_retains_separate_strokes(self):
        rows=self.rows([[0,0],[10,0]],[[0,0],[-10,4]],[[0,0],[-10,-4]])
        out,audit=m.join_existing(rows)
        self.assertEqual(len(out),3)
        self.assertEqual(audit,[])

    def test_gap_and_parallel_lines_not_bridged(self):
        rows=self.rows([[0,0],[10,0]],[[11,0],[20,0]],[[0,1],[10,1]])
        out,audit=m.join_existing(rows)
        self.assertEqual(len(out),3)
        self.assertFalse(audit)

    def test_cycles_and_tiny_pine_needles_retained(self):
        rows=self.rows([[0,0],[1,0]],[[4,4],[5,4],[5,5],[4,4]])
        out,_=m.join_existing(rows)
        self.assertEqual(m.edges([r['points'] for r in rows]),m.edges([r['points'] for r in out]))
        self.assertEqual(len(out),2)

    def test_compression_preserves_corners_reversals_and_endpoints(self):
        p=np.array([[0,0],[1,0],[2,0],[2,1],[2,2],[2,1]],float)
        np.testing.assert_array_equal(m.collinear(p),[[0,0],[2,0],[2,2],[2,1]])

    def test_contain_square_in_landscape_paper_does_not_stretch(self):
        scale,offset=m.contain_transform((1254,1254),(297,210),10)
        self.assertAlmostEqual(scale*1254,190)
        np.testing.assert_allclose(offset,[53.5,10])
        with self.assertRaises(ValueError):
            m.contain_transform((1254,1254),(20,20),10)

    def test_order_keeps_geometry_and_never_worsens_travel(self):
        rows=self.rows([[90,0],[100,0]],[[0,0],[10,0]],[[40,0],[50,0]])
        out=m.order_paths(rows)
        self.assertLessEqual(m.air_length(out),m.air_length(rows))
        self.assertEqual(m.edges([r['points'] for r in rows]),m.edges([r['points'] for r in out]))

    def site(self):
        d=m.calibration_template()
        d.update(measured_at='2026-11-02 manual test fixture',user_index=2,tool_index=0,
                 rpy_deg=[0,0,0],z_increases_down=True,
                 paper_corners_touch_xyz=dict(TL=[0,0,-40],TR=[200,0,-40],BR=[200,150,-40],BL=[0,150,-40]),
                 paper_center_touch_xyz=[100,75,-40],
                 black_station=dict(x=-50,y=0,surface_z=-30,rim_z=-50,dip_mm=1,dwell_s=1,drip_s=2),
                 brush_test=dict(samples=[dict(press_mm=.1,width_mm=.5,continuous=True)],selected_press_mm=.1,
                                 draw_velocity_percent=20,global_speed_factor=20),
                 paper_lift_mm=5,station_clearance_mm=20,reach_limits_xyz=[[-400,400],[-400,400],[-140,-10]],site_clearance_checked=True)
        return d

    def test_blank_calibration_rejected_valid_fixture_accepted_offline(self):
        with patch('socket.socket',side_effect=AssertionError('No network allowed')):
            self.assertFalse(m.validate_calibration(m.calibration_template())['valid'])
            self.assertTrue(m.validate_calibration(self.site())['valid'])

    def test_bad_paper_plane_or_unknown_pressure_not_exportable(self):
        bad=self.site();bad['paper_center_touch_xyz'][2]= -39
        self.assertFalse(m.validate_calibration(bad)['valid'])
        bad=self.site();bad['brush_test']['selected_press_mm']=.2
        self.assertFalse(m.validate_calibration(bad)['valid'])
        bad=self.site();bad['paper_corners_touch_xyz']['TR']=[200,150,-40]
        self.assertFalse(m.validate_calibration(bad)['valid'])

    def test_station_reach_and_z_direction_checked(self):
        bad=self.site();bad['black_station']['rim_z']= -20
        self.assertFalse(m.validate_calibration(bad)['valid'])
        bad=self.site();bad['black_station']['x']=500
        self.assertFalse(m.validate_calibration(bad)['valid'])
        inverse=self.site();inverse['z_increases_down']=False
        inverse['black_station'].update(surface_z=-50,rim_z=-30)
        self.assertTrue(m.validate_calibration(inverse)['valid'])

    def test_existing_executor_rejects_unbound_plan_without_connection(self):
        import run_mountain_balanced
        with patch('socket.socket',side_effect=AssertionError('No network allowed')):
            with self.assertRaisesRegex(ValueError,'Requires reviewed balanced'):
                run_mountain_balanced.load_reviewed(dict(schema='full_ink_trial_unbound_v1',paths=[],ready_for_robot=False))


if __name__=='__main__':
    unittest.main()
