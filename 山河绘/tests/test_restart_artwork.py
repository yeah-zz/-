"""Offline guards for scoped progress resets and faster air transfers."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from painting_session import Session, save, session_lock
from restart_json_painting import reset_selected_progress
from run_mountain_balanced import Motion
from geometry import CanvasMap
from config import station_approach_z, z_for_lift


class RestartTests(unittest.TestCase):
    def test_reset_only_selected_progress_keeps_history(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            manifest=dict(registration={},stages={'black':[{'id':'s0'}]})
            first=Session(folder,folder/'first.json',manifest)
            other=Session(folder,folder/'other.json',manifest)
            first.data.update(status='paused',completed={'black':1});first.persist()
            other.persist()
            untouched=other.path.read_bytes()
            self.assertEqual(reset_selected_progress([first.source],folder),1)
            retired=json.loads(first.path.read_text(encoding='utf-8'))
            self.assertTrue(retired['abandoned'])
            self.assertEqual(retired['completed']['black'],1)
            self.assertEqual(other.path.read_bytes(),untouched)
            reopened=Session(folder,first.source,manifest)
            self.assertEqual(reopened.count('black'),0)
            self.assertTrue(list(folder.glob(first.path.stem+'_retired_*.json')))

    def test_paused_but_open_session_cannot_reset(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            session=Session(folder,folder/'first.json',dict(registration={},stages={'black':[]}))
            session.data['status']='paused';session.persist()
            previous=session.path.read_bytes()
            with session_lock(folder):
                with self.assertRaises(FileExistsError):
                    reset_selected_progress([session.source],folder)
            self.assertEqual(session.path.read_bytes(),previous)

    def test_uncertain_motion_cannot_reset(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            session=Session(folder,folder/'first.json',dict(registration={},stages={'black':[]}))
            session.data['status']='uncertain';session.persist()
            previous=session.path.read_bytes()
            with self.assertRaises(ValueError):
                reset_selected_progress([session.source],folder)
            self.assertEqual(session.path.read_bytes(),previous)

    def test_new_sheet_confirmation_retires_uncertain_history(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            session=Session(folder,folder/'first.json',dict(registration={},stages={'black':[]}))
            session.data.update(status='uncertain',active='black',pending_move={'id':'unknown'})
            session.persist()
            self.assertEqual(reset_selected_progress([session.source],folder,
                             controller_stopped_confirmed=True),1)
            record=json.loads(session.path.read_text(encoding='utf-8'))
            self.assertTrue(record['abandoned']['controller_stopped_confirmed'])
            self.assertEqual(record['pending_move'],{'id':'unknown'})
            with session_lock(folder):
                with self.assertRaises(FileExistsError):
                    reset_selected_progress([session.source],folder,controller_stopped_confirmed=True)

    def test_air_speed_does_not_change_drawing_or_cup_clearance(self):
        motion=Motion(Mock(),CanvasMap((1505,1049)),(0,100,-100),80)
        batches=[]
        continuous_modes=[]
        def capture(points, *, continuous=False):
            batches.append(points)
            continuous_modes.append(continuous)
            motion.current=points[-1]
        motion.send=capture
        motion.dip({'pal_index':0})
        points=[p for batch in batches for p in batch]
        self.assertEqual(next(p for p in points if p['tag']=='dip-in')['velocity'],80)
        self.assertEqual(next(p for p in points if p['tag']=='dip-in')['z'],station_approach_z())
        self.assertEqual(next(p for p in points if p['tag']=='dip-out')['velocity'],60)
        self.assertFalse(any(continuous_modes))
        batches.clear()
        continuous_modes.clear()
        stroke=dict(points=[[500,400],[520,400]],press=[.36,.36],speed=[30.,30.],
                    layer=1,pal_index=0,name='test',tail_mm=0.,lift_mm=2.5,
                    horizontal_velocity_percent=80,lift_velocity_percent=60)
        motion.stroke(stroke)
        self.assertEqual(continuous_modes,[False,True])
        self.assertEqual(batches[0][0]['tag'],'station-return-high')
        self.assertEqual(batches[0][0]['velocity'],80)
        self.assertEqual(batches[0][0]['z'],station_approach_z())
        draw=next(p for p in batches[-1] if p['tag'].startswith('draw:'))
        self.assertEqual(draw['velocity'],30.)
        self.assertAlmostEqual(batches[-1][-1]['z'],z_for_lift(2.5))
        batches.clear()
        motion.stroke(stroke)
        self.assertEqual(len(batches),1)
        self.assertEqual(batches[0][0]['velocity'],80)
        self.assertAlmostEqual(batches[0][0]['z'],z_for_lift(2.5))
        self.assertFalse(any(p['z']==station_approach_z() for p in batches[0]))


if __name__=='__main__':
    unittest.main()
