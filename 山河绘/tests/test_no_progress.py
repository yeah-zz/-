"""Memory-only progress retains pause/continue but never checkpoints strokes."""
from pathlib import Path
import json,tempfile,unittest
from unittest.mock import patch
from painting_session import Session


class NoProgressTests(unittest.TestCase):
    def manifest(self):
        return dict(registration={},stages={'black':[{'id':'s0'},{'id':'s1'}]})

    def test_no_disk_write_at_stroke_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            s=Session(folder,Path(folder)/'painting.json',self.manifest(),save_progress=False)
            s.begin('black')
            before=s.path.read_bytes()
            with patch('painting_session.save',side_effect=AssertionError('No per-stroke write')):
                s.complete_stroke('black',1)
                s.complete_stroke('black',2)
            self.assertEqual(s.count('black'),2)
            self.assertEqual(s.path.read_bytes(),before)
            s.parked()
            record=json.loads(s.path.read_text())
            self.assertNotIn('completed',record)
            self.assertNotIn('manifest',record)
            self.assertFalse(record['resumable'])

    def test_reopen_starts_at_zero_and_abnormal_exit_is_still_guarded(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'painting.json'
            s=Session(folder,path,self.manifest(),save_progress=False)
            s.begin('black');s.complete_stroke('black',1)
            with self.assertRaises(ValueError):Session(folder,path,self.manifest(),save_progress=False)
            s.parked()
            next_run=Session(folder,path,self.manifest(),save_progress=False)
            self.assertEqual(next_run.count('black'),0)

    def test_existing_saved_pause_can_handoff_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'painting.json'
            old=Session(folder,path,self.manifest())
            old.begin('black');old.complete_stroke('black',1);old.parked()
            memory=Session(folder,path,self.manifest(),save_progress=False)
            self.assertEqual(memory.count('black'),1)
            memory.begin('black');memory.complete_stroke('black',2);memory.parked()
            self.assertEqual(Session(folder,path,self.manifest(),save_progress=False).count('black'),0)


if __name__=='__main__':unittest.main()
