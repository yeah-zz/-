import sys,unittest
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from resume_pavilion_deferred_20261103 import select_strokes,scope
from config import ArmConfig

class DeferredTests(unittest.TestCase):
    def test_prefix_and_other_order_preserved(self):
        old=[dict(id=i,points=[[i,0],[i,1]]) for i in range(8)];copy=deepcopy(old)
        new,kept=select_strokes(old,3,[4,6]);self.assertEqual(new[:3],old[:3]);self.assertEqual(kept,[0,1,2,3,5,7]);self.assertEqual(old,copy)
    def test_completed_candidate_cannot_be_deferred(self):
        with self.assertRaises(ValueError):select_strokes([dict(id=i) for i in range(8)],3,[1])
    def test_repeated_candidate_rejected(self):
        with self.assertRaises(ValueError):select_strokes([dict(id=i) for i in range(8)],3,[4,4])
    def test_speed_scope_only_and_restored(self):
        before=ArmConfig.SPEED_FACTOR
        with scope():self.assertEqual(ArmConfig.SPEED_FACTOR,70)
        self.assertEqual(ArmConfig.SPEED_FACTOR,before)

if __name__=='__main__':unittest.main()
