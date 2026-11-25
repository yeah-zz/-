import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from approved_execution_refine_20261107 import schedule_contact_aware,mask_for,hit_start

class ContactScheduleTests(unittest.TestCase):
    def row(self,p,mid=0,phase=0):return dict(points=np.array(p,float),module=mid,phase=phase)
    def test_airborne_start_is_not_ink_contact(self):
        ink=mask_for([[20,20],[40,20]],1.2)
        self.assertEqual(hit_start(np.array([[20,30],[40,30]]),ink),0)
        self.assertGreater(hit_start(np.array([[20,20],[20,40]]),ink),0)
    def test_object_and_phase_order_and_each_curve_once(self):
        rows=[self.row([[10,10],[30,10]],0,0),self.row([[20,10],[20,30]],0,1),self.row([[50,40],[70,40]],1,0)]
        ref=[dict(curve=i,module=r['module'],phase=r['phase']) for i,r in enumerate(rows)]
        result=schedule_contact_aware(rows,[],ref)
        self.assertEqual([(a['module'],a['phase']) for a in result],[(0,0),(0,1),(1,0)])
        self.assertEqual(sorted(a['curve'] for a in result),[0,1,2])
    def test_does_not_modify_curve_geometry(self):
        rows=[self.row([[10,10],[25,35],[40,10]]),self.row([[40,10],[50,20]])]
        copies=[r['points'].copy() for r in rows]
        route=schedule_contact_aware(rows,[],[dict(module=0,phase=0)]*2)
        for r,p in zip(rows,copies):self.assertTrue(np.array_equal(r['points'],p))
        self.assertEqual(len(route),2)
    def test_intersection_can_start_from_clear_endpoint(self):
        rows=[self.row([[10,10],[40,10]],0,0),self.row([[40,10],[40,30]],0,1)]
        route=schedule_contact_aware(rows,[],[dict(module=0,phase=0),dict(module=0,phase=1)])
        self.assertTrue(route[1]['reverse'])

if __name__=='__main__':unittest.main()
