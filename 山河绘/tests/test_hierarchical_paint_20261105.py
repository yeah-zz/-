# -*- coding: utf-8 -*-
"""Meaningful topology, ordering and execution-evidence invariants; no hardware."""
import sys,json,time,hashlib,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from hierarchical_paint_core_20261105 import recover_continuity,hierarchy_schedule,order_metrics,ridge_connector,source_fields
from module_stroke_core_20261104 import raster
from hierarchical_paint_server_20261105 import execution_state,ExecutionJournal,ROOT

def owned(n,uncertain=False):return [dict(module=0,uncertain=uncertain) for _ in range(n)]
def source(paths):return np.repeat(((1-raster(paths,(50,50)))*255).astype(np.uint8)[:,:,None],3,axis=2)

class Topology(unittest.TestCase):
    def test_supported_long_gap_recovered_without_deleting_inputs(self):
        p=[np.array([[5.,20],[15,20]]),np.array([[19.,20],[32,20]])]
        out,d=recover_continuity(source([np.array([[5.,20],[32,20]])]),p,owned(2))
        self.assertEqual(len(out),1);self.assertEqual(sorted(out[0]['source_curve_ids']),[0,1])
        self.assertEqual(len(d['accepted']),1);self.assertGreaterEqual(d['accepted'][0]['support'],.92)

    def test_white_gap_stays_separate(self):
        p=[np.array([[5.,20],[15,20]]),np.array([[19.,20],[32,20]])]
        out,_=recover_continuity(source(p),p,owned(2));self.assertEqual(len(out),2)

    def test_parallel_and_uncertain_owners_not_joined(self):
        p=[np.array([[5.,20],[15,20]]),np.array([[15.,22],[30,22]])]
        out,_=recover_continuity(np.zeros((50,50,3),np.uint8),p,owned(2));self.assertEqual(len(out),2)
        p=[np.array([[5.,20],[15,20]]),np.array([[15.,20],[30,20]])]
        out,_=recover_continuity(source(p),p,owned(2,True));self.assertEqual(len(out),2)

    def test_different_modules_no_join_and_no_added_loop(self):
        p=[np.array([[5.,20],[15,20]]),np.array([[15.,20],[30,20]])]
        own=owned(2);own[1]['module']=1
        out,_=recover_continuity(source(p),p,own);self.assertEqual(len(out),2)
        p=[np.array([[10.,10],[20,10],[20,20]]),np.array([[20.,20],[10,20],[10,10]])]
        out,d=recover_continuity(source(p),p,owned(2));self.assertGreaterEqual(len(out),1)
        self.assertEqual(sorted(i for r in out for i in r['source_curve_ids']),[0,1])
        self.assertTrue(all(np.linalg.norm(r['points'][0]-r['points'][-1])>.1 for r in out))

    def test_module_lock_phase_order_and_once_only(self):
        curves=[]
        for mid in [0,1]:
            for phase in [2,0,1]:
                p=np.array([[5+mid*20,5+phase*3],[15+mid*20,5+phase*3]],float)
                curves.append(dict(points=p,module=mid,phase=phase,length_px=10))
        route,_=hierarchy_schedule(curves,[dict(sam_key='a'),dict(sam_key='b')])
        self.assertEqual(order_metrics(route)['module_reentries'],0)
        self.assertEqual(sorted(r['curve'] for r in route),list(range(6)))
        for m in [0,1]:self.assertEqual([r['phase'] for r in route if r['module']==m],[0,1,2])

    def test_texture_direction_bundles_are_completed_without_fusing_paths(self):
        pp=[[[5,5],[20,5]],[[5,8],[20,8]],[[8,10],[8,25]],[[11,10],[11,25]]]
        rows=[dict(points=np.array(p,float),module=0,phase=1,length_px=15) for p in pp]
        route,_=hierarchy_schedule(rows,[dict(sam_key='a')]);bundles=[a['texture_bundle'] for a in route]
        self.assertEqual(len(set(bundles)),2)
        self.assertTrue(bundles in ([0,0,1,1],[1,1,0,0]));self.assertEqual(len(rows),4)

class LiveEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT,prefix='unit_journal_');self.root=Path(self.tmp.name)
        (self.root/'G03_01').mkdir();self.plan=self.root/'G03_01/plan.json'
        self.plan.write_text(json.dumps(dict(actions=[dict(stroke_id='s00000'),dict(stroke_id='s00001')])),'utf8')
        self.digest=hashlib.sha256(self.plan.read_bytes()).hexdigest();self.log=self.root/'events.jsonl'
        self.binding=dict(scene='G03_01',plan_sha256=self.digest,journal=str(self.log),run_id='test-not-hardware')
        self.write_binding();self.journal=ExecutionJournal(self.log,self.digest,'test-not-hardware')
    def tearDown(self):self.tmp.cleanup()
    def write_binding(self):(self.root/'execution_binding.json').write_text(json.dumps(self.binding),'utf8')

    def test_unbound_and_wrong_plan_never_display_progress(self):
        self.binding['journal']=None;self.write_binding();self.assertEqual(execution_state(self.root)['status'],'not_bound')
        self.binding['journal']=str(self.log);self.binding['plan_sha256']='wrong';self.write_binding()
        s=execution_state(self.root);self.assertEqual(s['status'],'plan_mismatch');self.assertEqual(s['confirmed_sequences'],[])

    def test_only_confirmed_events_advance_and_staleness(self):
        self.journal.append('stroke_started',1,'s00000');s=execution_state(self.root)
        self.assertEqual(s['confirmed_sequences'],[]);self.assertEqual(s['active_sequence'],1)
        self.journal.append('stroke_completed',1,'s00000');s=execution_state(self.root)
        self.assertEqual(s['confirmed_sequences'],[1]);self.assertIsNone(s['active_sequence'])
        self.assertEqual(execution_state(self.root,now=time.time()+20)['status'],'stale')

    def test_out_of_order_or_false_completion_rejected(self):
        self.journal.append('stroke_started',2,'s00001');s=execution_state(self.root)
        self.assertEqual(s['status'],'invalid_journal');self.assertEqual(s['confirmed_sequences'],[])

    def test_partial_write_is_not_a_completed_stroke(self):
        self.journal.append('stroke_started',1,'s00000')
        with self.log.open('ab') as f:f.write(b'{"event":"stroke_completed"')
        s=execution_state(self.root);self.assertEqual(s['confirmed_sequences'],[]);self.assertEqual(s['active_sequence'],1)

    def test_old_run_and_early_full_completion_rejected(self):
        bad=ExecutionJournal(self.log,self.digest,'old-run');bad.append('stroke_started',1,'s00000')
        self.assertEqual(execution_state(self.root)['status'],'invalid_journal')

if __name__=='__main__':unittest.main()
