import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cv2,numpy as np
from full_image_ownership_core_20261105 import source_ink,ink_graph,read_paths,junction_votes,automatic_queries

class FullImageOwnershipTests(unittest.TestCase):
    def test_blank_image_has_no_ink(self):
        self.assertFalse(source_ink(np.full((96,96,3),255,np.uint8)).any())

    def test_disconnected_gaps_are_never_bridged(self):
        b=np.zeros((80,100),np.uint8);b[40,5:40]=1;b[40,45:90]=1
        g=ink_graph(b);rows,pending=read_paths(g,{},[0]*len(g['chains']),1)
        self.assertEqual(len(rows),2)
        self.assertTrue(all(np.ptp(r['points'][:,0])<50 for r in rows))

    def test_no_model_votes_never_join_crossing(self):
        b=np.zeros((81,81),np.uint8);b[40,10:71]=1;b[10:71,40]=1
        g=ink_graph(b);links,diag=junction_votes(g,[],[0]*len(g['chains']),1)
        self.assertFalse(links);self.assertEqual(len(g['chains']),4)

    def test_mutual_ownership_preserves_two_crossing_strokes(self):
        b=np.zeros((81,81),np.uint8);b[40,10:71]=1;b[10:71,40]=1
        g=ink_graph(b);pred=[]
        for i,r in enumerate(g['chains']):
            m=np.zeros((81,81),np.float32)
            if np.ptp(r['points'][:,0])>np.ptp(r['points'][:,1]):m[38:43,:]=1
            else:m[:,38:43]=1
            pred.append(dict(mask=m,presence=1.,origin=np.array([-.5,-.5]),side=81.,query={'chain':i}))
        links,diag=junction_votes(g,pred,[0]*4,1);rows,pending=read_paths(g,links,[0]*4,1)
        self.assertEqual(len(rows),2);self.assertEqual(sum(r['learned_junctions'] for r in rows),2)
        self.assertTrue(all(min(np.ptp(r['points'],axis=0))<1 for r in rows))

    def test_full_mask_ambiguous_at_y_not_forced(self):
        b=np.zeros((81,81),np.uint8)
        for end in ((10,10),(70,10),(40,70)):cv2.line(b,(40,40),end,1)
        g=ink_graph(b);pred=[dict(mask=np.ones((81,81),np.float32),presence=1.,origin=np.array([-.5,-.5]),side=81.,query={'chain':i}) for i in range(len(g['chains']))]
        links,diag=junction_votes(g,pred,[0]*len(g['chains']),1)
        self.assertFalse(links)

    def test_cross_object_pair_never_joined(self):
        b=np.zeros((81,81),np.uint8);b[40,10:71]=1;b[10:71,40]=1
        g=ink_graph(b);pred=[dict(mask=np.ones((81,81),np.float32),presence=1.,origin=np.array([-.5,-.5]),side=81.,query={'chain':i}) for i in range(4)]
        links,_=junction_votes(g,pred,list(range(4)),1)
        self.assertFalse(links)

    def test_automatic_queries_use_no_references(self):
        b=np.zeros((80,100),np.uint8);b[40,5:90]=1
        q=automatic_queries(ink_graph(b),1)
        self.assertGreater(len(q),2);self.assertTrue(all(not z['reference_used'] for z in q))

if __name__=='__main__':unittest.main()
