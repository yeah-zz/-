"""Check requested pressures and movement from black needles to green paint."""
import json
from pathlib import Path
import unittest
import numpy as np
from web.catalog import ARTWORKS

ROOT=Path(__file__).resolve().parents[1]


class PineSourceRevisionTests(unittest.TestCase):
    def test_requested_pressures_green_leaves_and_ridges(self):
        path=ROOT/ARTWORKS[0][3]
        bundle=json.loads(path.read_text(encoding='utf-8'))
        revision=bundle['workflow']['source_revision']
        prior=json.loads((path.parent.parent/revision['baseline']).read_text(encoding='utf-8'))
        original={s['id']:s for s in prior['ink_job']['meta']['ink_execution']['base_strokes']}
        strokes=bundle['ink_job']['meta']['ink_execution']['base_strokes']
        self.assertFalse(any(s['role']=='pine_needle' for s in strokes))
        leaves=[s for s in bundle['workflow']['modules']['color']['strokes'] if s.get('role')=='leaf_color']
        if revision.get('leaf_mode')=='solid_canopy_pads':
            self.assertEqual(leaves,[])
            pads=[s for s in bundle['workflow']['modules']['color']['strokes'] if s.get('role')=='canopy_fill']
            self.assertEqual(len(pads),revision['canopy_fill_strokes'])
            self.assertEqual({s['canopy_index'] for s in pads},set(range(9)))
            self.assertEqual([s['canopy_index'] for s in pads],sorted(s['canopy_index'] for s in pads))
            for s in pads:
                self.assertEqual(s['color'],'green');self.assertEqual(set(s['press']),{.20})
                self.assertLessEqual(s['length_mm'],48.00001)
                self.assertEqual(s['drawing_region'],'整棵松树')
                self.assertEqual(s['path_style'],'in-mask-continuous-serpentine')
        else:self.assertEqual(len(leaves),len(revision['black_leaf_ids_transferred_to_green']))
        for leaf in leaves:
            old=original[leaf['id'].removeprefix('green-leaf-')]
            self.assertEqual(leaf['points'],old['points'])
            self.assertEqual(leaf['color'],'green');self.assertEqual(set(leaf['press']),{.20})
        ridges=[s for s in strokes if s['role']=='far_outline']
        self.assertEqual(len(ridges),10)
        for s in strokes:
            if s['role']=='title':
                self.assertEqual(s['points'],original[s['id']]['points'])
                self.assertEqual(s['press'],original[s['id']]['press'])
            elif s['role'] in ('texture','far_texture','far_outline'):self.assertEqual(set(s['press']),{.15})
            elif s['role']=='trunk':self.assertEqual(set(s['press']),{.30})
            elif s['role']=='branch':self.assertLessEqual(set(s['press']),{.20,.30})
            elif s.get('source_edge_index') in (6,7,9,10,11,12,13,14):self.assertEqual(set(s['press']),{.25})
            elif s.get('source_edge_index') in (0,1,2,3,4,5):self.assertEqual(set(s['press']),{.30})
        # All original black IDs are accounted for; no silent loss of structures.
        self.assertEqual(set(original),{s['id'] for s in strokes if s['id'] in original}|
            (set(revision['removed_ids'])|set(revision.get('canopy_removed_branch_ids',[])))&set(original)|set(revision['black_leaf_ids_transferred_to_green']))
        transform=revision['source_transform']
        restored=(np.array(next(s for s in ridges if s['id']=='mist-ridge-00')['points'])-transform['offset'])/transform['scale']
        self.assertTrue(np.allclose(restored[0],[356,351]))
        self.assertTrue(np.allclose(restored[-1],[592,309]))


if __name__=='__main__':unittest.main()
