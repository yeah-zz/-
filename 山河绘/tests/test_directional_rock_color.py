"""Candidate paths and safe idle-only publication, never connect hardware."""
import json,copy,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from web.catalog import ARTWORKS
import pending_artwork

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/ARTWORKS[0][3]
PENDING=SOURCE.parent/'待应用_顺皴刷色'


class DirectionalTests(unittest.TestCase):
    def test_direction_and_no_trunk_fill_preserve_canopy_and_title(self):
        old=json.loads(SOURCE.read_text(encoding='utf-8'));new=json.loads((PENDING/'painting.json').read_text(encoding='utf-8'))
        from scripts.directional_rock_color import mask_poly,TRUNK,pavilion_mask
        trunk=mask_poly((1086,1448),TRUNK)
        tr=new['workflow']['source_revision']['source_transform']
        washes=[s for s in new['workflow']['modules']['color']['strokes'] if s.get('role')=='terrain_wash']
        self.assertTrue(washes)
        scale=new['ink_job']['meta']['ink_execution']['mm_per_px']
        inset=new['workflow']['source_revision']['terrain_centerline_inset_mm']
        radius=int(np.ceil(inset/(scale*tr['scale'])))+8
        pavilion=pavilion_mask((1086,1448),new['ink_job']['meta']['ink_execution']['base_strokes'],tr['scale'],np.array(tr['offset']),radius)
        self.assertTrue(any(s.get('connected_passes',1)>1 for s in washes))
        self.assertTrue(any(s.get('terrain_region_index')==11 and s['color']=='blue' for s in washes))
        self.assertTrue(any(s.get('terrain_region_index')==12 and s['color']=='green' for s in washes))
        for s in washes:
            self.assertTrue(all(v==.30 for v in s['press']))
            q=np.rint((np.array(s['points'])-tr['offset'])/tr['scale']).astype(int)
            self.assertFalse(np.any(trunk[q[:,1],q[:,0]]))
            self.assertFalse(np.any(pavilion[q[:,1],q[:,0]]))
            self.assertLessEqual(s['length_mm'],60.001)
            if s.get('connected_passes',1)>1:
                dy=np.diff(np.array(s['points'])[:,1])
                self.assertTrue(np.any(dy>0) and np.any(dy<0))
            else:
                displacement=np.array(s['points'][-1])-s['points'][0]
                guide=np.array(s['source_guide'][-1])-s['source_guide'][0]
                self.assertGreater(np.dot(displacement,guide),0)
        for owner in ('old','new'):
            b=old if owner=='old' else new
            leaves=[s for s in b['workflow']['modules']['color']['strokes'] if s.get('role')=='canopy_fill']
            if owner=='old':old_leaves=leaves
            else:self.assertEqual(leaves,old_leaves)
        old_title={s['id']:s for s in old['ink_job']['meta']['ink_execution']['base_strokes'] if s.get('role')=='title'}
        new_title={s['id']:s for s in new['ink_job']['meta']['ink_execution']['base_strokes'] if s.get('role')=='title'}
        self.assertEqual(old_title,new_title)
        self.assertEqual(sum(s.get('role')=='bark_texture' for s in new['ink_job']['meta']['ink_execution']['base_strokes']),3)
        for s in new['ink_job']['meta']['ink_execution']['base_strokes']:
            if s.get('role')=='bark_texture':
                self.assertGreater(s['length_mm'],15)
                self.assertTrue(np.all(np.diff(np.array(s['points'])[:,1])>0))
                self.assertTrue(all(v==.15 for v in s['press']))

    def test_pending_install_blocked_by_active_lock_then_validates_and_installs(self):
        with tempfile.TemporaryDirectory() as tmp,patch('robot.E6Robot.__init__',side_effect=AssertionError('Hardware forbidden')):
            root=Path(tmp);path=root/'待画作品'/'test'/'任务资料'/'painting.json';path.parent.mkdir(parents=True)
            path.write_bytes(SOURCE.read_bytes());pending=path.parent/'待应用_顺皴刷色';pending.mkdir()
            for n in ('painting.json','应用记录.json','原图色带禁涂掩膜.npz'):(pending/n).write_bytes((PENDING/n).read_bytes())
            # Isolated proposed update must remain testable after the production
            # pending update has been installed or its marker was acknowledged.
            spec=json.loads((pending/'应用记录.json').read_text(encoding='utf-8'))
            spec.update(applied=False,baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest())
            (pending/'应用记录.json').write_text(json.dumps(spec),encoding='utf-8')
            logs=root/'robot_runs';logs.mkdir();lock=logs/'painting_session.lock';lock.write_text('current task')
            original=path.read_bytes()
            with patch.object(pending_artwork,'ROOT',root),patch('scripts.report_subject_order.main') as report:
                self.assertFalse(pending_artwork.apply_pending(path));self.assertEqual(path.read_bytes(),original)
                lock.unlink()
                self.assertTrue(pending_artwork.apply_pending(path));report.assert_called_once()
                installed=json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(installed['workflow']['source_revision']['terrain_fill'],'source_aligned_curved_ribbons')
                self.assertTrue(json.loads((pending/'应用记录.json').read_text(encoding='utf-8'))['applied'])
                self.assertFalse(pending_artwork.apply_pending(path))
                self.assertTrue((path.parent.parent/'历史版本'/'20261006_顺皴刷色前'/'painting.json').exists())

    def test_saved_contact_segments_and_turns_respect_protected_source(self):
        from scripts.validate_source_washes import audit_candidate
        report=audit_candidate(PENDING/'painting.json',SOURCE)
        self.assertGreater(report['terrain_contact_samples_checked'],50000)
        self.assertGreaterEqual(report['minimum_centerline_to_region_boundary_mm'],2.1566)
        self.assertGreater(report['minimum_blue_green_centerline_gap_mm'],2)
        self.assertLessEqual(report['max_terrain_contact_length_mm'],50)
        self.assertTrue(report['black_canopy_water_and_title_preserved'])
