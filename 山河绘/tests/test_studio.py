"""Offline draft generation, editing, revisions and non-destructive publication."""
import base64
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

from web.studio import Studio, options, reorder, read


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        import threading
        self.app=Mock(lock=threading.RLock())
        self.studio=Studio(self.app,self.root,self.root/'data')

    def sample(self):
        import cv2
        import numpy as np
        image=np.full((180,240,3),255,np.uint8)
        cv2.line(image,(35,135),(170,40),(0,0,0),3)
        cv2.line(image,(65,45),(180,135),(0,0,0),2)
        cv2.rectangle(image,(100,65),(145,95),(170,120,65),-1)
        cv2.rectangle(image,(65,100),(100,125),(65,150,65),-1)
        raw=cv2.imencode('.png',image)[1].tobytes()
        return self.studio.upload(dict(name='新画作.png',data=base64.b64encode(raw).decode()))

    def step(self,record,action,settings=None):
        result=self.studio.start(dict(id=record['id'],revision=record['revision'],action=action,
                                      settings=settings or record['settings']))
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            record=self.studio.public(self.studio.record(result['id']))
            if not record['busy']:
                if record['error']:
                    diagnostics=list((self.studio.folder/record['id']/'versions').glob('*/diagnostics.json'))
                    self.fail(record['error']+'\n'+json.dumps(read(diagnostics[-1]),ensure_ascii=False))
                return record
            time.sleep(.05)
        self.fail('Offline planning did not complete')

    def test_complete_offline_pipeline_edits_and_publishes_materials(self):
        record=self.sample()
        self.assertEqual(record['stage'],'uploaded')
        self.assertIsNone(self.studio.scene(record['id']))
        record=self.step(record,'extract')
        self.assertGreater(record['counts']['black'],1)
        initial=self.studio.scene(record['id'])
        self.assertEqual(initial['color'],[])
        # Editing keeps the selected existing paths and their pressures intact.
        ids=[s['id'] for s in initial['outline']]
        change=copy.deepcopy(record['settings']);change.update(edit_only=True,excluded=[ids[0]],ordered=list(reversed(ids[1:])),order='original')
        record=self.step(record,'outline',change)
        edited=self.studio.scene(record['id'])
        self.assertEqual([s['id'] for s in edited['outline']],list(reversed(ids[1:])))
        self.assertEqual(edited['outline'][0]['points'],initial['outline'][-1]['points'])
        record=self.step(record,'color',record['settings']|{'colors':['blue','green'],'edit_only':False})
        scene=self.studio.scene(record['id'])
        self.assertGreater(len(scene['color']),0)
        self.assertEqual(scene['outline'],edited['outline'])
        record=self.step(record,'validate')
        self.assertTrue(record['checked'])
        previous=set(self.root.rglob('*'))
        published=self.studio.publish({'id':record['id'],'revision':record['revision']})
        path=self.root/published['published']['path']
        self.assertTrue(path.exists())
        self.assertTrue(path.is_relative_to(self.root/'待画作品'/'新作品'))
        bundle=read(path)
        self.assertEqual(bundle['workflow']['reference_image'],bundle['ink_job']['meta']['image'])
        self.assertTrue(Path(bundle['ink_job']['meta']['image']).is_file())
        self.app.add.assert_called_once()
        self.assertFalse(self.app.add.call_args.kwargs['featured'])
        # Later edits create a draft revision and never overwrite the published bundle.
        committed=path.read_bytes()
        record=self.step(self.studio.public(self.studio.record(record['id'])),'outline',record['settings']|{'edit_only':True,'excluded':[],'ordered':[]})
        self.assertFalse(record['checked'])
        self.assertEqual(record['stage'],'outlined')
        self.assertEqual(path.read_bytes(),committed)
        self.assertFalse((self.root/'robot_runs').exists())
        reloaded_app=Mock(lock=__import__('threading').RLock())
        Studio(reloaded_app,self.root,self.root/'empty-drafts')
        reloaded_app.add.assert_called_once()
        self.assertEqual(reloaded_app.add.call_args.args[0],path)

    def test_invalid_upload_options_revision_and_stage_are_rejected(self):
        with self.assertRaises(ValueError):self.studio.upload({'name':'script.py','data':'AA=='})
        with self.assertRaises(ValueError):options({'pressure_scale':float('nan')})
        with self.assertRaises(ValueError):options({'pitch':True})
        with self.assertRaises(ValueError):options({'colors':['blue','blue']})
        with self.assertRaises(ValueError):self.studio.record('../painting')
        record=self.sample()
        with self.assertRaises(ValueError):self.studio.start({'id':record['id'],'revision':99,'action':'extract'})
        with self.assertRaises(ValueError):self.studio.start({'id':record['id'],'revision':0,'action':'validate'})
        with self.assertRaises(ValueError):self.studio.publish({'id':record['id'],'revision':0})

    def test_reorder_rejects_unknown_or_duplicate_ids(self):
        strokes=[dict(id='a',points=[[1,4],[2,4]]),dict(id='b',points=[[4,1],[5,1]])]
        self.assertEqual([s['id'] for s in reorder(strokes,options({'order':'top'}))],['b','a'])
        self.assertEqual(strokes[0]['id'],'a')
        with self.assertRaises(ValueError):reorder(strokes,options({'ordered':['a','a']}))
        with self.assertRaises(ValueError):reorder(strokes,options({'excluded':['missing']}))

    def test_rename_and_abandon_keep_existing_materials(self):
        record = self.sample()
        renamed = self.studio.rename(dict(id=record['id'], revision=record['revision'], title=' 松山新画 '))
        self.assertEqual(renamed['title'], '松山新画')
        self.assertEqual(renamed['revision'], record['revision']+1)
        with self.assertRaises(ValueError):
            self.studio.rename(dict(id=record['id'], revision=renamed['revision'], title='  '))
        self.studio.abandon(dict(id=record['id'], revision=renamed['revision']))
        self.assertTrue(read(self.studio.folder/record['id']/'record.json')['abandoned'])
        self.assertTrue((self.studio.folder/record['id']/'参考原图.png').exists())
        with self.assertRaises(ValueError):
            self.studio.start(dict(id=record['id'], revision=renamed['revision'], action='extract'))

    def test_abandon_terminates_only_draft_process(self):
        record = self.sample()
        internal = self.studio.record(record['id'])
        process = Mock()
        process.poll.return_value = None
        internal.update(busy=True, _process=process)
        self.studio.abandon(dict(id=record['id'], revision=record['revision']))
        process.terminate.assert_called_once()
        self.assertFalse(read(self.studio.folder/record['id']/'record.json')['busy'])

    def test_empty_color_route_also_supports_ink_only(self):
        record = self.step(self.sample(), 'extract')
        before = self.studio.scene(record['id'])['outline']
        record = self.step(record, 'color', record['settings']|{'colors': []})
        self.assertEqual(self.studio.scene(record['id'])['outline'], before)
        self.assertEqual(self.studio.scene(record['id'])['color'], [])
        record = self.step(record, 'validate')
        self.assertTrue(record['checked'])

    def test_ink_only_preserves_edited_outline_and_can_publish(self):
        record = self.step(self.sample(), 'extract')
        record = self.step(record, 'color')
        outline = self.studio.scene(record['id'])['outline']
        self.assertTrue(self.studio.scene(record['id'])['color'])
        record = self.step(record, 'ink_only')
        scene = self.studio.scene(record['id'])
        self.assertEqual(scene['outline'], outline)
        self.assertEqual(scene['color'], [])
        self.assertEqual(record['settings']['colors'], [])
        record = self.step(record, 'validate')
        published = self.studio.publish({'id': record['id'], 'revision': record['revision']})
        bundle = read(self.root/published['published']['path'])
        self.assertEqual(bundle['workflow']['modules']['color']['strokes'], [])
        self.assertTrue(bundle['workflow']['ready_for_execution'])
        # A draft can be added again after its published artwork was removed.
        self.app.deleted_artworks = {published['published']['artworkId']: {}}
        republished = self.studio.publish({'id': record['id'], 'revision': record['revision']})
        self.assertNotEqual(republished['published']['artworkId'], published['published']['artworkId'])
        self.assertTrue((self.root/published['published']['path']).exists())


if __name__=='__main__':unittest.main()
