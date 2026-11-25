"""Verify all five permutations and simulate final-title execution without hardware."""
from collections import Counter
import contextlib
import io
import json
from itertools import groupby
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from web.catalog import ARTWORKS
from scripts.order_selected_artworks import signature, ORDERS
from web.execution import prepare
from run_json_painting import load_artwork, run_file, manifest_for
from painting_session import Console
from web import server

ROOT=Path(__file__).resolve().parents[1]


class SubjectOrderTests(unittest.TestCase):
    def setUp(self):
        import job
        from config import ArmConfig
        for change in (patch.multiple(job, dip_waypoints=job.dip_waypoints, TRANSIT_SPEED=job.TRANSIT_SPEED,
                                      stroke_waypoints=job.stroke_waypoints),
                       patch('web.execution._base_motion', None),
                       patch.object(ArmConfig,'SPEED_FACTOR',ArmConfig.SPEED_FACTOR)):
            change.start()
            self.addCleanup(change.stop)
    def test_all_five_keep_every_motion_and_complete_subjects_and_parents(self):
        for number,(folder,title,_,source) in enumerate(ARTWORKS):
            with self.subTest(title=title):
                package=ROOT/'待画作品'/folder
                old=json.loads((package/'历史版本'/'20261006_笔序调整前'/'painting.json').read_text(encoding='utf-8'))
                new=json.loads((ROOT/source).read_text(encoding='utf-8'))
                old_ink=old['ink_job']['meta']['ink_execution']['base_strokes']
                new_ink=new['ink_job']['meta']['ink_execution']['base_strokes']
                revision=new['workflow'].get('source_revision')
                if revision:
                    from scripts.consolidate_selected_artworks import digest,stroke_data
                    self.assertEqual(revision['stroke_data_sha256'],digest(stroke_data(new)))
                    self.assertEqual(Counter(map(signature,[s for s in old_ink if s.get('role')=='title'])),Counter(map(signature,[s for s in new_ink if s.get('role')=='title'])))
                else:
                    self.assertEqual(Counter(map(signature,old_ink)),Counter(map(signature,new_ink)))
                    self.assertEqual(Counter(map(signature,old['workflow']['modules']['color']['strokes'])),Counter(map(signature,new['workflow']['modules']['color']['strokes'])))
                stages={'black':new_ink}
                for s in new['workflow']['modules']['color']['strokes']:stages.setdefault(s['color'],[]).append(s)
                for stage,ss in stages.items():
                    regions=[s['drawing_region'] for s in ss]
                    self.assertEqual([ORDERS[number].index(r) for r in regions],sorted(ORDERS[number].index(r) for r in regions))
                    self.assertEqual(len([r for r,_ in groupby(regions)]),len(set(regions)))
                parents=[s.get('parent_stroke_id',s['id']) for s in new_ink]
                self.assertEqual(len([p for p,_ in groupby(parents)]),len(set(parents)))
                for _,ss in groupby(new_ink,key=lambda s:s.get('parent_stroke_id',s['id'])):
                    segments=[s.get('segment',0) for s in ss]
                    self.assertEqual(segments,sorted(segments))
                characters=[s['title_character'] for s in new_ink if s.get('role')=='title']
                self.assertEqual([c for c,_ in groupby(characters)],list('山河绘'))
                self.assertTrue(new['workflow']['execution_order']['title_last'])
                self.assertEqual(new['workflow']['execution_order']['color_mode'],'one_color_all_subjects')

    def test_all_five_execute_body_colors_then_title_and_keep_memory_only_counts(self):
        for folder,title,_,source in ARTWORKS:
            with self.subTest(title=title),tempfile.TemporaryDirectory() as temporary,patch('robot.E6Robot.__init__',side_effect=AssertionError('Hardware forbidden')),contextlib.redirect_stdout(io.StringIO()):
                path=ROOT/source;prepare(path);artwork=load_artwork(path)
                expected=manifest_for(artwork[2],artwork[1],artwork[3])
                calls=[]
                def ink_runner(cache,control,start_index):
                    data=json.loads(Path(cache).read_text(encoding='utf-8'))
                    strokes=data['meta']['ink_execution']['base_strokes']
                    self.assertEqual(strokes,expected['stages'][control.stage])
                    if control.stage=='title':self.assertEqual(calls,['black','blue','green'])
                    else:self.assertTrue(all(s.get('role')!='title' for s in strokes))
                    calls.append(control.stage)
                    for i in range(start_index+1,control.total+1):control.progress(i)
                    return 0
                def color_runner(*args,control,start_index,module,color,**kwargs):
                    self.assertEqual(calls[0],'black');self.assertNotIn('title',calls)
                    selected=[s for s in module['strokes'] if s['color']==color]
                    self.assertEqual(selected,expected['stages'][color])
                    calls.append(color)
                    for i in range(start_index+1,control.total+1):control.progress(i)
                    return 0
                answers=iter(['title','blue','green','title'])
                console=Console(io.StringIO(),ask=lambda _:next(answers),poll=lambda:False)
                code=run_file(path,console,Path(temporary),confirmed=True,save_progress=False,
                              prepared_artwork=artwork,fresh_run=True,ink_runner=ink_runner,
                              color_runner=color_runner,station_reader=lambda _:object())
                self.assertEqual(code,0)
                self.assertEqual(calls,['black','blue','green','title'])
                self.assertEqual(sum(map(len,expected['stages'].values())),len(artwork[2])+len(artwork[1]['modules']['color']['strokes']))
                record=json.loads(next(Path(temporary).glob('painting_session_*.json')).read_text(encoding='utf-8'))
                self.assertEqual(record['status'],'complete');self.assertNotIn('completed',record)

    def test_web_can_title_before_colors_and_demo_keeps_selected_order(self):
        from tests import test_web_workbench as fixtures
        fixture=fixtures.WorkbenchTests();fixture.setUp()
        try:
            bundle=fixture.bundle
            bundle['workflow']['execution_order']={'title_last':True}
            for i,s in enumerate(bundle['ink_job']['meta']['ink_execution']['base_strokes']):s['role']='title' if i==1 else 'coarse'
            server.write(fixture.source,bundle);fixture.app.add(fixture.source,'试画',ident=fixture.ident)
            app=fixture.app
            app.start_demo(fixture.ident);app.tick()
            self.assertEqual(app.state()['completed']['title'],0)
            app.action({'command':'stage','stage':'title'});app.tick()
            self.assertEqual(app.state()['status'],'waiting')
            self.assertEqual(app.state()['completed']['title'],1)
            self.assertEqual(app.state()['completed']['blue'],0)
            app.action({'command':'stage','stage':'blue'});app.tick()
            self.assertEqual(app.state()['status'],'complete')
            self.assertEqual(app.state()['stageOrder'],['black','title','blue'])
        finally:fixture.tearDown()


if __name__=='__main__':unittest.main()
