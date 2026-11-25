"""Per-artwork drawing-only changes, command caps and process isolation."""
import copy,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from color_profile import ColorProfile
from web import execution
from web.motion_profile import apply_to_artwork,DEFAULTS

class ArtworkDrawBoostTests(unittest.TestCase):
    def test_color_only_boost_caps_and_preserves_contact_and_geometry(self):
        ink={'meta':{}};meta={}
        module={'motion_candidate':ColorProfile(draw_velocity_percent=40,travel_velocity_percent=80).to_dict(),
                'strokes':[{'points':[[1,2],[3,4],[5,6]],'press':[.2]*3,'speed':[10,40,45]}]}
        base=(ink,{'modules':{'color':module}},[],meta)
        boosted=copy.deepcopy(base);boosted[1]['speed_revision']={'paper_draw_multiplier':2}
        original=copy.deepcopy(boosted)
        a=apply_to_artwork(base,DEFAULTS)[1]['modules']['color'];b=apply_to_artwork(boosted,DEFAULTS)[1]['modules']['color']
        self.assertEqual(a['strokes'][0]['speed'],[24,96,100])
        self.assertEqual(b['strokes'][0]['speed'],[48,100,100])
        self.assertEqual(boosted,original)
        for k in a['motion_candidate']:
            if k not in ('draw_velocity_percent','draw_velocity_max'):self.assertEqual(a['motion_candidate'][k],b['motion_candidate'][k])
        self.assertEqual(a['strokes'][0]['points'],b['strokes'][0]['points'])
        self.assertEqual(a['strokes'][0]['press'],b['strokes'][0]['press'])

    def test_ink_title_boost_does_not_leak_or_speed_up_contacts(self):
        points=[{'velocity':v,'pen_up':up,'tag':tag} for v,up,tag in
                [(40,False,'draw:s'),(10,False,'tail:s'),(30,False,'down:s'),(40,True,'hover:s'),(60,True,'lift:s')]]
        arm=types.SimpleNamespace(SPEED_FACTOR=70)
        job=types.SimpleNamespace(TRANSIT_SPEED=80,stroke_waypoints=lambda:copy.deepcopy(points),
                                  dip_waypoints=lambda:[{'velocity':20,'tag':'dip'},{'velocity':20,'tag':'dip-out'}])
        with tempfile.TemporaryDirectory() as temp,patch('web.execution._base_motion',None),patch.dict('sys.modules',{'config':types.SimpleNamespace(ArmConfig=arm),'job':job}):
            p=Path(temp)/'job.json'
            for multiplier,expected in [(2,[100,48,15,40,60]),(2,[100,48,15,40,60]),(1,[96,24,15,40,60])]:
                p.write_text(json.dumps({'workflow':{'speed_revision':{'paper_draw_multiplier':multiplier}}}),encoding='utf-8')
                execution.prepare(p)
                self.assertEqual([x['velocity'] for x in job.stroke_waypoints()],expected)
                self.assertEqual([x['velocity'] for x in job.dip_waypoints()],[5,10])
                self.assertEqual(arm.SPEED_FACTOR,70)

    def test_invalid_multiplier_rejected_and_default_preserved(self):
        for value in [True,0,3,float('inf'),float('nan'),'2']:
            with self.subTest(value=value),self.assertRaises(ValueError):execution.paper_draw_scale({'speed_revision':{'paper_draw_multiplier':value}})
        self.assertEqual(execution.paper_draw_scale({}),2.4)
        self.assertEqual(execution.paper_draw_scale({'speed_revision':None}),2.4)

if __name__=='__main__':unittest.main()
