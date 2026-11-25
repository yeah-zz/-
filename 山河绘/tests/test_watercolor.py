"""Watercolor planning invariants; entirely offline."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import imgio
import main
import watercolor as wc


class WatercolorTests(unittest.TestCase):
    def fixture(self):
        im = np.full((160, 220, 3), 255, np.uint8)
        for y in range(20, 140):
            alpha = .55*(140-y)/120+.04
            im[y, 15:95] = np.asarray(wc.PALETTE['blue'])*alpha+255*(1-alpha)
        im[25:130, 115:205] = (85, 202, 250)
        cv2.line(im, (55, 0), (55, 159), (0, 0, 0), 3)
        return im

    def test_white_and_neutral_ink_are_not_pigment(self):
        im=np.full((60,90,3),255,np.uint8)
        im[10:30,10:30]=(20,20,20)
        im[35:55,10:30]=(220,220,220)
        ink,fields=wc.analyze(im,.3)
        self.assertTrue(ink[20,20]); self.assertFalse(any(mask.any() for mask,_ in fields.values()))

    def test_full_footprints_stay_in_color_and_outside_ink_gap(self):
        im=self.fixture(); scale=.3; gap=.6
        strokes,ink,fields=wc.plan_paths(im,scale,3.,gap)
        self.assertTrue(strokes)
        self.assertEqual({s['color'] for s in strokes},{'blue','yellow'})
        for s in strokes:
            radius=int(np.ceil(s['brush_width_mm_estimate']/(2*scale)))
            ink_radius=radius+int(np.ceil(gap/scale))
            blocked=cv2.dilate(ink.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*ink_radius+1,)*2))>0
            valid=cv2.erode(fields[s['color']][0].astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2),
                borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)&~blocked
            for a,b in zip(np.asarray(s['points']),np.asarray(s['points'])[1:]):
                self.assertTrue(wc.inside_segment(valid,a,b))
            self.assertIsNone(s['pressure_mm'])

    def test_streamlines_curve_and_never_cross_blank_channel(self):
        yy,xx=np.mgrid[:100,:120]
        field=np.stack([(yy-50)/65.,np.ones_like(xx)],axis=2).astype(float)
        field/=np.linalg.norm(field,axis=2,keepdims=True)
        valid=np.ones((100,120),bool); valid[70:75,:]=False
        path=wc.trace((60,45),valid,field,1.,100)
        self.assertGreater(len(path),10)
        self.assertLess(path[:,1].max(),70)
        vectors=np.diff(path,axis=0)
        angles=np.arctan2(vectors[:,1],vectors[:,0])
        self.assertGreater(float(np.ptp(angles)),.3)

    def test_taper_and_fading_are_profiles_not_pressure(self):
        points=np.c_[np.full(25,10.),np.arange(25.)]
        field=np.tile(np.linspace(.65,.04,30)[:,None],(1,30))
        alpha,widths,recipe=wc.stroke_profile(points,field,3.)
        self.assertGreater(alpha[0],alpha[-1])
        self.assertLess(widths[-1],max(widths)/2)
        self.assertLess(widths[0],max(widths))
        self.assertLessEqual(max(widths),3.)

    def test_preview_does_not_copy_reference_color(self):
        im=np.full((70,100,3),(180,210,245),np.uint8)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'empty.png'
            wc.render(im,dict(strokes=[],mm_per_px=.3),path)
            self.assertTrue(np.all(imgio.imread(str(path))==255))

    def test_schedule_light_to_dark_and_washes_between_colors(self):
        strokes=[dict(id=c+r,color=c,recipe=r) for c in wc.COLOR_ORDER for r in wc.RECIPE_LEVELS]
        seq=wc.schedule(strokes)
        paint=[s for s in seq if s['kind']=='paint_design_layer']
        self.assertEqual([(s['color'],s['recipe']) for s in paint],[(c,r) for c in wc.COLOR_ORDER for r in wc.RECIPE_LEVELS])
        self.assertEqual(sum(s['kind']=='manual_wash_wipe_and_verify_clean' for s in seq),2)
        for s in seq:
            if s['kind']=='manual_load_calibrated_tone': self.assertIsNone(s['water_ratio'])

    def test_minimum_brush_never_generates_wider_stroke(self):
        strokes,_,_=wc.plan_paths(self.fixture()[::2,::2],.6,.3,.1)
        self.assertTrue(strokes)
        self.assertTrue(all(s['brush_width_mm_estimate']<=.3 for s in strokes))

    def test_cli_outputs_and_all_hardware_gates(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); source=root/'reference.png'; plan=root/'plan.json'; preview=root/'preview.png'
            imgio.imwrite(str(source),self.fixture())
            args=['--mode','watercolor','--image',str(source),'--job-out',str(plan),'--preview-out',str(preview),'--diagnostics']
            self.assertEqual(main.main(args),0)
            data=json.loads(plan.read_text(encoding='utf8'))
            self.assertFalse(data['ready_for_execution']); self.assertNotIn('waypoints',data)
            self.assertIsNone(data['existing_paper_registration'])
            self.assertTrue(preview.is_file()); self.assertTrue((root/'preview_paths.png').is_file())
            self.assertEqual(main.main(args),2) # no overwriting a reviewed design
            for flag in ['--execute','--probe','--calibrate-z','--calibrate-ink','--brush-test','--teach','--test-strokes']:
                with patch.object(main,'cmd_probe') as robot:
                    self.assertEqual(main.main(args+[flag]),2); robot.assert_not_called()
            self.assertEqual(main.main(args+['--ink-job','artifacts/mountain/mountain_v6_job.json']),2)

if __name__=='__main__': unittest.main()
