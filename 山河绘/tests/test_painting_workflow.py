"""Offline regression tests for independent modules and sparse deposition."""
import contextlib
import io
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import imgio
import main
import watercolor
import sparse_color
import painting_workflow as workflow
import outline_pass
import mountain
import vision
from color_profile import ColorProfile


class PaintingTests(unittest.TestCase):
    def test_shared_ink_extraction_excludes_dark_red_paint(self):
        image = np.full((180, 280, 3), 255, np.uint8)
        cv2.line(image, (20, 90), (250, 90), (12, 12, 12), 5)
        cv2.circle(image, (140, 35), 18, (0, 0, 100), -1)
        mask = vision.neutral_ink_mask(image, gray_limit=210)
        self.assertEqual(int(mask[35, 140]), 0)
        self.assertEqual(int(mask[90, 140]), 1)
        outline = outline_pass.build_plan(image, .3)
        self.assertTrue(outline['strokes'])
        self.assertTrue(all(min(p[1] for p in stroke['points']) > 55
                            for stroke in outline['strokes']))
        mountains, _ = mountain.build_strokes(image, .3)
        self.assertTrue(all(min(p[1] for p in stroke['points']) > 55
                            for stroke in mountains))

    def fixture(self):
        im = np.full((120,180,3),255,np.uint8)
        im[15:105,15:95] = (210,180,145)
        im[20:100,115:165] = (130,210,245)
        cv2.line(im,(50,25),(50,95),(15,15,15),1)
        return im

    def test_per_component_cap_and_single_pass(self):
        for limit in (1,2,4):
            strokes, regions = sparse_color.plan_paths(self.fixture(),.3,max_strokes=limit)
            counts = Counter(s['region_id'] for s in strokes)
            self.assertTrue(counts)
            self.assertTrue(all(n<=limit for n in counts.values()))
            self.assertEqual(len(regions),2)
            self.assertEqual(sum(r['stroke_count'] for r in regions),len(strokes))
            ids = [i for step in sparse_color.schedule(strokes) for i in step.get('stroke_ids',[])]
            self.assertEqual(Counter(ids),Counter(s['id'] for s in strokes))

    def test_configurable_sparse_count_and_invalid_limits(self):
        strokes, _ = sparse_color.plan_paths(self.fixture(), .3, max_strokes=4)
        self.assertGreater(max(Counter(s['region_id'] for s in strokes).values()), 2)
        for limit in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                sparse_color.plan_paths(self.fixture(), .3, max_strokes=limit)

    def test_existing_ink_clearance(self):
        im=self.fixture(); scale=.3; gap=.9
        strokes,_=sparse_color.plan_paths(im,scale,gap=gap,respect_ink=True)
        ink,_=watercolor.analyze(im,scale)
        for s in strokes:
            radius=int(np.ceil(s['brush_width_mm_estimate']/(2*scale)))+int(np.ceil(gap/scale))
            blocked=cv2.dilate(ink.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2))>0
            for a,b in zip(s['points'],s['points'][1:]):
                self.assertTrue(watercolor.inside_segment(~blocked,np.array(a),np.array(b)))

    def test_selection_order_and_shared_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.png'; imgio.imwrite(str(source),self.fixture())
            with patch('outline_pass.build_plan',side_effect=AssertionError('outline must not run')):
                plan,work=workflow.build_plan(source,'color',work_size=180)
                output=Path(d)/'color.png'
                workflow.save_outputs(plan,work,Path(d)/'plan.json',output)
                rendered=imgio.imread(str(output))
                self.assertGreater(int(rendered.min()),35)
                with self.assertRaises(ValueError): workflow.save_outputs(plan,work,Path(d)/'plan.json',output)
            with patch('sparse_color.plan_paths',side_effect=AssertionError('color must not run')):
                plan,_=workflow.build_plan(source,'outline',work_size=180)
                self.assertEqual(list(plan['modules']),['outline'])
            for order in ('color,outline','outline,color'):
                plan,_=workflow.build_plan(source,order,work_size=180)
                self.assertEqual(plan['stages'],order.split(','))
                self.assertEqual(plan['modules']['color']['respect_existing_ink'],order.startswith('outline'))
                self.assertEqual(plan['modules']['color']['image_size'],plan['modules']['outline']['image_size'])
                self.assertIn('observe_previous_stage_dry',[s['kind'] for s in plan['sequence']])
                self.assertFalse(plan['ready_for_execution'])

    def test_invalid_inputs_and_no_hardware_dispatch(self):
        for stages in ('','color,color','color,wrong'):
            with self.assertRaises(ValueError): workflow.parse_stages(stages)
        with self.assertRaises(ValueError): ColorProfile(press_mm=3.0).to_dict()
        with self.assertRaises(ValueError): ColorProfile(draw_velocity_percent=0).to_dict()
        with patch('main.cmd_probe',side_effect=AssertionError('hardware called')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.main(['--mode','painting','--probe']),2)
            self.assertEqual(main.main(['--mode','painting','--execute']),2)


if __name__=='__main__': unittest.main()
