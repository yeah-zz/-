import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from pigment_station import PigmentStation
from .test_color_experiment import ExperimentTests
from plum_blossom import build_plan, petal_plan, _radial_petals
from run_color_experiment import module_lines
from color_profile import ColorProfile
from project_paths import PAINTING_DIR, PLUM_REFERENCE_IMAGE


def cup(**over):
    """当前标定的黑墨杯（直接读 pigment_measurements.json）。

    以前这里写死过一组杯子坐标，换笔/挪盘重标后就会拿"旧高度"去撞门限，
    失败信息还看不出是夹具过期。现在夹具跟着真表走。
    """
    saved = json.loads((PAINTING_DIR / 'pigment_measurements.json').read_text(encoding='utf-8-sig'))['black']
    values = dict(x=saved['x'], y=saved['y'], surface_z=saved['surface_z'], rim_z=saved['rim_z'])
    values.update(over)
    return PigmentStation(**values)


class PlumTests(unittest.TestCase):
    def test_station_clearance_and_invalid_input(self):
        cup().validate()
        for s in [PigmentStation(100,100,-57,-80),PigmentStation(-79,31,-57,-110),
                  PigmentStation(-79,31,float('nan'),-80),PigmentStation(-79,31,-90,-80)]:
            with self.assertRaises(ValueError): s.validate()

    def test_auto_dip_is_initial_and_repeated_then_returns_high(self):
        with tempfile.TemporaryDirectory() as d,patch('run_color_experiment.time.sleep'):
            code,robot=ExperimentTests().execute(d,station=cup(strokes_per_load=1))
            self.assertEqual(code,0)
            record=json.loads(next(Path(d).glob('color_experiment_*.json')).read_text())
            self.assertEqual(len(record['dips']),2)
            names=[s['name'] for s in record['steps']]
            self.assertLess(names.index('cup-dip'),names.index('trial-0:touch'))
            self.assertEqual(names[-1],'return-loading-high')
            # 进出颜料杯走专用的速度（不跟纸面上的移动一起提速）
            # ★ 按【步骤名】筛，不要按 'v=NN' 子串也不要按杯心坐标：
            #   子串法在 dip 速度与 travel 速度撞成同一个数字时误判；
            #   坐标法会把"起点/终点也在黑盘上方"的 raise-before-dip / return-loading-high 算进来。
            #   （两种坑 2026-09-26 都踩过）
            dip_v=ColorProfile().dip_velocity_percent
            travel_v=ColorProfile().travel_velocity_percent
            steps=record['steps']
            cup_steps=[s for s in steps if s['name'] in ('cup-high','cup-dip','cup-withdraw')]
            self.assertEqual(len(cup_steps),6)                       # 2 次蘸色 × 3 段杯内移动
            self.assertTrue(all(s['velocity']==dip_v for s in cup_steps), cup_steps[:2])
            back=[s for s in steps if s['name']=='return-loading-high']
            self.assertTrue(back and back[0]['velocity'] in (travel_v, 1))   # 回纸外高位用移动档

    def test_reference_paths_fit_and_two_colors_only(self):
        plan,_=build_plan(PAINTING_DIR/'参考图'/'plum_blossom_reference.png')
        self.assertEqual(plan['stages'],['outline','color'])
        for stage,color in [('outline','ink'),('color','red')]:
            lines,skipped=module_lines(plan['modules'][stage],color,ColorProfile().to_dict())
            self.assertTrue(lines); self.assertFalse(skipped)
        self.assertEqual({s['color'] for s in plan['modules']['color']['strokes']},{'red'})

    def test_robot_ready_reference_is_simpler_and_executable(self):
        plan,_=build_plan(PLUM_REFERENCE_IMAGE)
        ink=plan['modules']['outline']['strokes']
        red=plan['modules']['color']['strokes']
        self.assertLess(len(ink),50)
        self.assertEqual(len(red),17)  # 三朵各五瓣，加两个花苞
        self.assertEqual(sum(1 for region in {s['region_id'] for s in red}
                             if sum(s['region_id']==region for s in red)==5),3)
        for stage,color in [('outline','ink'),('color','red')]:
            lines,skipped=module_lines(plan['modules'][stage],color,ColorProfile().to_dict())
            self.assertTrue(lines)
            self.assertFalse(skipped)

    def test_petals_are_five_short_press_and_drag_strokes(self):
        plan,_=build_plan(PAINTING_DIR/'参考图'/'plum_blossom_reference.png')
        strokes=plan['modules']['color']['strokes']
        self.assertTrue(strokes)
        self.assertLess(len(strokes),35)
        for s in strokes:
            self.assertEqual(len(s['points']),len(s['press']))
            self.assertEqual(len(s['points']),3)
            self.assertGreaterEqual(min(s['press']),0.)
            self.assertLessEqual(max(s['press']),ColorProfile().press_mm+1e-9)
            self.assertGreaterEqual(s['length_mm'],.59)
            self.assertLessEqual(s['length_mm'],3.01)
            self.assertGreater(s['press'][0],s['press'][1])
            self.assertGreater(s['press'][1],s['press'][2])
        red_lines,_=module_lines(plan['modules']['color'],'red',ColorProfile().to_dict())
        self.assertTrue(all(len(l['draw_speeds'])==len(l['xy'])-1 for l in red_lines))

    def test_dense_reference_preserves_small_five_petal_flowers(self):
        reference = PAINTING_DIR/'参考图'/'plum_blossom_full_reference.png'
        plan,_ = build_plan(reference)
        strokes = plan['modules']['color']['strokes']
        regions = {s['region_id'] for s in strokes}
        self.assertGreaterEqual(sum(sum(s['region_id']==region for s in strokes)==5
                                    for region in regions), 12)
        self.assertTrue(any(s['length_mm'] < 1. for s in strokes))  # 花苞短点
        self.assertEqual(plan['modules']['color']['open_area_mode'],'auto')
        self.assertGreater(plan['modules']['color']['open_area_mm2'],10.)
        self.assertLess(plan['modules']['color']['open_area_mm2'],25.)

    def test_synthetic_petal_stroke_follows_its_major_axis(self):
        mask=np.zeros((240,240),np.uint8)
        cv2.ellipse(mask,(120,120),(50,14),35,0,360,1,-1)      # 长轴 35° 的花瓣状
        strokes=petal_plan(mask,0.2,style='strokes')
        self.assertTrue(strokes)
        pts=np.asarray(strokes[0]['points'])
        angle=math.degrees(math.atan2(float(pts[-1][1]-pts[0][1]),
                                      float(pts[-1][0]-pts[0][0]))) % 180.0
        diff=min(abs(angle-35.),180.-abs(angle-35.))
        self.assertLess(diff,15)                               # 顺着花瓣，而不是水平扫描

    def test_touching_flower_is_split_into_five_petals(self):
        mask=np.zeros((240,240),np.uint8)
        centre=np.array([120.,120.])
        for degrees in (-90,-18,54,126,198):
            angle=math.radians(degrees)
            tip=centre+np.array([math.cos(angle),math.sin(angle)])*43.
            cv2.ellipse(mask,tuple(np.rint(tip).astype(int)),(31,17),degrees,0,360,1,-1)
        petals=_radial_petals(mask,centre,0.25)
        self.assertEqual(len(petals),5)
        self.assertTrue(all(int(p.sum())>100 for p in petals))
        self.assertTrue(np.array_equal(np.minimum(1,sum(petals)),mask))

    def test_tiny_blob_becomes_a_dab(self):
        mask=np.zeros((120,120),np.uint8)
        cv2.circle(mask,(60,60),4,1,-1)                        # 小到只该点一下
        strokes=petal_plan(mask,0.2,style='strokes')
        self.assertEqual(len(strokes),1)
        self.assertEqual(len(strokes[0]['points']),2)          # 极短一小笔
        self.assertLess(strokes[0]['length_mm'],2.0)


if __name__=='__main__': unittest.main()
