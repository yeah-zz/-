import unittest
from unittest.mock import Mock, patch
import json
from pathlib import Path
import tempfile
import numpy as np
import contextlib
import io
from config import CanvasConfig, ArmConfig
import test_brush_pressure as p


class PressureTests(unittest.TestCase):
    def samples(self):
        corners=CanvasConfig.ROBOT_CORNERS
        x=sum(q[0] for q in corners)/4-10
        y=sum(q[1] for q in corners)/4-12
        return p.build_samples(x,y,[.05,.10,.15,.20])

    def test_four_separate_rows(self):
        samples=self.samples()
        self.assertEqual([s['press_mm'] for s in samples],[.05,.10,.15,.20])
        self.assertEqual(len({s['y'] for s in samples}),4)
        for s in samples:
            self.assertAlmostEqual(s['path']['xy'][1][0]-s['path']['xy'][0][0],20)

    def test_invalid_pressure_order_and_limit_rejected(self):
        s=self.samples()[0]
        for values in ([.2,.1],[.1,.1],[.51],[-.1],[float('nan')]):
            with self.assertRaises(ValueError): p.build_samples(s['x'],s['y'],values)

    def test_outside_paper_rejected_before_execution(self):
        with self.assertRaises(ValueError): p.build_samples(1e5,1e5,[.1])

    def test_cancel_before_first_does_not_run(self):
        runner=Mock()
        with contextlib.redirect_stdout(io.StringIO()):
            p.run_samples(self.samples(),ask=lambda _: '',runner=runner)
        runner.assert_not_called()

    def test_one_line_then_stop(self):
        runner=Mock(return_value=0)
        answers=iter(['RUN',''])
        with contextlib.redirect_stdout(io.StringIO()):
            p.run_samples(self.samples(),ask=lambda _: next(answers),runner=runner)
        self.assertEqual(runner.call_count,1)
        self.assertEqual(runner.call_args.kwargs['count'],1)
        self.assertAlmostEqual(runner.call_args.kwargs['profile'].press_mm,.05)
        self.assertFalse(runner.call_args.kwargs['save_progress'])

    def test_ten_levels_cover_actual_task_depth_at_black_speed(self):
        from config import z_for_press
        corners=CanvasConfig.ROBOT_CORNERS
        x=sum(q[0] for q in corners)/4-15
        y=sum(q[1] for q in corners)/4-22.5
        samples=p.build_samples(x,y,[i/100 for i in range(5,51,5)],length=30,spacing=5)
        self.assertEqual(len(samples),10)
        for sample in samples:
            self.assertEqual(sample['profile']['draw_velocity_percent'],30)
            self.assertEqual(sample['profile']['global_speed_factor'],ArmConfig.SPEED_FACTOR)
            self.assertAlmostEqual(sample['target_z_mm'],z_for_press(sample['press_mm']))
            targets=[pose[2] for name,pose in sample['path']['sequence'] if name in ('touch','draw')]
            self.assertTrue(all(abs(z-sample['target_z_mm'])<1e-9 for z in targets))
        with self.assertRaises(ValueError):p.build_samples(x,y,[i/100 for i in range(1,12)])

    def test_failed_line_blocks_next(self):
        runner=Mock(return_value=4)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(p.run_samples(self.samples(),ask=lambda _: 'RUN',runner=runner),4)
        self.assertEqual(runner.call_count,1)

    def continuous_samples(self):
        corners=CanvasConfig.ROBOT_CORNERS
        return p.build_samples(sum(q[0] for q in corners)/4-15,
            sum(q[1] for q in corners)/4-22.5,[i/100 for i in range(5,51,5)],length=30,spacing=5)

    def test_continuous_one_run_confirmation_and_cancel(self):
        from pigment_station import PigmentStation
        station=PigmentStation(x=-50,y=75,surface_z=-42,rim_z=-63,strokes_per_load=3)
        samples=self.continuous_samples()
        runner=Mock(return_value=0)
        ask=Mock(return_value='RUN')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(p.run_continuous_samples(samples,ask=ask,runner=runner,station_reader=lambda:station),0)
        ask.assert_called_once()
        runner.assert_called_once()
        kw=runner.call_args.kwargs
        self.assertEqual(kw['color'],'ink')
        self.assertFalse(kw['save_progress'])
        self.assertEqual(kw['profile'].ink_draw_speed_scale,1)
        self.assertEqual(kw['module']['reload_strokes_per_load'],3)
        self.assertEqual(len(kw['module']['strokes']),10)
        runner.reset_mock()
        with contextlib.redirect_stdout(io.StringIO()):
            p.run_continuous_samples(samples,ask=lambda _: '',runner=runner,station_reader=lambda:station)
        runner.assert_not_called()

    def test_continuous_ten_lines_fixed_depth_and_dips_before_1_4_7_10(self):
        import run_color_experiment as trial
        from color_profile import ColorProfile
        from pigment_station import PigmentStation
        from tests.test_color_experiment import ExperimentTests
        samples=self.continuous_samples()
        module,profile=p.build_continuous_module(samples)
        lines,skipped=trial.module_lines(module,'ink',ColorProfile(**profile).to_dict())
        self.assertFalse(skipped)
        self.assertEqual(len(lines),10)
        for sample,line in zip(samples,lines):
            self.assertEqual(line['draw_speeds'],[30])
            self.assertTrue(np.allclose(line['xy'],sample['path']['xy'],atol=.002))
            for name,pose in line['sequence']:
                if name in ('touch','draw'):
                    self.assertAlmostEqual(pose[2],sample['target_z_mm'])
                if name in ('hover','lift'):
                    self.assertAlmostEqual(abs(pose[2]-ArmConfig.PAPER_Z),2.5)
        station=PigmentStation(x=-50,y=75,surface_z=-42,rim_z=-63,strokes_per_load=3)
        # Even >60 seconds between calls must not trigger extra calibration dips.
        ticks=iter(range(0,100000,100))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(trial,'module_lines',return_value=(lines,[])), \
                patch.object(trial.time,'sleep'),patch.object(trial.time,'monotonic',side_effect=lambda:next(ticks)):
            code,_=ExperimentTests().execute(folder,station=station,module=module,
                profile=ColorProfile(**profile),color='ink')
            self.assertEqual(code,0)
            record=json.loads(next(Path(folder).glob('color_experiment_*.json')).read_text(encoding='utf-8'))
            self.assertEqual([d['before_stroke'] for d in record['dips']],
                ['pressure-01','pressure-04','pressure-07','pressure-10'])
            self.assertEqual(len(record['contacts']),10)
            self.assertTrue(all(c['completed'] for c in record['contacts']))


if __name__=='__main__': unittest.main()
