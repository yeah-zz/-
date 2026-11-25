import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import test_color_pressure_03 as sample
from config import CanvasConfig, z_for_press
from color_profile import ColorProfile


class ColorPressureSampleTests(unittest.TestCase):
    def setup_sample(self):
        center = np.asarray(CanvasConfig.ROBOT_CORNERS, float).mean(axis=0)
        profile = ColorProfile(press_mm=.3, draw_velocity_percent=100, draw_velocity_max=100)
        module, lines = sample.build_module(center[0]-26, center[1]-2.5, 'blue', profile)
        return module, lines, profile

    def test_physical_depth_and_comparable_geometry(self):
        module, lines, profile = self.setup_sample()
        self.assertEqual(len(lines), 12)
        for i,line in enumerate(lines):
            press = .2 if i<6 else .3
            self.assertAlmostEqual(np.linalg.norm(np.diff(line['xy'],axis=0)),20,places=3)
            for kind,pose in line['sequence']:
                if kind in ('touch','draw'):
                    self.assertAlmostEqual(pose[2],z_for_press(press))
            self.assertEqual(line['draw_speeds'],[100])
        np.testing.assert_allclose(np.asarray(lines[6]['xy'])-lines[0]['xy'],[[32,0],[32,0]],atol=.002)
        from run_color_experiment import needs_color_reload, color_reload_limits, long_stroke_reload_policy
        policy=long_stroke_reload_policy(module)
        limit,length=color_reload_limits(policy,20,20,6)
        self.assertFalse(needs_color_reload(0,5,100,20,10,limit,length_limit_mm=length))
        self.assertTrue(needs_color_reload(0,6,120,20,10,limit,length_limit_mm=length))

    def test_outside_or_nonfinite_rejected(self):
        for x,y in [(1e5,1e5),(float('nan'),10)]:
            with self.assertRaises(ValueError):
                sample.build_module(x,y,'blue',ColorProfile(press_mm=.3))

    def test_cancel_and_shared_job_lock_prevent_runner(self):
        module,_,profile=self.setup_sample();runner=Mock()
        with tempfile.TemporaryDirectory() as folder,patch.object(sample,'ROBOT_LOG_DIR',Path(folder)):
            self.assertEqual(sample.execute(module,'blue',profile,None,ask=lambda _: '',runner=runner),0)
            runner.assert_not_called()
            (Path(folder)/'painting_session.lock').write_text('existing')
            with self.assertRaises(FileExistsError):
                sample.execute(module,'blue',profile,None,ask=lambda _: 'RUN',runner=runner)
            runner.assert_not_called()
            self.assertEqual((Path(folder)/'painting_session.lock').read_text(),'existing')


if __name__=='__main__':unittest.main()
