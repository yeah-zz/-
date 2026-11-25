import contextlib
import io
import json
import unittest
from unittest.mock import patch

from project_paths import PAINTING_DIR
from scripts import run_plum_contours as entry


class PlumContourEntryTests(unittest.TestCase):
    def test_red_geometry_uses_green_physical_station_and_restores_shared_entry(self):
        original = entry.run_plum_stage.STAGES
        measured = json.loads((PAINTING_DIR/'pigment_measurements.json').read_text(encoding='utf-8-sig'))
        def check(arguments):
            self.assertEqual(entry.run_plum_stage.STAGES['color'][1:3], ('red', 'green'))
            station = entry.run_plum_stage.station_from_json('green')
            for field in ('x', 'y', 'surface_z', 'rim_z'):
                self.assertEqual(getattr(station, field), measured['green'][field])
            self.assertNotIn('--confirm-site', arguments)
            self.assertEqual(arguments[arguments.index('--plan')+1], str(entry.PLAN))
            return 0
        with patch.object(entry.run_plum_stage, 'main', side_effect=check), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(['--stage', 'color', '--check']), 0)
        self.assertIs(entry.run_plum_stage.STAGES, original)

    def test_black_keeps_current_black_station_and_check_cannot_move(self):
        def check(arguments):
            self.assertEqual(entry.run_plum_stage.STAGES['ink'][1:3], ('ink', 'black'))
            self.assertNotIn('--confirm-site', arguments)
            return 0
        with patch.object(entry.run_plum_stage, 'main', side_effect=check), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(['--stage', 'ink', '--check']), 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.main(['--stage', 'ink', '--check', '--confirm-site'])


if __name__=='__main__':
    unittest.main()
