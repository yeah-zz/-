# -*- coding: utf-8 -*-
import json
from pathlib import Path
import sys
import tempfile
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_one_hour_ink_20261102 as plan
import run_one_hour_ink_20261102 as runner
from config import ArmConfig
from run_mountain_balanced import load_reviewed


class OneHourTrialTests(TestCase):
    def test_speed_context_restores_default_even_on_exception(self):
        original=ArmConfig.SPEED_FACTOR
        with self.assertRaises(RuntimeError):
            with plan.session_speed():
                self.assertEqual(ArmConfig.SPEED_FACTOR,40)
                raise RuntimeError('fixture')
        self.assertEqual(ArmConfig.SPEED_FACTOR,original)

    def test_pilot_is_exact_prefix_and_every_segment_gets_ink(self):
        full=json.loads((plan.OUT/'G05_03_simple_full_job.json').read_text('utf-8'))
        pilot=json.loads((plan.OUT/'G05_03_first_two_job.json').read_text('utf-8'))
        self.assertEqual(pilot['meta']['ink_execution']['base_strokes'],full['meta']['ink_execution']['base_strokes'][:2])
        self.assertEqual(full['meta']['dips'],full['meta']['strokes'])
        self.assertEqual(pilot['meta']['dips'],2)
        with plan.session_speed(),patch('socket.socket',side_effect=AssertionError('No networking')):
            self.assertEqual(len(load_reviewed(full)[0]),full['meta']['strokes'])

    def test_incomplete_pilot_blocks_continuation_before_connection(self):
        with tempfile.TemporaryDirectory(dir=r'E:\Codex_Workspace_Data\CLIPascene_20261021\temp') as folder:
            p=Path(folder)/'failed.json'
            p.write_text(json.dumps(dict(success=False,completed_strokes=1,job_path=str(plan.OUT/'G05_03_first_two_job.json'))),encoding='utf-8')
            with patch.object(sys,'argv',['runner','--phase','finish','--task-dir',str(plan.OUT),'--pilot-log',str(p)]),patch.object(runner,'execute_reviewed_job') as execute:
                with self.assertRaisesRegex(ValueError,'did not finish'):
                    runner.run()
                execute.assert_not_called()

    def test_valid_continuation_skips_completed_first_two(self):
        with tempfile.TemporaryDirectory(dir=r'E:\Codex_Workspace_Data\CLIPascene_20261021\temp') as folder:
            p=Path(folder)/'success.json'
            p.write_text(json.dumps(dict(success=True,completed_strokes=2,job_path=str(plan.OUT/'G05_03_first_two_job.json'))),encoding='utf-8')
            with patch.object(sys,'argv',['runner','--phase','finish','--task-dir',str(plan.OUT),'--pilot-log',str(p)]),patch('builtins.input',return_value='RUN'),patch.object(runner,'execute_reviewed_job',return_value=0) as execute:
                self.assertEqual(runner.run(),0)
                execute.assert_called_once_with(str(plan.OUT/'G05_03_simple_full_job.json'),start_index=2)

    def test_direct_full_start_rejects_unverified_faster_contact_profile(self):
        with patch.object(sys,'argv',['runner','--phase','full','--task-dir',str(plan.OUT)]),patch.object(runner,'main') as execute:
            with self.assertRaisesRegex(ValueError,'unchanged operator-verified'):
                runner.run()
            execute.assert_not_called()

    def test_direct_full_start_accepts_verified_contact_profile_only(self):
        selected=plan.OUT.parent/'selected_complete_G03_01_20261102'
        with patch.object(sys,'argv',['runner','--phase','full','--task-dir',str(selected)]),patch.object(runner,'main',return_value=0) as execute:
            self.assertEqual(runner.run(),0)
            execute.assert_called_once_with(['--job',str(selected/'G03_01_complete_full_job.json')])
