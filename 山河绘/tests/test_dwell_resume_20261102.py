# -*- coding: utf-8 -*-
import json
from pathlib import Path
import sys
import tempfile
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_one_hour_ink_20261102 as runner

TASK=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\selected_complete_G03_01_20261102\fresh_paper_reload3_dwell2\dwell1_resume51')


class DwellResumeTests(TestCase):
    def test_confirmed_dip_checkpoint_preserves_all_completed_strokes(self):
        self.assertEqual(runner.validate_resume(TASK/'resume_checkpoint.json',TASK/'G03_01_complete_full_job.json'),51)

    def test_uncertain_paper_stroke_checkpoint_rejected(self):
        state=json.loads((TASK/'resume_checkpoint.json').read_text('utf-8'))
        prior=json.loads(Path(state['prior_execution_log']).read_text('utf-8'))
        prior['active_action']={'kind':'stroke','stroke_number':52}
        with tempfile.TemporaryDirectory(dir=r'E:\Codex_Workspace_Data\CLIPascene_20261021\temp') as tmp:
            root=Path(tmp);log=root/'uncertain.json';log.write_text(json.dumps(prior),encoding='utf-8')
            state['prior_execution_log']=str(log);check=root/'checkpoint.json';check.write_text(json.dumps(state),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'confirmed ink-loading boundary'):
                runner.validate_resume(check,TASK/'G03_01_complete_full_job.json')

    def test_continuation_only_task_refuses_first_stroke_replay(self):
        with patch.object(sys,'argv',['runner','--phase','full','--task-dir',str(TASK)]),patch.object(runner,'main') as physical:
            with self.assertRaisesRegex(ValueError,'continuation-only'):
                runner.run()
            physical.assert_not_called()

    def test_resume_delegates_to_existing_executor_at_index_51(self):
        argv=['runner','--phase','resume','--task-dir',str(TASK),'--checkpoint',str(TASK/'resume_checkpoint.json')]
        with patch.object(sys,'argv',argv),patch('builtins.input',return_value='RUN'),patch.object(runner,'execute_reviewed_job',return_value=0) as physical:
            self.assertEqual(runner.run(),0)
            physical.assert_called_once_with(str(TASK/'G03_01_complete_full_job.json'),start_index=51)
