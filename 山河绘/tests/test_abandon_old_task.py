import json
import os
from pathlib import Path
import tempfile
import unittest

from abandon_old_task import abandon
from experiment_resume import unresolved_records


@unittest.skipUnless(os.name == 'nt', 'Windows exclusive file sharing')
class AbandonTests(unittest.TestCase):
    def make_record(self, folder):
        path = Path(folder) / 'color_experiment_test.json'
        record = dict(success=False, pending_move={'name': 'draw'},
                      contacts=[dict(id='line-1', completed=False)])
        path.write_text(json.dumps(record), encoding='utf8')
        return path, record

    def test_archives_stale_lock_and_keeps_uncertain_motion_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path, before = self.make_record(folder)
            lock = Path(folder) / 'blue_wash_motion.lock'
            lock.write_text(str(path), encoding='utf8')
            self.assertEqual(abandon(folder), 1)
            after = json.loads(path.read_text(encoding='utf8'))
            self.assertEqual(after['pending_move'], before['pending_move'])
            self.assertEqual(after['contacts'], before['contacts'])
            self.assertFalse(after['success'])
            self.assertTrue(after['abandoned'])
            self.assertEqual(unresolved_records(folder), [])
            self.assertFalse(lock.exists())
            self.assertEqual(next((Path(folder)/'retired_locks').glob('*.lock')).read_text(), str(path))

    def test_active_legacy_runner_blocks_cleanup_without_record_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = self.make_record(folder)
            before = path.read_bytes()
            lock = Path(folder) / 'blue_wash_motion.lock'
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                with self.assertRaises(OSError):
                    abandon(folder)
                self.assertEqual(path.read_bytes(), before)
                self.assertTrue(lock.exists())
            finally:
                os.close(fd)

    def test_no_lock_and_repeat_are_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_record(folder)
            self.assertEqual(abandon(folder), 1)
            self.assertEqual(abandon(folder), 0)

    def test_black_and_color_retired_together_without_erasing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            logs = root / 'robot_runs'
            logs.mkdir()
            self.make_record(logs)
            runs = root / 'output' / 'drawing_runs'
            runs.mkdir(parents=True)
            failed = runs / 'failed_execution_log.json'
            record = dict(success=False, completed_strokes=12, total_strokes=30,
                          active_action={'kind': 'stroke', 'stroke_number': 13})
            failed.write_text(json.dumps(record), encoding='utf8')
            success = runs / 'success_execution_log.json'
            success.write_text(json.dumps(dict(record, success=True)), encoding='utf8')
            actual = runs / 'failed_actual.json'
            actual.write_text('{"trajectory": [1, 2]}', encoding='utf8')
            before_success, before_actual = success.read_bytes(), actual.read_bytes()
            self.assertEqual(abandon(logs, [root / 'output']), 2)
            after = json.loads(failed.read_text(encoding='utf8'))
            self.assertTrue(after['abandoned']['previous_sha256'])
            for key, value in record.items():
                self.assertEqual(after[key], value)
            self.assertEqual(success.read_bytes(), before_success)
            self.assertEqual(actual.read_bytes(), before_actual)
            self.assertEqual(abandon(logs, [root / 'output']), 0)

    def test_color_lock_also_prevents_black_record_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runs = root / 'drawing_runs'
            runs.mkdir()
            failed = runs / 'failed_execution_log.json'
            failed.write_text(json.dumps(dict(success=False, completed_strokes=1,
                                              total_strokes=10)), encoding='utf8')
            before = failed.read_bytes()
            fd = os.open(root / 'blue_wash_motion.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                with self.assertRaises(OSError):
                    abandon(root, [runs])
                self.assertEqual(failed.read_bytes(), before)
            finally:
                os.close(fd)
