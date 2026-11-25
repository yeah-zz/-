"""Durable progress must remain atomic when serialized in one write."""
import json
import os
import threading
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from painting_session import save


class SaveTests(unittest.TestCase):
    def test_compact_save_preserves_full_unicode_manifest_and_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'progress.json'
            data={'source':'飞瀑云崖','completed':{'black':12},
                  'manifest':{'stages':{'black':[{'points':[[1.25,2.5]],'press':[.36]}]}}}
            save(path,data)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),data)
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_fsync_failure_keeps_previous_committed_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'progress.json'
            save(path,{'completed':11})
            previous=path.read_bytes()
            with patch('painting_session.os.fsync',side_effect=OSError('disk sync failed')):
                with self.assertRaises(OSError):save(path,{'completed':12})
            self.assertEqual(path.read_bytes(),previous)

    def test_progress_is_synced_before_atomic_replace(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'progress.json'
            events=[]
            import os
            real_replace=os.replace
            def replace(*args):
                events.append('replace');real_replace(*args)
            with patch('painting_session.os.fsync',side_effect=lambda _:events.append('fsync')), \
                 patch('painting_session.os.replace',side_effect=replace):
                save(path,{'completed':12})
            self.assertEqual(events,['fsync','replace'])

    def test_windows_reader_contention_retries_commit_without_rewriting(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'progress.json'
            save(path, {'completed': 22})
            import os
            real_replace = os.replace
            busy = PermissionError('Reader temporarily holds snapshot')
            busy.winerror = 5
            calls = []
            def replace(source, target):
                self.assertEqual(json.loads(Path(source).read_text(encoding='utf-8')), {'completed': 23})
                calls.append(source)
                if len(calls) < 3:
                    self.assertEqual(json.loads(path.read_text()), {'completed': 22})
                    raise busy
                real_replace(source, target)
            with patch('painting_session.os.replace', side_effect=replace), patch('painting_session.time.sleep') as sleep, patch('painting_session.os.fsync', wraps=os.fsync) as sync:
                save(path, {'completed': 23})
            self.assertEqual(len(set(calls)), 1)
            self.assertEqual(sync.call_count, 1)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(json.loads(path.read_text()), {'completed': 23})

    def test_permanent_windows_denial_is_bounded_and_preserves_old_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'progress.json'
            save(path, {'completed': 22})
            busy = PermissionError('Still held')
            busy.winerror = 32
            with patch('painting_session.os.replace', side_effect=busy) as replace, patch('painting_session.time.sleep'):
                with self.assertRaises(PermissionError):
                    save(path, {'completed': 23})
            self.assertEqual(replace.call_count, 25)
            self.assertEqual(json.loads(path.read_text()), {'completed': 22})
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_disk_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'progress.json'
            save(path, {'completed': 22})
            with patch('painting_session.os.replace', side_effect=OSError('disk failure')) as replace, patch('painting_session.time.sleep') as sleep:
                with self.assertRaises(OSError):
                    save(path, {'completed': 23})
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()
            self.assertEqual(json.loads(path.read_text()), {'completed': 22})

    def test_concurrent_writers_have_independent_complete_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'progress.json'
            records = [{'completed': i, 'manifest': [i]*1000} for i in range(24)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda value: save(path, value), records))
            self.assertIn(json.loads(path.read_text()), records)
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    @unittest.skipUnless(os.name == 'nt', 'Windows sharing behavior')
    def test_actual_windows_reader_releases_during_commit_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'progress.json'
            save(path, {'completed': 22})
            reader = path.open('r', encoding='utf-8')
            release = threading.Timer(.1, reader.close)
            release.start()
            try:
                save(path, {'completed': 23})
            finally:
                release.join()
                reader.close()
            self.assertEqual(json.loads(path.read_text()), {'completed': 23})


if __name__=='__main__':unittest.main()
