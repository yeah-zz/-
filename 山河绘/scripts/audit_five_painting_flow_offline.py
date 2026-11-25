"""Check real web artwork stage transitions; stage executors are simulated."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from painting_session import Console
from run_json_painting import run_file
from web.catalog import ARTWORKS
from web.site_settings import prepared_artwork
from web.worker import calibrated_station


def main():
    rows = []
    with patch.object(socket, 'socket', side_effect=AssertionError('Network forbidden')), \
            patch.object(socket, 'create_connection', side_effect=AssertionError('Network forbidden')), \
            patch('robot.E6Robot.__init__', side_effect=AssertionError('Hardware forbidden')):
        for _, title, _, relative in ARTWORKS[:5]:
            path = ROOT / relative
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            effective_path = path
            pending_revisions = []
            # Check the actual pending gate without installing a new artwork.
            for name in ('待应用_顺皴刷色', '待应用_右峰黑线'):
                request = path.parent / name / '应用记录.json'
                if request.exists():
                    spec = json.loads(request.read_text(encoding='utf-8'))
                    if not spec.get('applied'):
                        assert hashlib.sha256(effective_path.read_bytes()).hexdigest() == spec['baseline_sha256'], request
                        effective_path = request.parent / 'painting.json'
                        assert hashlib.sha256(effective_path.read_bytes()).hexdigest() == spec['candidate_sha256'], request
                        pending_revisions.append(name)
            with patch('pending_artwork.apply_pending', return_value=False):
                artwork = prepared_artwork(effective_path, ROOT / 'web/data/site_settings.json')
            stages, prompts = [], []
            answers = iter(['blue', 'green', 'black'])

            def answer(prompt):
                prompts.append(prompt)
                return next(answers)

            def execute(*args, control, start_index, **kwargs):
                assert start_index == 0
                stages.append(dict(stage=control.stage, strokes=control.total))
                for index in range(1, control.total + 1):
                    control.progress(index)
                return 0

            output = io.StringIO()
            with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(output):
                code = run_file(path, console=Console(output, ask=answer, poll=lambda: False),
                                folder=folder, ink_runner=execute, color_runner=execute,
                                station_reader=calibrated_station, confirmed=True,
                                save_progress=False, fresh_run=True, prepared_artwork=artwork)
            assert code == 0
            assert [s['stage'] for s in stages] == ['black', 'blue', 'green', 'title']
            assert all(s['strokes'] > 0 for s in stages)
            assert len(prompts) == 3 and '绘画完成。' in output.getvalue()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
            rows.append(dict(title=title, stages=stages, change_color_prompts=len(prompts),
                             source_sha256=source_hash, pending_revisions=pending_revisions,
                             effective_sha256=hashlib.sha256(effective_path.read_bytes()).hexdigest(),
                             result='passed'))
            print(title, stages, 'PASS')
    report = dict(hardware_connected=False, stage_executors='simulated',
                  scope='Real web preparation, cup validation, whole-artwork stage orchestration; no dynamics or hardware validation.',
                  artworks=rows)
    target = ROOT / 'artifacts/motion_audit_20261009/five_painting_flow.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Five complete stage flows passed. No robot connected.')


if __name__ == '__main__':
    main()
