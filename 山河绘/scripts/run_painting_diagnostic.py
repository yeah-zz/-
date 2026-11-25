"""One supervised original-prefix attempt, parked after stroke 122.

Default only validates offline. --execute requires site recovery confirmation
outside this program. No alarm clearing, enable, controller reset or retry.
"""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from painting_session import Console
from run_json_painting import run_file, manifest_for
from web.site_settings import prepared_artwork


class DiagnosticConsole(Console):
    def __init__(self, limit):
        self.limit = limit
        self.completed = 0

    def progress(self, label, completed, total):
        self.completed = completed
        print(json.dumps(dict(event='progress', label=label, completed=completed,
                              total=total), ensure_ascii=False), flush=True)

    def poll(self):
        # Reached only at stroke boundaries. park() must confirm arrival before
        # ask() can quit; any alarm exits immediately without a parking move.
        return self.completed >= self.limit

    def ask(self, prompt):
        print(prompt, flush=True)
        return 'q'

    def say(self, message):
        print(message, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--limit', type=int, default=122)
    parser.add_argument('--source', type=Path, default=ROOT/'对照分析/诊断准备_20261007/原画稿_只读备份.json')
    parser.add_argument('--settings', type=Path, default=ROOT/'对照分析/诊断准备_20261007/现场参数_只读备份.json')
    args = parser.parse_args()
    artwork = prepared_artwork(args.source, args.settings)
    manifest = manifest_for(artwork[2], artwork[1], artwork[3])
    if not 1 <= args.limit <= len(manifest['stages']['black']):
        raise ValueError('Limit must be within original black stage')
    print('Original prefix validated; strokes/pressure/speed/compensation retained. Limit:', args.limit, flush=True)
    if not args.execute:
        print('OFFLINE ONLY: no robot connection or movement', flush=True)
        return 0
    # Read the existing app, do not create a second executor alongside it.
    with urllib.request.urlopen('http://127.0.0.1:8765/api/state', timeout=5) as response:
        state = json.load(response)
    runtime = state.get('runtime')
    if runtime and runtime.get('status') in ('running','starting','paused','waiting'):
        raise RuntimeError('Existing web task is active; do not start another executor')
    return run_file(args.source, console=DiagnosticConsole(args.limit),
                    confirmed=True, save_progress=False, fresh_run=True,
                    prepared_artwork=artwork)


if __name__ == '__main__':
    raise SystemExit(main())
