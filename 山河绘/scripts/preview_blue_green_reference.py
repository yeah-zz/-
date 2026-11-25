"""Generate an offline blue/green painting plan from a reference image."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import imgio
import job
from painting_workflow import save_outputs
from unified_painting import build_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / '输出' / ('松山曲水_蓝绿_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    source = out / '参考原图.png'
    shutil.copy2(args.reference, source)
    plan, work, lineart, waypoints, meta = build_plan(source, '山水')
    color = plan['modules']['color']
    color['strokes'] = [s for s in color['strokes'] if s['color'] in ('blue', 'green')]
    color['regions'] = [r for r in color['regions'] if r['color'] in ('blue', 'green')]
    color['stroke_count'] = len(color['strokes'])
    counts = Counter(s['color'] for s in color['strokes'])
    assert set(counts) == {'blue', 'green'}, counts
    plan['note'] = 'Offline trajectory preview: black outlines, blue mountain washes and green vegetation only.'
    prefix = out / '松山曲水_蓝绿'
    save_outputs(plan, work, str(prefix) + '_计划.json', str(prefix) + '_预览.png')
    imgio.imwrite(str(prefix) + '_提取线稿.png', lineart)
    job.save_job(str(prefix) + '_黑线任务.json', waypoints, meta)
    print('OUTPUT:', out, flush=True)
    print('BLACK:', len(plan['modules']['outline']['strokes']), 'COLORS:', dict(counts), flush=True)


if __name__ == '__main__':
    main()
