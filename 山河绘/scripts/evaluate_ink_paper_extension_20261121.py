# -*- coding: utf-8 -*-
"""Evaluate short one-sided extensions using frozen predictions and source review."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_ink_paper_regions_20261118 import data_for, region_masks, render
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_tracking_20261121 import ROOT as TRACKING
from evaluate_ink_paper_width_20261119 import REGIONS, ROOT as WIDTH, score
from gongbi_training.ink_paper_extension import extend_narrow_paper


ROOT = LAB/'paper_extension_trial_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/ink_paper_extension.py'
# Coordinates measured on the source-only enlarged grid before this trial.
# Rows are y, inclusive paper interval, left ink, right ink.
REVIEW = [
    [68, 153, 153, 151, 155],
    [71, 155, 155, 153, 157],
    [72, 155, 155, 153, 157],
    [73, 156, 156, 154, 157],
    [74, 156, 156, 154, 158],
    [75, 156, 157, 155, 158],
    [76, 157, 157, 155, 159],
    [77, 157, 157, 155, 159],
    [78, 158, 158, 156, 159],
    [79, 158, 158, 156, 159],
    [80, 158, 158, 157, 160],
    [81, 159, 159, 157, 160],
    [82, 159, 159, 157, 161],
]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    ROOT.mkdir(exist_ok=True)
    known = read(REGIONS/'protocol.json')
    extra = read(WIDTH/'extra_reference.json')
    last = read(TRACKING/'results.json')
    assert sha(TRACKING/'protocol.json') == last['protocol_sha256']
    names = ['L', *(r['id'] for r in known['source_references']), *extra['images']]
    protocol = dict(
        module=sha(MODULE), limit=6, source_review=REVIEW, review_scope='Known case, not independent.',
        known_regions=sha(REGIONS/'protocol.json'), protection=sha(WIDTH/'extra_reference.json'),
        inputs={f'{label}/{name}': sha(directory/f'{name}.npz')
                for label, directory in [('width', WIDTH), ('tracking', TRACKING)] for name in names},
        original_artworks={item['id']: sha(item['image']) for item in known['source_references']},
    )
    protocol_path = ROOT/'protocol.json'
    if protocol_path.exists():
        assert read(protocol_path) == protocol
    else:
        write_json(protocol_path, protocol)

    outcomes, changes = {}, {}
    for name in names:
        with np.load(WIDTH/f'{name}.npz') as stored:
            original = stored['baseline']
            narrow = stored['narrow']
        with np.load(TRACKING/f'{name}.npz') as tracked:
            double = tracked['roles']
        if name in extra['images']:
            info = extra['images'][name]
            assert sha(info['path']) == info['sha256']
            with Image.open(info['path']) as source:
                image = np.array(source.convert('RGB'))
        else:
            image, baseline, *_ = data_for(name, known['source_references'])
            np.testing.assert_array_equal(original, baseline)
        extended, edits = extend_narrow_paper(image, original, narrow, limit=protocol['limit'])
        assert np.all(extended[edits] == 0)
        np.testing.assert_array_equal(extended[~edits], narrow[~edits])
        changes[name] = int(edits.sum())
        outcomes[name] = image, dict(original=original, narrow=narrow, double=double, extended=extended), edits
        np.savez_compressed(ROOT/f'{name}.npz', roles=extended, edits=edits)

    panels, results = [], {}
    specs = known['regions'] + extra['regions']
    reviewed = {row[0]: row for row in REVIEW}
    for spec in specs + [dict(next(s for s in specs if s['id'] == 'HK_gap'),
                               id='HK_review', title='披麻图余段复核')]:
        image, variants, edits = outcomes[spec['image']]
        mask, rows = region_masks(image.shape[:2], spec)
        if spec['id'] == 'HK_review':
            rows = [reviewed.get(row[0], row) for row in rows]
            mask[:] = False
            for y, low, high, _, _ in rows:
                mask[y, low:high+1] = True
        stats = {name: score(roles, mask, rows) for name, roles in variants.items()}
        if spec['id'] != 'HK_review':
            assert stats['narrow'] == last['regions'][spec['id']]['scores']['narrow']
            assert stats['double'] == last['regions'][spec['id']]['scores']['tracked']
        name = spec['id']
        results[name] = dict(scores=stats, edits_inside=int((edits & mask).sum()))
        if rows:
            results[name]['new_open_rows'] = [int(y) for y, low, high, _, _ in rows
                                              if edits[y, low:high+1].any()]
        title = '短披麻亮纹待定' if name == 'AAM_streak' else spec['title']
        panels.append((dict(spec, title=title), image, mask, variants))

    headings = ['原图参考', '冻结模型', '上轮限宽', '双端追踪', '本轮单端延伸']
    render([*panels[:4], panels[-1]], ROOT/'comparison.png', headings)
    render(panels[4:-1], ROOT/'protection.png', headings)
    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    baseline_hashes = {k: sha(lock[k]) == lock[k+'_sha256'] for k in ('approved500', 'working489')}
    assert all(baseline_hashes.values())
    report = dict(regions=results, edits=changes, baseline_hashes=baseline_hashes,
                  protocol_sha256=sha(protocol_path), independent_validation=False,
                  adopted=False, robot_connected=False)
    write_json(ROOT/'results.json', report)
    print({name: (row['scores']['narrow'], row['scores']['extended'], row.get('new_open_rows'))
           for name, row in results.items() if 'gap' in name or name == 'HK_review'})
    print(changes)


if __name__ == '__main__':
    main()
