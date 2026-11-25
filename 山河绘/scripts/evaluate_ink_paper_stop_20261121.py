# -*- coding: utf-8 -*-
"""Evaluate the forward stop condition on locked extension trials."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_ink_paper_extension_20261121 import ROOT as EXT, REVIEW, write_json
from evaluate_ink_paper_regions_20261118 import data_for, region_masks, render
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_width_20261119 import REGIONS, ROOT as WIDTH, score
from validate_ink_paper_extension_20261121 import ROOT as VALIDATION
from gongbi_training.ink_paper_stop import guarded_paper_extension


ROOT = LAB/'paper_stop_trial_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/ink_paper_stop.py'


def evaluation_regions(specifications, outputs, previous):
    results, panels = {}, []
    review = {row[0]: row for row in REVIEW}
    for spec in specifications:
        name = spec['id']
        image, variants, rejected = outputs[spec['image']]
        mask, rows = region_masks(image.shape[:2], spec)
        if name == 'HK_review':
            rows = [review.get(row[0], row) for row in rows]
            mask[:] = False
            for y, low, high, _, _ in rows:
                mask[y, low:high+1] = True
        measured = {key: score(values, mask, rows) for key, values in variants.items()}
        for key in ('narrow', 'extended'):
            assert measured[key] == previous[name]['scores'][key]
        results[name] = dict(scores=measured, rejected={key: int((data & mask).sum()) for key, data in rejected.items()})
        display = dict(spec, title='短披麻亮纹待定') if name == 'AAM_streak' else spec
        panels.append((display, image, mask, variants))
    return results, panels


def main():
    ROOT.mkdir(exist_ok=True)
    known = read(REGIONS/'protocol.json')
    extra = read(WIDTH/'extra_reference.json')
    validation = read(VALIDATION/'protocol.json')
    extension = read(EXT/'protocol.json')
    extension_results = read(EXT/'results.json')
    validation_results = read(VALIDATION/'results.json')
    assert extension['module'] == sha(MODULE.with_name('ink_paper_extension.py'))
    assert sha(EXT/'protocol.json') == extension_results['protocol_sha256']
    assert sha(VALIDATION/'protocol.json') == validation_results['protocol_sha256']
    assert all(sha(path) == digest for path, digest in known['artifacts'].items())
    ranges = (1, 2)
    sources = {**extra['images'], **validation['images']}
    names = ['L', *(r['id'] for r in known['source_references']), *sources]
    pairs = {name: ((VALIDATION if name in validation['images'] else WIDTH)/f'{name}.npz',
                    (VALIDATION if name in validation['images'] else EXT)/f'{name}.npz')
             for name in names}
    locked = dict(module=sha(MODULE), extension=sha(EXT/'results.json'), validation=sha(VALIDATION/'results.json'),
                  arrays={name: [sha(path) for path in files] for name, files in pairs.items()},
                  lookaheads=list(ranges), independent=False)
    path = ROOT/'protocol.json'
    if path.exists():
        assert read(path) == locked
    else:
        write_json(path, locked)

    outputs, changes = {}, {}
    for name, (prior_path, edited_path) in pairs.items():
        if name in sources:
            item = sources[name]
            assert sha(item['path']) == item['sha256']
            with Image.open(item['path']) as source:
                image = np.array(source.convert('RGB'))
        else:
            image, _, *_ = data_for(name, known['source_references'])
        with np.load(prior_path) as prior, np.load(edited_path) as previous:
            original = prior['original' if name in validation['images'] else 'baseline']
            narrow = prior['narrow']
            unguarded = previous['extended' if name in validation['images'] else 'roles']
            proposed = previous['edits']
        variants = dict(narrow=narrow, extended=unguarded)
        rejections = {}
        totals = dict(unguided=int(proposed.sum()))
        for span in ranges:
            key = f'guard{span}'
            filtered, accepted, rejected = guarded_paper_extension(image, original, narrow, lookahead=span)
            np.testing.assert_array_equal(accepted | rejected, proposed)
            np.testing.assert_array_equal(filtered[~rejected], unguarded[~rejected])
            variants[key] = filtered
            rejections[key] = rejected
            totals[key] = int(accepted.sum())
        outputs[name] = image, variants, rejections
        changes[name] = totals
        np.savez_compressed(ROOT/f'{name}.npz', **variants, **{f'{key}_reject': value for key, value in rejections.items()})

    old_specs = known['regions'] + [dict(next(s for s in known['regions'] if s['id'] == 'HK_gap'),
                                        id='HK_review', title='披麻余段复核')] + extra['regions']
    old, old_panels = evaluation_regions(old_specs, outputs, extension_results['regions'])
    transferred, transfer_panels = evaluation_regions(validation['specs'], outputs, validation_results['regions'])
    headers = ['原图参考', '上轮限宽', '原单端延伸', '前看一步', '前看两步']
    render(old_panels, ROOT/'comparison.png', headers)
    render(transfer_panels, ROOT/'transfer_comparison.png', headers)
    audits = {}
    for source, audit in (
        ('previous', read(VALIDATION/'off_target_index.json')),
        ('transfer', read(VALIDATION/'heldout_off_target_index.json')['groups']),
    ):
        audits[source] = []
        for group in audit:
            name = group.get('image', 'qingpu20')
            _, _, rejected = outputs[name]
            coords = group['pixels'] if 'pixels' in group else group['coords']
            counts = {key: sum(bool(data[y, x]) for x, y in coords) for key, data in rejected.items()}
            audits[source].append(dict(id=group['id'], n=len(coords), rejected=counts))
    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key: sha(lock[key]) == lock[key+'_sha256'] for key in ('approved500', 'working489')}
    assert all(hashes.values())
    result = dict(regions={**old, **transferred}, edits=changes, audits=audits, baseline_hashes=hashes,
                  protocol_sha256=sha(path), independent_validation=False, adopted=False, robot_connected=False)
    write_json(ROOT/'results.json', result)
    print(json.dumps(changes, ensure_ascii=False))
    for name in ('HK_review','L_gap','Q_gap','Q_faint','F_wash'):
        print(name, {key: data.get('longest_8_connected_rows', data.get('ink_pixels'))
                     for key, data in result['regions'][name]['scores'].items()})


if __name__ == '__main__':
    main()
