# -*- coding: utf-8 -*-
"""Test sustained-loss stopping against the frozen one- and two-step guards."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_ink_paper_extension_20261121 import write_json
from evaluate_ink_paper_regions_20261118 import data_for, render
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_stop_20261121 import EXT, REGIONS, ROOT as STOP, VALIDATION, WIDTH, evaluation_regions
from gongbi_training.ink_paper_persistent_stop import persistent_paper_extension


ROOT = LAB/'paper_persistent_stop_trial_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/ink_paper_persistent_stop.py'


def main():
    ROOT.mkdir(exist_ok=True)
    known = read(REGIONS/'protocol.json')
    extra = read(WIDTH/'extra_reference.json')
    validation = read(VALIDATION/'protocol.json')
    extension = read(EXT/'protocol.json')
    prior = read(STOP/'results.json')
    stop_protocol = read(STOP/'protocol.json')
    assert sha(STOP/'protocol.json') == prior['protocol_sha256']
    assert stop_protocol['module'] == sha(MODULE.with_name('ink_paper_stop.py'))
    assert extension['module'] == sha(MODULE.with_name('ink_paper_extension.py'))
    assert all(sha(path) == digest for path, digest in known['artifacts'].items())
    sources = {**extra['images'], **validation['images']}
    names = ['L', *(r['id'] for r in known['source_references']), *sources]
    locked = dict(module=sha(MODULE), source_stop=sha(STOP/'results.json'),
                  arrays={name: sha(STOP/f'{name}.npz') for name in names}, independent=False)
    path = ROOT/'protocol.json'
    if path.exists():
        assert read(path) == locked
    else:
        write_json(path, locked)

    outputs, changes = {}, {}
    for name in names:
        if name in sources:
            item = sources[name]
            assert sha(item['path']) == item['sha256']
            with Image.open(item['path']) as source:
                image = np.array(source.convert('RGB'))
            source_dir = VALIDATION if name in validation['images'] else WIDTH
            with np.load(source_dir/f'{name}.npz') as saved:
                original = saved['original' if source_dir == VALIDATION else 'baseline']
        else:
            image, original, *_ = data_for(name, known['source_references'])
        with np.load(STOP/f'{name}.npz') as previous:
            variants = {key: previous[key] for key in ('narrow', 'extended', 'guard2')}
        narrow, extended = variants['narrow'], variants['extended']
        revised, accepted, rejected = persistent_paper_extension(image, original, narrow)
        np.testing.assert_array_equal(accepted | rejected, (narrow > 0) & (extended == 0))
        np.testing.assert_array_equal(revised[~rejected], extended[~rejected])
        variants['persistent'] = revised
        changes[name] = dict(accepted=int(accepted.sum()), rejected=int(rejected.sum()))
        outputs[name] = image, variants, dict(persistent=rejected)
        np.savez_compressed(ROOT/f'{name}.npz', roles=revised, accepted=accepted, rejected=rejected)

    development = known['regions'] + [dict(next(s for s in known['regions'] if s['id'] == 'HK_gap'),
                                         id='HK_review', title='披麻余段复核')] + extra['regions']
    local, local_panels = evaluation_regions(development, outputs, prior['regions'])
    transfer, transfer_panels = evaluation_regions(validation['specs'], outputs, prior['regions'])
    for name, item in {**local, **transfer}.items():
        assert item['scores']['guard2'] == prior['regions'][name]['scores']['guard2']
    labels = ['原图参考', '上轮限宽', '无停止规则', '原前看两步', '允许短暂变暗']
    render(local_panels, ROOT/'comparison.png', labels)
    render(transfer_panels, ROOT/'transfer_comparison.png', labels)

    audits = {}
    for category, entries in (
        ('previous', read(VALIDATION/'off_target_index.json')),
        ('transfer', read(VALIDATION/'heldout_off_target_index.json')['groups']),
    ):
        rows = []
        for group in entries:
            coords = group['pixels'] if 'pixels' in group else group['coords']
            image_name = group.get('image', 'qingpu20')
            _, _, masks = outputs[image_name]
            rejected = masks['persistent']
            rows.append(dict(id=group['id'], n=len(coords), rejected=sum(bool(rejected[y,x]) for x,y in coords)))
        audits[category] = rows
    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key: sha(lock[key]) == lock[key+'_sha256'] for key in ('approved500', 'working489')}
    assert all(hashes.values())
    data = dict(regions={**local, **transfer}, edits=changes, audits=audits, baseline_hashes=hashes,
                protocol_sha256=sha(path), independent_validation=False, adopted=False, robot_connected=False)
    write_json(ROOT/'results.json', data)
    print(json.dumps(changes))
    for name in ('HK_review', 'HK_gap', 'L_gap', 'Q_gap', 'Q_faint', 'F_wash'):
        print(name, {key: value.get('longest_8_connected_rows', value.get('ink_pixels'))
                     for key, value in data['regions'][name]['scores'].items()})
    print({name: sum(group['rejected'] for group in groups) for name, groups in audits.items()})


if __name__ == '__main__':
    main()
