# -*- coding: utf-8 -*-
"""Compare paired-ridge tracking with previous frozen paper-gap experiments."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_ink_paper_regions_20261118 import data_for, region_masks, render
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_width_20261119 import REGIONS, ROOT as WIDTH, score
from gongbi_training.ink_paper_tracking import track_narrow_paper


ROOT = LAB/'paper_tracking_nearest_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/ink_paper_tracking.py'


def main():
    ROOT.mkdir(exist_ok=True)
    known = read(REGIONS/'protocol.json')
    extra = read(WIDTH/'extra_reference.json')
    prior = read(WIDTH/'results.json')
    width_protocol = read(WIDTH/'protocol.json')
    assert sha(WIDTH/'protocol.json') == prior['protocol_sha256']
    assert sha(WIDTH/'extra_reference.json') == width_protocol['extra_reference_sha256']
    assert sha(REGIONS/'protocol.json') == width_protocol['known_region_sha256']
    assert all(sha(path) == digest for path, digest in known['artifacts'].items())
    names = ['L', *(r['id'] for r in known['source_references']), *extra['images']]
    inputs = {name: sha(WIDTH/f'{name}.npz') for name in names}
    protocol = dict(module_sha256=sha(MODULE), width_result_sha256=sha(WIDTH/'results.json'),
                    width_prediction_sha256=inputs, max_gap=6, independent=False)
    protocol_path = ROOT/'protocol.json'
    if protocol_path.exists():
        assert read(protocol_path) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf8')

    outputs, changes = {}, {}
    for name in names:
        with np.load(WIDTH/f'{name}.npz') as saved:
            original = saved['baseline']
            variants = {key: saved[key] for key in ('source', 'flank', 'narrow')}
        if name in extra['images']:
            item = extra['images'][name]
            assert sha(item['path']) == item['sha256']
            with Image.open(item['path']) as source:
                image = np.array(source.convert('RGB'))
        else:
            image, expected, *_ = data_for(name, known['source_references'])
            np.testing.assert_array_equal(original, expected)
        narrow = variants['narrow']
        deepest, _ = track_narrow_paper(image, original, narrow, max_gap=protocol['max_gap'])
        with np.load(LAB/'paper_tracking_trial_20261121'/f'{name}.npz') as first:
            np.testing.assert_array_equal(deepest, first['roles'])
        revised, tracked = track_narrow_paper(image, original, narrow, max_gap=protocol['max_gap'], nearest=True)
        np.testing.assert_array_equal(revised[~tracked], narrow[~tracked])
        assert np.all(revised[tracked] == 0)
        variants['tracked'] = revised
        variants['deepest'] = deepest
        outputs[name] = image, variants, tracked
        additional = (variants['source'] > 0) & (revised == 0)
        changes[name] = dict(tracked=int(tracked.sum()), total_beyond_source=int(additional.sum()))
        np.savez_compressed(ROOT/f'{name}.npz', roles=revised, tracked=tracked)

    results = {}
    panels = []
    for spec in known['regions'] + extra['regions']:
        image, variants, tracked = outputs[spec['image']]
        mask, rows = region_masks(image.shape[:2], spec)
        scores = {key: score(roles, mask, rows) for key, roles in variants.items()}
        for key in ('source', 'flank', 'narrow'):
            assert scores[key] == prior['scores'][spec['id']][key]
        results[spec['id']] = dict(scores=scores, newly_removed=int((mask & tracked).sum()))
        display_spec = dict(spec, title='短披麻亮纹待定') if spec['id'] == 'AAM_streak' else spec
        panels.append((display_spec, image, mask,
                       {key: variants[key] for key in ('flank', 'narrow', 'deepest', 'tracked')}))
    titles = ['原图参考', '局部纸色', '上轮限宽', '追踪最深墨线', '追踪最近墨线']
    render(panels[:len(known['regions'])], ROOT/'comparison.png', titles)
    render(panels[len(known['regions']):], ROOT/'protection.png', titles)

    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    baseline_hashes = {k: sha(lock[k]) == lock[k+'_sha256'] for k in ('approved500', 'working489')}
    assert all(baseline_hashes.values())
    result = dict(regions=results, changes=changes, baseline_hashes=baseline_hashes,
                  protocol_sha256=sha(protocol_path), independent_validation=False,
                  adopted=False, new_model_trained=False, robot_connected=False)
    (ROOT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    for name, result in results.items():
        print(name, result['scores']['tracked'], 'removed', result['newly_removed'])
    print(changes)


if __name__ == '__main__':
    main()
