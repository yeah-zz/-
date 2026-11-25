# -*- coding: utf-8 -*-
"""Test source-only paper entrances on the already reviewed region benchmark."""
import json
from pathlib import Path

import numpy as np

from evaluate_ink_paper_regions_20261118 import ROOT as REGIONS_ROOT, LAB, data_for, measure, read, region_masks, render, sha
from gongbi_training.ink_paper_connected import separate_connected_paper
from gongbi_training.ink_paper_source import separate_source_paper


ROOT = LAB / 'paper_source_trial_20261118'


def main():
    ROOT.mkdir(exist_ok=True)
    region_file = REGIONS_ROOT / 'protocol.json'
    locked = read(region_file)
    prior_results = read(REGIONS_ROOT / 'results.json')
    assert sha(region_file) == prior_results['protocol_sha256']
    assert all(sha(path) == digest for path, digest in locked['artifacts'].items())
    module = Path(__file__).resolve().parents[1] / 'gongbi_training/ink_paper_source.py'
    protocol = dict(region_protocol_sha256=sha(region_file), source_filter_sha256=sha(module),
                    change='Only replace model-background seeds with broad source-paper seeds. Keep contextual candidates and the three-pixel radius.',
                    score_scope='Known reference regions. Not independent stroke validation.')
    path = ROOT / 'protocol.json'
    if path.exists():
        assert read(path) == protocol, 'Keep the existing run reproducible.'
    else:
        path.write_text(json.dumps(protocol, indent=2), encoding='utf8')

    outputs, changes = {}, {}
    for name in ['L', *[example['id'] for example in locked['source_references']]]:
        image, original, _, context = data_for(name, locked['source_references'])
        model_connected, model_removed = separate_connected_paper(image, original)
        source_connected, source_removed = separate_source_paper(image, original)
        context_removed = (context == 0) & (original > 0)
        assert not np.any(source_removed & ~context_removed)
        assert np.array_equal(source_connected[~source_removed], original[~source_removed])
        variants = dict(frozen=original, context=context, model_entrance=model_connected, source_entrance=source_connected)
        outputs[name] = (image, variants)
        changes[name] = dict(context_removed=int(context_removed.sum()), model_entrance_removed=int(model_removed.sum()),
                             source_entrance_removed=int(source_removed.sum()),
                             source_only_edits=int((source_removed & ~model_removed).sum()),
                             model_only_edits=int((model_removed & ~source_removed).sum()))
        np.savez_compressed(ROOT / f'{name}.npz', roles=source_connected, removed=source_removed,
                            source_only=source_removed & ~model_removed)

    results, panels = {}, []
    for spec in locked['regions']:
        image, variants = outputs[spec['image']]
        support, rows = region_masks(image.shape[:2], spec)
        scores = {key: measure(labels, support, rows) for key, labels in variants.items()}
        for key, prior in [('frozen','frozen'), ('context','context'), ('model_entrance','connected')]:
            assert scores[key] == prior_results['regions'][spec['id']][prior]
        results[spec['id']] = scores
        panels.append((spec, image, support, variants))
    render(panels, destination=ROOT / 'comparison.png',
           titles=['原图参考', '冻结模型', '周围纸色', '模型纸面入口', '原图纸面入口'])
    baseline_lock = read(LAB / 'local_rock_patch_20261117/baseline_lock.json')
    baseline = {name: sha(baseline_lock[name]) == baseline_lock[name+'_sha256'] for name in ('approved500','working489')}
    assert all(baseline.values())
    result = dict(regions=results, changes=changes, protocol_sha256=sha(path), baseline_hashes=baseline,
                  independent_validation=False, adopted=False, robot_connected=False, new_model_trained=False)
    (ROOT / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
