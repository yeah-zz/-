# -*- coding: utf-8 -*-
"""Evaluate directional flank evidence on versioned narrow-gap references."""
import json
from pathlib import Path

import numpy as np

from evaluate_ink_paper_regions_20261118 import ROOT as REGIONS, data_for, measure, region_masks, render
from evaluate_ink_paper_seams_20261118 import LAB, UPSTREAM, read, sha, stats
from evaluate_ink_paper_source_20261118 import ROOT as SOURCE
from gongbi_training.ink_paper_flanks import separate_flank_paper


ROOT = LAB / 'paper_flank_trial_20261119'


def main():
    ROOT.mkdir(exist_ok=True)
    region_path = REGIONS / 'protocol.json'
    references = read(region_path)
    previous = read(SOURCE / 'results.json')
    assert sha(SOURCE / 'protocol.json') == previous['protocol_sha256']
    assert read(SOURCE / 'protocol.json')['region_protocol_sha256'] == sha(region_path)
    assert all(sha(path) == checksum for path, checksum in references['artifacts'].items())
    module = Path(__file__).resolve().parents[1] / 'gongbi_training/ink_paper_flanks.py'
    protocol = dict(region_protocol_sha256=sha(region_path), prior_source_result_sha256=sha(SOURCE / 'results.json'),
                    module_sha256=sha(module), scope='Known regions; local variant follows the strict failure diagnostic.')
    protocol_path = ROOT / 'protocol.json'
    if protocol_path.exists():
        assert read(protocol_path) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf8')

    outputs, changes = {}, {}
    for name in ['L', *[item['id'] for item in references['source_references']]]:
        image, baseline, _, _ = data_for(name, references['source_references'])
        with np.load(SOURCE / f'{name}.npz') as prior:
            source = prior['roles']
        strict_evidence, local_evidence = {}, {}
        strict, strict_removed, strict_added = separate_flank_paper(image, baseline, evidence=strict_evidence)
        local, local_removed, local_added = separate_flank_paper(image, baseline, strict=False, evidence=local_evidence)
        for revised, removed, added in [(strict, strict_removed, strict_added), (local, local_removed, local_added)]:
            assert np.array_equal(removed & ~added, (source == 0) & (baseline > 0))
            assert np.array_equal(revised[~removed], baseline[~removed])
            assert not np.any(added & (source == 0))
        variants = dict(frozen=baseline, source=source, strict=strict, local=local)
        outputs[name] = image, variants, strict_evidence, local_evidence, local_added
        changes[name] = dict(strict_added=int(strict_added.sum()), local_added=int(local_added.sum()))
        np.savez_compressed(ROOT / f'{name}.npz', roles=local, extra_removed=local_added,
                            strict_roles=strict, strict_extra_removed=strict_added)

    regions, stages, panels = {}, {}, []
    for spec in references['regions']:
        image, variants, strict_evidence, local_evidence, added = outputs[spec['image']]
        support, rows = region_masks(image.shape[:2], spec)
        scores = {name: measure(labels, support, rows) for name, labels in variants.items()}
        assert scores['source'] == previous['regions'][spec['id']]['source_entrance']
        regions[spec['id']] = scores
        stages[spec['id']] = {
            name: {key: int((mask & support).sum()) for key, mask in maps.items()}
            for name, maps in [('strict', strict_evidence), ('local', local_evidence)]
        }
        stages[spec['id']]['local_added_in_reference'] = int((added & support).sum())
        panels.append((spec, image, support, variants))
    render(panels, destination=ROOT/'comparison.png',
           titles=['原图参考', '冻结模型', '原图纸面入口', '两侧宽纸面', '局部纸色'])

    with np.load(UPSTREAM/'predictions/H_band_paper.npz') as prediction:
        image, baseline = prediction['image'], prediction['prediction']
    with np.load(LAB/'cunfa_band_checked_20261112/references/H_14.npz') as reference:
        assert np.array_equal(image, reference['image'])
        target = reference['band_targets']
        broader = reference['core_targets']
    local, _, additional = separate_flank_paper(image, baseline, strict=False)
    h14 = dict(band=stats(baseline, local, np.isin(target, [1, 2]), target == 0),
               wider_roles=stats(baseline, local, np.isin(broader, [1,2]), broader == 0),
               added_edits=int(additional.sum()), lost_known_wider_ink=int((additional & np.isin(broader, [1,2])).sum()))
    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    baseline_hashes = {name: sha(lock[name]) == lock[name+'_sha256'] for name in ('approved500','working489')}
    assert all(baseline_hashes.values())
    result = dict(regions=regions, stages=stages, changes=changes, H14=h14, baseline_hashes=baseline_hashes,
                  protocol_sha256=sha(protocol_path), adopted=False, independent_validation=False,
                  new_model_trained=False, robot_connected=False)
    (ROOT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({**changes, 'H14':h14, 'reference_stages':stages}, ensure_ascii=False))


if __name__ == '__main__':
    main()
