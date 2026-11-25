# -*- coding: utf-8 -*-
"""Check frozen one-sided extension on source-only reviewed new images."""
import json
from pathlib import Path

import numpy as np

from evaluate_ink_paper_extension_20261121 import ROOT as EXTENSION, write_json
from evaluate_ink_paper_regions_20261118 import region_masks, render
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_width_20261119 import REGIONS, ROOT as WIDTH, predict_extra, score
from gongbi_training.ink_paper_extension import extend_narrow_paper
from gongbi_training.ink_paper_flanks import separate_flank_paper
from gongbi_training.ink_paper_width import separate_narrow_paper


ROOT = LAB/'paper_extension_validation_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/ink_paper_extension.py'
SOURCES = {
    'qingpu20': LAB/'cunfa_cross_source_20261117/units/qingpu_20_0_named.png',
    'fuchun1875': LAB/'clear_rock_sources_20261117/views/F1875_wash_core.png',
}
# Source-only narrow strip with a dark crest immediately on either side.
ROWS = [
    [96, 210, 210, 208, 212], [97, 210, 210, 207, 212],
    [98, 209, 209, 207, 211], [99, 209, 209, 205, 211],
    [100, 208, 208, 205, 210], [101, 208, 208, 204, 210],
    [102, 207, 207, 203, 209], [103, 207, 207, 203, 209],
    [104, 206, 206, 202, 208], [105, 206, 206, 202, 208],
    [106, 205, 205, 201, 207], [107, 205, 205, 200, 207],
]
CHECKS = [
    dict(id='Q_gap', image='qingpu20', title='画谱窄缝', kind='gap', box=[187, 88, 224, 118], sections=ROWS),
    dict(id='Q_faint', image='qingpu20', title='画谱窄缝右侧淡墨', kind='protect', box=[197, 89, 231, 120],
         polygon=[[217, 98], [218, 100], [217, 104], [216, 108], [215, 109], [215, 106], [216, 102]]),
    dict(id='F_wash', image='fuchun1875', title='富春示范淡墨', kind='protect', box=[0, 0, 53, 58],
         polygon=[[12, 11], [38, 11], [43, 26], [38, 47], [12, 47], [8, 27]]),
]


def main():
    ROOT.mkdir(exist_ok=True)
    protocol = read(EXTENSION/'protocol.json')
    assert protocol['module'] == sha(MODULE)
    assert read(EXTENSION/'results.json')['protocol_sha256'] == sha(EXTENSION/'protocol.json')
    previous_images = set(protocol['original_artworks'].values())
    previous_images.update(r['sha256'] for r in read(WIDTH/'extra_reference.json')['images'].values())
    images = {key: dict(path=str(path), sha256=sha(path)) for key, path in SOURCES.items()}
    assert len({info['sha256'] for info in images.values()}) == len(images)
    assert all(info['sha256'] not in previous_images for info in images.values())
    prior_width = read(WIDTH/'protocol.json')
    assert prior_width['module_sha256'] == sha(MODULE.with_name('ink_paper_width.py'))
    frozen = dict(images=images, specs=CHECKS, extension_protocol=sha(EXTENSION/'protocol.json'),
                  original_references=sha(REGIONS/'protocol.json'), limit=protocol['limit'],
                  scope='New to the extension trial, sourced from existing project assets; source-only step before inference.',
                  expert_labels=False)
    path = ROOT/'protocol.json'
    if path.exists():
        assert read(path) == frozen
    else:
        write_json(path, frozen)
    extended_outputs = {}
    for name, (image, original) in predict_extra(frozen).items():
        flank, _, _ = separate_flank_paper(image, original, strict=False)
        narrow, _, _ = separate_narrow_paper(image, original)
        extended, edits = extend_narrow_paper(image, original, narrow, limit=frozen['limit'])
        np.testing.assert_array_equal(extended[~edits], narrow[~edits])
        options = dict(original=original, flank=flank, narrow=narrow, extended=extended)
        extended_outputs[name] = image, options, edits
        np.savez_compressed(ROOT/f'{name}.npz', **options, edits=edits)

    panels, outcomes = [], {}
    for spec in CHECKS:
        image, options, edits = extended_outputs[spec['image']]
        mask, rows = region_masks(image.shape[:2], spec)
        scores = {name: score(roles, mask, rows) for name, roles in options.items()}
        item = dict(scores=scores, extended_edits_inside=int((edits & mask).sum()), all_extension_edits=int(edits.sum()))
        if rows:
            item['rows_open_before'] = [int(y) for y, low, high, _, _ in rows
                                        if (options['narrow'][y, low:high+1] == 0).any()]
            item['rows_newly_open'] = [int(y) for y, low, high, _, _ in rows if edits[y, low:high+1].any()]
            item['rail_samples_removed'] = sum(bool(edits[y, col]) for y, _, _, l, r in rows for col in (l, r))
        outcomes[spec['id']] = item
        panels.append((spec, image, mask, options))
    render(panels, ROOT/'heldout_comparison.png', ['原图参考', '冻结模型', '局部纸色', '上轮限宽', '单端延伸'])
    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key: sha(lock[key]) == lock[key+'_sha256'] for key in ('approved500', 'working489')}
    assert all(hashes.values())
    write_json(ROOT/'results.json', dict(regions=outcomes, protocol_sha256=sha(path),
                                       baseline_hashes=hashes, adopted=False, independent_acceptance=False))
    print(json.dumps(outcomes, ensure_ascii=False))


if __name__ == '__main__':
    main()
