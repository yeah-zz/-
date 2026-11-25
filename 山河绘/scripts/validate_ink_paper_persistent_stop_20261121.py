# -*- coding: utf-8 -*-
"""Source-only checks on images unused by the persistent-stop rule."""
from pathlib import Path

import numpy as np

from evaluate_ink_paper_extension_20261121 import write_json
from evaluate_ink_paper_persistent_stop_20261121 import ROOT as DEVELOPED, MODULE
from evaluate_ink_paper_regions_20261118 import region_masks, render
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_stop_20261121 import EXT, REGIONS, ROOT as STOP, VALIDATION, WIDTH
from evaluate_ink_paper_width_20261119 import predict_extra, score
from gongbi_training.ink_paper_extension import extend_narrow_paper
from gongbi_training.ink_paper_persistent_stop import persistent_paper_extension
from gongbi_training.ink_paper_stop import guarded_paper_extension
from gongbi_training.ink_paper_width import separate_narrow_paper


ROOT = LAB/'paper_persistent_validation_20261121'
UNITS = LAB/'cunfa_cross_source_20261117/units'
SOURCES = dict(huang=UNITS/'ntm_huang.png',
               tang=LAB/'guohua_faint_sources_20261116/views/tang-yin_making-tea_3_input.png',
               qingpu21=UNITS/'qingpu_21_0_named.png')
CHECKS = [
    dict(id='H_upper', image='huang', title='山石窄缝上段', kind='gap', box=[334, 207, 366, 223],
         sections=[[212,349,350,346,353],[213,349,350,346,353],[214,348,349,345,352],
                   [215,348,349,344,352],[216,347,348,344,351],[217,347,347,343,350],
                   [218,346,346,342,350]]),
    dict(id='H_lower', image='huang', title='山石窄缝下段', kind='gap', box=[333, 220, 362, 237],
         sections=[[224,349,349,345,353],[225,349,349,345,353],[226,348,349,345,352],
                   [227,348,349,344,352],[228,347,349,344,351],[229,347,348,343,351],
                   [230,346,347,342,350],[231,345,346,341,350]]),
    dict(id='T_roof', image='tang', title='屋面相邻细线', kind='gap', box=[102, 320, 123, 340],
         sections=[[326,111,111,109,112],[327,111,112,110,113],[328,112,112,111,114],
                   [329,113,113,112,114],[330,113,114,112,115],[331,114,114,113,116],
                   [332,115,115,114,116],[333,116,116,114,117],[334,116,116,115,118],
                   [335,117,117,115,118]]),
    dict(id='H_join', image='huang', title='山石墨线汇合', kind='protect', box=[408, 107, 454, 145],
         polygon=[[418,121],[426,117],[440,117],[441,120],[430,121],[421,124],[418,128],[415,128]]),
    dict(id='Q_join', image='qingpu21', title='画谱墨线汇合', kind='protect', box=[209, 97, 254, 128],
         polygon=[[221,115],[229,111],[235,109],[239,109],[238,112],[233,114],[227,116],[221,117]]),
]


def main():
    ROOT.mkdir(exist_ok=True)
    trained_scope = read(DEVELOPED/'protocol.json')
    assert trained_scope['module'] == sha(MODULE)
    assert sha(DEVELOPED/'protocol.json') == read(DEVELOPED/'results.json')['protocol_sha256']
    assert read(STOP/'protocol.json')['module'] == sha(MODULE.with_name('ink_paper_stop.py'))
    assert read(EXT/'protocol.json')['module'] == sha(MODULE.with_name('ink_paper_extension.py'))
    assert read(WIDTH/'protocol.json')['module_sha256'] == sha(MODULE.with_name('ink_paper_width.py'))
    historical = set(read(EXT/'protocol.json')['original_artworks'].values())
    for folder, filename in ((WIDTH,'extra_reference.json'), (VALIDATION,'protocol.json')):
        historical.update(row['sha256'] for row in read(folder/filename)['images'].values())
    images = {name: dict(path=str(path), sha256=sha(path)) for name, path in SOURCES.items()}
    assert len({row['sha256'] for row in images.values()}) == len(images)
    assert all(row['sha256'] not in historical for row in images.values())
    locked = dict(images=images, specs=CHECKS, development=sha(DEVELOPED/'results.json'),
                  prior_regions=sha(REGIONS/'protocol.json'), expert_verified=False,
                  scope='Existing project assets newly used for this rule. Source sections locked before inference.')
    path = ROOT/'protocol.json'
    if path.exists():
        assert read(path) == locked
    else:
        write_json(path, locked)

    outputs, counts = {}, {}
    for name, (image, original) in predict_extra(locked).items():
        narrow, _, _ = separate_narrow_paper(image, original)
        extended, candidate = extend_narrow_paper(image, original, narrow)
        before, previous, _ = guarded_paper_extension(image, original, narrow)
        after, accepted, rejected = persistent_paper_extension(image, original, narrow)
        np.testing.assert_array_equal(accepted | rejected, candidate)
        np.testing.assert_array_equal(after[~rejected], extended[~rejected])
        versions = dict(narrow=narrow, extended=extended, guard2=before, persistent=after)
        counts[name] = dict(candidate=int(candidate.sum()), guard2=int(previous.sum()),
                            persistent=int(accepted.sum()), retained_again=int((accepted & ~previous).sum()))
        outputs[name] = image, original, versions, accepted
        np.savez_compressed(ROOT/f'{name}.npz', original=original, **versions,
                            candidate=candidate, accepted=accepted, rejected=rejected)

    results, panels = {}, []
    for spec in CHECKS:
        image, original, versions, accepted = outputs[spec['image']]
        mask, rows = region_masks(image.shape[:2], spec)
        stats = {name: score(labels, mask, rows) for name, labels in dict(original=original, **versions).items()}
        item = dict(scores=stats, accepted_in_scope=int((accepted & mask).sum()))
        if rows:
            item['narrow_open_rows'] = [int(y) for y,l,h,_,_ in rows if (versions['narrow'][y,l:h+1]==0).any()]
            item['new_open_rows'] = [int(y) for y,l,h,_,_ in rows if accepted[y,l:h+1].any()]
            item['rails_removed'] = sum(bool(accepted[y,x]) for y,_,_,l,r in rows for x in (l,r))
        results[spec['id']] = item
        panels.append((spec, image, mask, versions))
    render(panels, ROOT/'comparison.png', ['原图参考', '限宽', '原单端延伸', '前看两步', '允许短暂变暗'])
    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key: sha(lock[key]) == lock[key+'_sha256'] for key in ('approved500', 'working489')}
    assert all(hashes.values())
    report = dict(regions=results, edits=counts, baseline_hashes=hashes, protocol_sha256=sha(path),
                  independent_trial=True, independent_acceptance=False, adopted=False, robot_connected=False)
    write_json(ROOT/'results.json', report)
    for name, item in results.items():
        print(name,{version: ([s['open_rows_with_both_rails'],s['longest_8_connected_rows'],s['rows_with_both_rails']]
                              if 'rows' in s else s['ink_pixels']) for version,s in item['scores'].items()},
              item['accepted_in_scope'], item.get('new_open_rows'))
    print(counts)


if __name__ == '__main__':
    main()
