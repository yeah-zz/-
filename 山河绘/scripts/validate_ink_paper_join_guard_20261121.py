# -*- coding: utf-8 -*-
"""Prospective source-only checks of the refined paper-gap rule."""
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_ink_paper_extension_20261121 import write_json
from evaluate_ink_paper_join_guard_20261121 import MODULE, ROOT as DEVELOPED
from evaluate_ink_paper_regions_20261118 import region_masks, render
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_stop_20261121 import EXT, VALIDATION, WIDTH
from evaluate_ink_paper_width_20261119 import predict_extra, score
from validate_ink_paper_persistent_stop_20261121 import ROOT as PERSISTENT_VALIDATION
from gongbi_training.ink_paper_join_guard import protect_ink_joins
from gongbi_training.ink_paper_persistent_stop import persistent_paper_extension
from gongbi_training.ink_paper_width import separate_narrow_paper


ROOT = LAB/'paper_join_guard_validation_20261121'
UNITS = LAB/'cunfa_cross_source_20261117/units'
SOURCES = {
    'xia': (UNITS/'ntm_xia.png', [440,2700,1240,2950]),
    'axe': (LAB/'cunfa_role_reference_20261112/references/aam_large_axe.png', None),
    'mi': (UNITS/'ntm_mi.png', None),
}
CHECKS = [
    dict(id='X_gap', image='xia', title='水纹间的窄缝', kind='gap', axis='x',
         box=[580,2849,618,2871],
         sections=[[590,2858,2859,2856,2864],[591,2859,2860,2856,2864],[592,2859,2860,2856,2864],
                   [593,2859,2860,2856,2864],[594,2859,2861,2856,2865],[595,2859,2861,2857,2864],
                   [596,2859,2861,2856,2864],[597,2859,2861,2857,2864],[598,2859,2861,2857,2864],
                   [599,2860,2861,2857,2863],[600,2860,2861,2857,2864],[601,2860,2861,2857,2864],
                   [602,2860,2862,2858,2865],[603,2860,2862,2858,2865]]),
    dict(id='A_gap', image='axe', title='斧劈样图收窄亮缝', kind='gap', axis='y', box=[101,218,132,241],
         sections=[[223,112,116,108,122],[224,112,116,108,122],[225,112,115,108,122],
                   [226,112,114,108,121],[227,112,114,108,121],[228,112,113,108,120],
                   [229,112,113,108,119],[230,112,113,108,118]]),
    dict(id='A_join', image='axe', title='亮缝下方深墨汇合', kind='protect', box=[101,218,132,243],
         polygon=[[108,233],[118,233],[122,235],[121,239],[107,239],[105,236]]),
    dict(id='A_transition', image='axe', title='收窄灰边待定', kind='uncertain', box=[101,218,132,241],
         polygon=[[110,231],[115,231],[116,232],[110,232]]),
    dict(id='M_wash', image='mi', title='山体深墨面', kind='protect', box=[195,264,247,313],
         polygon=[[212,280],[223,279],[228,285],[226,294],[214,295],[209,289]]),
    dict(id='M_bright', image='mi', title='墨面内部亮点待定', kind='uncertain', box=[180,225,222,271],
         polygon=[[192,241],[196,241],[198,245],[197,253],[194,254],[192,249]]),
]


def freeze():
    prior = read(DEVELOPED/'protocol.json')
    assert sha(DEVELOPED/'protocol.json') == read(DEVELOPED/'results.json')['protocol_sha256']
    assert prior['module'] == sha(MODULE)
    assert read(EXT/'protocol.json')['module'] == sha(MODULE.with_name('ink_paper_extension.py'))
    assert read(WIDTH/'protocol.json')['module_sha256'] == sha(MODULE.with_name('ink_paper_width.py'))
    excluded = set(read(EXT/'protocol.json')['original_artworks'].values())
    for folder, filename in [(WIDTH,'extra_reference.json'), (VALIDATION,'protocol.json'),
                             (PERSISTENT_VALIDATION,'protocol.json')]:
        excluded.update(row['sha256'] for row in read(folder/filename)['images'].values())
    images = {}
    for name, (source, box) in SOURCES.items():
        digest = sha(source)
        assert digest not in excluded
        path = source
        if box:
            path = ROOT/f'{name}_input.png'
            if not path.exists():
                with Image.open(source) as image:
                    image.convert('RGB').crop(box).save(path)
        images[name] = dict(path=str(path), sha256=sha(path), source=str(source), source_sha256=digest, crop=box)
    assert len({item['source_sha256'] for item in images.values()}) == len(images)
    locked = dict(images=images, specs=CHECKS, development=sha(DEVELOPED/'results.json'),
                  span=3, check_dark_edge=True, independent_reviewer=False,
                  scope='Source coordinates fixed before inference on assets new to this rule.')
    protocol = ROOT/'protocol.json'
    if protocol.exists():
        assert read(protocol) == locked
    else:
        write_json(protocol, locked)
    return protocol, locked


def local_reference(spec, image, crop):
    dx, dy = (crop[:2] if crop else (0,0))
    view = dict(spec)
    box = spec['box']
    view['box'] = [box[0]-dx, box[1]-dy, box[2]-dx, box[3]-dy]
    transverse = spec.get('axis') == 'x'
    if spec['kind']=='gap':
        along, across = (dx,dy) if transverse else (dy,dx)
        view['sections'] = [[row-along, lo-across, hi-across, l-across, r-across]
                            for row,lo,hi,l,r in spec['sections']]
    else:
        view['polygon'] = [[x-dx,y-dy] for x,y in spec['polygon']]
    shape = image.shape[:2]
    mask, rows = region_masks(shape[::-1] if transverse else shape, view)
    return view, mask, rows, transverse


def main():
    ROOT.mkdir(exist_ok=True)
    protocol, locked = freeze()
    outputs, changes = {}, {}
    for name, (image, original) in predict_extra(locked).items():
        narrow, _, _ = separate_narrow_paper(image, original)
        before, proposed, _ = persistent_paper_extension(image, original, narrow)
        new, accepted, rejected = protect_ink_joins(image, original, narrow, before, span=locked['span'],
                                                    check_dark_edge=locked['check_dark_edge'])
        np.testing.assert_array_equal(proposed, accepted | rejected)
        np.testing.assert_array_equal(new[~rejected], before[~rejected])
        variants = dict(original=original, narrow=narrow, continuous=before, protected=new)
        outputs[name] = image, variants, proposed, rejected
        changes[name] = dict(proposed=int(proposed.sum()), restored=int(rejected.sum()), accepted=int(accepted.sum()))
        np.savez_compressed(ROOT/f'{name}.npz', **variants, proposed=proposed, rejected=rejected)
    panels, results = [], {}
    for spec in CHECKS:
        name = spec['image']
        image, variants, proposed, rejected = outputs[name]
        view, mask, rows, transverse = local_reference(spec, image, locked['images'][name]['crop'])
        region = mask.T if transverse else mask
        scores = {key: score(roles.T if transverse else roles, mask, rows) for key,roles in variants.items()}
        results[spec['id']] = dict(scores=scores, proposed_in_scope=int((region & proposed).sum()),
                                   restored_in_scope=int((region & rejected).sum()),
                                   axis=spec.get('axis'))
        panels.append((view, image, region, variants))
    render(panels, ROOT/'comparison.png', ['原图参考', '冻结模型', '限宽', '原持续判断', '最新灰墨保护'])
    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key: sha(lock[key]) == lock[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    result = dict(regions=results, edits=changes, baseline_hashes=hashes, protocol_sha256=sha(protocol),
                  independent_trial=True, independent_acceptance=False, adopted=False, robot_connected=False)
    write_json(ROOT/'results.json', result)
    print(changes)
    for name, item in results.items():
        print(name, item)


if __name__ == '__main__':
    main()
