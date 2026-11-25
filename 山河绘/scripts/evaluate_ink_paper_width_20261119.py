# -*- coding: utf-8 -*-
"""Compare width-limited paper edits on locked source references."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evaluate_ink_paper_flanks_20261119 import ROOT as FLANKS, SOURCE, REGIONS
from evaluate_ink_paper_regions_20261118 import data_for, measure, region_masks, render
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from gongbi_training.ink_paper_flanks import separate_flank_paper
from gongbi_training.ink_paper_width import separate_narrow_paper


ROOT = LAB / 'paper_width_trial_20261119'
MODULE = Path(__file__).resolve().parents[1] / 'gongbi_training/ink_paper_width.py'
CHECKS = [
    dict(id='Su_wash', image='Su', title='苏春生远山淡墨', box=[87,65,180,126], kind='protect',
         polygon=[[122,88],[131,83],[145,91],[153,97],[143,100],[126,97]]),
    dict(id='Su_edge', image='Su', title='苏春生远山墨边', box=[92,55,145,106], kind='protect',
         polygon=[[108,85],[116,73],[120,69],[125,68],[130,73],[134,78],[125,75],[118,79],[112,87]]),
    dict(id='AAM_streak', image='aam_short', title='短披麻笔内亮纹待定', box=[110,151,184,235],
         kind='uncertain', polygon=[[147,188],[151,180],[157,184],[157,197],[154,211],[149,219],[146,208]]),
]


def freeze_extra():
    paths = {
        'Su': LAB/'independent_teacher_intake_20261118/views/su_stage6_2.jpg',
        'aam_short': LAB/'cunfa_role_reference_20261112/references/aam_short_hemp.png',
    }
    checks = dict(regions=CHECKS,
                  images={name: dict(path=str(path), sha256=sha(path)) for name, path in paths.items()},
                  source_stage='Source image only; marked before testing the width-limited rules.',
                  uncertain_scope='Unresolved ownership of lighter strokes; excluded from error count.')
    path = ROOT/'extra_reference.json'
    if not path.exists():
        path.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf8')
    assert read(path) == checks
    return checks


def open_rows_diagonally(roles, support, rows):
    """Count connected rows with both rails, including diagonal paper steps."""
    opened = support & (roles == 0)
    for y, low, high, left, right in rows:
        if not (roles[y, left - 1:min(left + 2, low)].any()
                and roles[y, max(right - 1, high + 1):right + 2].any()):
            opened[y] = False
    _, _, stats, _ = cv2.connectedComponentsWithStats(opened.astype(np.uint8), connectivity=8)
    return int(stats[1:, cv2.CC_STAT_HEIGHT].max(initial=0))


def score(roles, support, rows):
    result = measure(roles, support, rows)
    if rows:
        result['longest_8_connected_rows'] = open_rows_diagonally(roles, support, rows)
    return result


def predict_extra(refs):
    import torch
    from cunfa_roles_v2_train_20261112 import Net, features
    from cunfa_band_checked_train_20261112 import Refiner
    from verify_paper_seam_transfer_20261118 import MODELS, UPSTREAM, lock_artifacts

    lock_artifacts()
    torch.set_num_threads(4)
    teacher, refiner = Net(False).eval(), Refiner().eval()
    teacher.load_state_dict(torch.load(MODELS/'flat_roles.pt', map_location='cpu', weights_only=False)['model'])
    refiner.load_state_dict(torch.load(UPSTREAM/'band_paper.pt', map_location='cpu', weights_only=False)['model'])
    predictions = {}
    for name, info in refs['images'].items():
        assert sha(info['path']) == info['sha256']
        with Image.open(info['path']) as file:
            image = np.array(file.convert('RGB'))
        with torch.inference_mode():
            input_batch = torch.from_numpy(features(image))[None]
            probabilities, _ = refiner(input_batch, teacher.probabilities(input_batch))
        predictions[name] = image, probabilities[0].argmax(0).numpy().astype(np.uint8)
    return predictions


def main():
    ROOT.mkdir(exist_ok=True)
    extra = freeze_extra()
    known = read(REGIONS/'protocol.json')
    flank_result = read(FLANKS/'results.json')
    assert all(sha(path) == hash_ for path, hash_ in known['artifacts'].items())
    flank_protocol = read(FLANKS/'protocol.json')
    assert sha(FLANKS/'protocol.json') == flank_result['protocol_sha256']
    assert sha(MODULE.with_name('ink_paper_flanks.py')) == flank_protocol['module_sha256']
    protocol = dict(extra_reference_sha256=sha(ROOT/'extra_reference.json'), module_sha256=sha(MODULE),
                    known_region_sha256=sha(REGIONS/'protocol.json'), flank_result_sha256=sha(FLANKS/'results.json'))
    protocol_path = ROOT/'protocol.json'
    if protocol_path.exists():
        assert read(protocol_path) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf8')
    predictions = {}
    for name in ['L', *[r['id'] for r in known['source_references']]]:
        image, baseline, _, _ = data_for(name, known['source_references'])
        predictions[name] = image, baseline
    predictions.update(predict_extra(extra))

    outputs, changes = {}, {}
    for name, (image, baseline) in predictions.items():
        flank, removed, flank_extra = separate_flank_paper(image, baseline, strict=False)
        source = baseline.copy()
        source[removed & ~flank_extra] = 0
        if name in {'L', *[r['id'] for r in known['source_references']]}:
            with np.load(SOURCE/f'{name}.npz') as prior:
                np.testing.assert_array_equal(source, prior['roles'])
            with np.load(FLANKS/f'{name}.npz') as prior:
                np.testing.assert_array_equal(flank, prior['roles'])
        narrow, kept, _ = separate_narrow_paper(image, baseline)
        joined, before_join, bridges = separate_narrow_paper(image, baseline, bridge=True)
        np.testing.assert_array_equal(kept, before_join)
        assert not np.any(kept & ~flank_extra)
        np.testing.assert_array_equal(narrow[~kept], source[~kept])
        np.testing.assert_array_equal(joined[~bridges], narrow[~bridges])
        variants = dict(source=source, flank=flank, narrow=narrow, joined=joined)
        outputs[name] = variants
        changes[name] = dict(flank=int(flank_extra.sum()), narrow=int(kept.sum()), bridges=int(bridges.sum()))
        np.savez_compressed(ROOT/f'{name}.npz', baseline=baseline, **variants, bridges=bridges)

    scores, known_panels, new_panels = {}, [], []
    for spec in known['regions'] + extra['regions']:
        name = spec['image']
        image, _ = predictions[name]
        support, rows = region_masks(image.shape[:2], spec)
        variants = outputs[name]
        scores[spec['id']] = {key: score(array, support, rows) for key, array in variants.items()}
        if rows:
            oracle = variants['source'].copy()
            oracle[support] = 0
            scores[spec['id']]['marked_corridor_oracle'] = score(oracle, support, rows)
        panels = known_panels if spec in known['regions'] else new_panels
        display_spec = dict(spec, title='短披麻亮纹待定') if spec['id'] == 'AAM_streak' else spec
        panels.append((display_spec, image, support, variants))
    titles = ['原图参考', '原图入口', '局部纸色', '限制窄缝中心', '允许补一个缺口']
    render(known_panels, ROOT/'comparison.png', titles)
    render(new_panels, ROOT/'extra_comparison.png', titles)
    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    baseline_hashes = {k: sha(lock[k]) == lock[k+'_sha256'] for k in ('approved500','working489')}
    assert all(baseline_hashes.values())
    result = dict(scores=scores, changes=changes, baseline_hashes=baseline_hashes,
                  protocol_sha256=sha(ROOT/'protocol.json'), independent_validation=False,
                  adopted=False, new_model_trained=False, robot_connected=False)
    (ROOT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
