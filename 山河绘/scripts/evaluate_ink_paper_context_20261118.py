# -*- coding: utf-8 -*-
"""Compare contextual paper checks with the frozen seam experiment."""
import json
from pathlib import Path

from evaluate_ink_paper_seams_20261118 import LAB, UPSTREAM, read, sha, stats
from gongbi_training.ink_paper_context import separate_contextual_paper

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = LAB / 'paper_context_trial_20261118'
PREVIOUS = LAB / 'paper_seam_trial_20261118'


def reference_masks(path, image, case):
    with np.load(path) as reference:
        assert np.array_equal(image, reference['image'])
        if case == 'H':
            band = reference['band_targets']
            roles = reference['core_targets']
            return {'H': (np.isin(band, [1, 2]), band == 0),
                    'H_legacy_roles': (np.isin(roles, [1, 2]), roles == 0)}
        target = reference['ink_targets']
    masks = {case: (target == 1, target == 0)}
    if case == 'L':
        directory = LAB / 'role_style_evidence_20261118'
        core, paper = np.zeros(target.shape, bool), np.zeros(target.shape, bool)
        for record in read(directory / 'reviewed.json')['records']:
            mask_path = Path(record['partial_mask'])
            assert sha(mask_path) == record['partial_mask_sha256']
            with np.load(mask_path) as data:
                core |= data['core']
                paper |= data['paper']
        masks['L_close_fragments'] = (core, paper & ~core)
    return masks


def comparison(panels):
    width, height = 275, 260
    board = Image.new('RGB', (5 * width, len(panels) * height + 38), '#f4f0e7')
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    labels = ['原图', '冻结模型', '上一轮拆分', '加纸色复核', '移除与保留位置']
    for n, (name, image, baseline, v1, current, removed, withheld, box) in enumerate(panels):
        delta = image.copy()
        for mask, color in [(removed, (219, 73, 42)), (withheld, (53, 132, 209))]:
            delta[mask] = (0.44 * image[mask] + 0.56 * np.array(color)).astype(np.uint8)
        others = []
        for prediction in [baseline, v1, current]:
            overlay = image.copy()
            mask = prediction > 0
            overlay[mask] = (0.52 * image[mask] + 0.48 * np.array([32, 130, 112])).astype(np.uint8)
            others.append(overlay)
        for j, (data, label) in enumerate(zip([image, *others, delta], labels)):
            part = Image.fromarray(data).crop(box)
            zoom = min((width - 14) / part.width, (height - 34) / part.height)
            part = part.resize((round(part.width * zoom), round(part.height * zoom)), Image.Resampling.NEAREST)
            x, y = j * width + 7, n * height
            draw.text((x, y + 4), f'{name} · {label}', font=font, fill='#283d35')
            board.paste(part, (x, y + 30))
    draw.text((8, board.height - 32), '橙色为此次仍移除的像素，蓝色为纸色证据不足而保留的像素', font=font, fill='#283d35')
    board.save(ROOT / 'comparison.png')


def main():
    ROOT.mkdir(exist_ok=True)
    previous_protocol = read(PREVIOUS / 'protocol.json')
    assert sha(UPSTREAM / 'band_paper.pt') == previous_protocol['candidate_sha256']
    module = Path(__file__).resolve().parents[1] / 'gongbi_training'
    assert sha(module / 'ink_paper_seams.py') == previous_protocol['filter_sha256']
    protocol = {'base_protocol_sha256': sha(PREVIOUS / 'protocol.json'),
                'context_filter_sha256': sha(module / 'ink_paper_context.py'),
                'rule': '31 px elliptical close, same 6-level tolerance, prior seam candidates only',
                'reference_usage': 'known L/F/H subsets for scoring only; no reference input',
                'independent_validation': False}
    protocol_path = ROOT / 'protocol.json'
    if protocol_path.exists():
        assert read(protocol_path) == protocol
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf8')

    paths = {entry['id']: Path(entry['path']) for entry in read(UPSTREAM / 'dataset.json')['records']}
    paths['H'] = LAB / 'cunfa_band_checked_20261112/references/H_14.npz'
    cases, changes, panels = {}, {}, []
    for case, path in paths.items():
        with np.load(UPSTREAM / f'predictions/{case}_band_paper.npz') as source:
            image, prediction = source['image'], source['prediction']
        with np.load(PREVIOUS / f'{case}.npz') as prior:
            prior_removed, prior_output = prior['removed'], prior['roles']
        current, removed, withheld = separate_contextual_paper(image, prediction)
        assert np.array_equal(prior_removed, removed | withheld) and not np.any(removed & withheld)
        assert np.array_equal(current[~removed], prediction[~removed])
        np.savez_compressed(ROOT / f'{case}.npz', roles=current, removed=removed, withheld=withheld)
        checked = np.zeros(prediction.shape, bool)
        for tag, (ink, paper) in reference_masks(path, image, case).items():
            cases[tag] = {'prior': stats(prediction, prior_output, ink, paper),
                          'context': stats(prediction, current, ink, paper)}
            checked |= ink | paper
        changes[case] = {'prior_removed': int(prior_removed.sum()), 'removed': int(removed.sum()),
                         'withheld': int(withheld.sum()), 'prior_unreviewed': int((prior_removed & ~checked).sum()),
                         'removed_unreviewed': int((removed & ~checked).sum())}
        if case in ('L', 'H'):
            title, box = ('窄线', (23, 239, 76, 315)) if case == 'L' else ('晕染边缘', (46, 200, 108, 251))
            panels.append((title, image, prediction, prior_output, current, removed, withheld, box))
    lock = read(LAB / 'local_rock_patch_20261117/baseline_lock.json')
    baseline = {key: sha(lock[key]) == lock[key + '_sha256'] for key in ('approved500', 'working489')}
    assert all(baseline.values())
    comparison(panels)
    result = dict(cases=cases, changes=changes, baseline_hashes=baseline, adopted=False,
                  independent_validation=False, new_model_trained=False, robot_connected=False)
    (ROOT / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
