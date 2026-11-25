# -*- coding: utf-8 -*-
"""Evaluate the paper seam filter against frozen development outputs."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import LAB, configure

configure()

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gongbi_training.ink_paper_seams import separate_visible_paper

ROOT = LAB / 'paper_seam_trial_20261118'
UPSTREAM = LAB / 'two_work_bands_20261118'


def read(path):
    return json.loads(Path(path).read_text('utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(prediction, corrected, ink, paper):
    return {
        'ink_core': int(ink.sum()),
        'ink_before': int(((prediction > 0) & ink).sum()),
        'ink_after': int(((corrected > 0) & ink).sum()),
        'near_paper': int(paper.sum()),
        'paper_false_before': int(((prediction > 0) & paper).sum()),
        'paper_false_after': int(((corrected > 0) & paper).sum()),
    }


def panel(items):
    labels = ('原图', '冻结候选', '纸缝拆分后', '本次移除位置')
    width, height = 300, 255
    board = Image.new('RGB', (width * 4, height * len(items)), '#f4f0e7')
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    for i, (name, image, before, after, removed, box) in enumerate(items):
        for j, (label, mask, color) in enumerate(zip(labels, (None, before > 0, after > 0, removed),
                                                    (None, (32, 130, 112), (32, 130, 112), (225, 72, 40)))):
            view = image.copy()
            if mask is not None:
                view[mask] = (0.48 * image[mask] + 0.52 * np.array(color)).astype(np.uint8)
            view = Image.fromarray(view).crop(box)
            scale = min((width - 16) / view.width, (height - 38) / view.height)
            view = view.resize((round(view.width * scale), round(view.height * scale)), Image.Resampling.NEAREST)
            draw.text((j * width + 8, i * height + 8), f'{name} · {label}', font=font, fill='#283d35')
            board.paste(view, (j * width + 8, i * height + 33))
    board.save(ROOT / 'comparison.png')


def main():
    ROOT.mkdir(exist_ok=True)
    protocol = {
        'candidate_source': str(UPSTREAM / 'band_paper.pt'),
        'candidate_sha256': sha(UPSTREAM / 'band_paper.pt'),
        'filter_sha256': sha(Path(__file__).resolve().parents[1] / 'gongbi_training/ink_paper_seams.py'),
        'reference_usage': 'reference arrays opened for scoring only, not passed to the filter',
        'test_scope': 'known training-side references L/F and previously inspected H14 development regression',
        'train_new_model': False,
    }
    assert protocol['candidate_sha256'] == read(UPSTREAM / 'checkpoint_freeze.json')['band_paper']
    protocol_file = ROOT / 'protocol.json'
    if protocol_file.exists():
        assert read(protocol_file) == protocol, 'Use a separate run for a changed filter or checkpoint'
    else:
        protocol_file.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf8')
    training = {r['id']: Path(r['path']) for r in read(UPSTREAM / 'dataset.json')['records']}
    training['H'] = LAB / 'cunfa_band_checked_20261112/references/H_14.npz'
    cases = {}
    figures = []
    for name, reference_path in training.items():
        with np.load(UPSTREAM / f'predictions/{name}_band_paper.npz') as data:
            image, prediction = data['image'], data['prediction']
        corrected, removed = separate_visible_paper(image, prediction)
        assert np.array_equal(corrected[~removed], prediction[~removed]) and (corrected[removed] == 0).all()
        np.savez_compressed(ROOT / f'{name}.npz', roles=corrected, removed=removed)
        with np.load(reference_path) as reference:
            assert np.array_equal(image, reference['image'])
            if name == 'H':
                ink = np.isin(reference['band_targets'], [1, 2])
                paper = reference['band_targets'] == 0
                core_roles = reference['core_targets']
                cases['H_legacy_roles'] = stats(prediction, corrected, np.isin(core_roles, [1, 2]), core_roles == 0)
            else:
                ink = reference['ink_targets'] == 1
                paper = reference['ink_targets'] == 0
        cases[name] = stats(prediction, corrected, ink, paper)
        cases[name]['removed_total'] = int(removed.sum())
        checked = ink | paper
        if name == 'H':
            checked |= core_roles != 255
        if name == 'L':
            fragment = LAB / 'role_style_evidence_20261118'
            refs = []
            for ref in read(fragment / 'reviewed.json')['records']:
                path = Path(ref['partial_mask'])
                assert sha(path) == ref['partial_mask_sha256']
                with np.load(path) as data:
                    refs.append((data['core'], data['paper']))
            core = refs[0][0] | refs[1][0]
            near = (refs[0][1] | refs[1][1]) & ~core
            cases['L_close_fragments'] = stats(prediction, corrected, core, near)
            checked |= core | near
            figures.append(('窄线局部', image, prediction, corrected, removed, (23, 239, 76, 315)))
        elif name == 'H':
            figures.append(('已知回归图', image, prediction, corrected, removed, (0, 0, image.shape[1], image.shape[0])))
        cases[name]['removed_outside_checked_pixels'] = int((removed & ~checked).sum())
    lock = read(LAB / 'local_rock_patch_20261117/baseline_lock.json')
    baseline = {k: sha(lock[k]) == lock[k + '_sha256'] for k in ('approved500', 'working489')}
    assert all(baseline.values())
    panel(figures)
    result = {'cases': cases, 'baseline_hashes': baseline, 'new_model_trained': False,
              'independent_validation': False, 'adopted': False, 'robot_connected': False}
    (ROOT / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
