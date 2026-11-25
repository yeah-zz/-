# -*- coding: utf-8 -*-
"""Review true triggered edits from frozen paper-gap runs."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_ink_paper_extension_20261121 import write_json
from evaluate_ink_paper_persistent_stop_20261121 import ROOT as PERSISTENT
from evaluate_ink_paper_regions_20261118 import data_for
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_stop_20261121 import EXT, REGIONS, ROOT as STOP, VALIDATION, WIDTH
from validate_ink_paper_persistent_stop_20261121 import ROOT as TRANSFER
from gongbi_training.ink_paper_persistent_stop import persistent_paper_extension
from gongbi_training.ink_paper_stop import guarded_paper_extension


ROOT = LAB/'paper_stop_challenge_20261121'
MODEL_ROOT = Path(__file__).resolve().parents[1]/'gongbi_training'
# One-based point indices matching the source-only review boards.
# Protection includes visibly shaded ink edges; it is deliberately conservative.
PROTECT = {
    'L09': ([7, 8], '深灰基部的连续墨色'),
    'L11': ([2], '浅条末端已进入相连灰墨'),
    'L15': ([1, 2], '深墨环内的灰墨过渡'),
    'L17': ([2, 3], '深墨与灰边相连处'),
    'L18': ([2], '倾斜墨线接入横向灰边'),
    'L19': ([2], '竖向深墨旁的灰色过渡'),
    'Q10': ([1], '顺着右侧墨线的灰墨边'),
    'Q11': ([1], '墨带内的淡墨过渡'),
    'Q12': ([2], '相邻墨线下端汇合处的灰边'),
    'T01': ([1], '屋面上方树枝的灰边'),
    'T04': ([1], '排线和横向墨线接合的灰边'),
}
PAPER = {
    'T05': ([5, 6], '可沿相邻屋面墨线之间的亮带连续定位'),
    'T06': ([1, 2], '两根横向墨线间连续亮带'),
    'T07': ([1], '竖线左侧与水平横梁之间的亮纸'),
}


def annotations(candidates):
    reviewed = []
    for group, spec in candidates.items():
        protective, protect_reason = PROTECT.get(group, ([], ''))
        paper, paper_reason = PAPER.get(group, ([], ''))
        assert not set(protective) & set(paper)
        assert all(0 < index <= len(spec['coords']) for index in [*protective, *paper])
        for index, (x, y) in enumerate(spec['coords'], start=1):
            if index in protective:
                label, reason = 'protect', protect_reason
            elif index in paper:
                label, reason = 'paper', paper_reason
            else:
                label, reason = 'unknown', '墨边、笔内亮纹或纸缝的归属待核'
            reviewed.append(dict(id=f'{group}-{index}', group=group, image=spec['image'],
                                 xy=[x, y], label=label, reason=reason))
    return reviewed


def comparison(sources, points):
    shown = [row for row in points if row['label'] == 'protect'
             and row['group'] in ('L09','L11','Q11','Q12','T01','T04')]
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    board = Image.new('RGB', (1160, 170 * len(shown) + 36), '#f5f1e7')
    pen = ImageDraw.Draw(board)
    for index, item in enumerate(shown):
        x, y = item['xy']
        image, versions = sources[item['image']]
        origin = (x - 8, y - 7)
        box = (*origin, x + 9, y + 8)
        scale = 9
        for column, (label, value) in enumerate([('原图', None), ('原单端延伸', versions['extended']),
                                                  ('前看两步', versions['guard2']), ('最新规则', versions['persistent'])]):
            left, top = column * 290 + 7, index * 170 + 2
            title = f"{item['id']} · {label}"
            pen.text((left, top + 1), title, font=font, fill='#263c34')
            crop = image[box[1]:box[3], box[0]:box[2]].copy()
            if value is not None:
                ink = value[box[1]:box[3], box[0]:box[2]] > 0
                crop[ink] = (.60 * crop[ink] + .40 * np.array([34, 139, 111])).astype(np.uint8)
            tile = Image.fromarray(crop).resize((crop.shape[1] * scale, crop.shape[0] * scale), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(tile)
            u, v = (x - origin[0]) * scale, (y - origin[1]) * scale
            draw.rectangle((u, v, u + scale - 1, v + scale - 1), outline='#fa8f30', width=2)
            board.paste(tile, (left, top + 25))
    pen.text((7, board.height - 30), '橙框为需保护的灰墨参考点，绿色覆盖为预测墨迹。', font=font, fill='#263c34')
    board.save(ROOT/'triggered_protection.png')


def main():
    candidate_path = ROOT/'challenge_candidates.json'
    refs = read(REGIONS/'protocol.json')
    source_protocol = read(ROOT/'screened_inputs.json')
    rows = annotations(read(candidate_path))
    assert sha(MODEL_ROOT/'ink_paper_persistent_stop.py') == read(PERSISTENT/'protocol.json')['module']
    assert sha(MODEL_ROOT/'ink_paper_stop.py') == read(STOP/'protocol.json')['module']
    sources = {
        'qingpu20': read(VALIDATION/'protocol.json')['images']['qingpu20'],
        'tang': read(TRANSFER/'protocol.json')['images']['tang'],
        **source_protocol['images'],
    }
    paths = dict(L=WIDTH/'L.npz', qingpu20=VALIDATION/'qingpu20.npz', tang=TRANSFER/'tang.npz',
                 axe_small=ROOT/'axe_small_screened.npz', fuchun600=ROOT/'fuchun600_screened.npz')
    locked = dict(candidates=sha(candidate_path), source_arrays={key: sha(path) for key, path in paths.items()},
                  annotations=rows, scope='Selected on old proposals; source-image analyst review, not independent gold.')
    protocol = ROOT/'review_protocol.json'
    if protocol.exists():
        assert read(protocol) == locked
    else:
        write_json(protocol, locked)

    outputs = {}
    for name, path in paths.items():
        if name == 'L':
            image, _, *_ = data_for(name, refs['source_references'])
        else:
            meta = sources[name]
            assert sha(meta['path']) == meta['sha256']
            with Image.open(meta['path']) as source:
                image = np.array(source.convert('RGB'))
        with np.load(path) as cached:
            original = cached['baseline' if name == 'L' else 'original']
            narrow = cached['narrow']
            if name != 'L':
                extended = cached['extended']
        if name == 'L':
            with np.load(EXT/'L.npz') as cached:
                extended = cached['roles']
        previous, _, _ = guarded_paper_extension(image, original, narrow)
        latest, accepted, rejected = persistent_paper_extension(image, original, narrow)
        np.testing.assert_array_equal(accepted | rejected, (narrow > 0) & (extended == 0))
        outputs[name] = image, dict(narrow=narrow, extended=extended, guard2=previous, persistent=latest)
    outcomes = []
    for row in rows:
        x, y = row['xy']
        _, versions = outputs[row['image']]
        flags = {name: bool(roles[y, x] > 0) for name, roles in versions.items()}
        assert flags['narrow'] and not flags['extended']
        outcomes.append(dict(row, ink=flags))
    stats = {}
    for name in ('protect', 'paper', 'unknown'):
        judged = [row for row in outcomes if row['label'] == name]
        stats[name] = dict(n=len(judged), kept={key: sum(int(row['ink'][key]) for row in judged)
                                             for key in ('narrow', 'extended', 'guard2', 'persistent')})
    comparison(outputs, outcomes)
    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key: sha(lock[key]) == lock[key+'_sha256'] for key in ('approved500', 'working489')}
    assert all(hashes.values())
    write_json(ROOT/'results.json', dict(points=outcomes, stats=stats, protocol_sha256=sha(protocol), baseline_hashes=hashes,
                                       independent_validation=False, adopted=False, robot_connected=False))
    print(stats)
    for row in outcomes:
        if row['label'] != 'unknown':
            print(row['id'], row['label'], row['ink']['guard2'], row['ink']['persistent'])


if __name__ == '__main__':
    main()
