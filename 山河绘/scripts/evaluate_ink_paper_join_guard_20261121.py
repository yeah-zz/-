# -*- coding: utf-8 -*-
"""Compare paper confirmation on triggered ink-edge edits and gap regressions."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audit_ink_paper_stop_challenge_20261121 import ROOT as CHALLENGE
from evaluate_ink_paper_extension_20261121 import REVIEW, write_json
from evaluate_ink_paper_persistent_stop_20261121 import ROOT as PERSISTENT
from evaluate_ink_paper_regions_20261118 import data_for, region_masks
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_stop_20261121 import REGIONS, VALIDATION, WIDTH
from evaluate_ink_paper_width_20261119 import score
from validate_ink_paper_persistent_stop_20261121 import ROOT as TRANSFER
from gongbi_training.ink_paper_join_guard import protect_ink_joins
from gongbi_training.ink_paper_persistent_stop import persistent_paper_extension


ROOT = LAB/'paper_join_guard_refined_20261121'
INITIAL = LAB/'paper_join_guard_trial_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/ink_paper_join_guard.py'


def prepare_sources():
    known = read(REGIONS/'protocol.json')
    extra = read(WIDTH/'extra_reference.json')
    validation = read(VALIDATION/'protocol.json')
    transfer = read(TRANSFER/'protocol.json')
    meta = {**extra['images'], **validation['images'], **transfer['images'],
            **read(CHALLENGE/'screened_inputs.json')['images']}
    paths = {}
    for name in ['L', *(r['id'] for r in known['source_references']), *extra['images']]:
        paths[name] = (WIDTH/f'{name}.npz', PERSISTENT/f'{name}.npz', 'baseline', 'roles')
    for name in validation['images']:
        paths[name] = (VALIDATION/f'{name}.npz', PERSISTENT/f'{name}.npz', 'original', 'roles')
    for name in transfer['images']:
        paths[name] = (TRANSFER/f'{name}.npz', TRANSFER/f'{name}.npz', 'original', 'persistent')
    for name in ('fuchun600','axe_small'):
        path = CHALLENGE/f'{name}_screened.npz'
        paths[name] = (path, path, 'original', None)
    review_region = dict(next(item for item in known['regions'] if item['id'] == 'HK_gap'),
                         id='HK_review', title='披麻余段复核')
    specs = [*known['regions'], review_region, *extra['regions'], *validation['specs'], *transfer['specs']]
    return known, meta, paths, specs


def comparison(images, rows, path):
    selected = [row for row in rows if (row['label']=='protect' and not row['ink']['persistent'])
                or row['label']=='paper']
    width, height = 260, 156
    board = Image.new('RGB', (width * 4, height * len(selected) + 33), '#f5f1e7')
    pen = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    for index, row in enumerate(selected):
        x, y = row['xy']
        image, variants = images[row['image']]
        crop = (slice(y-6, y+7), slice(x-8, x+9))
        for column, (name, title) in enumerate([(None, '原图'), ('persistent','原持续判断'),
                                                ('span3','两端确认亮纸'), ('edge','再检查灰墨边')]):
            source = image[crop].copy()
            if name:
                ink = variants[name][crop] > 0
                source[ink] = (.58*source[ink] + .42*np.array([34,139,111])).astype(np.uint8)
            tile = Image.fromarray(source).resize((153, 117), Image.Resampling.NEAREST)
            mark = ImageDraw.Draw(tile)
            color = '#ed8a2e' if row['label']=='protect' else '#3aaed1'
            mark.rectangle((72,54,80,62), outline=color, width=2)
            left, top = 8 + column * width, index * height
            pen.text((left, top+4), f"{row['id']} · {title}", font=font, fill='#263c34')
            board.paste(tile, (left, top+27))
    pen.text((8, board.height-30), '橙框为墨边保护点，蓝框为纸缝参考。绿色覆盖为预测墨迹。', font=font, fill='#263c34')
    board.save(path)


def main():
    ROOT.mkdir(exist_ok=True)
    source = read(CHALLENGE/'results.json')
    assert sha(CHALLENGE/'review_protocol.json') == source['protocol_sha256']
    assert sha(MODULE.with_name('ink_paper_persistent_stop.py')) == read(PERSISTENT/'protocol.json')['module']
    known, meta, paths, specs = prepare_sources()
    locked = dict(module=sha(MODULE), audit_sha=sha(CHALLENGE/'results.json'), spans=[1,3],
                  initial_result=sha(INITIAL/'results.json'), dark_edge_check=True,
                  arrays={key: [sha(item[0]), sha(item[1])] for key,item in paths.items()},
                  references=specs, scope='Known challenge points and previous image regressions.')
    path = ROOT/'protocol.json'
    if path.exists():
        assert read(path) == locked
    else:
        write_json(path, locked)
    outputs, all_edits = {}, {}
    for name, (prior, last, original_key, roles_key) in paths.items():
        if name in meta:
            info = meta[name]
            assert sha(info['path']) == info['sha256']
            with Image.open(info['path']) as source_image:
                image = np.array(source_image.convert('RGB'))
        else:
            image, _, *_ = data_for(name, known['source_references'])
        with np.load(prior) as old, np.load(last) as newest:
            original, narrow = old[original_key], old['narrow']
            persistent = newest[roles_key] if roles_key else persistent_paper_extension(image, original, narrow)[0]
        variants = dict(narrow=narrow, persistent=persistent)
        additions = (narrow > 0) & (persistent == 0)
        counts = dict(persistent=int(additions.sum()))
        for key, span in [('span1', 1), ('span3', 3), ('edge', 3)]:
            output, keep, reject = protect_ink_joins(
                image, original, narrow, persistent, span=span, check_dark_edge=key == 'edge',
            )
            np.testing.assert_array_equal(keep | reject, additions)
            np.testing.assert_array_equal(output[~reject], persistent[~reject])
            np.testing.assert_array_equal(output[reject], narrow[reject])
            variants[key] = output
            counts[key] = int(keep.sum())
        with np.load(INITIAL/f'{name}.npz') as earlier:
            for key in ('span1', 'span3'):
                np.testing.assert_array_equal(earlier[key], variants[key])
        outputs[name] = image, variants
        all_edits[name] = counts
        np.savez_compressed(ROOT/f'{name}.npz', **variants)

    points = []
    for item in source['points']:
        x,y = item['xy']
        _, variants = outputs[item['image']]
        ink = {key: bool(labels[y,x]) for key,labels in variants.items()}
        assert ink['persistent'] == item['ink']['persistent']
        points.append(dict(item, ink=ink))
    classes = {}
    for label in ('protect', 'paper', 'unknown'):
        group = [point for point in points if point['label'] == label]
        classes[label] = dict(n=len(group),
                              ink={name: sum(int(point['ink'][name]) for point in group)
                                   for name in ('narrow','persistent','span1','span3','edge')})

    measured = {}
    review = {row[0]:row for row in REVIEW}
    for spec in specs:
        image, variants = outputs[spec['image']]
        mask, rows = region_masks(image.shape[:2], spec)
        if spec['id']=='HK_review':
            rows = [review.get(row[0], row) for row in rows]
            mask[:] = False
            for y,low,high,_,_ in rows:
                mask[y,low:high+1] = True
        measured[spec['id']] = {key:score(labels,mask,rows) for key,labels in variants.items()}
    comparison(outputs, points, ROOT/'point_comparison.png')
    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key: sha(lock[key]) == lock[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    data = dict(points=points, classes=classes, regions=measured, edits=all_edits,
                baseline_hashes=hashes, protocol_sha256=sha(path), independent_validation=False,
                adopted=False, robot_connected=False)
    write_json(ROOT/'results.json', data)
    print(classes)
    for item in points:
        if item['label']=='protect' and not item['ink']['persistent']:
            print(item['id'], item['ink'])
    for name in ('HK_review','L_gap','T_roof','Q_gap','H_join','Q_join'):
        print(name,{k:v.get('longest_8_connected_rows', v.get('ink_pixels')) for k,v in measured[name].items()})


if __name__ == '__main__':
    main()
