# -*- coding: utf-8 -*-
"""Evaluate visible gap continuity and ink preservation on source-reviewed regions."""
import json
from pathlib import Path

from evaluate_ink_paper_seams_20261118 import LAB, UPSTREAM, read, sha
from gongbi_training.ink_paper_connected import separate_connected_paper

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = LAB / 'paper_seam_regions_20261118'
TRANSFER = LAB / 'paper_seam_transfer_20261118'
CONTEXT = LAB / 'paper_context_trial_20261118'
PREVIOUS = LAB / 'paper_seam_trial_20261118'
MOD = Path(__file__).resolve().parents[1] / 'gongbi_training'

# Each gap row has an inclusive paper interval and a nearby visible flank on
# each side. Use only the span whose paper and flanks can be traced in the source.
REGIONS = [
    dict(id='L_gap', image='L', title='训练图细线间隙', box=[25,235,61,288], kind='gap',
         sections=[[243,43,44,38,51],[248,42,44,37,49],[253,41,43,36,47],
                   [258,40,42,36,46],[263,39,41,35,45],[268,38,40,34,44],
                   [273,37,39,33,42],[278,36,38,32,41]]),
    dict(id='HK_gap', image='hkma_hemp', title='披麻练习线间隙', box=[143,54,172,90], kind='gap',
         sections=[[62,151,151,149,154],[66,153,153,150,155],[70,154,154,151,156],
                   [74,156,156,154,158],[78,158,158,156,159],[82,159,159,157,160]]),
    dict(id='AAM_gray', image='aam_long_hemp', title='长披麻灰墨区', box=[20,387,64,423], kind='protect',
         polygon=[[37,399],[40,396],[44,397],[44,400],[41,404],[38,410],[36,410],[36,405]]),
    dict(id='AAM_dry_candidate', image='aam_long_hemp', title='疑似笔内亮纹', box=[392,393,445,438], kind='uncertain',
         polygon=[[402,411],[414,409],[419,410],[415,415],[409,420],[405,419]]),
]


def freeze():
    transfer = read(TRANSFER / 'frozen.json')
    artifacts = dict(transfer['artifacts'])
    artifacts[str(MOD / 'ink_paper_connected.py')] = sha(MOD / 'ink_paper_connected.py')
    for directory in (PREVIOUS, CONTEXT):
        p = directory / 'L.npz'
        artifacts[str(p)] = sha(p)
    for row in transfer['examples']:
        p = TRANSFER / (row['id'] + '.npz')
        artifacts[str(p)] = sha(p)
    p = UPSTREAM / 'predictions/L_band_paper.npz'
    artifacts[str(p)] = sha(p)
    assert all(sha(p) == value for p, value in artifacts.items())
    locked = dict(regions=REGIONS, artifacts=artifacts, source_references=transfer['examples'],
                  unit='native image rows and four-connected open paper within reviewed corridors',
                  dry_claim='static bright pattern only; excluded from verified dry-brush scoring',
                  independent=False, reference_scope='assistant source review on previously seen material')
    file = ROOT / 'protocol.json'
    if file.exists():
        assert read(file) == locked
    else:
        file.write_text(json.dumps(locked, ensure_ascii=False, indent=2), encoding='utf8')
    return locked


def data_for(name, references):
    if name == 'L':
        with np.load(UPSTREAM / 'predictions/L_band_paper.npz') as source:
            image, original = source['image'], source['prediction']
        with np.load(PREVIOUS / 'L.npz') as first, np.load(CONTEXT / 'L.npz') as second:
            return image, original, first['roles'], second['roles']
    ref = next(r for r in references if r['id'] == name)
    assert sha(ref['image']) == ref['sha256']
    with Image.open(ref['image']) as source:
        image = np.array(source.convert('RGB'))
    with np.load(TRANSFER / (name + '.npz')) as source:
        return image, source['prediction'], source['seam'], source['context']


def region_masks(shape, spec):
    support = np.zeros(shape, np.uint8)
    if spec['kind'] != 'gap':
        cv2.fillPoly(support, [np.array(spec['polygon'], np.int32)], 1)
        return support.astype(bool), []
    coordinates = np.array(spec['sections'])
    ys = np.arange(coordinates[0, 0], coordinates[-1, 0] + 1)
    steps = np.rint(np.column_stack([np.interp(ys, coordinates[:, 0], coordinates[:, i])
                                     for i in range(1, 5)])).astype(int)
    rows = list(zip(ys, *steps.T))
    for y, low, high, _, _ in rows:
        support[y, low:high + 1] = 1
    return support.astype(bool), rows


def measure(roles, support, rows):
    ink = roles > 0
    if not rows:
        return {'pixels': int(support.sum()), 'ink_pixels': int((ink & support).sum())}
    open_paper = support & ~ink
    rails = []
    for y, low, high, left, right in rows:
        pair = ink[y, left - 1:min(left + 2, low)].any() and ink[y, max(right - 1, high + 1):right + 2].any()
        rails.append(bool(pair))
        if not pair:
            open_paper[y] = False
    count, _, info, _ = cv2.connectedComponentsWithStats(open_paper.astype(np.uint8), connectivity=4)
    longest = int(info[1:, cv2.CC_STAT_HEIGHT].max()) if count > 1 else 0
    full = bool(longest == len(rows))
    return dict(rows=len(rows), rows_with_both_rails=sum(rails),
                open_rows_with_both_rails=sum(bool(open_paper[y].any()) for y, *_ in rows),
                longest_connected_rows=longest, end_to_end_clear=full,
                gap_pixels=int(support.sum()), gap_ink_pixels=int((support & ink).sum()))


def render(panels, destination=None, titles=None):
    width, height = 225, 188
    figure = Image.new('RGB', (width * 5, height * len(panels) + 37), '#f5f1e7')
    pen = ImageDraw.Draw(figure)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    if titles is None:
        titles = ['原图参考', '冻结模型', '纸缝拆分', '周围纸色', '连通检验']
    for i, (spec, image, support, variants) in enumerate(panels):
        for j, mask in enumerate([support, *[v > 0 for v in variants.values()]]):
            visual = image.copy()
            tint = (218, 128, 51) if j == 0 else (34, 137, 111)
            visual[mask] = (0.56 * visual[mask] + 0.44 * np.array(tint)).astype(np.uint8)
            tile = Image.fromarray(visual).crop(spec['box'])
            scale = min((width - 10) / tile.width, (height - 34) / tile.height)
            tile = tile.resize((round(tile.width * scale), round(tile.height * scale)), Image.Resampling.NEAREST)
            x, y = width * j + 6, height * i
            pen.text((x, y + 4), f'{titles[j]} · {spec["title"]}' if j == 0 else titles[j], font=font, fill='#253c35')
            figure.paste(tile, (x, y + 28))
    pen.text((7, figure.height - 30), '橙色为核查范围，绿色为预测墨迹。末行亮纹归属待定。', font=font, fill='#253c35')
    figure.save(destination if destination is not None else ROOT / 'comparison.png')


def main():
    ROOT.mkdir(exist_ok=True)
    locked = freeze()
    names = ['L', *[r['id'] for r in locked['source_references']]]
    outputs = {}
    changes = {}
    for name in names:
        image, original, simple, context = data_for(name, locked['source_references'])
        connected, removed = separate_connected_paper(image, original)
        withheld = (original > 0) & (context == 0) & ~removed
        assert np.array_equal(removed | withheld, (original > 0) & (context == 0))
        variants = dict(frozen=original, seam=simple, context=context, connected=connected)
        outputs[name] = (image, variants)
        changes[name] = {'context_removed': int((removed | withheld).sum()),
                         'connected_removed': int(removed.sum()), 'withheld': int(withheld.sum())}
    results, panels = {}, []
    for spec in locked['regions']:
        image, variants = outputs[spec['image']]
        print(spec['id'], flush=True)
        support, rows = region_masks(image.shape[:2], spec)
        results[spec['id']] = {key: measure(labels, support, rows) for key, labels in variants.items()}
        panels.append((spec, image, support, variants))
    render(panels)
    lock = read(LAB / 'local_rock_patch_20261117/baseline_lock.json')
    baseline = {k: sha(lock[k]) == lock[k + '_sha256'] for k in ('approved500', 'working489')}
    assert all(baseline.values())
    result = dict(regions=results, changes=changes, protocol_sha256=sha(ROOT / 'protocol.json'),
                  baseline_hashes=baseline, independent_validation=False, adopted=False, robot_connected=False)
    (ROOT / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
