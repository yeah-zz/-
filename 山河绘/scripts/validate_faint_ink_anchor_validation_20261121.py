# -*- coding: utf-8 -*-
"""Evaluate frozen faint-ink anchor filtering on a source-reviewed holdout."""
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from screen_faint_ink_anchor_validation_20261121 import LAB, MODULE, ROOT, read, sha, write_json
from gongbi_training.faint_ink_anchor_guard import guard_faint_ink_anchors

# These judgments use the source image and direction-only candidate locations.
# Each entry identifies the one-based group, its clearly visible ink points, and notes.
REVIEW = {
    'mountain': {
        1: ((2, 3, 4, 5), '岩面斜下的灰色细墨带'),
        2: ((), '浅岩面过渡'),
        3: ((), '坡地墨色与底色交叠'),
        5: ((), '岩面灰阶渐变'),
        6: ((), '邻近深墨和淡墨，边界不明'),
    },
    'rounded': {
        1: ((), '较宽浅色弧形痕迹'),
        2: ((2, 3, 4, 5), '深岩线下的斜向细墨中心'),
        3: ((1, 2, 4, 5), '圆石右侧浅弧线'),
        4: ((), '横向淡墨与色面交界'),
        5: ((), '两条较浅灰带靠近，归属不清'),
        6: ((), '较宽浅灰过渡'),
        7: ((2, 3), '弧线外侧的短淡墨带'),
        8: ((), '浅墨边缘与相邻弧线分界不清'),
        10: ((1, 2, 3), '顺着两端墨迹的浅横线'),
        11: ((1, 2, 3), '斜墨带的连续灰脊'),
        14: ((1, 2), '同一条细斜墨带'),
        18: ((), '近邻暗点，残墨与图像噪声未分'),
    },
    'silk': {
        1: ((), '长横带可能是岸线或底材纹理'),
        2: ((), '在设色的远景边界上'),
        3: ((), '横带邻近建筑和岸线，待核'),
    },
    'folded': {1: ((), '岩层浅灰带与纸色交界模糊')},
    'branches': {1: ((), '树枝下方浅灰难与底材区分')},
}


def review_protocol(manifest):
    cases = read(ROOT/'screen.json')
    references = []
    for name, groups in REVIEW.items():
        for group, (positive, note) in groups.items():
            coordinates = cases[name]['groups'][group - 1]['xy']
            assert set(positive).issubset(range(1, len(coordinates) + 1))
            for index, xy in enumerate(coordinates, 1):
                references.append(dict(image=name, group=group, point=index, xy=xy,
                                       label='faint' if index in positive else 'unknown', note=note))
    return dict(screen_sha256=sha(ROOT/'screen.json'),
                screen_protocol_sha256=sha(ROOT/'screen_protocol.json'),
                mask_sha256={name:sha(ROOT/f'{name}.npz') for name in manifest['images']},
                references=references,
                selection='Reviewed original images at direction-only proposals before anchor inference. '
                          'No confidently labeled background locations in the proposal review.')


def render(images, points, destination, *, sections=None, columns=None, standalone=()):
    if sections is None:
        sections = [('mountain', 1, '岩面斜线'), ('rounded', 2, '圆石淡线'), ('rounded', 3, '圆石弧线'),
                    ('rounded', 10, '岩层横线'), ('silk', 1, '横带归属待核')]
    if columns is None:
        columns = ((None, '原图'), ('baseline', '原始提取'), ('direction', '方向筛选'), ('anchor', '起点复核'))
    plate = Image.new('RGB', (1040, 146*len(sections)+29), '#f5f1e7')
    pen = ImageDraw.Draw(plate)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    for row, (name, group, title) in enumerate(sections):
        source, masks = images[name]
        selected = [p for p in points if p['image']==name and p['group']==group]
        xs, ys = zip(*(p['xy'] for p in selected))
        height, width = source.shape[:2]
        left, top = max(0,min(xs)-11),max(0,min(ys)-8)
        right, bottom = min(width,max(xs)+12),min(height,max(ys)+9)
        window = np.s_[top:bottom, left:right]
        scale = min(248//(right-left), 110//(bottom-top))
        for col, (variant, label) in enumerate(columns):
            view = source[window].copy()
            if variant in standalone:
                view[:] = 248
                view[masks[variant][window]>0] = [29,105,77]
            elif variant:
                ink = masks['baseline'][window]>0
                added = (masks[variant][window]>0)&~ink
                view[ink] = .62*view[ink]+.38*np.array([34,137,111])
                view[added] = .27*view[added]+.73*np.array([37,148,202])
            size = ((right-left)*scale, (bottom-top)*scale)
            crop = Image.fromarray(view).resize(size, Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(crop)
            for point in selected:
                x, y = point['xy']
                x, y = (x-left)*scale, (y-top)*scale
                color = {'faint':'#ff932f', 'ink':'#ff932f', 'paper':'#d45085'}.get(point['label'], '#b8a22d')
                draw.rectangle((x,y,x+scale-1,y+scale-1), outline=color)
            x, y = col*260+6, row*146+4
            pen.text((x,y), f'{title} · {label}', font=font, fill='#263c34')
            plate.paste(crop, (x,y+25))
    legend = '橙框为墨线参考，黄框为待核，蓝色为补点，绿色为原始提取。'
    if any(point['label']=='paper' for point in points):
        legend = '橙框为淡墨参考，粉框为纸纹，黄框为待核，蓝色为补点，绿色为原始提取。'
    if standalone:
        legend = '橙框为墨线参考，粉框为留白，黄框为待核。绿色为各栏对应的掩膜或轨迹。'
    pen.text((8,plate.height-24), legend,
             font=font, fill='#263c34')
    plate.save(destination)


def main():
    output = Path(os.environ.get('FAINT_INK_ANCHOR_VALIDATION_OUTPUT', str(ROOT)))
    output.mkdir(parents=True, exist_ok=True)
    manifest = read(ROOT/'screen_protocol.json')
    assert sha(MODULE)==manifest['anchor_rule']
    protocol = review_protocol(manifest)
    protocol_path = output/'review_protocol.json'
    if protocol_path.exists():
        assert read(protocol_path)==protocol
    else:
        write_json(protocol_path, protocol)

    images, changes = {}, {}
    for name, meta in manifest['images'].items():
        assert sha(meta['path'])==meta['sha256'] and sha(meta['source'])==meta['source_sha256']
        with Image.open(meta['path']) as opened:
            image = np.array(opened.convert('RGB'))
        with np.load(ROOT/f'{name}.npz') as cached:
            baseline, direction, additions = (cached[key] for key in ('baseline','direction','additions'))
        anchor, keep, reject = guard_faint_ink_anchors(image, baseline, direction, **manifest['options'])
        np.testing.assert_array_equal(keep | reject, additions)
        np.testing.assert_array_equal(anchor[~reject], direction[~reject])
        assert not anchor[reject].any()
        changes[name] = dict(direction=int(additions.sum()), anchor=int(keep.sum()))
        images[name] = image, dict(baseline=baseline, direction=direction, anchor=anchor)
        np.savez_compressed(output/f'{name}_anchor.npz', anchor=anchor, keep=keep, reject=reject)

    points = []
    for point in protocol['references']:
        x,y = point['xy']
        _, masks = images[point['image']]
        ink = {key:bool(mask[y,x]) for key,mask in masks.items()}
        assert not ink['baseline'] and ink['direction']
        points.append(dict(point, ink=ink))

    metrics = {}
    for label in ('faint', 'unknown'):
        items = [point for point in points if point['label']==label]
        metrics[label] = dict(n=len(items), retained=sum(p['ink']['anchor'] for p in items))
    render(images, points, output/'comparison.png')
    locks = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:sha(locks[key])==locks[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    result = dict(points=points, metrics=metrics, changes=changes, baseline_hashes=hashes,
                  protocol_sha256=sha(protocol_path), independent_trial=True, independent_acceptance=False,
                  adopted=False, robot_connected=False)
    write_json(output/'results.json', result)
    print(metrics)
    print(changes)


if __name__=='__main__':
    main()
