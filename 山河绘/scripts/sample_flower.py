"""小样计划：把整幅梅花里的一朵花、或一段主干单独抽出来（预览 / 真机试画都用它）。

只做规划 + 预览，不连机械臂、不发运动。

    python scripts/sample_flower.py                        # 取花瓣最多的那朵花（含附近枝干）
    python scripts/sample_flower.py --list                 # 列出各朵花
    python scripts/sample_flower.py --trunk                # 改取最粗的一段主干
    python scripts/sample_flower.py --list                 # 先查当前图的 region_id
    python scripts/sample_flower.py --flower 2 --no-ink    # 只要红花瓣

真机试画用 scripts/run_sample_flower.py（同一个 --flower/--trunk 参数）。
"""
# --- path bootstrap: this script lives in scripts/, put the project root on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import math
from collections import defaultdict

import numpy as np

from project_paths import PLUM_REFERENCE_IMAGE, OUTPUT_DIR
from plum_blossom import build_plan
from painting_workflow import save_outputs


def flower_groups(module):
    """按 region_id（= 一朵花）分组红笔触。"""
    groups = defaultdict(list)
    for s in module['strokes']:
        groups[int(s.get('region_id', 0))].append(s)
    return groups


def bounds(items):
    pts = np.concatenate([np.asarray(s['points'], float) for s in items])
    return pts.min(axis=0), pts.max(axis=0)


def describe(groups, scale):
    rows = []
    for fid, strokes in groups.items():
        lo, hi = bounds(strokes)
        rows.append(dict(id=fid, strokes=len(strokes),
                         length_mm=sum(float(s['length_mm']) for s in strokes),
                         size_mm=(hi - lo) * scale, centre=(lo + hi) / 2))
    return rows


def in_window(points, lo, hi):
    pts = np.asarray(points, float)
    return (pts[:, 0].max() >= lo[0] and pts[:, 0].min() <= hi[0]
            and pts[:, 1].max() >= lo[1] and pts[:, 1].min() <= hi[1])


def build_sample(plan, flower=None, trunk=False, trunk_count=4, margin_mm=25., no_ink=False):
    """返回 (小样计划, tag, 窗口尺寸mm, 笔数统计)。"""
    red, ink = plan['modules']['color'], plan['modules']['outline']
    scale = plan['mm_per_px']
    groups = flower_groups(red)
    rows = describe(groups, scale)

    picked_red, picked_ink, tag = [], [], 'empty'
    if trunk:
        wide = sorted(ink['strokes'], key=lambda s: -float(s.get('image_width_mm', 0)))[:max(1, trunk_count)]
        lo, hi = bounds(wide)
        margin_px = margin_mm / scale
        lo, hi = lo - margin_px, hi + margin_px
        picked_ink = [s for s in ink['strokes'] if in_window(s['points'], lo, hi)]
        picked_red = [s for s in red['strokes'] if in_window(s['points'], lo, hi)]
        tag = 'trunk'
        width_txt = '/'.join('%.1f' % float(s.get('image_width_mm', 0)) for s in wide)
        extra = '主干窗口：最粗 %d 条（图像宽度 %s mm）外扩 %.0f mm' % (len(wide), width_txt, margin_mm)
    else:
        if flower is None:
            chosen = max(rows, key=lambda r: (r['strokes'], r['length_mm']))
        else:
            chosen = next((r for r in rows if r['id'] == flower), None)
            if chosen is None:
                raise ValueError('没有 region_id=%s 这朵花' % flower)
        picked_red = groups[chosen['id']]
        lo, hi = bounds(picked_red)
        margin_px = margin_mm / scale
        lo, hi = lo - margin_px, hi + margin_px
        picked_ink = [] if no_ink else [s for s in ink['strokes'] if in_window(s['points'], lo, hi)]
        tag = 'flower%d' % chosen['id']
        extra = '花 region %d：花瓣笔 %d 条，跨度 %.1f x %.1f mm' % (
            chosen['id'], chosen['strokes'], chosen['size_mm'][0], chosen['size_mm'][1])

    geometry = dict(image_size=plan['image_size'], image_corners=plan['image_corners'], mm_per_px=scale)
    sample = dict(format='plum-experiment-v1', ready_for_execution=False, **geometry,
                  stages=['outline', 'color'],
                  modules=dict(outline=dict(ink, strokes=picked_ink, stroke_count=len(picked_ink)),
                               color=dict(red, strokes=picked_red, stroke_count=len(picked_red))),
                  note='%s sample; one-stroke-per-petal petals, thin skeleton branches' % tag)
    counts = dict(ink=len(picked_ink), red=len(picked_red),
                  ink_mm=sum(float(s['length_mm']) for s in picked_ink),
                  red_mm=sum(float(s['length_mm']) for s in picked_red),
                  note=extra)
    return sample, tag, (hi - lo) * scale, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--reference', default=str(PLUM_REFERENCE_IMAGE))
    parser.add_argument('--flower', type=int, default=None, help='region_id；不给就取花瓣最多的那朵')
    parser.add_argument('--trunk', action='store_true', help='改取最粗的一段主干')
    parser.add_argument('--trunk-count', type=int, default=4, help='用最粗的几条笔迹圈定主干窗口')
    parser.add_argument('--margin-mm', type=float, default=25.0, help='窗口外扩多少毫米')
    parser.add_argument('--no-ink', action='store_true', help='只要红花瓣')
    parser.add_argument('--list', action='store_true', help='只列出各朵花，不出图')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)

    plan, work = build_plan(args.reference)
    scale = plan['mm_per_px']
    rows = describe(flower_groups(plan['modules']['color']), scale)
    print('整幅：黑线 %d 条 / 红笔 %d 条，共 %d 朵（region_id）'
          % (plan['modules']['outline']['stroke_count'],
             plan['modules']['color']['stroke_count'], len(rows)))
    for r in sorted(rows, key=lambda r: r['id']):
        print('  region %2d: 花瓣笔 %2d 条，笔长 %6.1f mm，跨度 %4.1f x %4.1f mm'
              % (r['id'], r['strokes'], r['length_mm'], r['size_mm'][0], r['size_mm'][1]))
    if args.list:
        return 0

    sample, tag, size_mm, counts = build_sample(
        plan, flower=args.flower, trunk=args.trunk, trunk_count=args.trunk_count,
        margin_mm=args.margin_mm, no_ink=args.no_ink)
    out = _os.path.splitext(args.out or str(OUTPUT_DIR / ('sample_' + tag)))[0]
    save_outputs(sample, work, out + '.json', out + '.png')
    print('')
    print(counts['note'])
    print('小样[%s]：窗口 %.0f x %.0f mm | 黑笔 %d 条（%.1f mm）| 红花瓣 %d 条（%.1f mm）'
          % (tag, size_mm[0], size_mm[1], counts['ink'], counts['ink_mm'], counts['red'], counts['red_mm']))
    from run_plum_experiment import read_station
    red_load = read_station('红色颜料杯', 'red').strokes_per_load
    ink_load = read_station('黑色墨盘', 'black').strokes_per_load
    print('预估蘸色次数约 %d 次；真机试画约 %d~%d 分钟（含换笔头）'
          % (math.ceil(counts['red'] / red_load) + math.ceil(counts['ink'] / ink_load),
             max(1, counts['ink'] // 20), max(2, counts['red'] // 20 + 2)))
    print('预览:', out + '.png')
    print('任务:', out + '.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
