"""Offline sparse variant of the reviewed creek plan; black paths stay byte-identical."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import json
import shutil
from collections import Counter
from unittest.mock import patch
import numpy as np
import cv2
import vision
import imgio
from painting_workflow import save_outputs
from run_color_from_job import load_plan


def spread(strokes, limit):
    if len(strokes) <= limit:
        return strokes
    centers = np.array([np.mean(s['points'], axis=0) for s in strokes])
    lengths = np.array([s['length_mm'] for s in strokes])
    weight = .7 + .3 * np.clip(lengths / max(float(np.median(lengths)), .01), 0, 2)
    chosen = [int(np.argmax(lengths))]
    distances = np.full(len(strokes), np.inf)
    for _ in range(limit - 1):
        distances = np.minimum(distances, np.linalg.norm(centers - centers[chosen[-1]], axis=1))
        score = distances * weight
        score[chosen] = -1
        chosen.append(int(np.argmax(score)))
    # Preserve execution order and the complete shape of each retained stroke.
    return [strokes[i] for i in sorted(chosen)]


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / '输出/群峰溪桥_预览_20260927_235606'
    out = root / '输出/群峰溪桥_疏淡上色_20260928'
    out.mkdir(exist_ok=False)
    plan = json.loads((source / '群峰溪桥_计划.json').read_text(encoding='utf-8'))
    original = copy.deepcopy(plan)
    work, _ = vision.prepare_image(vision.load_image(str(source / '参考原图.png')), work_size=1800)
    old = plan['modules']['color']['strokes']
    selected = []
    rejected = 0
    for color, limit in [('blue', 220), ('yellow', 110), ('green', 45)]:
        candidates = [s for s in old if s['color'] == color]
        if color == 'yellow':
            filtered = []
            for s in candidates:
                pts = np.rint(s['points']).astype(int)
                pixels = work[pts[:, 1], pts[:, 0]].astype(float)
                b, g, r = pixels.T
                if np.mean((g-r > 4) & (g-b > 2)) > .35:
                    rejected += 1
                    continue
                filtered.append(s)
            candidates = filtered
        selected.extend(spread(candidates, limit))
    module = plan['modules']['color']
    module['strokes'] = selected
    module['stroke_count'] = len(selected)
    scale = module['mm_per_px']
    for region in module.get('regions', []):
        strokes = [s for s in selected if s['region_id'] == region['id']]
        region['stroke_count'] = len(strokes)
        region['status'] = 'planned_sparse' if strokes else 'sparse_skip'
        region.pop('planned_coverage_fraction', None)
    plan['sparse_variant'] = dict(source_plan=str(source / '群峰溪桥_计划.json'),
        note='Spatially spaced subset of original paths; pressure, speed, opacity and width unchanged.',
        yellow_on_green_removed=rejected, counts=dict(Counter(s['color'] for s in selected)))
    assert plan['modules']['outline'] == original['modules']['outline']
    old_by_id = {s['id']: s for s in old}
    assert all(s == old_by_id[s['id']] for s in selected)
    prefix = out / '群峰溪桥_疏淡'
    shutil.copy2(source / '群峰溪桥_黑线任务.json', str(prefix) + '_黑线任务.json')
    shutil.copy2(source / '参考原图.png', out / '参考原图.png')
    save_outputs(plan, work, str(prefix) + '_计划.json', str(prefix) + '_预览.png')
    with patch('robot.E6Robot.__init__', side_effect=RuntimeError('Offline preview only')):
        load_plan(str(prefix) + '_计划.json')
    before_length = sum(s['length_mm'] for s in old)
    after_length = sum(s['length_mm'] for s in selected)
    report = dict(before=dict(Counter(s['color'] for s in old)), after=plan['sparse_variant']['counts'],
        before_length_mm=round(before_length, 1), after_length_mm=round(after_length, 1),
        stroke_reduction_percent=round(100*(1-len(selected)/len(old)), 1),
        black_json_byte_identical=(source / '群峰溪桥_黑线任务.json').read_bytes() == Path(str(prefix)+'_黑线任务.json').read_bytes(),
        offline_validation='passed', yellow_on_green_removed=rejected)
    (out / '调整记录.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True))


if __name__ == '__main__':
    main()
