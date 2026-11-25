"""Generate and validate plum contour modules; no hardware connection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime
import cv2
import numpy as np

from color_profile import ColorProfile
from project_paths import PAINTING_DIR, OUTPUT_DIR
from plum_contours import build_plan, render, save_image
from run_color_experiment import module_lines


def main(argv=None):
    parser = argparse.ArgumentParser(description='梅花纯描边离线任务：粗枝双钩、花瓣红勾、小苞点缀')
    parser.add_argument('--image', type=Path,
                        default=PAINTING_DIR/'参考图'/'plum_contour_reference_20260930.png')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    out = args.output or OUTPUT_DIR/('plum_contour_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if out.exists():
        raise ValueError('Output folder already exists: '+str(out))
    plan, work = build_plan(args.image)
    plan['station_keys'] = {'outline': 'black', 'color': 'green'}
    plan['station_note'] = 'User requested red pigment to use the green physical station; run with run_plum_contours.py'
    audit, motions = {}, {}
    for key, color in [('outline', 'ink'), ('color', 'red')]:
        module = plan['modules'][key]
        lines, skipped = module_lines(module, color, ColorProfile().to_dict())
        if skipped or len(lines)!=module['stroke_count']:
            raise ValueError('Paths were skipped: '+str(skipped))
        kinds = {}
        for s in module['strokes']:
            kinds[s['kind']] = kinds.get(s['kind'], 0)+1
        audit[key] = dict(stroke_count=len(lines), skipped=skipped, kinds=kinds,
            length_mm=round(sum(s['length_mm'] for s in module['strokes']), 1),
            press_range_mm=[min(min(s['press']) for s in module['strokes']),
                            max(max(s['press']) for s in module['strokes'])])
        motions[key] = dict(format='plum-contour-motion-audit-v1', color=color,
            note='Offline XYZ contact/lift sequences; actual execution uses the design module and cup workflow',
            lines=lines)
    audit['flower_count'] = sum(r['kind']=='flower' for r in plan['modules']['color']['regions'])
    audit['bud_count'] = sum(r['kind']=='bud' for r in plan['modules']['color']['regions'])
    audit['station_keys'] = plan['station_keys']
    audit['preview_limit'] = 'Width from current black-ink brush table; red diffusion and actual transfer unverified'
    out.mkdir(parents=True)
    for name, data in [('plan', plan), ('validation', audit), *plan['modules'].items(),
                       ('outline_motion', motions['outline']), ('color_motion', motions['color'])]:
        (out/(name+'.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    save_image(out/'source_fitted.png', work)
    save_image(out/'preview.png', render(plan, work))
    save_image(out/'paths.png', render(plan, work, calibrated=False))
    for key in plan['stages']:
        save_image(out/(key+'.png'), render(plan, work, stage=key))
    panels = []
    for title, panel in [('SOURCE / uniformly fitted', work), ('PATHS / geometry only', render(plan, work, False)),
                          ('BRUSH WIDTH / estimated', render(plan, work))]:
        bar = np.full((38, panel.shape[1], 3), 250, np.uint8)
        cv2.putText(bar, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .7, (45,45,45), 1, cv2.LINE_AA)
        panels.append(np.vstack([bar, panel]))
    save_image(out/'comparison.png', np.hstack(panels))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print('输出:', out.resolve())
    return 0


if __name__=='__main__':
    raise SystemExit(main())
