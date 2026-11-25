"""Load an existing workflow JSON and draw only operator-selected colors."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from project_paths import PROJECT_DIR
from geometry import CanvasMap
from color_profile import ColorProfile
from run_color_experiment import module_lines, reload_stroke_limit
from run_mountain_balanced import load_reviewed
from color_selection import paint_selected_colors


def load_plan(path, ink_job=None):
    path = Path(path)
    plan = json.loads(path.read_text(encoding='utf-8-sig'))
    if plan.get('format') != 'painting-workflow-design-v1':
        raise ValueError('请选择完整“计划.json”，不是黑线任务JSON。')
    # Old plans store their calibration in the paired black job, so verify it
    # without executing any black strokes. Do not infer missing paper settings.
    if ink_job is None:
        if not path.stem.endswith('_计划'):
            raise ValueError('无法定位配套黑线任务，请用 --ink-job 指定。')
        ink_job = path.with_name(path.stem[:-3] + '_黑线任务.json')
    return validate_plan(plan, ink_job)


def validate_plan(plan, ink_job):
    module = plan['modules']['color']
    reload_stroke_limit(module)
    _, _, _, meta = load_reviewed(ink_job)
    if (plan.get('image_size') != meta['image_size']
            or plan.get('reference_image') != meta['image']):
        raise ValueError('计划与配套黑线任务不是同一张图。')
    mapper = CanvasMap(module['image_size'], module['image_corners'])
    if (module['image_size'] != plan['image_size']
            or module['image_corners'] != plan['image_corners']
            or abs(mapper.mm_per_px-module['mm_per_px']) > 1e-8):
        raise ValueError('上色与黑线的画布映射不一致。')
    expected = CanvasMap(meta['image_size'])
    if not np.allclose(mapper.matrix, expected.matrix, rtol=0, atol=1e-8):
        raise ValueError('上色与黑线坐标变换不一致。')
    profile = ColorProfile(**module['motion_candidate']).to_dict()
    for color in set(s['color'] for s in module['strokes']):
        if color not in ('red','yellow','blue','blue2','green'): raise ValueError('暂不支持该颜色：'+color)
        lines, skipped = module_lines(module,color,profile)
        if skipped or not lines: raise ValueError('上色路径超出当前画布或为空。')
    return plan


def main(argv=None):
    from launcher import setup_console
    setup_console()
    os.chdir(PROJECT_DIR)
    parser = argparse.ArgumentParser(description='读取完整计划JSON，只画你选择的颜色，不重画黑线。')
    parser.add_argument('--job', required=True)
    parser.add_argument('--ink-job', help='配套黑线任务，用于核对标定，不执行黑线')
    parser.add_argument('--check', action='store_true', help='只离线校验')
    parser.add_argument('--no-progress', action='store_true', help='不保存上色续画进度')
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.job, args.ink_job)
        if args.no_progress:
            plan['save_progress'] = False
        print('任务:',args.job)
        print('彩色笔触:',len(plan['modules']['color']['strokes']))
        print('本命令只上色。请确认这是对应的已勾线纸张，位置不变。')
        if args.check:
            print('离线校验通过，没有连接机械臂。')
            return 0
        return paint_selected_colors(plan)
    except (ValueError,KeyError,TypeError,OSError,EOFError,KeyboardInterrupt) as exc:
        print('停止:',exc)
        return 2


if __name__ == '__main__': raise SystemExit(main())
