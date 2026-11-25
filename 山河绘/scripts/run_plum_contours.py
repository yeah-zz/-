"""Current-project plum contours; red pigment uses the user-selected green cup.

No flag means offline validation/description. --confirm-site delegates to the
existing guarded execution route; station overrides exist only in this process.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from color_profile import ColorProfile
from run_color_experiment import module_lines
from scripts import run_plum_stage

PLAN = ROOT/'输出'/'梅花描边_20260930'/'plan.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description='当前山河绘梅花任务：黑枝描边；红色借用绿色工位')
    parser.add_argument('--stage', required=True, choices=('ink', 'color'))
    parser.add_argument('--check', action='store_true', help='只离线检查（默认），不连接机械臂')
    parser.add_argument('--confirm-site', action='store_true', help='现场确认后实际运动')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--speed-factor', type=int)
    parser.add_argument('--press-scale', type=float)
    parser.add_argument('--press-floor', type=float)
    args = parser.parse_args(argv)
    if args.check and args.confirm_site:
        parser.error('--check 和 --confirm-site 不能同时使用')
    if not PLAN.is_file():
        parser.error('找不到项目内任务：'+str(PLAN))
    # "red" selects red stroke geometry; "green" selects the physical cup.
    stages = dict(run_plum_stage.STAGES)
    stages['color'] = ('color', 'red', 'green', '红色描边（借用绿色工位）')
    key, color, station, _ = stages[args.stage]
    data = json.loads(PLAN.read_text(encoding='utf-8'))
    module = data['modules'][key]
    if args.press_scale is not None or args.press_floor is not None:
        module, _ = run_plum_stage.retune_press(module, key, color,
            scale=1. if args.press_scale is None else args.press_scale, floor=args.press_floor)
    lines, skipped = module_lines(module, color, ColorProfile().to_dict())
    if skipped or len(lines)!=len(module['strokes']):
        parser.error('存在被边距检查跳过的路径：'+str(skipped))
    print('项目：', ROOT)
    print('任务：', PLAN)
    print('离线路径检查：%d 笔，跳过 0 笔；颜料工位 key=%s' % (len(lines), station))
    if args.stage=='color':
        print('花瓣颜色为红色；使用 green 的杯心、液面和杯沿。该工位本次放红色颜料。')
        kinds = [s.get('kind') for s in module['strokes']]
        print('花瓣轮廓 %d 笔，小苞点缀 %d 笔。' %
              (kinds.count('petal_contour'), kinds.count('bud_dab')))
    forwarded = ['--stage', args.stage, '--plan', str(PLAN)]
    if args.confirm_site:
        forwarded.append('--confirm-site')
    if args.list:
        forwarded.append('--list')
    for option in ('speed_factor', 'press_scale', 'press_floor'):
        value = getattr(args, option)
        if value is not None:
            forwarded.extend(['--'+option.replace('_', '-'), str(value)])
    original_stages = run_plum_stage.STAGES
    try:
        run_plum_stage.STAGES = stages
        return run_plum_stage.main(forwarded)
    finally:
        run_plum_stage.STAGES = original_stages


if __name__=='__main__':
    raise SystemExit(main())
