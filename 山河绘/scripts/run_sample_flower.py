"""真机小样：只画一朵红梅（含附近那一小段枝干），用来在生宣上定参数。

流程与整幅梅花完全一致，只是笔画少：先蘸黑画枝 -> 纸外高位停下等你换笔头 -> 再蘸红画花瓣。
起点必须是纸外高位停车点（先用 go_to_start.py 到位）；起始检查、到位核对、报警处理、
运动锁和 robot_runs 记录都由现有的上色执行器负责，本脚本不放松任何门限。

    python scripts/run_sample_flower.py                     # 花瓣最多的那朵
    python scripts/run_sample_flower.py --flower 2 --margin-mm 30
    python scripts/run_sample_flower.py --no-ink            # 只画红花瓣（不用换笔）
    python scripts/run_sample_flower.py --trunk --trunk-count 2 --margin-mm 25

只读预演（不出图、不动机械臂）：--preview-only
"""
# --- path bootstrap: scripts/ 和项目根都放上 sys.path ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))
_sys.path.insert(0, _HERE)

import argparse
from datetime import datetime

import sample_flower as sf
from color_profile import ColorProfile
from config import ArmConfig, station_approach_z
from park_color_loading import target_pose
from painting_workflow import save_outputs
from plum_blossom import build_plan
from project_paths import OUTPUT_DIR, PLUM_REFERENCE_IMAGE, PROJECT_DIR
from robot import RobotError
from run_color_experiment import module_lines, run as run_stage
from run_plum_experiment import read_station


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--flower', type=int, default=None, help='region_id；不给就取花瓣最多的那朵')
    parser.add_argument('--trunk', action='store_true', help='改画一段主干（看粗细，不是看花）')
    parser.add_argument('--trunk-count', type=int, default=2)
    parser.add_argument('--margin-mm', type=float, default=25.0, help='带上花周围多少毫米的枝干')
    parser.add_argument('--no-ink', action='store_true', help='只画红花瓣')
    parser.add_argument('--preview-only', action='store_true', help='只生成预览和任务，不发运动')
    parser.add_argument('--petal-style', choices=('dots', 'strokes'), default='dots',
                        help='dots=每朵五笔短拖（默认），strokes=旧的多笔铺色')
    parser.add_argument('--petal-spacing', type=float, default=None,
                        help='每瓣内笔迹间距 mm（当前 1.55；纸洇得越厉害间距可增大）')
    parser.add_argument('--petal-max-lines', type=int, default=None,
                        help='每瓣最多几条笔迹（当前最多 4）')
    parser.add_argument('--petal-press', type=float, default=None,
                        help='花瓣峰值下压 mm（五点短拖默认 2.50，旧铺色默认 0.90；不得超过 ColorProfile.press_mm）')
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)

    _os.chdir(PROJECT_DIR)
    try:
        import plum_blossom
        if args.petal_style == 'dots' and (args.petal_spacing is not None or args.petal_max_lines is not None):
            raise ValueError('--petal-spacing/--petal-max-lines 仅用于 --petal-style strokes')
        if args.petal_spacing is not None:
            plum_blossom.PETAL_LINE_SPACING_MM = args.petal_spacing
        if args.petal_max_lines is not None:
            plum_blossom.PETAL_MAX_LINES = args.petal_max_lines
        if args.petal_press is not None:
            if not 0 < args.petal_press <= ColorProfile().press_mm:
                raise ValueError('--petal-press 必须在 0~%.2f mm 内' % ColorProfile().press_mm)
            if args.petal_style == 'dots':
                plum_blossom.PETAL_DOT_PRESS_MM = args.petal_press
            else:
                plum_blossom.PETAL_PRESS_MM = args.petal_press
        if args.petal_style == 'dots':
            print('花瓣参数：每朵五笔；每瓣先压 %.2f mm，再向花心短拖 %.1f mm'
                  % (plum_blossom.PETAL_DOT_PRESS_MM, plum_blossom.PETAL_DOT_LENGTH_MM))
        else:
            print('花瓣参数：间距 %.2f mm / 每瓣最多 %d 条 / 峰值下压 %.2f mm'
                  % (plum_blossom.PETAL_LINE_SPACING_MM, plum_blossom.PETAL_MAX_LINES,
                     plum_blossom.PETAL_PRESS_MM))
        plan, work = build_plan(str(PLUM_REFERENCE_IMAGE), petal_style=args.petal_style)
        sample, tag, size_mm, counts = sf.build_sample(
            plan, flower=args.flower, trunk=args.trunk, trunk_count=args.trunk_count,
            margin_mm=args.margin_mm, no_ink=args.no_ink)
        # 只读校验整条轨迹，越界时在预览阶段就报错，不等输入 RED 后才发现。
        profile = ColorProfile().to_dict()
        if counts['ink']:
            module_lines(sample['modules']['outline'], 'ink', profile)
        if counts['red']:
            module_lines(sample['modules']['color'], 'red', profile)
        out = _os.path.splitext(args.out or str(
            OUTPUT_DIR / ('sample_run_%s_%s' % (tag, datetime.now().strftime('%Y%m%d_%H%M%S')))))[0]
        save_outputs(sample, work, out + '.json', out + '.png')
        print('')
        print(counts['note'])
        print('小样[%s]：窗口 %.0f x %.0f mm | 黑笔 %d 条（%.1f mm）| 红花瓣 %d 条（%.1f mm）'
              % (tag, size_mm[0], size_mm[1], counts['ink'], counts['ink_mm'],
                 counts['red'], counts['red_mm']))
        print('预览:', out + '.png')
        print('任务:', out + '.json')
        red_press = max((max(stroke.get('press', [0])) for stroke in
                         sample['modules']['color']['strokes']), default=0.)
        print('User/Tool: %d %d  纸面 Z: %.4f  站点高位: %.4f  红瓣峰值下压 %.2f mm'
              % (ArmConfig.USER_INDEX, ArmConfig.TOOL_INDEX, ArmConfig.PAPER_Z,
                 station_approach_z(), red_press))
        print('起始和结束位（纸外高位停车点）:', tuple(round(v, 4) for v in target_pose()))
        print('起点要求（2026-09-26 已放宽）：纸外 >=20 mm、姿态 = 标定值、')
        print('  Z 在装载高度 %.4f 的 -60 ~ +20 mm 内；不必凑到停车点 10 mm 内。' % station_approach_z())
        if args.preview_only:
            print('--preview-only：到此结束，未连接机械臂、未发送任何指令。')
            return 0
        if counts['ink']:
            black = read_station('黑色墨盘', 'black')
            print('确认：原纸配准与黑笔高度有效，杯内深度足够且路径清空，现场有人看护。')
            if input('输入 BLACK 开始黑色阶段: ').strip() != 'BLACK':
                return 0
            if run_stage(confirm_site=True, module=sample['modules']['outline'],
                         color='ink', station=black) != 0:
                return 4
            print('黑色完成，机械臂已在纸外高位停止；不会按倒计时自行继续。')
            print('现在换笔头、等墨干，并测量另一个红色杯的坐标。')
            if input('换笔头完成、纸面高度一致且墨已干，输入 CHANGED；其他输入结束: ').strip() != 'CHANGED':
                return 0
        red = read_station('红色颜料杯', 'red')
        print('确认：红杯内有足够液深、笔头不滴液、双手及工具移开运动路径。')
        if input('输入 RED 开始红色阶段: ').strip() != 'RED':
            return 0
        return run_stage(profile=ColorProfile(), confirm_site=True,
                         module=sample['modules']['color'], color='red', station=red)
    except (ValueError, OSError, EOFError, KeyboardInterrupt, RobotError) as exc:
        print('停止:', exc)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
