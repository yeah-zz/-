"""笔压梯尺：在同一张纸上、当前 PAPER_Z 下逐档试笔宽与含色量。

为什么需要它：--calibrate-z 是"肉眼判断笔尖是否刚好碰到纸面"，对软毫长锋笔
这个判断非常含糊 —— 差 0.3~0.5 mm 肉眼看不出，但落纸就是"有的地方有线、
有的地方全断"。所以改用【墨迹】来定标：同一支笔、同一张纸、同一个 PAPER_Z，
画若干条等压直线，压力从低到高，哪一档开始出现连续实线，一眼就知道。

    python scripts/press_ladder.py                 # 只生成计划 JSON（离线）
    python scripts/press_ladder.py --color red --list   # 红颜料专用（0.30~0.95 mm）；可用 --levels 加到 1.30

真机执行（走和画梅花完全相同的执行器、同样的门限）：

    python scripts/run_plum_stage.py --stage color --plan 输出/press_ladder_red_XXXX.json --strokes-per-load 1 --confirm-site

线画在纸的右下角空处（按当前参考图的构图），不碰已经画好的梅花。
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))

import argparse
import json
from datetime import datetime

import cv2
import numpy as np

from geometry import CanvasMap
from plum_blossom import build_plan
from project_paths import OUTPUT_DIR, PLUM_REFERENCE_IMAGE
from color_profile import ColorProfile
from config import ShuimoConfig

# 每档：压力 mm。从最轻到最重，画在纸上时自下而上排列。
INK_LEVELS = [0.12, 0.25, 0.40, 0.50]
RED_LEVELS = [0.30, 0.50, 0.70, 0.95]

# 右下角空白区（用户坐标 mm）：每条 40 mm 长，行距 10 mm
X0, X1 = 25.0, 65.0
Y_BOTTOM = 58.0
Y_STEP = 10.0
LENGTH_MM = 40.0
POINT_STEP_MM = 1.5


def robot_to_image(mapper, xy_mm):
    """用户坐标 mm -> 图像像素（CanvasMap 只有正变换，这里用同一个单应矩阵求逆）。"""
    inv = np.linalg.inv(mapper.matrix)
    arr = np.asarray(xy_mm, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, inv).reshape(-1, 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description='生成笔压梯尺计划（离线，不发运动）')
    parser.add_argument('--color', choices=('ink', 'red'), default='ink', help='墨色或红颜料')
    parser.add_argument('--levels', help='逗号分隔的笔压 mm；不填时按所选颜色用默认档位')
    parser.add_argument('--list', action='store_true', help='打印每一行对应的笔压')
    args = parser.parse_args(argv)

    levels = RED_LEVELS if args.color == 'red' else INK_LEVELS
    if args.levels:
        levels = [float(v) for v in args.levels.replace('，', ',').split(',') if v.strip()]
    ceiling = ColorProfile().press_mm if args.color == 'red' else ShuimoConfig.PRESS_MAX_MM
    if not levels or not all(0 <= p <= ceiling for p in levels):
        parser.error('笔压必须在 0~%.2f mm 内（%s）' % (ceiling, args.color))

    plan, _work = build_plan(PLUM_REFERENCE_IMAGE)
    mapper = CanvasMap(tuple(plan['image_size']), tuple(plan['image_corners']))
    n_pts = int(round(LENGTH_MM / POINT_STEP_MM)) + 1

    strokes = []
    print('%-4s %-22s %-9s %s' % ('#', 'id', '笔压mm', '纸面坐标 (X0,Y) -> (X1,Y)'))
    for i, press in enumerate(levels):
        y = Y_BOTTOM + i * Y_STEP
        xy = np.column_stack([np.linspace(X0, X1, n_pts), np.full(n_pts, y)])
        pts = robot_to_image(mapper, xy)
        strokes.append(dict(
            id='%s-ladder-%02d' % (args.color, i), name='笔压梯尺 %.2fmm' % press,
            source='press_ladder', color='red' if args.color == 'red' else 'ink',
            points=np.round(pts, 3).tolist(),
            press=[round(float(press), 4)] * n_pts,
            speed=[float(ColorProfile().draw_velocity_percent if args.color == 'red' else 28.0)] * n_pts,
            length_mm=LENGTH_MM, role='press-ladder', brush='fine', lift_mm=4.0, tail_mm=0.0,
            pal_index=0, layer=0, pass_=1,
            press_mean=round(float(press), 4), press_max=round(float(press), 4),
            width_mm=round(float(__import__('shuimo').press_to_width_mm(press)), 3)))
        print('%-4d %-22s %-9.2f (%.0f,%.0f) -> (%.0f,%.0f)'
              % (i, strokes[-1]['id'], press, X0, y, X1, y))

    key = 'color' if args.color == 'red' else 'outline'
    module = dict(format='sparse-color-design-v1' if args.color == 'red' else 'outline-design-v1',
                  ready_for_execution=False,
                  image_size=list(plan['image_size']), image_corners=list(plan['image_corners']),
                  mm_per_px=plan['mm_per_px'], strokes=strokes, stroke_count=len(strokes),
                  execution_profile='existing_ink_profile_unchanged')
    # run_plum_stage.py --plan 读取 modules.color 或 modules.outline。
    plan_out = dict(format='plum-experiment-v1', ready_for_execution=False,
                    image_size=list(plan['image_size']),
                    image_corners=list(plan['image_corners']),
                    mm_per_px=plan['mm_per_px'], stages=[key],
                    modules={key: module},
                    note='press ladder: measured deposition on this paper with this brush and pigment')
    out = OUTPUT_DIR / ('press_ladder_%s_%s.json' % (args.color, datetime.now().strftime('%Y%m%d_%H%M%S')))
    out.write_text(json.dumps(plan_out, ensure_ascii=False, indent=1), encoding='utf-8')
    print()
    print('计划:', out)
    stage = 'color' if args.color == 'red' else 'ink'
    extra = ' --strokes-per-load 1' if args.color == 'red' else ''
    print('跑法: uv run python scripts/run_plum_stage.py --stage %s --plan "%s"%s --confirm-site'
          % (stage, out, extra))
    print('！（会在纸右下角空处真画 %d 条 40mm 直线；每条线旁标注上面的压力档位）' % len(strokes))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
