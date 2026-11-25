"""Offline plum composition preview: no robot connection, no hardware, no motion.

Builds the exact plan that run_plum_experiment.py uses (black branches + red petal
deposits) for project_paths.PLUM_REFERENCE_IMAGE and writes the design JSON and
previews into 输出/. Safe to run any time; nothing is sent to the controller.
"""
# --- path bootstrap: this script lives in scripts/, put the project root on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
from datetime import datetime
from pathlib import Path

import imgio
from project_paths import PLUM_REFERENCE_IMAGE, OUTPUT_DIR
from plum_blossom import build_plan
from painting_workflow import save_outputs
from taper_preview import render_full


def main(argv=None):
    parser = argparse.ArgumentParser(description='离线生成梅花机械臂计划与两种笔痕预览')
    parser.add_argument('--reference', type=Path, default=PLUM_REFERENCE_IMAGE,
                        help='参考图路径；默认使用当前简化梅花图')
    parser.add_argument('--open-area-mm2', type=float,
                        help='可选：手动覆盖自动识别的开放花面积门槛，单位 mm²')
    args = parser.parse_args(argv)
    if args.open_area_mm2 is not None and not 0 < args.open_area_mm2 <= 1000:
        parser.error('--open-area-mm2 必须在 0~1000 mm² 内')
    reference = args.reference
    if not reference.is_file():
        print('找不到参考图: %s' % reference)
        return 2
    plan, work = build_plan(reference, open_area_mm2=args.open_area_mm2)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = OUTPUT_DIR / ('plum_preview_' + stamp + '.json')
    preview = out.with_suffix('.png')
    pressure_preview = out.with_name(out.stem + '_pressure.png')
    if pressure_preview.exists():
        raise ValueError('预览文件已存在: %s' % pressure_preview)
    save_outputs(plan, work, out, preview)
    if not imgio.imwrite(str(pressure_preview), render_full(plan, plan['mm_per_px'])):
        raise OSError('无法写入逐点笔压预览: %s' % pressure_preview)
    ink = plan['modules']['outline']
    red = plan['modules']['color']
    print('参考图      : %s' % reference)
    print('纸张比例    : %.4f mm/px，画布 %.1f x %.1f mm'
          % (plan['mm_per_px'], plan['image_size'][0] * plan['mm_per_px'],
             plan['image_size'][1] * plan['mm_per_px']))
    print('黑枝轮廓笔触: %d 条' % ink['stroke_count'])
    print('红瓣短笔触  : %d 条' % red['stroke_count'])
    print('输出计划    : %s' % out)
    print('输出预览    : %s' % preview)
    print('逐点笔压预览: %s' % pressure_preview)
    print('（离线预览，未连接机械臂；黄色花蕊按设计省略）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
