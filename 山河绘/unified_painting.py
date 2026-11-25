"""Shared reference -> black outlines -> supervised color workflow."""
from datetime import datetime
from pathlib import Path
import json
import os

import cv2
import numpy as np
import imgio
import vision
import watercolor
import ink_control
import job
import shuimo
from config import ArmConfig, CanvasConfig, ShuimoConfig, PALETTE
from geometry import CanvasMap, check_waypoints
from color_profile import ColorProfile
from painting_workflow import save_outputs
from project_paths import PROJECT_DIR, OUTPUT_DIR, ROBOT_LOG_DIR


def outline_image(work, scale, pad=None):
    from reference_lines import extract
    return extract(work, scale, pad=pad)


def build_plan(reference, subject=None, outline_engine='source-constrained-v2'):
    """One planner for every reference; subject is a legacy metadata-only argument."""
    # One preparation and one mapping for BOTH passes, including padding.
    work, info = vision.prepare_image(vision.load_image(str(reference)), work_size=1800)
    if info.get('used_corners'):
        raise ValueError('当前黑线执行器使用整图映射；请先裁好输入图，不要使用自定义 IMAGE_CORNERS。')
    mapper = CanvasMap(info['prepared_size'], info['image_corners'])
    if outline_engine == 'source-constrained-v2':
        from source_ink import build_strokes
        strokes, stats = build_strokes(work, mapper.mm_per_px, pad=info.get('pad'))
    elif outline_engine == 'legacy':
        # Explicit comparison/recovery option; existing saved jobs are untouched.
        from original_d_adapter import build_strokes as build_original
        from scripts.preview_contrast_landscape import width_strokes
        _legacy, stats = build_original(work, mapper.mm_per_px, pad=info.get('pad'),
                                        source_image=vision.load_image(str(reference)))
        strokes, new_stats = width_strokes(work, mapper.mm_per_px, pad=info.get('pad'))
        stats.update(new_stats)
        stats['detail'] = 'width-layered; legacy tracer used for comparison stats only'
    else:
        raise ValueError('Unknown outline engine: ' + str(outline_engine))
    print('提线器 %s: %d 条' % (stats['algorithm'], len(strokes)))
    lineart = np.full_like(work, 255)
    for stroke in strokes:
        cv2.polylines(lineart, [np.rint(stroke['points']).astype(np.int32)],
                      False, (0, 0, 0), 1, cv2.LINE_AA)
    if not strokes:
        raise ValueError('未提取到可画的轮廓，请使用浅底、线条清楚的参考图。')
    policy = ink_control.InkPolicy()
    strokes = ink_control.split_long_strokes(strokes, mapper.mm_per_px, policy)
    strokes = ink_control.order_for_ink(strokes, mapper.mm_per_px, policy)
    for i, stroke in enumerate(strokes):
        stroke['id'] = 'outline-%05d' % i
    steps, ink_stats = ink_control.build_plan(strokes, shuimo.ink_palette_index(), mapper.mm_per_px)
    conditioned = [step['stroke'] for step in steps if step['kind'] == 'stroke']
    # Reserve the planned black footprint as well as original black pixels.
    import landscape_color
    ink_mask = landscape_color.analyze(work, mapper.mm_per_px)[0].astype(np.uint8)
    for stroke in conditioned:
        cv2.polylines(ink_mask, [np.rint(stroke['points']).astype(np.int32)], False, 1,
                      max(1, int(np.ceil(stroke['width_mm'] / mapper.mm_per_px))))
    colors, regions = landscape_color.plan_paths(
        work, mapper.mm_per_px, width=1., ink_mask=ink_mask.astype(bool))
    from green_accents import plan_accents
    colors.extend(plan_accents(work, mapper.mm_per_px, conditioned))
    # The old 2.50 mm profile is specific to pressed plum dots, not generic washes.
    profile = ColorProfile(press_mm=min(.22, float(ShuimoConfig.PRESS_MAX_MM)))
    geometry = dict(image_size=list(info['prepared_size']), mm_per_px=mapper.mm_per_px,
                    image_corners=mapper.image_corners)
    outline = dict(format='outline-design-v1', strokes=conditioned, **geometry)
    color = dict(format='sparse-color-design-v1', strokes=colors, regions=regions,
                 palette_bgr=watercolor.PALETTE, motion_candidate=profile.to_dict(), **geometry)
    # Apply the existing color executor's margin before previewing, so preview
    # does not promise paths which the executor would subsequently omit.
    from run_color_experiment import module_lines
    allowed = set()
    skipped = []
    for name in (*watercolor.COLOR_ORDER, 'green'):
        lines, omitted = module_lines(color, name, profile.to_dict())
        allowed.update(line['id'] for line in lines)
        skipped.extend(omitted)
    color['strokes'] = [s for s in colors if s['id'] in allowed]
    color['skipped_margin_ids'] = skipped
    color['stroke_count'] = len(color['strokes'])
    plan = dict(format='painting-workflow-design-v1', ready_for_execution=False,
                subject='reference', reference_image=str(Path(reference).resolve()),
                stages=['outline', 'color'], modules=dict(outline=outline, color=color),
                sequence=[dict(kind='run_module', module='outline'),
                          dict(kind='observe_previous_stage_dry', automatic_seconds=None),
                          dict(kind='manual_wash_wipe_and_load', next_stage='color'),
                          dict(kind='run_module', module='color')],
                note='Shared reference tracing and area-driven color planning for all subjects; optical width and opacity are approximate.',
                **geometry)
    waypoints, _ = job.steps_to_waypoints(steps, mapper)
    from run_mountain_v4_robot import safe_transfers
    waypoints = safe_transfers(waypoints)
    ok, problems = check_waypoints(waypoints)
    if not ok:
        raise ValueError('黑线轨迹超出可达范围: ' + str(problems))
    meta = dict(mode='shuimo-framework', profile='mountain', ink_policy='balanced',
                image=str(Path(reference).resolve()), image_size=geometry['image_size'],
                canvas_corners=[list(p) for p in CanvasConfig.ROBOT_CORNERS],
                paper_z=ArmConfig.PAPER_Z, strokes=len(strokes), quality_stats=stats,
                ink_execution=dict(base_strokes=strokes, policy=ink_stats['ink_policy'],
                    mm_per_px=mapper.mm_per_px,
                    calibration=dict(rpy=list(ArmConfig.RPY), user=ArmConfig.USER_INDEX,
                        tool=ArmConfig.TOOL_INDEX, speed_factor=ArmConfig.SPEED_FACTOR,
                        palette=PALETTE)))
    return plan, work, lineart, waypoints, meta


def color_station(name, ask=input):
    """Use only this color's measured cup; never borrow another color's XY/Z."""
    from pigment_station import PigmentStation, read_measurements
    saved = read_measurements()
    if (saved['user'] != ArmConfig.USER_INDEX or saved['tool'] != ArmConfig.TOOL_INDEX
            or saved['rpy'] != list(ArmConfig.RPY)):
        raise ValueError('颜料杯记录的坐标系/姿态已改变，请重新标定。')
    label = {'blue2': '淡蓝', 'blue': '蓝色', 'yellow': '黄色', 'red': '红色', 'green': '绿色'}[name]
    values = saved.get(name)
    if values and all(values.get(k) is not None for k in ('x', 'y', 'surface_z', 'rim_z')):
        print(label + '杯的已保存测量值:', values)
        reply = ''  # Use this color's saved, validated station without another prompt.
    else:
        reply = ask(label + '杯未标定，请输入实测 X Y 液面Z 杯沿Z（留空取消）：').strip()
        if not reply:
            raise ValueError(label + '没有杯位，已取消，不会开始勾线。')
    if reply:
        coords = list(map(float, reply.split()))
        if len(coords) != 4:
            raise ValueError('需要四个实测数值。')
        values = dict(zip(('x', 'y', 'surface_z', 'rim_z'), coords))
    keys = ('x', 'y', 'surface_z', 'rim_z', 'dip_mm', 'dwell_s', 'drip_s', 'strokes_per_load',
            'level_compensation_enabled', 'level_every_dips', 'level_step_mm', 'level_max_mm',
            'level_every_length_mm')
    return PigmentStation(**{k: values[k] for k in keys if k in values}).validate()


def execute_stages(plan, ink_path, stations, ask=input, ink_runner=None, color_runner=None, start_confirmed=False):
    """Only a successful black pass and explicit dry confirmation permit color."""
    from run_mountain_balanced import execute_reviewed_job
    from run_color_experiment import run
    ink_runner = ink_runner or execute_reviewed_job
    color_runner = color_runner or run
    if not start_confirmed and ask('确认预览、新纸、笔尖高度和墨盘，输入 BLACK 开始勾线；其他输入只保留预览：').strip() != 'BLACK':
        return 0
    result = ink_runner(ink_path)
    if result != 0:
        return result
    module = plan['modules']['color']
    if not module['strokes']:
        print('黑线完成；输入图没有可执行的上色笔触，结束。')
        return 0
    print('黑线已完成并抬笔，等待你手动换色。')
    from color_selection import paint_selected_colors
    return paint_selected_colors(plan, stations=stations, ask=ask, runner=color_runner)

def main(subject=None):
    os.chdir(PROJECT_DIR)
    try:
        reference = input('输入图片路径（可拖入图片，留空返回）：').strip().strip('"').strip("'")
        if not reference:
            return 0
        plan, work, lineart, waypoints, meta = build_plan(reference, subject)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        stem = OUTPUT_DIR / (Path(reference).stem + '_' + stamp)
        plan_path = Path(str(stem) + '_计划.json')
        preview_path = Path(str(stem) + '_预览.png')
        ink_path = Path(str(stem) + '_黑线任务.json')
        save_outputs(plan, work, plan_path, preview_path)
        imgio.imwrite(str(stem) + '_提取线稿.png', lineart)
        job.save_job(ink_path, waypoints, meta)
        from run_json_painting import make_bundle
        from painting_session import save
        bundle_path = Path(str(stem) + '_完整绘画.json')
        save(bundle_path, make_bundle(plan, dict(format='arm-painting job', version=1,
                                                meta=meta, waypoints=waypoints)))
        print('下次只需提供这个 JSON:', bundle_path)
        from run_mountain_balanced import load_reviewed
        load_reviewed(ink_path)
        print('预览图:', preview_path)
        print('提取线稿:', str(stem) + '_提取线稿.png')
        print('流程：原图提线 → 黑墨勾线/自动补墨 → 等墨干并换色 → 自动蘸色铺色。')
        print('所有题材共用提线和按色区面积铺色；颜色为现有颜料近似，笔宽与浓淡仍需试纸。')
        module = plan['modules']['color']
        for name in (*watercolor.COLOR_ORDER, 'green'):
            print(name, '铺色笔数:', sum(s['color'] == name for s in module['strokes']))
        if input('只看预览按回车；准备实际绘画请输入 PAINT：').strip() != 'PAINT':
            return 0
        # Read each cup only after the operator selects that color.
        from run_json_painting import run_file
        return run_file(bundle_path, confirmed=True)
    except (ValueError, OSError, EOFError, KeyboardInterrupt) as exc:
        print('已停止:', exc)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
