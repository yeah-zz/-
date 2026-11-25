"""独立上色对照小样：左0.2mm，右0.3mm；默认离线，不改正式画稿。"""
import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from color_profile import ColorProfile
from config import ArmConfig, CanvasConfig, z_for_press
from geometry import CanvasMap
from project_paths import PROJECT_DIR, ROBOT_LOG_DIR
from run_color_experiment import module_lines, run


def load_setup(color):
    from web.catalog import ARTWORKS
    from web.site_settings import apply, load_record
    from web.motion_profile import apply_to_artwork
    from pigment_station import PigmentStation
    settings = PROJECT_DIR / 'web/data/site_settings.json'
    if not settings.exists():
        raise ValueError('未找到网页保存的现场参数，请先在网页保存纸面和颜料杯标定。')
    record = apply(load_record(settings))  # 仅本进程，不写回设置
    b = json.loads((PROJECT_DIR / ARTWORKS[4][3]).read_text(encoding='utf-8'))
    ink = b['ink_job']
    artwork = apply_to_artwork((ink, b['workflow'], [], ink['meta']), record['motion_profile'])
    profile = dict(artwork[1]['modules']['color']['motion_candidate'], press_mm=.3)
    station = PigmentStation(**record['measurements'][color]).validate()
    return ColorProfile(**profile), station


def build_module(x, y, color, profile):
    """Two 20 x 5 mm blocks; 6 separate passes each, 1 mm center spacing."""
    if color not in ('blue', 'green') or not all(math.isfinite(v) for v in (x, y)):
        raise ValueError('颜色或坐标无效。')
    mapper = CanvasMap((1000, 700))
    inverse = np.linalg.inv(mapper.matrix)
    polygon = np.asarray(CanvasConfig.ROBOT_CORNERS, np.float32)
    strokes = []
    for block, press in enumerate((.2, .3)):
        for row in range(6):
            xy = np.array([[x+32*block, y+row], [x+32*block+20, y+row]], float)
            if row % 2:
                xy = xy[::-1].copy()
            # Check whole segment, including for an unusual/nonconvex calibration.
            for q in np.linspace(xy[0], xy[1], 41):
                if cv2.pointPolygonTest(polygon, tuple(map(float, q)), True) < 20:
                    raise ValueError('测试区必须全部位于标定画布内，距边缘至少20mm；请调整X/Y。')
            points = cv2.perspectiveTransform(xy.reshape(-1,1,2), inverse).reshape(-1,2).tolist()
            strokes.append(dict(id=f'pressure-{press:.2f}-row-{row+1}', color=color,
                points=points, press=[press]*2, speed=[profile.draw_velocity_percent]*2,
                length_mm=20., pressure_mm=press))
    module = dict(format='sparse-color-design-v1', image_size=list(mapper.image_size),
        image_corners=mapper.image_corners, mm_per_px=mapper.mm_per_px,
        strokes=strokes, reload_strokes_per_load=6,
        long_stroke_reload=dict(min_length_mm=30., very_long_length_mm=50.,
            strokes_per_load=2, very_long_strokes_per_load=2,
            max_length_mm=160., short_max_length_mm=180.))
    lines, skipped = module_lines(module, color, profile.to_dict())
    if skipped or len(lines) != 12:
        raise ValueError('小样路径检查未通过。')
    return module, lines


def save_preview(module, lines, profile, station, x, y, color):
    from PIL import Image, ImageDraw
    out = PROJECT_DIR / '输出' / ('上色下压对照_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    im = Image.new('RGB', (900, 280), 'white');d = ImageDraw.Draw(im)
    d.text((25,20), 'LEFT: 0.20 mm     RIGHT: 0.30 mm | +X right, +Y down', fill='black')
    d.text((25,45), 'Each: 20 x 5 mm, 6 passes, spacing 1 mm. Path diagram, NOT predicted paint width.', fill='black')
    for line in lines:
        xy = [(45+(q[0]-x)*15, 100+(q[1]-y)*15) for q in line['xy']]
        d.line(xy, fill='#648aa6' if color=='blue' else '#719776', width=2)
    d.text((25,225), f'Paper Z={ArmConfig.PAPER_Z:.4f}; start XY=({x:.3f}, {y:.3f}); color={color}', fill='black')
    im.save(out/'位置预览.png')
    from dataclasses import asdict
    (out/'测试计划.json').write_text(json.dumps(dict(module=module, profile=profile.to_dict(),
        station=asdict(station), paper_z=ArmConfig.PAPER_Z, x=x, y=y,
        target_z=[z_for_press(.2), z_for_press(.3)]), ensure_ascii=False, indent=2), encoding='utf-8')
    return out


def execute(module, color, profile, station, ask=input, runner=run):
    from painting_session import session_lock, Console
    from color_selection import ColorProgressControl
    # Same lock as complete web jobs, including their pauses and stage switches.
    with session_lock(ROBOT_LOG_DIR):
        if ask('新纸已放在原标定位置、测试区空白、杯位与纸面高度正确；输入 RUN 开始，回车取消：').strip().upper() != 'RUN':
            return 0
        console = Console()
        console.say('按P可在当前笔完成后暂停；按提示C继续或Q退出。')
        control = ColorProgressControl(console, '0.2/0.3mm上色对照', 12)
        return runner(module=module, color=color, profile=profile, station=station,
                      confirm_site=True, save_progress=False, log_dir=ROBOT_LOG_DIR, control=control)


def main(argv=None):
    from launcher import setup_console
    setup_console()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true', help='核对位置并输入RUN后真机测试')
    p.add_argument('--color', choices=('blue','green'), default='blue')
    p.add_argument('--x', type=float, help='左块起点，机器人用户坐标mm；须同时提供Y')
    p.add_argument('--y', type=float)
    args = p.parse_args(argv)
    try:
        if (args.x is None) != (args.y is None):
            raise ValueError('X、Y须同时提供。')
        profile, station = load_setup(args.color)
        center = np.asarray(CanvasConfig.ROBOT_CORNERS, float).mean(axis=0)
        x,y = (float(center[0]-26), float(center[1]-2.5)) if args.x is None else (args.x,args.y)
        if args.execute and args.x is None:
            raw = input(f'左块起点 X Y [{x:.3f} {y:.3f}]（回车使用画布中央，新纸）：').strip()
            if raw:
                x,y = map(float, raw.split())
        module, lines = build_module(x,y,args.color,profile)
        out = save_preview(module,lines,profile,station,x,y,args.color)
        print(f'预览和计划：{out}')
        print(f'左块0.20mm，右块0.30mm；总范围X={x:.3f}~{x+52:.3f}，Y={y:.3f}~{y+5:.3f}。')
        print(f'纸面Z={ArmConfig.PAPER_Z:.4f}；落笔Z分别为{z_for_press(.2):.4f}、{z_for_press(.3):.4f}。')
        print(f'颜色={args.color}；杯XY=({station.x:.4f},{station.y:.4f})，液面Z={station.surface_z:.4f}。')
        print(f'行笔v={profile.draw_velocity_percent}，全局={profile.global_speed_factor}%；每块6笔，开始及换块自动蘸色；超时/暂停会额外补蘸。')
        if not args.execute:
            print('仅离线预览，没有连接机械臂。加 --execute 进入测试；正式画稿和网页参数不变。')
            return 0
        return execute(module,args.color,profile,station)
    except FileExistsError:
        print('已有绘画任务占用机械臂。请先结束该任务再测试，不要删除运行锁。')
        return 2
    except (ValueError,OSError,EOFError,KeyboardInterrupt) as exc:
        print('测试停止：',exc)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
