"""Small supervised pressure ladder. Default is offline preview only."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import math
import os

import cv2
import numpy as np
from config import ArmConfig, CanvasConfig, ShuimoConfig, z_for_press
from color_profile import ColorProfile
from project_paths import PROJECT_DIR, OUTPUT_DIR, ROBOT_LOG_DIR
from run_color_experiment import plan_lines, run, module_lines

# Current 354-stroke artwork carries v30 on every black drawing point.
# JobConfig.DOWN_SPEED is the touchdown default, not this artwork's draw speed.
BLACK_DRAW_VELOCITY_PERCENT = 30


def build_samples(x, y, pressures, length=20., spacing=8.):
    values = [x, y, length, spacing, *pressures]
    if not pressures or not all(math.isfinite(v) for v in values):
        raise ValueError('坐标、下压量必须是有限数值。')
    if len(pressures) > 10 or pressures != sorted(set(pressures)):
        raise ValueError('下压档位须从小到大、不重复，最多10档。')
    ceiling = min(.50, float(ShuimoConfig.PRESS_MAX_MM))
    if any(p <= 0 or p > ceiling for p in pressures):
        raise ValueError('本小样下压须大于0且不超过 %.2f mm。' % ceiling)
    if not 5 <= spacing <= 20:
        raise ValueError('行距须在5~20mm之间。')
    result = []
    for i, press in enumerate(pressures):
        row_y = y + i * spacing
        lines = plan_lines(x, row_y, count=1, length=length, spacing=spacing, press=press)
        result.append(dict(number=i+1, press_mm=press, x=x, y=row_y,
                           length_mm=length, paper_z_mm=ArmConfig.PAPER_Z,
                           target_z_mm=z_for_press(press),
                           profile=asdict(ColorProfile(press_mm=press,
                               draw_velocity_percent=BLACK_DRAW_VELOCITY_PERCENT,
                               global_speed_factor=ArmConfig.SPEED_FACTOR,
                               travel_velocity_percent=80,
                               touchdown_velocity_percent=30,
                               lift_velocity_percent=60)),
                           path=lines[0]))
    return result


def build_continuous_module(samples):
    from geometry import CanvasMap
    if not samples:
        raise ValueError('试线为空。')
    mapper=CanvasMap((1000,700))
    inverse=np.linalg.inv(mapper.matrix)
    strokes=[]
    for sample in samples:
        xy=np.asarray(sample['path']['xy'],dtype=np.float64).reshape(-1,1,2)
        points=cv2.perspectiveTransform(xy,inverse).reshape(-1,2).tolist()
        strokes.append(dict(id='pressure-%02d'%sample['number'],points=points,
            press=[sample['press_mm']]*2,speed=[BLACK_DRAW_VELOCITY_PERCENT]*2))
    module=dict(format='pressure-test-design-v1',image_size=list(mapper.image_size),
        image_corners=mapper.image_corners,mm_per_px=mapper.mm_per_px,strokes=strokes,
        reload_strokes_per_load=3,reload_policy='every_n_strokes')
    profile=dict(samples[0]['profile'],press_mm=max(s['press_mm'] for s in samples),
        ink_draw_speed_scale=1.,interstroke_lift_mm=2.5,horizontal_velocity_percent=80)
    lines,skipped=module_lines(module,'ink',ColorProfile(**profile).to_dict())
    if skipped or len(lines)!=len(samples):raise ValueError('试线超出画布。')
    for sample,line in zip(samples,lines):
        if not np.allclose(sample['path']['xy'],line['xy'],atol=.002,rtol=0):
            raise ValueError('试线坐标映射不一致。')
    return module,profile


def read_ink_station():
    from pigment_station import PigmentStation
    saved=json.loads((PROJECT_DIR/'pigment_measurements.json').read_text(encoding='utf-8-sig'))
    if (saved['user']!=ArmConfig.USER_INDEX or saved['tool']!=ArmConfig.TOOL_INDEX
            or saved['rpy']!=list(ArmConfig.RPY)):
        raise ValueError('黑墨杯的坐标系或笔姿态与当前配置不一致。')
    # Use a copy: never change the artwork's global ink loading settings.
    station=PigmentStation(**saved['black'])
    return replace(station,strokes_per_load=3,level_compensation_enabled=False).validate()


def run_continuous_samples(samples,ask=input,runner=run,station_reader=read_ink_station,console=None):
    import contextlib
    import io
    from painting_session import Console, PAUSED
    from color_selection import ColorProgressControl
    module,profile=build_continuous_module(samples)
    station=station_reader()
    print('自动蘸黑墨位置：X=%.4f Y=%.4f，液面Z=%.4f，蘸入%.2fmm。' %
          (station.x,station.y,station.surface_z,station.dip_mm))
    print('将连续画%d条；开始前蘸墨，之后每完成3条补蘸一次。' % len(samples))
    if ask('确认新纸、试线区域、触纸Z和黑墨杯已准备好，输入 RUN 开始全部试线：').strip().upper()!='RUN':
        return 0
    console=console or Console(ask=ask)
    console.say('按P：完成当前条后暂停；C继续，Q退出。暂停继续会先补蘸一次。')
    control=ColorProgressControl(console,'下压试线',len(samples))
    diagnostics=io.StringIO()
    with contextlib.redirect_stdout(diagnostics):
        result=runner(profile=ColorProfile(**profile),confirm_site=True,module=module,
            color='ink',station=station,log_dir=ROBOT_LOG_DIR,save_progress=False,control=control)
    console.say('')
    if result==0:console.say('全部试线完成；下压从0.05mm逐条增加，按编号比较线宽。正式参数未修改。')
    elif result==PAUSED:console.say('已退出试线，未保存续画进度。')
    else:console.say('\n'.join(diagnostics.getvalue().splitlines()[-12:]))
    return result


def save_preview(samples, prefix, continuous=False):
    canvas = np.full((150+90*len(samples), 1000, 3), 255, np.uint8)
    cv2.putText(canvas, 'PRESSURE TEST - path layout, NOT predicted ink width',
                (25, 40), cv2.FONT_HERSHEY_SIMPLEX, .65, (30,30,30), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Paper Z = %.4f | +X drawing; rows along +Y' % ArmConfig.PAPER_Z,
                (25, 80), cv2.FONT_HERSHEY_SIMPLEX, .6, (30,30,30), 1, cv2.LINE_AA)
    for i, sample in enumerate(samples):
        y = 140 + i*90
        cv2.putText(canvas, '%d: %.2f mm   XY=(%.2f, %.2f)' %
                    (sample['number'],sample['press_mm'],sample['x'],sample['y']),
                    (25,y),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,30,30),1,cv2.LINE_AA)
        cv2.line(canvas,(530,y-5),(920,y-5),(30,30,30),2,cv2.LINE_AA)
    import imgio
    image_path = Path(str(prefix)+'_位置示意.png')
    if not imgio.imwrite(str(image_path),canvas): raise OSError('无法保存示意图')
    plan_path = Path(str(prefix)+'_计划.json')
    plan_path.write_text(json.dumps(dict(format='pressure-test-v1', paper_z=ArmConfig.PAPER_Z,
        user=ArmConfig.USER_INDEX, tool=ArmConfig.TOOL_INDEX, rpy=list(ArmConfig.RPY),
        auto_dip=continuous, reload_strokes_per_load=3 if continuous else None,
        samples=samples),ensure_ascii=False,indent=2),encoding='utf-8')
    print('位置示意:',image_path)
    print('测试计划:',plan_path)


def run_samples(samples, ask=input, runner=run):
    for sample in samples:
        print('\n第%d条：下压 %.2f mm，长 %.1f mm，起点 X=%.3f Y=%.3f' %
              (sample['number'],sample['press_mm'],sample['length_mm'],sample['x'],sample['y']))
        print('触纸基准 Z=%.4f；本行落笔目标 Z=%.4f；整条保持该下压量。' %
              (sample['paper_z_mm'], sample['target_z_mm']))
        print('确认本行空白，使用黑墨；手动蘸墨并沥去余液，每行尽量保持相同的供墨条件。')
        if ask('输入 RUN 只画这一条；回车或其他输入结束：').strip().upper() != 'RUN':
            return 0
        result = runner(sample['x'],sample['y'],count=1,length=sample['length_mm'],
                        profile=ColorProfile(**sample['profile']),confirm_site=True,log_dir=ROBOT_LOG_DIR,
                        save_progress=False)
        if result != 0: return result
        print('本条完成并已抬笔到纸外高位。观察线宽/断墨及笔毛弯曲，再决定是否继续。')
    print('全部小样完成。触纸Z和压力参数没有写回config.py；按试画结果再决定正式参数。')
    return 0


def main(argv=None):
    from launcher import setup_console
    setup_console()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='确认一次后连续试线；配合--manual-each可逐条确认')
    parser.add_argument('--read-paper-z',action='store_true',help='只读当前笔尖Z，无运动；请先手动调到刚触纸')
    parser.add_argument('--paper-z',type=float,help='本次试验的触纸基准，不修改配置文件')
    parser.add_argument('--x',type=float)
    parser.add_argument('--y',type=float)
    parser.add_argument('--pressures',default='0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50')
    parser.add_argument('--length',type=float,default=30.,help='试线长度，5至30mm')
    parser.add_argument('--spacing',type=float,default=5.,help='行距，5至20mm')
    parser.add_argument('--manual-each',action='store_true',help='逐条RUN、手动蘸墨；默认一次RUN连续试线')
    args=parser.parse_args(argv)
    os.chdir(PROJECT_DIR)
    if args.read_paper_z:
        if args.execute: parser.error('--read-paper-z不能与--execute一起用')
        from robot import E6Robot
        from run_mountain_v4_robot import connect_readonly
        robot=E6Robot()
        try:
            connect_readonly(robot)
            pose=robot.get_pose()
            if pose is None or len(pose)!=6 or not all(math.isfinite(v) for v in pose):
                raise ValueError('未读取到有效位姿')
            print('当前位姿:',pose)
            print('当前笔尖 Z=%.4f；只有你已手动调到刚触纸时，才可作为PAPER_Z。' % pose[2])
            return 0
        finally:
            robot.close(disable=False)
    original_z=ArmConfig.PAPER_Z
    try:
        paper_z=args.paper_z
        if args.execute and paper_z is None:
            raw=input('本次刚触纸的 Z [%.4f]（回车沿用，改值只影响本次小样）：' % original_z).strip()
            paper_z=float(raw) if raw else original_z
        if paper_z is not None:
            if not math.isfinite(paper_z): raise ValueError('触纸Z必须是有限数值')
            ArmConfig.PAPER_Z=paper_z
        if (args.x is None) != (args.y is None): raise ValueError('X、Y须同时提供')
        corners=np.asarray(CanvasConfig.ROBOT_CORNERS,float)
        pressures=[float(v) for v in args.pressures.split(',')]
        x=float(corners[:,0].mean()-args.length/2) if args.x is None else args.x
        y=float(corners[:,1].mean()-(len(pressures)-1)*args.spacing/2) if args.y is None else args.y
        if args.execute and args.x is None:
            raw=input('第一条起点 X Y [%.3f %.3f]（回车用画布中央小样区）：' % (x,y)).strip()
            if raw: x,y=map(float,raw.split())
        samples=build_samples(x,y,pressures,length=args.length,spacing=args.spacing)
        OUTPUT_DIR.mkdir(exist_ok=True)
        prefix=OUTPUT_DIR/('下压小样_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        continuous=not args.manual_each
        if continuous:build_continuous_module(samples)
        save_preview(samples,prefix,continuous=continuous)
        print('纸面 Z=%.4f；每行%.1fmm，沿用户坐标+X，行距%.1fmm。' %
              (ArmConfig.PAPER_Z,args.length,args.spacing))
        print('行笔速度v%d，全局倍率%d%%，与当前黑线一致；固定下压，不使用蘸墨后的80%%笔压补偿。' %
              (BLACK_DRAW_VELOCITY_PERCENT,ArmConfig.SPEED_FACTOR))
        print('一次RUN连续试线，每3条自动蘸墨。' if continuous else '逐条RUN，手动蘸墨。')
        if not args.execute:
            print('离线预览完成，没有连接机械臂。双击“下压小样测试.bat”进入连续试线。' if continuous else
                  '离线预览完成，没有连接机械臂；加--execute --manual-each进入逐条试画。')
            return 0
        return run_continuous_samples(samples) if continuous else run_samples(samples)
    except (ValueError,OSError,EOFError,KeyboardInterrupt) as exc:
        print('停止:',exc)
        return 2
    finally:
        ArmConfig.PAPER_Z=original_z


if __name__=='__main__': raise SystemExit(main())
