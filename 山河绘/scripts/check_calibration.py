"""只读核对当前 config.py 的标定值，和"已经实画成功那次任务"的快照是否一致。

不连接机械臂、不发送任何指令、不写任何文件。

它回答：这套数字能不能信？和上次画成功时是不是同一套？
它不能回答：纸有没有挪、笔有没有换、墨有没有加水 —— 那些只能现场重新标定。
"""
# --- path bootstrap: this script lives in scripts/, put the project root on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
from pathlib import Path

from config import (ArmConfig, CanvasConfig, ReachConfig, ShuimoConfig, PALETTE,
                    validate, transit_z, station_approach_z)
from geometry import CanvasMap

ROOT = Path(__file__).resolve().parent.parent
JOB = ROOT / 'artifacts' / 'mountain' / 'mountain_v6_job.json'
ACTUAL = ROOT / 'artifacts' / 'mountain' / 'mountain_v6_job_actual.json'
OK, BAD = '  [OK] ', '  [!!] '


def normalized(value):
    return json.loads(json.dumps(value))


def main():
    if not JOB.is_file():
        print('找不到参考任务: %s' % JOB)
        return 2
    meta = json.loads(JOB.read_text(encoding='utf8'))['meta']
    snap = meta['ink_execution']
    cal = snap['calibration']

    print('=' * 72)
    print(' 标定核对（只读，不连机械臂）')
    print(' 基准任务: artifacts/mountain/mountain_v6_job.json')
    if ACTUAL.is_file():
        print(' 该任务有真机执行记录: %s' % ACTUAL.name)
    print('=%s' % ('=' * 70))

    failures = []

    def check(label, current, expected, why):
        same = normalized(current) == normalized(expected)
        print('%s%-24s %s' % (OK if same else BAD, label, '一致' if same else '不一致'))
        if not same:
            failures.append(label)
            print('        当前值: %s' % json.dumps(normalized(current), ensure_ascii=False))
            print('        快照值: %s' % json.dumps(normalized(expected), ensure_ascii=False))
        print('        %s' % why)

    print('')
    print('-- 和"实画成功那套"比对 ------------------------------------')
    check('ArmConfig.RPY', list(ArmConfig.RPY), cal['rpy'],
          '姿态角；超 ±0.15 度程序就拒绝发运动')
    check('USER_INDEX', ArmConfig.USER_INDEX, cal['user'], '用户坐标系索引')
    check('TOOL_INDEX', ArmConfig.TOOL_INDEX, cal['tool'], '工具坐标系索引（毛笔 TCP）')
    check('SPEED_FACTOR', ArmConfig.SPEED_FACTOR, cal['speed_factor'], '全局速度倍率')
    check('PALETTE', PALETTE, cal['palette'], '调色板：每格的 dip / surface_z / 笔档')
    check('ArmConfig.PAPER_Z', ArmConfig.PAPER_Z, meta['paper_z'],
          '纸面高度；换笔后必重标')
    check('ROBOT_CORNERS', [list(p) for p in CanvasConfig.ROBOT_CORNERS],
          [list(p) for p in meta['canvas_corners']], '画布四角 XY')
    mapper = CanvasMap(meta['image_size'])
    check('画布比例 mm/px', round(mapper.mm_per_px, 12), round(snap['mm_per_px'], 12),
          '图像像素 -> 毫米的换算比例')

    print('')
    print('-- 几何 / 可达性自检 ---------------------------------------')
    problems = validate(need_stations=True)
    if problems:
        for item in problems:
            print('%s%s' % (BAD, item))
        failures.append('validate')
    else:
        print('%s配置自检通过：无占位坐标，笔压与可达范围一致' % OK)

    deepest = ArmConfig.PAPER_Z + ShuimoConfig.PRESS_MAX_MM
    good = ReachConfig.Z_MAX >= deepest
    print('%s最深绘制 Z = PAPER_Z %.2f + PRESS_MAX %.2f = %.2f ；Z_MAX = %.2f %s'
          % (OK if good else BAD, ArmConfig.PAPER_Z, ShuimoConfig.PRESS_MAX_MM, deepest,
             ReachConfig.Z_MAX, '够用' if good else '不够，每个绘制点都会被判越界'))
    print('        换笔后 PAPER_Z 会平移，这个值必须跟着改')
    if not good:
        failures.append('Z_MAX')

    tz = transit_z()
    good = tz >= ReachConfig.Z_MIN
    print('%s站点间移动高度 transit_z = %.2f（纸面上方 %.0f mm）；Z_MIN = %.2f %s'
          % (OK if good else BAD, tz, ArmConfig.TRANSIT_LIFT_MM, ReachConfig.Z_MIN,
             '可达' if good else '高于 Z_MIN，会被判越界'))
    if not good:
        failures.append('transit_z')

    print('%s高杯接近高度 station_approach_z = %.2f' % (OK, station_approach_z()))
    print('        必须高于水盂/墨盘/杯沿，否则横移会撞翻容器')

    print('')
    print('-- 高位与容器的实际间隙（红梅用的两个杯子）------------------')
    print('    Z 增大 = 下降；所以"比杯沿高多少" = 杯沿Z - 高位Z')
    pm = ROOT / 'pigment_measurements.json'
    if not pm.is_file():
        print('%s找不到 %s' % (BAD, pm.name))
        failures.append('pigment_measurements.json')
    else:
        cups = json.loads(pm.read_text(encoding='utf-8-sig'))
        high, tz2 = station_approach_z(), transit_z()
        for label, key in (('黑墨杯', 'black'), ('红色杯', 'red')):
            cup = cups.get(key)
            if not cup:
                continue
            rim, surf = cup.get('rim_z'), cup.get('surface_z')
            print('  %s  dip=(%.1f, %.1f)' % (label, cup['x'], cup['y']))
            print('        杯沿 Z=%.4f   液面 Z=%.4f' % (rim, surf))
            print('        接近高位 %.2f  ->  比杯沿高 %.1f mm' % (high, rim - high))
            print('        横移高度 %.2f  ->  比杯沿高 %.1f mm' % (tz2, rim - tz2))
            if rim - high < 15:
                failures.append('%s 杯沿间隙' % label)
                print('%s      间隙不足 15 mm，PigmentStation.validate() 会拒绝' % BAD)
        print('  这些间隙是"当前容器 + 当前液面"下的值；换杯子、挪盘子、加墨之后要重算。')

        # ---- 黑盘在两处各存了一份，必须一致 ----
        #   梅花画：蘸墨读 pigment_measurements.json，起点/停车点读 config.py 的 PALETTE[0]
        #   山水/试笔/--brush-test/--calibrate-ink：读 PALETTE[0]
        #   两处不一致 -> "起点要求你在 A 点、蘸墨却扎到 B 点"，或两条路径蘸深不同。
        black = cups.get('black') or {}
        if PALETTE:
            px, py = PALETTE[0]['dip']
            pz = float(PALETTE[0]['surface_z'])
            same_xy = abs(px - float(black.get('x', 1e9))) < 1e-6 and abs(py - float(black.get('y', 1e9))) < 1e-6
            same_z = abs(pz - float(black.get('surface_z', 1e9))) < 1e-6
            if same_xy and same_z:
                print('%s黑盘两处一致：PALETTE[0] == json black（XY 与液面）' % OK)
            else:
                failures.append('PALETTE[0] 与 json black 不一致')
                print('%s黑盘两处不一致 —— 必须同步' % BAD)
                if not same_xy:
                    print('        PALETTE[0]["dip"] = (%.4f, %.4f)    json black xy = (%.4f, %.4f)'
                          % (px, py, float(black.get('x', float('nan'))), float(black.get('y', float('nan')))))
                if not same_z:
                    print('        PALETTE[0]["surface_z"] = %.4f    json black.surface_z = %.4f'
                          % (pz, float(black.get('surface_z', float('nan')))))
                print('        规则：以 json 为准（梅花流程唯一读它），把 config.py 的 PALETTE[0] 改成同样的值')

    print('')
    print('=' * 72)
    print(' 结论')
    print('=' * 72)
    if failures:
        print('  %d 项和"实画成功那套"不一致: %s' % (len(failures), ', '.join(failures)))
        print('  改动不一定是错的（可能是换笔/挪纸后重标过），但必须逐项确认；')
        print('  不一致时 run_mountain_balanced.py / run_blue_wash.py 会直接拒绝执行。')
    else:
        print('  当前 config.py 和"2026-09-24 实画成功的那套"完全一致。')
        print('  说明数字没被误改，可以作为继续工作的基准。')
    print('')
    print('  以下"物理上还对不对"离线证明不了，必须现场确认：')
    print('   - 四角是否真的对准纸角   -> uv run python main.py --probe（空跑不落笔，肉眼看）')
    print('   - 笔尖是否刚好触纸       -> uv run python main.py --calibrate-z（换笔后必跑）')
    print('   - 颜料液面高度           -> uv run python main.py --calibrate-ink（加墨/加水后必跑）')
    print('   - 机械臂当前姿态差多少   -> uv run python prepare_robot.py --job artifacts/mountain/mountain_v6_job.json')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())