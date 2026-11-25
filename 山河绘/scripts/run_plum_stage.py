"""非交互地跑梅花的【单个阶段】—— 给"先画黑、洗笔、再画红"这种分段流程用。

用法（两段分开跑，中间由人洗笔）：

    python scripts/run_plum_stage.py --stage ink                    # 只读：打印计划，不连机械臂
    python scripts/run_plum_stage.py --stage ink   --confirm-site   # 真机画黑色阶段
    python scripts/run_plum_stage.py --stage color --confirm-site   # 洗好笔、蘸好红再画红色阶段

★ 从已保存的计划 JSON 跑（和跑山体 --job 一样，可以手改轻重）：

    python scripts/plum_preview.py                                  # 先生成计划 JSON
    python scripts/run_plum_stage.py --stage ink --plan 输出/plum_preview_XXXX.json --list
    #   ↑ 打印每一条笔触的 id / 长度 / 笔压，照着 id 去 JSON 里改 press
    python scripts/run_plum_stage.py --stage ink --plan 输出/plum_preview_XXXX.json --confirm-site

    不改 JSON 也能整体调：--press-scale 1.2（全部加重 20%）、--press-floor 0.45
    （把所有低于 0.45 的下压抬到 0.45 —— 专治"细枝太淡、纸上没墨"）。
    注意：--plan / --press-scale / --press-floor 都会改变笔压，
    因此和 --resume 互斥（改了笔压就不存在"同一批笔触"，续画核对必然不通过）。

`--confirm-site` 的含义与 run_plum_experiment.py 里那句输入确认完全一样：
    现场有人看护、手放急停；笔/纸/垫板与当前标定一致；高位往返路径上没有
    杯沿、夹具、线缆或人；纸面配准有效、杯内液深足够。

真机执行走 run_color_experiment.run 这同一条路径，门槛一个都没少：
    validate_start（距停车点 <=10mm、纸外 >=20mm、Z 在起始窗口内）
    PigmentStation.validate（离纸边 >=20mm、杯沿比高位低 >=15mm、液面低于杯沿）
    check_waypoints（可达范围）、robot_runs/blue_wash_motion.lock（同时只允许一个脚本动）
"""
import argparse
import copy
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import shuimo
from color_profile import ColorProfile
from config import ArmConfig, ShuimoConfig, station_approach_z
from experiment_resume import resume_suffix, unresolved_records
from park_color_loading import target_pose
from plum_blossom import (PETAL_DOT_LENGTH_MM, PETAL_DOT_PRESS_MM,
                          PETAL_MAX_LINES, PETAL_PRESS_MM, build_plan)
from project_paths import OUTPUT_DIR, PLUM_REFERENCE_IMAGE, PROJECT_DIR, ROBOT_LOG_DIR
from run_color_experiment import run
from run_plum_experiment import station_from_json

STAGES = {
    "ink": ("outline", "ink", "black", "黑色枝干"),
    "color": ("color", "red", "red", "红色花瓣"),
}


def load_plan(path, key):
    """读一份保存下来的设计计划 JSON（scripts/plum_preview.py 写出来的那种）。

    这是跑山体时 `--job` 的等价物：JSON 就是真正的输入，改了就生效。
    path 还可以写 `latest`（或用 OUTPUT_DIR 目录名），自动取最新的一份预览计划。
    """
    if str(path).strip().lower() in ("latest", "newest", "最新"):
        found = sorted(OUTPUT_DIR.glob("plum_preview_*.json"),
                       key=lambda q: q.stat().st_mtime)
        if not found:
            raise ValueError("输出目录里还没有 plum_preview_*.json，先跑 scripts/plum_preview.py")
        p = found[-1]
        print("  （--plan latest -> %s）" % p.name)
    else:
        p = Path(path)
        if p.is_dir():
            found = sorted(p.glob("plum_preview_*.json"), key=lambda q: q.stat().st_mtime)
            if not found:
                raise ValueError("%s 里没有 plum_preview_*.json" % p)
            p = found[-1]
    if not p.is_file():
        raise ValueError("找不到计划文件 %s" % p)
    data = json.loads(p.read_text(encoding="utf-8"))
    modules = data.get("modules")
    if not isinstance(modules, dict) or key not in modules:
        raise ValueError("这个 JSON 不是梅花设计计划（缺 modules.%s）：%s" % (key, p))
    if not modules[key].get("strokes"):
        raise ValueError("计划里 modules.%s 没有笔触：%s" % (key, p))
    return data, p


def retune_press(module, key, color, scale=1.0, floor=None):
    """按倍数/下限改笔压，并把派生的 press_mean / press_max / width_mm 一起刷新。

    直接改 JSON 里的 press 数组也完全可以；这个函数只是省去逐条手算。
    """
    ceiling = float(ShuimoConfig.PRESS_MAX_MM) if key == "outline" else float(ColorProfile().press_mm)
    out = copy.deepcopy(module)
    touched = 0
    for s in out["strokes"]:
        presses = [float(v) for v in (s.get("press") or [])]
        if not presses:
            continue
        if scale != 1.0:
            presses = [v * scale for v in presses]
        if floor is not None:
            presses = [max(float(floor), v) for v in presses]
        if max(presses) > ceiling + 1e-9:
            raise ValueError("%s 的笔压 %.3f 超过上限 %.3f（%s 阶段），拒绝执行"
                             % (s["id"], max(presses), ceiling, color))
        if min(presses) < 0:
            raise ValueError("%s 的笔压出现负值，拒绝执行" % s["id"])
        s["press"] = [round(v, 4) for v in presses]
        s["press_max"] = round(max(presses), 4)
        s["press_mean"] = round(sum(presses) / len(presses), 4)
        s["width_mm"] = round(float(shuimo.press_to_width_mm(s["press_mean"])), 3)
        touched += 1
    return out, touched


def list_strokes(module):
    print("")
    print("  %-20s %7s %7s %7s %7s" % ("id", "长度mm", "笔压min", "笔压med", "笔压max"))
    order = sorted(module["strokes"], key=lambda s: -float(s.get("length_mm", 0.0)))
    for s in order:
        presses = np.asarray(s.get("press") or [0.0], float)
        print("  %-20s %7.1f %7.2f %7.2f %7.2f"
              % (s["id"], float(s.get("length_mm", 0.0)),
                 presses.min(), float(np.median(presses)), presses.max()))
    print("  （改 press= 数组就是改这一条的下压：Z = 纸面 Z + press；"
          "想让某条粗一点就把它的数调大）")


def describe(stage, module, station=None):
    key, color, station_key, label = STAGES[stage]
    strokes = module["strokes"]
    station = station if station is not None else station_from_json(station_key)
    peaks = [float(np.max(s["press"])) for s in strokes]
    total = sum(float(s.get("length_mm", 0.0)) for s in strokes)
    print("=" * 72)
    print(" 梅花 · %s 阶段" % label)
    print("=" * 72)
    print("  笔触 %d 条，总长 %.1f mm；笔压 %.2f~%.2f mm" % (len(strokes), total, min(peaks), max(peaks)))
    print("  纸面 Z=%.4f  站点高位 %.4f  最深 Z=%.4f" % (
        ArmConfig.PAPER_Z, station_approach_z(), ArmConfig.PAPER_Z + max(peaks)))
    print("  蘸色杯 (%.4f, %.4f) 液面 %.4f  蘸深 %.2f mm  停留 %.1f s  沥墨 %.1f s  每 %d 笔一蘸"
          % (station.x, station.y, station.surface_z, station.dip_mm,
             station.dwell_s, station.drip_s, station.strokes_per_load))
    print("  预计蘸色 %d 次" % max(1, int(np.ceil(len(strokes) / float(station.strokes_per_load)))))
    if stage == "color":
        if module.get('drawing_style') == 'contours_and_small_dabs':
            kinds = [s.get('kind') for s in strokes]
            print("  新描边方案：花瓣轮廓 %d 笔，小花苞点缀 %d 笔（按 JSON 执行）"
                  % (kinds.count('petal_contour'), kinds.count('bud_dab')))
        elif module.get('petal_style') == 'dots':
            print("  花瓣：每朵五笔；每瓣深压后短拖 %.1f mm，峰值下压 %.2f mm"
                  % (PETAL_DOT_LENGTH_MM, PETAL_DOT_PRESS_MM))
        else:
            print("  花瓣：一瓣最多 %d 笔，峰值下压 %.2f mm" % (PETAL_MAX_LINES, PETAL_PRESS_MM))
    print("  起点/终点（纸外高位停车点）:", tuple(round(v, 4) for v in target_pose()))
    print("  ! 起点要求（2026-09-26 已放宽）：纸外 >=20 mm、姿态 = 标定值、")
    print("    Z 在装载高度 %.4f 的 -60 ~ +20 mm 内。不必凑到停车点 10 mm 内，"
          % station_approach_z())
    print("    离装载点 >150 mm 只会提示，不会拒绝。")
    return module, color, station, label


def pick_resume_record(module, color, profile=None, record=None):
    """找出一条中断记录，并算出还剩哪些笔触（用 resume_suffix 的同一套核对）。

    record 给了就用它；没给就要求"未解决的中断记录"恰好一条。
    ★ 如果这条记录本身就是"续画后的记录"（它只记录了当时剩下的那批笔触），
      这里会把计划裁剪成同一批，否则 resume_suffix 的逐笔比对必然不通过。
    任何不一致都会抛 ValueError —— 与 run_color_experiment.run 里的判据完全相同。
    """
    import json
    from color_profile import ColorProfile
    from run_color_experiment import module_lines
    if record:
        path = Path(record)
        if not path.is_file():
            raise ValueError("找不到续画记录 %s" % path)
    else:
        blocked = unresolved_records(ROBOT_LOG_DIR)
        if len(blocked) != 1:
            raise ValueError("未解决的中断记录有 %d 条；已人工解除过的记录请用 --resume-record 指定" % len(blocked))
        path = blocked[0][0]
    old = json.loads(path.read_text(encoding="utf-8"))
    wanted = [s["id"] for s in old.get("lines") or []]
    have = [s["id"] for s in module["strokes"]]
    if wanted and have != wanted:
        keep = set(wanted)
        trimmed = [s for s in module["strokes"] if s["id"] in keep]
        if [s["id"] for s in trimmed] != wanted:
            raise ValueError("中断记录里的笔触与当前计划对不上，不能续画")
        module = dict(module, strokes=trimmed, stroke_count=len(trimmed))
        print("  （这条中断记录本身就是续画记录，只含 %d 笔；计划已裁到同一批）" % len(trimmed))
    lines, _ = module_lines(module, color, (profile or ColorProfile()).to_dict())
    remaining, audit = resume_suffix(path, lines)
    return path, remaining, audit, module


def main(argv=None):
    parser = argparse.ArgumentParser(description="跑梅花的单个阶段（黑 / 红）")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES),
                        help="ink=黑色枝干，color=红色花瓣")
    parser.add_argument("--confirm-site", action="store_true",
                        help="现场确认（有人看护、急停在手、路径清空、标定有效）才真机执行")
    parser.add_argument("--resume", action="store_true",
                        help="续画唯一那条未解决的中断记录：跳过已完成的笔触，只画剩下的")
    parser.add_argument("--resume-record", help="指定续画记录文件（人工解除过 pending_move 的用这个）")
    parser.add_argument("--speed-factor", type=int,
                        help="只改全局 SpeedFactor 来整体调速（不动指令速度，因此不影响续画比对）。"
                             "ColorProfile 默认 %d；嫌快就填 40~60" % ColorProfile().global_speed_factor)
    parser.add_argument("--plan",
                        help="跑这份已保存的设计计划 JSON（scripts/plum_preview.py 的输出），"
                             "不再重新生成计划。改 JSON 里的 press 就是改轻重，和跑山体 --job 一样。"
                             "写 latest 就自动取最新的一份")
    parser.add_argument("--press-scale", type=float,
                        help="整体缩放笔压，例如 1.2 = 全部加重 20%（与 --resume 互斥）")
    parser.add_argument("--press-floor", type=float,
                        help="笔压下限，例如 0.45 = 把所有低于 0.45 的下压抬到 0.45"
                             "（细枝太淡、纸上没墨时用；与 --resume 互斥）")
    parser.add_argument("--strokes-per-load", type=int, choices=(1, 2),
                        help="临时指定每蘸一次画几笔；点瓣小样建议 1（与续画互斥）")
    parser.add_argument("--list", action="store_true", dest="list_strokes",
                        help="列出每条笔触的 id / 长度 / 笔压，方便照着改 JSON")
    args = parser.parse_args(argv)
    os.chdir(PROJECT_DIR)
    try:
        if (args.plan or args.press_scale is not None or args.press_floor is not None) \
                and (args.resume or args.resume_record):
            raise ValueError("--plan/--press-scale/--press-floor 会改变笔压，"
                             "和 --resume 不能一起用（改了笔压就不存在'同一批笔触'）")
        if args.strokes_per_load is not None and (args.resume or args.resume_record):
            raise ValueError("--strokes-per-load 会改变蘸色路线，不能用于续画")
        profile = None
        if args.speed_factor:
            if not 1 <= args.speed_factor <= 100:
                raise ValueError("--speed-factor 必须在 1~100 之间")
            profile = replace(ColorProfile(), global_speed_factor=args.speed_factor)
        key, color = STAGES[args.stage][0], STAGES[args.stage][1]
        if args.plan:
            plan, plan_file = load_plan(args.plan, key)
            module = plan["modules"][key]
            print("  计划来源：%s（已保存的计划，不重新生成；改这个文件里的 press 才生效）"
                  % plan_file.name)
        else:
            plan, _work = build_plan(PLUM_REFERENCE_IMAGE)
            module = plan["modules"][key]
            print("  计划来源：参考图实时生成（要手改轻重就先跑 scripts/plum_preview.py）")
        if args.press_scale is not None or args.press_floor is not None:
            module, touched = retune_press(module, key, color,
                                           scale=1.0 if args.press_scale is None else args.press_scale,
                                           floor=args.press_floor)
            print("  笔压调整：倍数 %s，下限 %s，影响 %d 条笔触"
                  % ("-" if args.press_scale is None else args.press_scale,
                     "-" if args.press_floor is None else args.press_floor, touched))
        station = station_from_json(STAGES[args.stage][2])
        if args.strokes_per_load is not None:
            station = replace(station, strokes_per_load=args.strokes_per_load)
        module, color, station, label = describe(args.stage, module, station=station)
        if args.list_strokes:
            list_strokes(module)
        if profile:
            print("  速度：全局 SpeedFactor %d%%（指令速度保持 %.0f/%.0f/%.0f）"
                  % (profile.global_speed_factor, profile.draw_velocity_percent,
                     profile.travel_velocity_percent, profile.dip_velocity_percent))
        resume_record = None
        if args.resume or args.resume_record:
            resume_record, remaining, audit, module = pick_resume_record(
                module, color, profile=profile, record=args.resume_record)
            print("")
            print("  续画记录: %s" % resume_record.name)
            print("  已跳过 %d 笔，剩余 %d 笔" % (len(audit["skipped_ids"]), len(remaining)))
            print("  ! 前提：原纸/笔/垫板都没有移位；笔尖已抬起、机械臂已回到纸外高位。")
        if not args.confirm_site:
            print("")
            print("--confirm-site：以上只是计划，未连接机械臂、未发送任何指令。")
            print("确认现场后加 --confirm-site 再跑。")
            return 0
        return run(confirm_site=True, module=module, color=color, station=station,
                   resume_record=resume_record, profile=profile)
    except (ValueError, OSError, EOFError, KeyboardInterrupt) as exc:
        print("停止:", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
