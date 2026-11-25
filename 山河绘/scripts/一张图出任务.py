"""一张图出任务 · 端到端流水线(唯一推荐入口)

输入一张参考图,产出:
  ① 黑线任务 json    —— 好提取器(全分辨率 + 按墨线宽度分层)+ 下压补到落墨区间
  ② 上色计划 json    —— 稠密蛇形铺色 + 逐点色阶(不是稀疏点缀版)
  ③ 完整绘画 bundle  —— run_json_painting 直接可用
  ④ 合成预览 png     —— 同时含黑线与上色
  ⑤ 提取线稿 png     —— 只看黑线

用法:
    python scripts\\一张图出任务.py --image 参考图\\你的图.png
    python scripts\\一张图出任务.py --image <图> --name 作品名 --check

它会依次调用(每一步都是独立进程,失败立即可见):
    unified_painting.py                  提线 + 建计划
    scripts\\黑线下压补到落墨区间.py       下压补偏移
    scripts\\build_filled_landscape.py    稠密铺色 + 合成预览
"""
import argparse, glob, json, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '输出'
PY = sys.executable

# 稠密铺色参数(沿用层峦松水_双蓝长笔版验证过的那套)
PITCH_MM = 0.774
SHRINK_MM = 0.4
COLOUR_PRESS = '0.12,0.32'
TRAVEL = 60
MAX_STROKE = 300
RELOAD_PLAN = 20          # 计划里的每蘸笔数(运行时限 1~40;生成器 CLI 限 1~8)
TARGET_INK_STROKES = 1200  # 简化目标笔数(553 笔的黑线阶段约 30 分钟)


def run(cmd, stdin=None, label=''):
    print('\n$ %s' % ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), input=stdin,
                       text=True, capture_output=True, encoding='utf-8', errors='replace')
    tail = (p.stdout or '').strip().splitlines()[-14:]
    for line in tail:
        print('  | ' + line)
    if p.returncode != 0:
        print('  !! %s 失败 (exit %d)' % (label or cmd[1], p.returncode))
        for line in (p.stderr or '').strip().splitlines()[-12:]:
            print('  E ' + line)
        raise SystemExit('步骤失败: %s' % (label or cmd[1]))
    return p.stdout or ''


def newest(pat, after=None):
    fs = [Path(f) for f in glob.glob(str(OUT / pat))]
    if after is not None:
        fs = [f for f in fs if f.stat().st_mtime >= after]
    fs.sort(key=lambda f: f.stat().st_mtime)
    return fs[-1] if fs else None


def press_report(job):
    d = json.loads(Path(job).read_text(encoding='utf-8-sig'))
    bs = d['meta']['ink_execution']['base_strokes']
    ps = np.asarray([v for s in bs for v in (s.get('press') or [])], float)
    return dict(strokes=len(bs), lo=float(ps.min()), p50=float(np.median(ps)),
                hi=float(ps.max()), below=100.0 * float(np.mean(ps < 0.13)),
                length_mm=float(sum(s.get('length_mm', 0) for s in bs)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--name', default=None, help='作品名(默认用图片文件名)')
    ap.add_argument('--check', action='store_true', help='最后跑两个离线校验')
    ap.add_argument('--keep-src', action='store_true', help='保留中间的铺色源目录')
    a = ap.parse_args(argv)

    img = Path(a.image)
    if not img.is_absolute():
        img = ROOT / img
    if not img.is_file():
        raise SystemExit('图片不存在: %s' % img)
    name = a.name or img.stem
    t0 = datetime.now().timestamp()
    print('=' * 66)
    print('一张图出任务:  %s' % img.name)
    print('作品名      :  %s' % name)
    print('解释器      :  %s' % PY)
    print('=' * 66)

    # ---- ① 提线 + 建计划 ----
    run([PY, 'unified_painting.py'], stdin='%s\n\n\n' % img, label='提线/建计划')

    ink = newest('%s_*_黑线任务.json' % img.stem, t0)
    plan = newest('%s_*_计划.json' % img.stem, t0)
    lineart = newest('%s_*_提取线稿.png' % img.stem, t0)
    bundle = newest('%s_*_完整绘画.json' % img.stem, t0)
    if not ink or not plan:
        raise SystemExit('没找到提线产出(黑线任务/计划)')
    r = press_report(ink)
    print('\n提线: %d 笔, 总长 %.0f mm, 下压 p50 %.4f, 低于 0.13mm 的占 %.1f%%'
          % (r['strokes'], r['length_mm'], r['p50'], r['below']))

    # ---- ② 建铺色源目录 ----
    src = OUT / ('%s_铺色源' % name)
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    shutil.copy2(img, src / '参考原图.png')
    shutil.copy2(ink, src / ('%s_黑线任务.json' % name))
    shutil.copy2(plan, src / ('%s_计划.json' % name))
    print('\n铺色源目录: %s' % src)

    # ---- ②b 黑线简化到可上机 ----
    # 顺序要求: 必须先简化再补偏移。反了会把密线压成黑块(飞瀑松崖 6474 笔那次)。
    src_ink = src / ('%s_黑线任务.json' % name)
    run([PY, 'scripts/黑线简化到可上机.py', '--job', src_ink,
         '--target-strokes', TARGET_INK_STROKES, '--apply'], label='黑线简化')
    r1 = press_report(src_ink)
    print('\n简化后: %d 笔, 总长 %.0f mm, 下压 p50 %.4f'
          % (r1['strokes'], r1['length_mm'], r1['p50']))

    # ---- ③ 下压补到落墨区间 ----
    run([PY, 'scripts/黑线下压补到落墨区间.py',
         '--job', src / ('%s_黑线任务.json' % name), '--apply'], label='下压补偏移')
    r2 = press_report(src / ('%s_黑线任务.json' % name))
    print('\n补偏移后: 下压 p50 %.4f, 低于 0.13mm 的占 %.1f%%' % (r2['p50'], r2['below']))

    # ---- ④ 稠密铺色 ----
    run([PY, 'scripts/build_filled_landscape.py',
         '--src', src, '--out-name', '%s_铺色版' % name,
         '--pitch-mm', PITCH_MM, '--shrink-mm', SHRINK_MM,
         '--colour-press-range', COLOUR_PRESS,
         '--travel-velocity', TRAVEL, '--max-stroke-mm', MAX_STROKE,
         '--reload-strokes', 8], label='稠密铺色')

    d = sorted([p for p in OUT.glob('%s_铺色版_*' % name) if p.is_dir()],
               key=lambda p: p.stat().st_mtime)[-1]
    fplan = next(d.glob('*_计划.json'))
    fink = next(d.glob('*_黑线任务.json'))

    # 计划里的每蘸笔数提到 20(运行时上限 40)
    pj = json.loads(fplan.read_text(encoding='utf-8-sig'))
    pj['modules']['color']['reload_strokes_per_load'] = RELOAD_PLAN
    fplan.write_text(json.dumps(pj, ensure_ascii=False, indent=1), encoding='utf8')

    # ---- ⑤ 报告 ----
    from collections import Counter
    st = pj['modules']['color']['strokes']
    print('\n' + '=' * 66)
    print('产出目录: %s' % d)
    print('-' * 66)
    print('黑线任务 : %s  (%.1f MB)' % (fink.name, fink.stat().st_size / 1e6))
    print('上色计划 : %s  (%.1f MB)' % (fplan.name, fplan.stat().st_size / 1e6))
    for extra in d.glob('*_预览.png'):
        print('合成预览 : %s' % extra.name)
    if lineart:
        print('提取线稿 : %s' % lineart.name)
    if bundle:
        print('完整绘画 : %s' % bundle.name)
    print('-' * 66)
    print('黑线 %d 笔, 下压 p50 %.3f mm(低于 0.13 的占 %.1f%%)'
          % (r2['strokes'], r2['p50'], r2['below']))
    print('上色 %d 笔:' % len(st))
    for c, n in Counter(s['color'] for s in st).most_common():
        sl = [x for x in st if x['color'] == c]
        print('   %-7s %3d 笔  总长 %6.0f mm' % (c, n, sum(s.get('length_mm', 0) for s in sl)))

    if a.check:
        print('-' * 66)
        for cmd, lab in (([PY, 'run_mountain_balanced.py', '--job', fink, '--check'], '黑线'),
                         ([PY, 'run_color_from_job.py', '--job', fplan, '--check'], '上色')):
            p = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), text=True,
                               capture_output=True, encoding='utf-8', errors='replace')
            last = (p.stdout or '').strip().splitlines()[-1:] or ['(无输出)']
            print('%s 校验: %s' % (lab, last[0]))

    if not a.keep_src:
        shutil.rmtree(src, ignore_errors=True)
        print('(已清理铺色源目录;--keep-src 可保留)')
    print('=' * 66)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
