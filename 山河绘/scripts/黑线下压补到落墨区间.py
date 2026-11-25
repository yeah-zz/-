"""黑线下压补到落墨区间 · 通用工具

问题:提线器按原图墨线的【实际宽度】算下压,细线的下压只有 0.06~0.10 mm。
而实测的落墨阈值是 >= 0.13 mm(见 项目交接_20260930.md / 梅花那轮实测),
<= 0.10 mm 整档不落墨 —— 不补偏移,大部分线画不出来。

做法:整批加一个常数偏移(保持原本的粗细分布形状),把 p50 抬到目标值。
      偏移后再按 --max-press 夹一次。

用法:
  python scripts\\黑线下压补到落墨区间.py --job <黑线任务.json>              # 只报告
  python scripts\\黑线下压补到落墨区间.py --job <...> --apply               # 写回(留备份)
  python scripts\\黑线下压补到落墨区间.py --job <...> --apply --target-p50 0.30
"""
import argparse, json, shutil, sys
from datetime import datetime
from pathlib import Path
import numpy as np

MARK_MM = 0.13          # 实测落墨阈值
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def stats(press):
    a = np.asarray(press, float)
    if a.size == 0:
        return None
    return dict(n=int(a.size), lo=float(a.min()), p50=float(np.median(a)), hi=float(a.max()),
                below=100.0 * float(np.mean(a < MARK_MM)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True, help='黑线任务 json')
    ap.add_argument('--apply', action='store_true', help='写回文件(默认只报告)')
    ap.add_argument('--offset', type=float, default=None, help='显式偏移 mm(默认自动算)')
    ap.add_argument('--target-p50', type=float, default=0.28, help='目标 p50(默认 0.28)')
    ap.add_argument('--max-press', type=float, default=0.80, help='下压上限(默认 0.80)')
    a = ap.parse_args(argv)

    F = Path(a.job)
    d = json.loads(F.read_text(encoding='utf-8-sig'))
    bs = d['meta']['ink_execution']['base_strokes']
    before = stats([v for s in bs for v in (s.get('press') or [])])
    print('文件      : %s' % F)
    print('笔触      : %d 条' % len(bs))
    print('下压(前)  : %.4f ~ %.4f  p50 %.4f  低于 %.2fmm 的点 %.1f%%' % (
        before['lo'], before['hi'], before['p50'], MARK_MM, before['below']))

    off = a.offset if a.offset is not None else max(0.0, a.target_p50 - before['p50'])
    print('偏移      : +%.3f mm %s' % (off, '(显式)' if a.offset is not None else '(自动: p50 -> %.2f)' % a.target_p50))

    n_clip = 0
    for s in bs:
        if not s.get('press'):
            continue
        new = []
        for v in s['press']:
            nv = float(v) + off
            if nv > a.max_press:
                nv = a.max_press
                n_clip += 1
            new.append(round(nv, 4))
        s['press'] = new
        s['press_max'] = round(max(new), 4)
        s['press_mean'] = round(float(np.mean(new)), 4)

    after = stats([v for s in bs for v in (s.get('press') or [])])
    print('下压(后)  : %.4f ~ %.4f  p50 %.4f  低于 %.2fmm 的点 %.1f%%' % (
        after['lo'], after['hi'], after['p50'], MARK_MM, after['below']))
    if n_clip:
        print('  夹到上限 %d 个点' % n_clip)

    # 同步 waypoints 里 draw 段的 z(它存的是绝对 Z)
    PZ = d['meta'].get('paper_z')
    if PZ is not None:
        n_z = 0
        for q in d['waypoints']:
            tag = str(q.get('tag', ''))
            if tag.startswith(('down:', 'draw:')) and q.get('z') is not None:
                q['z'] = round(float(q['z']) + off, 4)
                n_z += 1
        print('waypoints : 同步 %d 个点的 z(+%.3f, paper_z=%s)' % (n_z, off, PZ))
    else:
        print('waypoints : 无 paper_z, 未同步 z')

    if a.apply:
        bak = F.with_name(F.name + '.bak-' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        shutil.copy2(F, bak)
        F.write_text(json.dumps(d, ensure_ascii=False), encoding='utf8')
        print('已写回。备份: %s' % bak.name)
    else:
        print('(未写回;加 --apply 生效)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
