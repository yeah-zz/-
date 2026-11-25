"""黑线简化到可上机 · 通用工具

问题:好提取器在"柔和渲染型"参考图上会产出几千条线(飞瀑松崖 6474 条),
      直接补下压偏移会糊成一整块黑。层峦松水那条链上是有"简化"这一环的
      (1763 -> 553),这里把它抽成通用工具。

做法:保留 source_broad_rail(宽笔道)+ 长度达标的 source_center,
      然后【重建 waypoints】(机器人路径是从 base_strokes 推出来的,只改笔触数组没用)。

阈值可以:
  --min-length-mm 4.0      固定阈值(层峦松水当年用的值)
  --target-strokes 1200    自动挑阈值,保留最长的 N 条

顺序要求:必须先简化,再补下压偏移 —— 反了会把密线压成黑块。

用法:
  python scripts\\黑线简化到可上机.py --job <黑线任务.json>                    # 只报告
  python scripts\\黑线简化到可上机.py --job <...> --target-strokes 1200 --apply
"""
import argparse, json, shutil, sys
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import job, ink_control, shuimo
from geometry import CanvasMap, check_waypoints
from run_mountain_v4_robot import safe_transfers


def pick_threshold(olds, a):
    L = np.asarray([s.get('length_mm', 0.0) for s in olds], float)
    n = L.size
    if a.min_length_mm is not None:
        return float(a.min_length_mm), 'fixed'
    if not a.target_strokes or a.target_strokes >= n:
        return 0.0, 'no-op'
    keep = np.sort(L)[::-1][:a.target_strokes]
    return float(keep[-1]), 'target'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--min-length-mm', type=float, default=None,
                    help='固定长度阈值(不传则用 --target-strokes)')
    ap.add_argument('--target-strokes', type=int, default=1200,
                    help='目标笔数,自动挑阈值(默认 1200)')
    ap.add_argument('--pine-region', action='store_true',
                    help='启用层峦松水那种"松树区域放宽到 2.4mm"的启发式(默认关)')
    a = ap.parse_args(argv)

    F = Path(a.job)
    d = json.loads(F.read_text(encoding='utf-8-sig'))
    meta = d['meta']
    snap = meta['ink_execution']
    old = snap['base_strokes']
    scale = float(snap['mm_per_px'])
    h, w = int(meta['image_size'][1]), int(meta['image_size'][0])

    rails = [s for s in old if s.get('role') == 'source_broad_rail']
    cent = [s for s in old if s.get('role') != 'source_broad_rail']
    L = np.asarray([s.get('length_mm', 0.0) for s in old], float)
    print('文件      : %s' % F.name)
    print('原件      : %d 条 (宽笔道 %d + 中心线 %d)   总长 %.0f mm'
          % (len(old), len(rails), len(cent), L.sum()))

    th, mode = pick_threshold(old, a)
    print('阈值      : %.2f mm  (%s)  松树区域放宽: %s'
          % (th, mode, '开' if a.pine_region else '关'))

    base = []
    for s in old:
        if s.get('role') == 'source_broad_rail':
            base.append(s)
            continue
        cut = th
        if a.pine_region:
            x, y = np.mean(s['points'], axis=0) / [w, h]
            if x < .45 and .35 < y < .72:
                cut = min(th, 2.4) if th else 2.4
        if s.get('length_mm', 0.0) >= cut:
            base.append(s)

    Lb = np.asarray([s.get('length_mm', 0.0) for s in base], float)
    print('简化后    : %d 条   总长 %.0f mm (保留 %.1f%% 长度)'
          % (len(base), Lb.sum(), 100.0 * Lb.sum() / max(L.sum(), 1e-9)))
    if len(base) < 20:
        raise SystemExit('简化后太少(%d 条), 阈值不对, 已中止' % len(base))

    # ---- 重建 waypoints ----
    policy = ink_control.InkPolicy(**snap['policy'])
    base = ink_control.split_long_strokes(base, scale, policy)
    base = ink_control.order_for_ink(base, scale, policy)
    for i, s in enumerate(base):
        s['id'] = 'outline-%05d' % i
    steps, _ = ink_control.build_plan(base, shuimo.ink_palette_index(), scale, policy)
    snap['base_strokes'] = base
    meta['strokes'] = len(base)

    new_outline = [v['stroke'] for v in steps if v['kind'] == 'stroke']
    if 'modules' in d and 'outline' in d['modules']:
        d['modules']['outline']['strokes'] = new_outline
        d['modules']['outline']['stroke_count'] = len(new_outline)
    else:
        # 黑线任务只含 meta+waypoints; 轮廓副本在同目录的 _计划.json 里, 一并同步
        for sib in sorted(F.parent.glob('*_计划.json')):
            try:
                pj = json.loads(sib.read_text(encoding='utf-8-sig'))
            except Exception:
                continue
            if 'modules' in pj and 'outline' in pj['modules']:
                pj['modules']['outline']['strokes'] = new_outline
                pj['modules']['outline']['stroke_count'] = len(new_outline)
                pj['simplification'] = dict(source_black_count=len(old), black_count=len(base),
                                            min_length_mm=th, mode=mode,
                                            note='ink simplified; outline copy synced')
                if a.apply:
                    shutil.copy2(sib, sib.with_name(sib.name + '.bak-' +
                                 datetime.now().strftime('%Y%m%d_%H%M%S')))
                    sib.write_text(json.dumps(pj, ensure_ascii=False), encoding='utf8')
                    print('已同步计划 : %s (outline %d 条)' % (sib.name, len(new_outline)))
                else:
                    print('计划待同步 : %s -> outline %d 条 (加 --apply 生效)' % (sib.name, len(new_outline)))

    mapper = CanvasMap(meta['image_size'])
    wp, _ = job.steps_to_waypoints(steps, mapper)
    wp = safe_transfers(wp)
    ok, msg = check_waypoints(wp)
    print('waypoints : %d 个   检查 %s %s' % (len(wp), 'OK' if ok else '失败', msg if not ok else ''))
    if not ok:
        raise SystemExit('waypoints 检查未通过: %s' % msg)
    d['waypoints'] = wp
    d['simplification'] = dict(source_black_count=len(old), black_count=len(base),
                               min_length_mm=th, mode=mode, pine_region=bool(a.pine_region),
                               note='Short ink texture removed; broad rails always kept. waypoints rebuilt.')

    ps = np.asarray([v for s in base for v in (s.get('press') or [])], float)
    if ps.size:
        print('下压      : %.4f ~ %.4f  p50 %.4f  (低于 0.13mm 占 %.1f%%)'
              % (ps.min(), ps.max(), np.median(ps), 100.0 * float(np.mean(ps < 0.13))))

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
