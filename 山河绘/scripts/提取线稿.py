r"""提取线稿 · 唯一推荐入口。

底层是 scripts/preview_contrast_landscape.py 的 width_strokes():
  algorithm = 'full-resolution-width-and-in-mask-rails'
  全分辨率 + 按原图墨线宽度分层(role=source_center / source_broad_rail, layer=0/1/2)

其它提线器(reference_lines.extract / original_d_tracer / unified_painting)已弃用,
见 docs\提线器清单.md。注意 original_d_tracer 不能删 —— 本入口的底层依赖它。

用法:
  python scripts\提取线稿.py --image 参考图\飞瀑松崖_20260930.png
  python scripts\提取线稿.py --image <图> --save-strokes 输出\xxx_线稿.json
"""
import argparse, importlib.util, json, sys
from collections import Counter
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CANVAS_W_MM = 296.5      # 画布宽(用户坐标 mm);换画布要改这里


def load_width_strokes():
    spec = importlib.util.spec_from_file_location(
        'pcl', ROOT / 'scripts' / 'preview_contrast_landscape.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.width_strokes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='参考图路径')
    ap.add_argument('--preview', default=None, help='线稿预览输出 png(可省)')
    ap.add_argument('--save-strokes', default=None, help='把提线结果存成 json(可省)')
    a = ap.parse_args(argv)

    import vision
    from shuimo import press_to_width_mm

    width_strokes = load_width_strokes()
    ref = Path(a.image)
    work, _ = vision.prepare_image(vision.load_image(str(ref)), work_size=1800)
    H, W = work.shape[:2]
    scale = CANVAS_W_MM / W
    strokes, stats = width_strokes(work, scale)

    roles = Counter(s.get('role') for s in strokes)
    layers = Counter(s.get('layer') for s in strokes)
    print('图片      : %s' % ref)
    print('工作图    : %dx%d   %.5f mm/px' % (W, H, scale))
    print('algorithm : %s' % stats.get('algorithm'))
    print('笔触      : %d 条' % len(strokes))
    print('role      : %s' % dict(roles))
    print('layer     : %s' % dict(sorted(layers.items(), key=lambda x: str(x[0]))))

    if a.preview:
        can = np.full((H, W, 3), 255, np.uint8)
        for s in strokes:
            pts = np.rint(np.asarray(s['points'], float)).astype(np.int32)
            if len(pts) < 2:
                continue
            wm = [press_to_width_mm(float(v)) for v in s['press']]
            for k in range(len(pts) - 1):
                t = max(1, int(round((wm[k] + wm[k+1]) / 2 / scale)))
                cv2.line(can, tuple(pts[k]), tuple(pts[k + 1]), (36, 34, 40), t, cv2.LINE_8)
        ok, buf = cv2.imencode('.png', can)
        if not ok:
            raise RuntimeError('imencode failed')
        buf.tofile(str(a.preview))
        print('预览      : %s' % a.preview)

    if a.save_strokes:
        Path(a.save_strokes).write_text(
            json.dumps(dict(algorithm=stats.get('algorithm'), image=str(ref),
                            work_size=[W, H], mm_per_px=scale,
                            count=len(strokes), strokes=strokes),
                       ensure_ascii=False), encoding='utf8')
        print('笔触 JSON : %s' % a.save_strokes)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
