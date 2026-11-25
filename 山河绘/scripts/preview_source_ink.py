"""Reproducible offline precision tracing comparison; never imports a robot."""
from pathlib import Path
import argparse
import json
import os
import sys
import hashlib
import time
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path(r'E:\Codex_Workspace_Data\CLIPascene_20261021')
for key, value in {'TEMP': STORAGE/'temp', 'TMP': STORAGE/'temp',
                   'PIP_CACHE_DIR': STORAGE/'cache'/'pip', 'TORCH_HOME': STORAGE/'cache'/'torch',
                   'HF_HOME': STORAGE/'cache'/'huggingface', 'XDG_CACHE_HOME': STORAGE/'cache',
                   'CUDA_CACHE_PATH': STORAGE/'cache'/'cuda'}.items():
    os.environ[key] = str(value)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from source_ink import extract, InkConfig


def read(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def save(path, image):
    cv2.imencode('.png', image)[1].tofile(str(path))


def paths_raster(paths, shape, thickness=1):
    out = np.zeros(shape[:2], np.uint8)
    for p in paths:
        cv2.polylines(out, [np.rint(p['points']).astype(np.int32)], False, 255, thickness)
    return out


def svg(paths, width, height):
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<g fill="none" stroke="#111" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">']
    for p in paths:
        coords = ' '.join(f'{x:.2f},{y:.2f}' for x, y in p['points'])
        lines.append(f'<polyline points="{coords}"/>')
    return '\n'.join(lines + ['</g>', '</svg>'])


def clip_centerlines(svg_file, shape):
    """Re-render this experiment's native M/C paths at the same 1 px width.

    Mapping is the documented 224 -> 1110 square, then remove 180 px top pad.
    No re-optimization, thinning, stroke filtering or new curve fitting.
    """
    root = ET.parse(svg_file).getroot()
    factor = 1110 / float(root.attrib['width'])
    canvas = np.zeros(shape[:2], np.uint8)
    for node in root.findall('.//{*}path'):
        tokens = re.findall(r'[A-Za-z]|[-+]?(?:\d*\.?\d+)(?:[eE][-+]?\d+)?', node.attrib['d'])
        if len(tokens) != 10 or tokens[0] != 'M' or tokens[3] != 'C':
            raise ValueError('Expected a single native cubic M/C curve')
        control = np.array([float(x) for k, x in enumerate(tokens) if k not in (0, 3)]).reshape(4, 2)
        control *= factor
        control[:, 1] -= 180
        n = max(12, int(np.ceil(np.linalg.norm(np.diff(control, axis=0), axis=1).sum())))
        t = np.linspace(0, 1, n)[:, None]
        p = ((1-t)**3*control[0] + 3*(1-t)**2*t*control[1] +
             3*(1-t)*t**2*control[2] + t**3*control[3])
        cv2.polylines(canvas, [np.rint(p).astype(np.int32)], False, 255, 1)
    return cv2.cvtColor(255-canvas, cv2.COLOR_GRAY2BGR)


def draw_text(image, xy, text, size=24, fill=(45, 52, 50)):
    ImageDraw.Draw(image).text(xy, text, font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size), fill=fill)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'offline_previews'/'clipascene_20261021'/'input_landscape.png')
    parser.add_argument('--output', type=Path, default=ROOT/'offline_previews'/'source_ink_v2_20261022')
    parser.add_argument('--strong', type=float, default=InkConfig.strong_contrast)
    parser.add_argument('--weak', type=float, default=InkConfig.weak_contrast)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = read(args.input)
    if source is None:
        raise ValueError(f'Cannot read {args.input}')
    started = time.perf_counter()
    result = extract(source, InkConfig(strong_contrast=args.strong, weak_contrast=args.weak))
    result['stats']['extraction_seconds'] = time.perf_counter() - started
    result['stats']['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result['stats']['opencv_version'] = cv2.__version__
    result['stats']['numpy_version'] = np.__version__
    out = args.output
    save(out/'input.png', source)
    save(out/'ink_mask.png', result['mask'])
    save(out/'centerlines.png', cv2.cvtColor(255-result['raster'], cv2.COLOR_GRAY2BGR))
    save(out/'weak_evidence.png', result['weak_mask'])
    overlay = (source.astype(float) * .6 + 255 * .4).astype(np.uint8)
    overlay[cv2.dilate(result['raster'], np.ones((2, 2), np.uint8)) > 0] = (45, 40, 225)
    save(out/'overlay.png', overlay)
    serial = {k: result[k] for k in ('paths', 'dots', 'stats')}
    serial['input'] = str(args.input.resolve())
    (out/'paths.json').write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'centerlines.svg').write_text(svg(result['paths'], source.shape[1], source.shape[0]), encoding='utf-8')
    baseline_file = ROOT/'offline_previews'/'clipascene_20261021'/'existing_strokes.json'
    same_input = args.input.resolve() == (baseline_file.parent/'input_landscape.png').resolve()
    if same_input:
        baseline = paths_raster(json.loads(baseline_file.read_text(encoding='utf-8')), source.shape)
        save(out/'baseline_centerlines.png', cv2.cvtColor(255-baseline, cv2.COLOR_GRAY2BGR))
        # Fixed, documented windows, unchanged across old/new extraction.
        regions = [('山脊', (460, 0, 880, 305)), ('松树分叉', (0, 170, 490, 490)),
                   ('亭子屋顶与立柱', (900, 285, 1110, 485))]
        views = [source, cv2.cvtColor(255-baseline, cv2.COLOR_GRAY2BGR),
                 cv2.cvtColor(255-result['raster'], cv2.COLOR_GRAY2BGR), overlay]
        panel = Image.new('RGB', (2040, 1330), (245, 244, 239))
        draw_text(panel, (32, 20), '原图约束提线 v2 / 三个局部同坐标对照', 34)
        draw_text(panel, (32, 75), '红线为算法轨迹叠加；不是人工真值。统一线宽展示，不添加毛笔纹理。', 23)
        for col, label in enumerate(('原图', '旧提线器', 'v2 中心线', 'v2 叠加原图')):
            draw_text(panel, (32 + col*505, 120), label, 26)
        for row, (label, box) in enumerate(regions):
            top = 175 + row*375
            draw_text(panel, (32, top), label, 25)
            x0, y0, x1, y1 = box
            for col, view in enumerate(views):
                im = Image.fromarray(cv2.cvtColor(view[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
                factor = min(480 / im.width, 325 / im.height)
                im = im.resize((round(im.width*factor), round(im.height*factor)), Image.Resampling.LANCZOS)
                panel.paste(im, (32 + col*505, top+42))
        panel.save(out/'detail_comparison.png')
        overview = Image.new('RGB', (1920, 1510), (245, 244, 239))
        draw_text(overview, (32, 22), '原图约束提线 v2 / 整图对照', 36)
        draw_text(overview, (32, 78), '仅检验路径位置和连接；点苔候选另存，不计入线条；未生成机械臂任务。', 24)
        for i, (label, view) in enumerate(zip(('原图', '旧提线器', 'v2 中心线', 'v2 叠加原图'), views)):
            x, y = 32 + (i % 2)*950, 140 + (i // 2)*670
            draw_text(overview, (x, y), label, 28)
            im = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)).resize((900, 608), Image.Resampling.LANCZOS)
            overview.paste(im, (x, y+45))
        overview.save(out/'comparison_overview.png')
        clip_file = baseline_file.parent/'layer2_paths128_seed0'/'best_iter.svg'
        if clip_file.exists():
            methods = Image.new('RGB', (1920, 625), (245, 244, 239))
            draw_text(methods, (32, 20), '同一张原图：CLIPascene 与原图约束提线 v2', 34)
            draw_text(methods, (32, 75), '真实运行结果；统一等宽线条展示。v2 保留源图坐标，不重新生成构图。', 23)
            clip_view = clip_centerlines(clip_file, source.shape)
            save(out/'clip_centerlines_1px.png', clip_view)
            method_views = [source, clip_view, cv2.cvtColor(255-result['raster'], cv2.COLOR_GRAY2BGR)]
            for j, (label, view) in enumerate(zip(('原图', '之前的 CLIPascene / 第 2 层', '新版 v2 / 中心线'), method_views)):
                draw_text(methods, (32+j*630, 125), label, 25)
                im = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)).resize((600, 405), Image.Resampling.LANCZOS)
                methods.paste(im, (32+j*630, 172))
            draw_text(methods, (32, 592), '仅为单张山水样本；尚无人工完整真值，不报告真实画作提线准确率。', 20)
            methods.save(out/'method_comparison.png')
        (out/'regions.json').write_text(json.dumps(regions, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['stats'], ensure_ascii=True, indent=2))
    print(str(out.resolve()))


if __name__ == '__main__':
    main()
