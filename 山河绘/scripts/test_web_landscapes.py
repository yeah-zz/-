"""Fixed-parameter v2 evaluation on four newly downloaded museum paintings.

No project reference images, algorithm changes, per-image tuning or robot calls.
All inputs/results live under the dedicated E-drive experiment directory.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import os
import shutil
import sys
import time
import argparse
import threading

ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path(r'E:\Codex_Workspace_Data\Landscape_Web_Test_20261022')
for key, suffix in {'TEMP': 'temp', 'TMP': 'temp', 'PIP_CACHE_DIR': 'cache/pip',
                    'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface',
                    'XDG_CACHE_HOME': 'cache', 'CUDA_CACHE_PATH': 'cache/cuda'}.items():
    os.environ[key] = str(STORAGE/suffix)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from source_ink import extract, InkConfig

OUT = STORAGE/'results'
# Crops selected from the source images BEFORE extraction, to remove mounting.
# Stored in museum web-image coordinates, scaled to the print image exactly.
# No calligraphy or seals within the painting are erased.
WINDOWS = {
    '01_zha_shibiao': dict(web_size=[1263, 803], box=[26, 31, 1234, 726],
                           detail=[.09, .10, .79, .88]),
    '02_wang_yuanqi': dict(web_size=[500, 893], box=[14, 12, 486, 880],
                           detail=[.05, .28, .96, .58]),
    '03_wang_meng': dict(web_size=[1263, 888], box=[12, 13, 1253, 881],
                           detail=[.42, .48, .83, .91]),
    '04_chen_ruyan': dict(web_size=[1263, 346], box=[93, 4, 1077, 340],
                           detail=[.03, .04, .52, .94]),
}


def save(path, bgr):
    cv2.imencode('.png', bgr)[1].tofile(str(path))


def text(canvas, xy, value, size=25, fill=(42, 49, 46)):
    ImageDraw.Draw(canvas).text(xy, value, font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size), fill=fill)


def place(canvas, image, box):
    x, y, w, h = box
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ratio = min(w/image.width, h/image.height)
    im = image.resize((round(image.width*ratio), round(image.height*ratio)), Image.Resampling.LANCZOS)
    canvas.paste(im, (x+(w-im.width)//2, y+(h-im.height)//2))


def as_svg(paths, w, h):
    result = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
              '<g fill="none" stroke="#111" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">']
    for path in paths:
        coords = ' '.join(f'{x:.2f},{y:.2f}' for x, y in path['points'])
        result.append(f'<polyline points="{coords}"/>')
    return '\n'.join(result + ['</g>', '</svg>'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Reuse completed results and recorded timeout cases')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = json.loads((STORAGE/'sources.json').read_text(encoding='utf-8'))
    cfg = InkConfig()  # Identical values to the previous v2 experiment.
    module_hash = hashlib.sha256((ROOT/'source_ink.py').read_bytes()).hexdigest()
    records, previews = [], []
    for source in sources:
        key = source['key']
        original_path = Path(source['downloaded_file'])
        assert original_path.resolve().is_relative_to(STORAGE.resolve()) if hasattr(Path, 'is_relative_to') else str(original_path.resolve()).startswith(str(STORAGE.resolve()) + os.sep)
        assert hashlib.sha256(original_path.read_bytes()).hexdigest() == source['sha256']
        window = WINDOWS[key]
        im = Image.open(original_path).convert('RGB')
        original_size = list(im.size)
        crop = [round(v * im.size[k % 2]/window['web_size'][k % 2]) for k, v in enumerate(window['box'])]
        im = im.crop(crop)
        crop_size = list(im.size)
        factor = min(1., 1800/max(im.size))
        im = im.resize((round(im.width*factor), round(im.height*factor)), Image.Resampling.LANCZOS)
        work = np.asarray(im)[:, :, ::-1].copy()
        folder = OUT/key
        folder.mkdir(parents=True, exist_ok=True)
        if args.resume and (folder/'failure.json').exists():
            failure = json.loads((folder/'failure.json').read_text(encoding='utf-8-sig'))
            save(folder/'input.png', work)
            record = dict(source=source, original_size=original_size, crop_box_in_download=crop,
                          crop_size=crop_size, work_size=list(im.size), resize_factor=factor,
                          same_config_for_all=True, config=asdict(cfg), source_ink_sha256=module_hash,
                          status=failure['status'], failure=failure, stats=None, seconds=None,
                          accuracy_evaluated=False, ready_for_execution=False)
            (folder/'experiment.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            records.append(record)
            previews.append((source, [work, None, None]))
            print('RECORDED_TIMEOUT', key, flush=True)
            continue
        if args.resume and (folder/'experiment.json').exists():
            record = json.loads((folder/'experiment.json').read_text(encoding='utf-8'))
            assert record['source_ink_sha256'] == module_hash and record['config'] == asdict(cfg)
            lineart = cv2.imdecode(np.fromfile(str(folder/'centerlines.png'), np.uint8), cv2.IMREAD_COLOR)
            overlay = cv2.imdecode(np.fromfile(str(folder/'overlay.png'), np.uint8), cv2.IMREAD_COLOR)
            assert lineart is not None and overlay is not None and (folder/'comparison.png').exists()
            records.append(record)
            previews.append((source, [work, lineart, overlay]))
            print('REUSED', key, flush=True)
            continue
        print('RUNNING', key, work.shape[1], work.shape[0], flush=True)
        save(folder/'input.png', work)
        def timed_out():
            failure = dict(key=key, status='stopped_after_runtime_limit',
                           reason='Extraction exceeded the 600 second per-image runtime budget',
                           elapsed_lower_bound_seconds=600, completed_extraction_saved=False)
            (folder/'failure.json').write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding='utf-8')
            print('TIMEOUT', key, flush=True)
            os._exit(124)
        timer = threading.Timer(600, timed_out)
        timer.daemon = True
        timer.start()
        start = time.perf_counter()
        try:
            result = extract(work, cfg)
        finally:
            timer.cancel()
        seconds = time.perf_counter() - start
        lineart = cv2.cvtColor(255-result['raster'], cv2.COLOR_GRAY2BGR)
        overlay = np.rint(work.astype(float)*.65+255*.35).astype(np.uint8)
        overlay[cv2.dilate(result['raster'], np.ones((2, 2), np.uint8)) > 0] = (45, 40, 225)
        save(folder/'input.png', work)
        save(folder/'centerlines.png', lineart)
        save(folder/'overlay.png', overlay)
        save(folder/'ink_mask.png', result['mask'])
        save(folder/'weak_evidence.png', result['weak_mask'])
        record = dict(source=source, original_size=original_size, crop_box_in_download=crop,
                      crop_size=crop_size, work_size=list(im.size), resize_factor=factor,
                      preprocessing='crop mounting and resize long edge <=1800; no ink enhancement or text removal',
                      same_config_for_all=True, config=asdict(cfg), seconds=seconds,
                      source_ink_sha256=module_hash, opencv_version=cv2.__version__,
                      numpy_version=np.__version__, stats=result['stats'],
                      status='completed',
                      accuracy_evaluated=False, ready_for_execution=False)
        serial = dict(paths=result['paths'], dots=result['dots'], experiment=record)
        (folder/'paths.json').write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding='utf-8')
        (folder/'centerlines.svg').write_text(as_svg(result['paths'], im.width, im.height), encoding='utf-8')
        (folder/'experiment.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        views = [work, lineart, overlay]
        panel = Image.new('RGB', (1920, 1440), (245, 244, 239))
        text(panel, (30, 20), source['label']+' / '+source['test_category'], 34)
        text(panel, (30, 75), f"同一套 v2 参数；{im.width}×{im.height}；{len(result['paths'])} 条几何路径。", 23)
        text(panel, (30, 113), '全图与固定局部对照；题跋和印章保留；红色为算法轨迹叠加。', 22)
        for col, (label, view) in enumerate(zip(('原图', 'v2 中心线', '叠加原图'), views)):
            text(panel, (30+630*col, 165), label, 26)
            place(panel, view, (30+630*col, 208, 600, 600))
        x0, y0, x1, y1 = [round(v*im.size[k % 2]) for k, v in enumerate(window['detail'])]
        text(panel, (30, 842), '局部放大（提取前指定区域）', 26)
        for col, view in enumerate(views):
            place(panel, view[y0:y1, x0:x1], (30+630*col, 894, 600, 480))
        text(panel, (30, 1395), 'Cleveland Museum of Art / CC0；仅离线几何测试，未校准毛笔，也未生成执行任务。', 21)
        panel.save(folder/'comparison.png')
        previews.append((source, views))
        records.append(record)
        print(json.dumps(dict(key=key, paths=len(result['paths']), seconds=round(seconds, 2),
                              compact_marks=len(result['dots'])), ensure_ascii=True), flush=True)
    (OUT/'experiments.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    overview = Image.new('RGB', (2040, 2110), (245, 244, 239))
    text(overview, (32, 22), '网上新图测试 / 四张真实山水画 / 固定 v2 参数', 36)
    text(overview, (32, 80), '仅去装裱边、统一长边上限；保留题跋与印章；没有逐图调参和人工修线。', 23)
    for col, label in enumerate(('网上原图', '提取中心线', '轨迹叠加')):
        text(overview, (32+670*col, 128), label, 27)
    for row, (source, views) in enumerate(previews):
        y = 186+row*472
        text(overview, (32, y), source['label']+' · '+source['test_category'], 26)
        for col, view in enumerate(views):
            if view is None:
                text(overview, (75+670*col, y+170), '运行超过 10 分钟，已停止', 25)
                text(overview, (75+670*col, y+215), '未获得完整提线结果', 23)
            else:
                place(overview, view, (32+670*col, y+44, 630, 398))
    text(overview, (32, 2075), '真实算法输出；没有人工真值，不报告准确率。细节见每张图的 comparison.png。', 21)
    overview.save(OUT/'overview.png')
    snapshot = OUT/'code_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in [ROOT/'source_ink.py', ROOT/'original_d_tracer.py', Path(__file__).resolve(), ROOT/'scripts'/'fetch_web_landscapes.py']:
        shutil.copy2(path, snapshot/path.name)
    print('DONE', str(OUT/'overview.png'), flush=True)


if __name__ == '__main__':
    main()
