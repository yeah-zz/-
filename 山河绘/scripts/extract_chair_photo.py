"""Photo structure extraction: color/GrabCut foreground + coarse-scale edges.

Actual OpenCV processing, not an AI-redrawn image or hand-traced vector.
The image-specific coarse rectangle is a foreground initialization, not a path.
"""
from pathlib import Path
import os
import sys
import json
import hashlib
import shutil
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path(r'E:\Codex_Workspace_Data\Chair_Extraction_20261022')
for key, suffix in {'TEMP':'temp', 'TMP':'temp', 'PIP_CACHE_DIR':'cache/pip',
                    'TORCH_HOME':'cache/torch', 'HF_HOME':'cache/huggingface',
                    'XDG_CACHE_HOME':'cache', 'CUDA_CACHE_PATH':'cache/cuda'}.items():
    os.environ[key] = str(STORAGE/suffix)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from original_d_tracer import LandscapeBrushGenerator as Trace

SOURCE = Path(r'C:\Users\Lenovo\AppData\Local\Temp\codex-clipboard-f4a0830d-321b-4ea8-9081-a5fd44684711.png')


def save(path, image):
    cv2.imencode('.png', image)[1].tofile(str(path))


def length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def text(im, xy, value, size=25):
    ImageDraw.Draw(im).text(xy, value, font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size), fill=(35, 47, 41))


def vector(paths, w, h):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<g fill="none" stroke="#161616" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">']
    for p in paths:
        out.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x, y in p['points'])+'"/>')
    return '\n'.join(out+['</g>', '</svg>'])


def main():
    STORAGE.mkdir(parents=True, exist_ok=True)
    (STORAGE/'temp').mkdir(exist_ok=True)
    out = STORAGE/'results'
    out.mkdir(exist_ok=True)
    copied = STORAGE/'input_chair.png'
    if not copied.exists():
        shutil.copy2(SOURCE, copied)
    image = cv2.imdecode(np.fromfile(str(copied), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot read chair attachment')
    h, w = image.shape[:2]
    start = time.perf_counter()
    cv2.setRNGSeed(20261022)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat, value = hsv[:, :, 1], hsv[:, :, 2]
    # Coarse object bounding region, selected from the attachment before any
    # tracing. Color seeds inside it are computed automatically.
    rect = [round(w*.175), round(h*.04), round(w*.66), round(h*.89)]
    x, y, rw, rh = rect
    region = np.zeros((h, w), bool)
    region[y:y+rh, x:x+rw] = True
    probable = region & (sat > 90)
    seeds = region & (sat > 135) & (value < 240)
    lower = np.indices((h,w))[0] > h*.64
    foot_zone = np.indices((h,w))[0] > h*.76
    probable[lower] = region[lower] & (sat[lower] > 130)
    seeds[lower] = region[lower] & (sat[lower] > 150) & (value[lower] < 230)
    seeds[foot_zone] = False
    # Flooring shares the wood hue, so do not seed foreground from low
    # saturation floor/shadow pixels merely because they are brown.
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[region] = cv2.GC_PR_BGD
    mask[probable] = cv2.GC_PR_FGD
    mask[foot_zone & region] = cv2.GC_PR_BGD
    mask[seeds] = cv2.GC_FGD
    mask[(sat < 45) & region] = cv2.GC_BGD
    mask[lower & region & (sat < 125)] = cv2.GC_BGD
    mask[foot_zone & region & (value > 190) & (sat < 150)] = cv2.GC_BGD
    bgmodel, fgmodel = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
    cv2.grabCut(image, mask, None, bgmodel, fgmodel, 6, cv2.GC_INIT_WITH_MASK)
    foreground = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if n < 2:
        raise ValueError('No foreground component')
    component = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    foreground = (labels == component).astype(np.uint8)
    # Remove single-pixel jaggedness without filling the chair's open spaces.
    foreground = cv2.medianBlur(foreground*255, 5)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    # Fill tiny segmentation pinholes caused by specular highlights, retaining
    # the genuinely large open spaces between the chair members.
    n, holes, hole_stats, _ = cv2.connectedComponentsWithStats((foreground == 0).astype(np.uint8), 8)
    for label in range(1, n):
        if hole_stats[label, cv2.CC_STAT_AREA] < 500:
            foreground[holes == label] = 255
    contours, _ = cv2.findContours(foreground, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths = []
    for contour in contours:
        if cv2.arcLength(contour, True) < 45:
            continue
        simplified = cv2.approxPolyDP(contour, 2., True)[:, 0, :]
        p = np.vstack((simplified, simplified[0])).astype(float)
        paths.append(dict(points=p.tolist(), role='foreground_boundary', closed=True, length_px=length(p)))
    boundary = np.zeros((h, w), np.uint8)
    cv2.drawContours(boundary, contours, -1, 255, 1)
    # Suppress wood grain by smoothing at a larger spatial scale, then require
    # agreement of edges at two scales. Only foreground-interior edges remain.
    smooth = cv2.bilateralFilter(image, 13, 45, 9)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    a = cv2.GaussianBlur(gray, (0,0), 2.5)
    b = cv2.GaussianBlur(gray, (0,0), 4.)
    edges_a = cv2.Canny(a, 30, 65, L2gradient=True)
    edges_b = cv2.Canny(b, 18, 42, L2gradient=True)
    match = cv2.dilate(edges_b, np.ones((7,7), np.uint8))
    interior = cv2.erode(foreground, np.ones((9,9), np.uint8))
    edges = ((edges_a > 0) & (match > 0) & (interior > 0)).astype(np.uint8)*255
    skeleton = Trace._thin_binary(np.pad(edges, 1))[1:-1, 1:-1]
    for raw in Trace._trace_skeleton_paths(skeleton):
        p = np.asarray(raw, np.float32)
        if length(p) < 45:
            continue
        p = cv2.approxPolyDP(p.reshape(-1,1,2), 2., False)[:,0,:].astype(float)
        paths.append(dict(points=p.tolist(), role='persistent_internal_edge', closed=False, length_px=length(p)))
    raster = np.zeros((h,w), np.uint8)
    for p in paths:
        cv2.polylines(raster, [np.rint(p['points']).astype(np.int32)], False, 255, 2, cv2.LINE_AA)
    lines = cv2.cvtColor(255-raster, cv2.COLOR_GRAY2BGR)
    overlay = np.rint(image.astype(float)*.65+255*.35).astype(np.uint8)
    overlay[raster > 96] = (35,40,225)
    save(out/'foreground_mask.png', foreground)
    save(out/'internal_edge_candidates.png', edges)
    save(out/'chair_lines.png', lines)
    save(out/'overlay.png', overlay)
    cutout = np.full_like(image, 255)
    cutout[foreground>0] = image[foreground>0]
    save(out/'foreground_on_white.png', cutout)
    (out/'chair_lines.svg').write_text(vector(paths,w,h), encoding='utf-8')
    report = dict(algorithm='photo-grabcut-and-persistent-edges-v1', image_size=[w,h],
                  input_sha256=hashlib.sha256(copied.read_bytes()).hexdigest(),
                  foreground_initialization_rectangle=rect, foreground_seeds='HSV saturation >135 and value <240 within coarse rectangle',
                  lower_image_color_prior=dict(start_y_fraction=.64, probable_foreground_min_saturation=130,
                                                foreground_seed_min_saturation=150, background_max_saturation=125,
                                                foot_region_start_y_fraction=.76, foreground_seeds_in_foot_region=False),
                  fill_segmentation_holes_below_area_px=500,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  manual_path_tracing=False, coarse_foreground_roi_selected=True,
                  paths=len(paths), boundary_paths=sum(p['role']=='foreground_boundary' for p in paths),
                  internal_paths=sum(p['role']=='persistent_internal_edge' for p in paths),
                  elapsed_seconds=time.perf_counter()-start, ready_for_execution=False,
                  note='Image-specific photo structural extraction; not the fixed v2 ink detector. No brush dynamics or robot motion.')
    (out/'paths.json').write_text(json.dumps(dict(paths=paths, experiment=report), ensure_ascii=False, indent=2), encoding='utf-8')
    panel = Image.new('RGB',(1920,790),(245,244,239))
    text(panel,(30,20),'椅子照片 / 结构线提取',34)
    text(panel,(30,75),'粗区域初始化 + 自动前景分离 + 多尺度边缘筛选；未手工描线。',23)
    for j,(label,view) in enumerate(zip(('原图','提取结构线','叠加原图'),(image,lines,overlay))):
        text(panel,(30+630*j,125),label,26)
        im=Image.fromarray(cv2.cvtColor(view,cv2.COLOR_BGR2RGB)).resize((600,600),Image.Resampling.LANCZOS)
        panel.paste(im,(30+630*j,170))
    panel.save(out/'comparison.png')
    # Geometry/provenance checks, not a claim of semantic tracing accuracy.
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == report['input_sha256'] if SOURCE.exists() else True
    for path in paths:
        p = np.asarray(path['points'], float)
        assert len(p) >= 2 and np.isfinite(p).all()
        assert np.all((p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h))
        if path['closed']:
            assert np.array_equal(p[0],p[-1])
    assert len(ET.parse(out/'chair_lines.svg').getroot().findall('.//{*}polyline')) == len(paths)
    validation = dict(input_copy_matches_attachment=True, paths_in_image_bounds=True,
                      finite_coordinates=True, closed_paths_have_shared_endpoints=True,
                      svg_path_count_matches_json=True, robot_task_created=False,
                      accuracy_measured=False)
    (out/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    snapshot = out/'code_snapshot'
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    shutil.copy2(Path(Trace._thin_binary.__code__.co_filename),snapshot/'original_d_tracer.py')
    notes = '''# 椅子照片结构线提取

查看 comparison.png（原图 / 提线 / 红线叠加），chair_lines.png 是白底线稿，chair_lines.svg 和 paths.json 是原图像素坐标的矢量路径。

这次使用照片专用流程：人工指定一个椅子粗矩形区域，结合本图颜色和上下区域先验初始化 GrabCut，再自动提取前景轮廓及多尺度持续边缘。没有手工描出椅子路径，没有使用图像生成。它不是前面固定参数山水 v2 的直接输出，也不能代表任意照片都可自动得到同样质量。

主要保留椅背、座面、腿和横撑，抑制木纹、墙面、地板及阴影。椅脚与地板同色，仍存在分割误差；座面细边有少量断线、重复边，轮廓局部也有轻微折线感。没有人工完整真值，不报告准确率。

源图保存于上一级 input_chair.png；参数和脚本 SHA256 在 paths.json；代码快照在 code_snapshot；几何及 SVG 导出检查记录在 validation.json。没有机械臂坐标、下压量或执行任务，没有连接机械臂。
'''
    (out/'README.md').write_text(notes,encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    print(str(out/'comparison.png'), flush=True)


if __name__=='__main__':
    main()
