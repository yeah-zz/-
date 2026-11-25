# -*- coding: utf-8 -*-
"""
vision.py - 图像 -> 色块 -> 宽笔触（项目第 1、2 步）
====================================================================

【算法流程】
  1. 预处理：等比缩放到工作分辨率并留白填满画布长宽比（或按用户标注的四点）
  2. 纸色检测：图像四周边缘像素的中位色 = 纸色，用于识别留白（云、纸底）
  3. 光照校正：逐通道除以大尺度模糊，消除照片的明暗渐变
  4. 颜色分区：标准 Lab 空间 KMeans（用 cv2.kmeans，无需 scikit-learn），
     再把每个簇吸附到调色板中最接近的颜色；接近纸色的簇标记为留白
  5. 笔触生成：每个色块求主轴方向，沿主轴做平行扫掠线，线距 = 笔宽 × 系数，
     扫掠线在色块内的连续段 = 一条宽笔触（过长会切段）

【为什么用扫掠线】
  目标效果是"色块 + 大笔触"。扫掠线天然保证铺满、不重不漏，实现简单且鲁棒，
  细长形状（树枝）用主轴方向的扫掠线也只会得到 1~2 笔，效果正确。
"""

import math

import cv2
import numpy as np

from config import (
    BRUSH_PRESETS,
    CanvasConfig,
    PALETTE,
    VisionConfig,
    brush_of,
)
import imgio


# =====================================================================
# 1. 颜色空间与图像工具
# =====================================================================
def to_std_lab(bgr):
    """
    OpenCV 的 8 位 Lab（L 0~255，a/b 偏移 128）转标准 Lab：
    L* 0~100，a*/b* 约 -128~127。这样距离阈值（色差）才有直观含义。
    """
    lab_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = np.empty_like(lab_cv)
    lab[..., 0] = lab_cv[..., 0] * (100.0 / 255.0)
    lab[..., 1] = lab_cv[..., 1] - 128.0
    lab[..., 2] = lab_cv[..., 2] - 128.0
    return lab


def rgb_to_std_lab(rgb):
    """单个 RGB -> 标准 Lab（调色板用）"""
    patch = np.zeros((1, 1, 3), dtype=np.uint8)
    patch[0, 0] = (int(rgb[2]), int(rgb[1]), int(rgb[0]))   # RGB -> BGR
    return to_std_lab(patch).reshape(3)


def load_image(path):
    img = imgio.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法读取图片: %s" % path)
    return img


def canvas_aspect():
    """纸张绘制区域的长宽比（宽 / 高）"""
    corners = CanvasConfig.ROBOT_CORNERS
    return canvas_size_mm()[0] / max(1e-6, canvas_size_mm()[1])


def canvas_size_mm():
    """纸张绘制区域的 (宽, 高) mm"""
    c = CanvasConfig.ROBOT_CORNERS
    top = math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1])
    bottom = math.hypot(c[2][0] - c[3][0], c[2][1] - c[3][1])
    left = math.hypot(c[3][0] - c[0][0], c[3][1] - c[0][1])
    right = math.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1])
    return (0.5 * (top + bottom), 0.5 * (left + right))


def prepare_image(bgr, work_size=None):
    """
    整理成"用于分割与绘制的工作图"：
      - 若 CanvasConfig.IMAGE_CORNERS 已给出：不缩放不留白，直接用原图，
        四点坐标就是用户标注的像素坐标（此时 WORK_SIZE 不生效）
      - 否则：先按画布长宽比留白（FIT_MODE），再等比缩放到 WORK_SIZE
    返回 (work_bgr, info)
    """
    h, w = bgr.shape[:2]

    if CanvasConfig.IMAGE_CORNERS is not None:
        info = {
            "prepared_size": (w, h),
            "scale": 1.0,
            "padded": False,
            "pad": (0, 0, 0, 0),
            "image_corners": [(float(p[0]), float(p[1])) for p in CanvasConfig.IMAGE_CORNERS],
            "used_corners": True,
        }
        return bgr, info

    target_aspect = canvas_aspect()
    img_aspect = float(w) / float(h)
    # pad_orig 记录"原图像素"下的填充边宽，最后换算到工作图尺度。
    # 水墨路线（shuimo.py）要用它把填充区排除出白底占比统计 —— 否则一张
    # 米黄宣纸的画会被白边抬高占比，误走"浅墨滞后连接"分支。
    pad_orig = (0, 0, 0, 0)
    if CanvasConfig.FIT_MODE == "stretch":
        fit = bgr
    elif img_aspect > target_aspect:
        new_h = max(h, int(round(w / target_aspect)))
        pad_top = (new_h - h) // 2
        pad_bottom = new_h - h - pad_top
        fit = cv2.copyMakeBorder(bgr, pad_top, pad_bottom, 0, 0,
                                 cv2.BORDER_CONSTANT, value=CanvasConfig.PAD_COLOR_BGR)
        pad_orig = (pad_top, pad_bottom, 0, 0)
    else:
        new_w = max(w, int(round(h * target_aspect)))
        pad_left = (new_w - w) // 2
        pad_right = new_w - w - pad_left
        fit = cv2.copyMakeBorder(bgr, 0, 0, pad_left, pad_right,
                                 cv2.BORDER_CONSTANT, value=CanvasConfig.PAD_COLOR_BGR)
        pad_orig = (0, 0, pad_left, pad_right)

    fh, fw = fit.shape[:2]
    scale = min(1.0, float(work_size or VisionConfig.WORK_SIZE) / float(max(fh, fw)))
    if scale < 1.0:
        work = cv2.resize(fit, (max(1, int(round(fw * scale))),
                                max(1, int(round(fh * scale)))),
                          interpolation=cv2.INTER_AREA)
    else:
        work = fit

    pad_work = tuple(int(round(v * scale)) for v in pad_orig)
    info = {
        "prepared_size": (work.shape[1], work.shape[0]),
        "scale": scale,
        "padded": (fw != w or fh != h),
        "pad": pad_work,
        "image_corners": None,
        "used_corners": False,
    }
    return work, info


def detect_paper_lab(bgr):
    """
    四周边缘像素的中位色 = 纸色（标准 Lab）。
    若检测结果太暗（L* < 60，说明四周是桌面或背景），退回纯白并标记不可信。
    """
    h, w = bgr.shape[:2]
    band = max(2, int(round(min(h, w) * 0.03)))
    strips = [bgr[0:band, :, :], bgr[h - band:h, :, :],
              bgr[:, 0:band, :], bgr[:, w - band:w, :]]
    pixels = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0)
    median_bgr = np.median(pixels, axis=0).astype(np.uint8)
    lab = to_std_lab(median_bgr.reshape(1, 1, 3)).reshape(3)
    # "可信"的含义：四周颜色本身就像纸（浅、低彩度）。
    # 如果一幅画铺满整幅、四周是天空/背景色，这里会判为不可信，
    # 于是只按"亮且低彩度"的规则识别留白，避免把天空当成纸。
    chroma = float(math.hypot(float(lab[1]), float(lab[2])))
    trusted = bool(lab[0] >= VisionConfig.PAPER_L_MIN - 5.0 and
                   chroma <= VisionConfig.PAPER_CHROMA_MAX + 6.0)
    if not trusted and lab[0] < 60.0:
        lab = np.array([100.0, 0.0, 0.0], dtype=np.float32)
    return lab, trusted, tuple(int(v) for v in median_bgr)


def normalize_lighting(bgr):
    """逐通道除以大尺度模糊，消除光照渐变（按通道均值回缩，保持白平衡）"""
    out = np.empty_like(bgr, dtype=np.float32)
    sigma = float(VisionConfig.LIGHTING_SIGMA_PX)
    for c in range(3):
        channel = bgr[..., c].astype(np.float32)
        background = np.maximum(cv2.GaussianBlur(channel, (0, 0), sigma), 1.0)
        out[..., c] = channel * float(np.mean(channel)) / background
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def neutral_ink_mask(bgr, gray_limit=210):
    """Extract low-chroma ink while excluding saturated paint, even when it is dark.

    The near-black channel-spread allowance keeps antialiased black contours.
    Callers choose their own brightness limit for line art or pale ink.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.linalg.norm(lab[:, :, 1:] - 128.0, axis=2)
    spread = np.ptp(bgr.astype(np.int16), axis=2)
    neutral = (chroma <= 18.0) | ((gray <= 80) & (spread <= 22))
    return ((gray < gray_limit) & neutral).astype(np.uint8)


# =====================================================================
# 2. 颜色分区
# =====================================================================
def palette_lab_array():
    return np.array([rgb_to_std_lab(e["rgb"]) for e in PALETTE], dtype=np.float32)


def _assign_nearest(pixels, centers, chunk=50000):
    """把一个 (N,3) 像素数组分配到最近的中心，分块计算省内存。返回 (N,) int32"""
    n = pixels.shape[0]
    labels = np.empty(n, dtype=np.int32)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        block = pixels[start:stop]
        # (b, k) 距离平方
        diff = block[:, None, :] - centers[None, :, :]
        dist = np.einsum("bkc,bkc->bk", diff, diff)
        labels[start:stop] = np.argmin(dist, axis=1).astype(np.int32)
    return labels


def segment(work_bgr, paper_lab):
    """
    返回 (labels, report)
      labels: (H, W) int32，值为调色板索引，-1 = 留白（不画）
      report: 每个簇的吸附情况，用于打印与调参
    """
    h, w = work_bgr.shape[:2]
    pal_lab = palette_lab_array()
    lab_full = to_std_lab(work_bgr).reshape(-1, 3)
    n_pixels = lab_full.shape[0]

    sample_n = int(min(VisionConfig.CLUSTER_SAMPLE, n_pixels))
    rng = np.random.RandomState(0)
    if sample_n < n_pixels:
        index = rng.choice(n_pixels, sample_n, replace=False)
    else:
        index = np.arange(n_pixels)
    sample = np.ascontiguousarray(lab_full[index], dtype=np.float32)

    k = max(2, len(PALETTE) + int(VisionConfig.EXTRA_CLUSTERS))
    k = int(min(k, max(2, sample.shape[0])))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # ★ cv2.kmeans 的 k-means++ 初始化用的是 OpenCV 自己的内部 RNG，
    #   不受上面那个 np.random.RandomState 控制。不固定它的话，
    #   同一个进程里连跑两次会得到不同的聚类中心 -> 色块分区不同 ->
    #   预览图和任务 JSON 都跟着变，而且没有任何提示。
    #   launcher.py 正是在同一个进程里反复调用 main()，所以必须在这里固定种子。
    cv2.setRNGSeed(20240607)
    _compactness, _lab, centers = cv2.kmeans(
        sample, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.ascontiguousarray(centers, dtype=np.float32)

    labels_all = _assign_nearest(np.ascontiguousarray(lab_full, dtype=np.float32), centers)

    # 每个簇：先判留白，再吸附到调色板
    cluster_to_pal = {}
    report = []
    paper_chroma = float(math.hypot(float(paper_lab[1]), float(paper_lab[2])))
    paper_is_paper_like = bool(float(paper_lab[0]) >= VisionConfig.PAPER_L_MIN - 5.0 and
                               paper_chroma <= VisionConfig.PAPER_CHROMA_MAX + 6.0)
    for i in range(centers.shape[0]):
        c = centers[i]
        L = float(c[0])
        chroma = float(math.hypot(float(c[1]), float(c[2])))
        d_paper = float(np.linalg.norm(c - paper_lab))
        is_paper = bool(L >= VisionConfig.PAPER_L_MIN and
                        chroma <= VisionConfig.PAPER_CHROMA_MAX)
        # 只有"四周确实像纸"时，才允许用"接近纸色"这条更宽松的规则
        if paper_is_paper_like and d_paper <= VisionConfig.PAPER_TOLERANCE:
            is_paper = True
        if is_paper:
            cluster_to_pal[i] = -1
            report.append({"cluster": i, "snap": "留白", "L": L, "chroma": chroma,
                           "d_paper": d_paper, "d_pal": 0.0})
        else:
            d_pal = np.linalg.norm(pal_lab - c, axis=1)
            j = int(np.argmin(d_pal))
            cluster_to_pal[i] = j
            report.append({"cluster": i, "snap": PALETTE[j]["name"], "L": L,
                           "chroma": chroma, "d_paper": d_paper,
                           "d_pal": float(d_pal[j])})

    labels = np.full((h, w), -1, dtype=np.int32)
    flat = labels.reshape(-1)
    for i, j in cluster_to_pal.items():
        if j >= 0:
            flat[labels_all == i] = j

    # 统计：面积加权平均色差（调色板覆盖好不好）+ 留白比例
    counts = np.bincount(labels_all, minlength=centers.shape[0])
    weight_sum = 0.0
    delta_sum = 0.0
    for item in report:
        index = item["cluster"]
        weight = float(counts[index]) if index < counts.shape[0] else 0.0
        weight_sum += weight
        delta_sum += weight * float(item.get("d_pal", 0.0))
    segment_stats = {
        "mean_delta_e": (delta_sum / weight_sum) if weight_sum > 0 else 0.0,
        "paper_ratio": float(np.mean(labels < 0)),
    }
    return labels, report, segment_stats


# =====================================================================
# 3. 色块 -> 宽笔触
# =====================================================================
def _principal_angle(mask):
    """2x2 协方差矩阵解析解求主轴角度（弧度）"""
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return 0.0
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    cxx = float(np.mean(x * x))
    cyy = float(np.mean(y * y))
    cxy = float(np.mean(x * y))
    return 0.5 * math.atan2(2.0 * cxy, cxx - cyy)


def _resolve_angle(fill_mode, mask):
    """把 PALETTE 里的 fill 设置解析成弧度"""
    mode = fill_mode
    if mode is None:
        mode = VisionConfig.FILL_DIRECTION
    if isinstance(mode, (int, float)):
        return math.radians(float(mode))
    text = str(mode).lower()
    if text == "h":
        return 0.0
    if text == "v":
        return math.pi / 2.0
    if text == "d1":
        return math.pi / 4.0
    if text == "d2":
        return -math.pi / 4.0
    return _principal_angle(mask)


def _make_stroke(points, pal_index, width_mm, length_mm, brush_name):
    entry = PALETTE[pal_index]
    return {
        "points": [(float(p[0]), float(p[1])) for p in points],
        "pal_index": int(pal_index),
        "width_mm": float(width_mm),
        "length_mm": float(length_mm),
        "layer": int(entry["layer"]),
        "name": entry["name"],
        "brush": brush_name,
    }


def _fit_brush_name(entry, mask, mm_per_px):
    """
    按色块的实际粗细自动降档选笔：
      - 用距离变换估计色块的"典型厚度"
      - 从该颜色指定的档位往下找，选不超过厚度的最粗一档
      - 这样细线（树干）不会被 22mm 的大笔糊掉，大色块又不会用细笔慢慢磨
    返回笔档名（BRUSH_PRESETS 的键）。
    """
    limit = float(brush_of(entry)["width_mm"])
    candidates = [(name, preset) for name, preset in BRUSH_PRESETS.items()
                  if float(preset["width_mm"]) <= limit + 1e-9]
    if not candidates:
        return entry.get("brush", "mid")
    candidates.sort(key=lambda item: float(item[1]["width_mm"]))
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    values = distance[mask]
    if values.size == 0:
        return candidates[0][0]
    thickness_mm = 2.0 * float(np.percentile(values, 60)) * mm_per_px
    target = max(2.0, thickness_mm * 0.9)
    picked = candidates[0]
    for name, preset in candidates:
        if float(preset["width_mm"]) <= target:
            picked = (name, preset)
    return picked[0]


def _thin(mask):
    """
    Zhang-Suen 细化：把墨迹削成单像素中心线（向量化实现）。
    用于细长笔迹（兰叶、竹枝、树干）——这些必须沿中心线运笔，
    而不能用平行扫掠线（扫掠线会处处横切细线，只能得到一堆短墨点）。
    """
    image = (mask > 0).astype(np.uint8)
    while True:
        changed = False
        for phase in (0, 1):
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]
            center = image[1:-1, 1:-1]
            count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8) +
                ((p3 == 0) & (p4 == 1)).astype(np.uint8) +
                ((p4 == 0) & (p5 == 1)).astype(np.uint8) +
                ((p5 == 0) & (p6 == 1)).astype(np.uint8) +
                ((p6 == 0) & (p7 == 1)).astype(np.uint8) +
                ((p7 == 0) & (p8 == 1)).astype(np.uint8) +
                ((p8 == 0) & (p9 == 1)).astype(np.uint8) +
                ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if phase == 0:
                side_a = p2 * p4 * p6
                side_b = p4 * p6 * p8
            else:
                side_a = p2 * p4 * p8
                side_b = p2 * p6 * p8
            remove = ((center == 1) & (count >= 2) & (count <= 6) &
                      (transitions == 1) & (side_a == 0) & (side_b == 0))
            if np.any(remove):
                center[remove] = 0        # center 是 image 的视图，直接改到原图
                changed = True
        if not changed:
            break
    return image * 255


def _trace_paths(skeleton, epsilon=1.2):
    """
    把骨架图拆成若干条路径（端点/分叉点之间的一段算一条）。
    对角线捷径在有正交邻居时忽略，避免拐角处产生大量伪分叉。

    epsilon 是 approxPolyDP 的简化容差（像素）：
      - 色块路线用默认 1.2（点少、执行快）
      - 水墨路线（shuimo.py）用 0.65，要贴合原墨线的转折，不能被抹圆
    """
    ys, xs = np.nonzero(skeleton > 0)
    pixels = set(zip(xs.tolist(), ys.tolist()))
    if not pixels:
        return []

    def links(point):
        x, y = point
        found = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy):
                    continue
                candidate = (x + dx, y + dy)
                if candidate not in pixels:
                    continue
                if dx and dy and ((x + dx, y) in pixels or (x, y + dy) in pixels):
                    continue
                found.append(candidate)
        return found

    graph = {point: links(point) for point in pixels}
    nodes = set(point for point, adjacent in graph.items() if len(adjacent) != 2)
    visited = set()

    def make_edge(a, b):
        return (a, b) if a <= b else (b, a)

    paths = []

    def walk(start, following):
        path = [start]
        previous, current = start, following
        visited.add(make_edge(previous, current))
        while True:
            path.append(current)
            if current in nodes and current != start:
                break
            choices = [q for q in graph[current]
                       if q != previous and make_edge(current, q) not in visited]
            if not choices:
                break
            following = choices[0]
            visited.add(make_edge(current, following))
            previous, current = current, following
        return path

    for node in nodes:
        for adjacent in graph[node]:
            if make_edge(node, adjacent) not in visited:
                paths.append(walk(node, adjacent))
    for point, adjacent_points in graph.items():
        for adjacent in adjacent_points:
            if make_edge(point, adjacent) not in visited:
                paths.append(walk(point, adjacent))

    simplified = []
    for path in paths:
        if len(path) < 2:
            continue
        contour = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(contour, float(epsilon), False)
        points = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if len(points) >= 2:
            simplified.append(points)
    return simplified


def _skeleton_strokes(mask, pal_index, mm_per_px, width_mm, brush_name):
    """细长笔迹：沿中心线生成笔触（过长按长度切段）"""
    skeleton = _thin(mask)
    paths = _trace_paths(skeleton)
    max_len_mm = float(VisionConfig.MAX_STROKE_LEN_MM)
    min_len_mm = float(VisionConfig.MIN_STROKE_LEN_MM)
    strokes = []
    for path in paths:
        # 累计长度，按 MAX_STROKE_LEN_MM 切段
        segment = [path[0]]
        segment_len = 0.0
        for i in range(1, len(path)):
            step = math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            segment_len += step * mm_per_px
            segment.append(path[i])
            if segment_len >= max_len_mm:
                if segment_len >= min_len_mm:
                    strokes.append(_make_stroke(segment, pal_index, width_mm,
                                                segment_len, brush_name))
                segment = [path[i]]
                segment_len = 0.0
        if len(segment) >= 2 and segment_len >= min_len_mm:
            strokes.append(_make_stroke(segment, pal_index, width_mm,
                                        segment_len, brush_name))
    return strokes


def region_to_strokes(mask, pal_index, mm_per_px, rng, brush_name=None):
    """把一个色块的 mask 转成若干条宽笔触（brush_name 由 _fit_brush_name 决定）"""
    entry = PALETTE[pal_index]
    if brush_name is None:
        brush_name = entry.get("brush", "mid")
    brush = BRUSH_PRESETS.get(brush_name, BRUSH_PRESETS["mid"])
    width_mm = float(brush["width_mm"])

    # 细长笔迹（比笔还细）-> 骨架模式；大色块 -> 平行扫掠模式
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    inside_values = distance[mask]
    if inside_values.size:
        thickness_mm = 2.0 * float(np.percentile(inside_values, 60)) * mm_per_px
        if thickness_mm < width_mm * 1.2:
            return _skeleton_strokes(mask, pal_index, mm_per_px, width_mm, brush_name)
    spacing_px = max(2.0, width_mm * float(VisionConfig.SPACING_FACTOR) / mm_per_px)
    step_px = max(1.0, float(VisionConfig.SAMPLE_STEP_MM) / mm_per_px)

    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return []

    angle = _resolve_angle(entry.get("fill"), mask)
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    centre = pts.mean(axis=0)
    along = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    normal = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float64)

    proj_along = (pts - centre) @ along
    proj_normal = (pts - centre) @ normal
    half_len = float(np.max(np.abs(proj_along))) + 4.0 * step_px

    h, w = mask.shape
    strokes = []
    max_pts = max(2, int(float(VisionConfig.MAX_STROKE_LEN_MM) / mm_per_px / step_px))
    min_len_mm = float(VisionConfig.MIN_STROKE_LEN_MM)
    overshoot_px = float(VisionConfig.OVERSHOOT_MM) / mm_per_px
    bow_px = float(VisionConfig.STROKE_BOW_MM) / mm_per_px
    jitter = math.radians(float(VisionConfig.STROKE_ANGLE_JITTER_DEG))

    offsets = np.arange(float(proj_normal.min()) + spacing_px * 0.5,
                        float(proj_normal.max()) + spacing_px * 0.5,
                        spacing_px)
    if offsets.size == 0:
        offsets = np.array([0.0])

    for offset in offsets:
        jitter_angle = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
        ca, sa = math.cos(jitter_angle), math.sin(jitter_angle)
        direction = np.array([along[0] * ca - along[1] * sa,
                              along[0] * sa + along[1] * ca])
        side = np.array([-direction[1], direction[0]])
        base = centre + side * offset

        steps = np.arange(-half_len, half_len, step_px)
        if steps.size < 2:
            continue
        pos = base[None, :] + steps[:, None] * direction[None, :]
        xi = np.rint(pos[:, 0]).astype(np.int32)
        yi = np.rint(pos[:, 1]).astype(np.int32)
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        inside = np.zeros(steps.shape[0], dtype=bool)
        if np.any(valid):
            inside[valid] = mask[yi[valid], xi[valid]]

        run = np.nonzero(inside)[0]
        if run.size < 2:
            continue
        splits = np.nonzero(np.diff(run) > 1)[0]
        groups = np.split(run, splits + 1)

        for group in groups:
            if group.size < 2:
                continue
            seg_len_mm = float((group[-1] - group[0]) * step_px * mm_per_px)
            if seg_len_mm < min_len_mm:
                continue

            start = 0
            while start < group.size - 1:
                stop = min(group.size, start + max_pts)
                seg = group[start:stop]
                if seg.size >= 2:
                    # 注意：steps 里存的是"相对中心的有符号位置"(-half_len .. +half_len)，
                    # 所以沿线位置必须取 steps[下标]，不能用 下标*step_px（那会整体偏移 half_len）。
                    u0 = float(steps[seg[0]])
                    u1 = float(steps[seg[-1]])
                    # 笔痕的纵向范围就是起笔点到收笔点（毛笔拖过去，末端是平头，
                    # 不会像圆头笔刷那样沿行笔方向多伸出半个笔宽），
                    # 所以这里只做一点"手绘出头"的外延，不做回缩。
                    u0 = float(steps[seg[0]]) - overshoot_px
                    u1 = float(steps[seg[-1]]) + overshoot_px
                    # 注意：这里绝对不能用 continue —— 它会跳过下面的 start 推进，造成死循环。
                    if u1 > u0:
                        length_mm = (u1 - u0) * mm_per_px
                        n_pts = int(round(length_mm / 2.0))
                        n_pts = max(3, min(14, n_pts))
                        bow = bow_px * (1.0 if rng.uniform(0.0, 1.0) < 0.5 else -1.0)
                        line = []
                        for k in range(n_pts):
                            f = float(k) / float(n_pts - 1)
                            u = u0 + (u1 - u0) * f
                            bend = bow * math.sin(math.pi * f)
                            point = base + direction * u + side * bend
                            line.append((float(point[0]), float(point[1])))
                        strokes.append(_make_stroke(line, pal_index, width_mm,
                                                    length_mm, brush_name))
                if stop >= group.size:
                    break
                start = stop - 1

    return strokes


# =====================================================================
# 4. 总入口
# =====================================================================
def build_strokes(work_bgr, mm_per_px):
    """
    工作图 -> (strokes, stats)
    stats 里带 labels（分区结果），供调参时输出彩色分区图
    """
    if VisionConfig.NORMALIZE_LIGHTING:
        work_bgr = normalize_lighting(work_bgr)

    paper_lab, paper_trusted, paper_bgr = detect_paper_lab(work_bgr)
    labels, report, segment_stats = segment(work_bgr, paper_lab)

    rng = np.random.RandomState(20240607)
    strokes = []
    kept = []
    skipped = 0
    for pal_index in range(len(PALETTE)):
        mask = (labels == pal_index)
        area_px = int(mask.sum())
        if area_px == 0:
            continue
        area_mm2 = float(area_px) * mm_per_px * mm_per_px
        if area_mm2 < float(VisionConfig.MIN_REGION_AREA_MM2):
            skipped += 1
            continue

        mask = mask.astype(np.uint8)
        close_px = int(max(1, round(float(VisionConfig.MORPH_CLOSE_MM) / mm_per_px)))
        open_px = int(max(1, round(float(VisionConfig.MORPH_OPEN_MM) / mm_per_px)))
        if close_px >= 2:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (close_px * 2 + 1, close_px * 2 + 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if open_px >= 2:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (open_px * 2 + 1, open_px * 2 + 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        mask_bool = mask > 0
        if not np.any(mask_bool):
            continue
        # 按色块粗细自动选笔（细部用细笔，大块用宽笔）
        brush_name = _fit_brush_name(PALETTE[pal_index], mask_bool, mm_per_px)
        region = region_to_strokes(mask_bool, pal_index, mm_per_px, rng, brush_name)
        strokes.extend(region)

        # 覆盖自检：把该色块的笔触按笔宽栅格化，计算
        #   coverage  = 色块被笔触盖住的比例（要接近 1.0，太低说明漏画）
        #   overshoot = 笔触画到色块外面的比例（笔越宽越大，0.3 以内观感正常）
        coverage = 0.0
        overshoot = 0.0
        width_mm = float(BRUSH_PRESETS[brush_name]["width_mm"])
        if region:
            thickness = int(max(1, round(width_mm / mm_per_px)))
            painted = np.zeros(mask_bool.shape, dtype=np.uint8)
            for item in region:
                pts = np.array([[int(round(p[0])), int(round(p[1]))]
                                for p in item["points"]], dtype=np.int32)
                if pts.shape[0] >= 2:
                    cv2.polylines(painted, [pts], False, 255, thickness, cv2.LINE_8)
            painted_bool = painted > 0
            region_px = int(mask_bool.sum())
            painted_px = int(painted_bool.sum())
            covered_px = int(np.count_nonzero(painted_bool & mask_bool))
            coverage = covered_px / float(max(1, region_px))
            overshoot = (painted_px - covered_px) / float(max(1, painted_px))

        kept.append({"name": PALETTE[pal_index]["name"],
                     "area_mm2": round(area_mm2, 1),
                     "strokes": len(region),
                     "brush": brush_name,
                     "width_mm": round(width_mm, 1),
                     "coverage": round(coverage, 3),
                     "overshoot": round(overshoot, 3)})

    stats = {
        "paper_lab": [round(float(v), 2) for v in paper_lab],
        "paper_trusted": paper_trusted,
        "paper_bgr": paper_bgr,
        "mm_per_px": mm_per_px,
        "clusters": report,
        "mean_delta_e": segment_stats["mean_delta_e"],
        "paper_ratio": segment_stats["paper_ratio"],
        "regions": kept,
        "skipped_regions": skipped,
        "labels": labels,
    }
    return strokes, stats


def render_label_map(labels, out_path):
    """输出彩色分区图（调色板颜色，留白为白），用于判断分区是否正确"""
    h, w = labels.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for i, entry in enumerate(PALETTE):
        rgb = entry["rgb"]
        canvas[labels == i] = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
    if not imgio.imwrite(out_path, canvas):
        raise IOError("分区图写出失败: %s" % out_path)
    return out_path


def print_stats(stats):
    """打印分区报告"""
    print("[分区] 毫米/像素 = %.4f" % stats["mm_per_px"])
    print("[分区] 纸色 Lab=%s 可信=%s 中位色 BGR=%s"
          % (stats["paper_lab"], stats["paper_trusted"], stats["paper_bgr"]))
    print("[分区] 聚类 -> 调色板:")
    for item in stats["clusters"]:
        if item["snap"] == "留白":
            print("   簇%-2d -> 留白      L*=%.1f 彩度=%.1f 距纸色=%.1f"
                  % (item["cluster"], item["L"], item["chroma"], item["d_paper"]))
        else:
            print("   簇%-2d -> %-4s     L*=%.1f 彩度=%.1f 色差=%.1f"
                  % (item["cluster"], item["snap"], item["L"],
                     item["chroma"], item["d_pal"]))
    print("[分区] 采用的色块:")
    low_coverage = []
    for r in stats["regions"]:
        print("   %-4s 面积=%8.1f mm2  %2d 笔  档=%-4s 宽=%4.1fmm  覆盖=%4.0f%%  出头=%3.0f%%"
              % (r["name"], r["area_mm2"], r["strokes"], r["brush"], r["width_mm"],
                 r["coverage"] * 100.0, r["overshoot"] * 100.0))
        if r["coverage"] < 0.85:
            low_coverage.append(r["name"])
    if low_coverage:
        print("[诊断] 覆盖率不足的色块: %s" % "、".join(low_coverage))
        print("       处理办法：把该颜色的 brush 换细一档，或把 VisionConfig.SPACING_FACTOR 调小。")
    if stats["skipped_regions"]:
        print("[分区] 跳过较小色块 %d 个（面积 < %.0f mm2）"
              % (stats["skipped_regions"], VisionConfig.MIN_REGION_AREA_MM2))
    print("[诊断] 留白占比 %.1f%%，面积加权平均色差 %.1f"
          % (stats["paper_ratio"] * 100.0, stats["mean_delta_e"]))
    if not stats["paper_trusted"]:
        print("[诊断] 图片四周不是浅色纸底（纸色检测不可信）：")
        print("       如果这是照片，建议在 CanvasConfig.IMAGE_CORNERS 里圈出画面区域，")
        print("       或调低 VisionConfig.PAPER_L_MIN 让浅色背景被判为留白。")
    if stats["mean_delta_e"] > 14.0:
        print("[诊断] 平均色差偏大（>14）：原图颜色和 PALETTE 差得比较多，")
        print("       建议按原图主色在 config.py 的 PALETTE 里增加/调整颜色。")
    if stats["paper_ratio"] < 0.05:
        print("[诊断] 几乎检测不到留白：确认这是线稿/插画而不是实景照片；")
        print("       若是照片，考虑调低 PAPER_L_MIN 或提高 PAPER_TOLERANCE。")
