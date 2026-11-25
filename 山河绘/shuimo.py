# -*- coding: utf-8 -*-
"""
shuimo.py - 水墨框架（第一遍：勾墨线）图像 -> 带逐点笔压的长笔触
====================================================================

【做什么】
  输入一张水墨范图 -> 提取墨线中心线 -> 输出"每个点都有笔压"的长笔触。
  这是"先勾框架、再上色"里的第一遍，全部用墨，只有一种"颜色"。

【和 vision.py 的分工】
  vision.py  ：色块路线（KMeans 分区 + 平行扫掠铺色），目标是"色块 + 大笔触感"
  shuimo.py  ：水墨路线（墨线骨架 + 逐点笔压），目标是"一笔之内有粗细浓淡"
  两条路线互不干扰，共用 geometry / robot / job 执行层与安全检查。

【算法链（移植自 DobotE6Calligraphy/landscape_painting.py，已适配本项目的像素坐标）】
  1. 高斯模糊 -> Otsu 反向阈值 -> 深墨蒙版
  2. 白底占比 >= 70% 时做"浅墨滞后连接"，把同一笔里的淡墨段保留下来。
     不做这一步，一笔淡墨会被阈值切成好几段，机械臂就只能不停地抬笔点按。
  3. Zhang-Suen 细化 -> 单像素骨架（复用 vision._thin）
  4. distanceTransform -> width_map：每个骨架点的原始墨宽
  5. 逐连通域追踪骨架路径（严禁跨空白区域合笔）
  6. 交叉点按切线方向续接 +（白底图）方向约束的缺口桥接
  7. 按人类作画次序排序：主体长轮廓 -> 中等结构线 -> 短皴线
  8. 骨架局部密度 -> 密集区减压加速（这就是"飞白/枯笔"的来源）
  9. 按弧长重采样 -> 逐点笔压 + 逐点速度（起笔 ramp / 行笔起伏 / 收笔 ramp）

【坐标与单位约定（重要）】
  - points 是**工作图像素坐标**，由 job.py 经 CanvasMap.to_robot() 换成机械臂毫米
  - press  直接是毫米下压量（正值 = 往纸里压），交给 config.z_for_press() 换 Z
  - speed  是 DashBoard 的 v 速度百分比
  这样 shuimo.py 完全不需要知道用户坐标系和纸面高度。

【为什么行笔中笔压不许降到 0 以下】
  press < 0 意味着笔尖抬离纸面 -> 墨线断成几截。旧引擎在收尾处做了 clip
  （landscape_painting.py:856-858），这里保留同样的安全约束。
"""

import math

import cv2
import numpy as np

from config import PALETTE, ShuimoConfig
import vision


INK_NAME = "墨黑"          # 框架这一遍用的调色板条目（决定预览颜色与蘸墨格）
STROKE_NAME = "墨线"
# 手抖扰动的随机源（固定种子 -> 同一张图每次跑出来一样，便于对比调参）
_JITTER = np.random.RandomState(20240607)


# =====================================================================
# 0. 笔档换算
# =====================================================================
def ink_palette_index():
    """框架用的调色板索引：优先找 INK_NAME，找不到就用最暗的一格"""
    for index, entry in enumerate(PALETTE):
        if entry.get("name") == INK_NAME:
            return index
    if not PALETTE:
        raise ValueError("PALETTE 为空，无法确定墨色")
    darkest = min(range(len(PALETTE)),
                  key=lambda i: sum(PALETTE[i]["rgb"]))
    return int(darkest)


def _press_width_table():
    """把 ShuimoConfig.BRUSHES 整理成按 press 升序的 (press[], width[]) 插值表"""
    rows = sorted((float(p["press_mm"]), float(p["width_mm"]))
                  for p in ShuimoConfig.BRUSHES.values())
    if not rows:
        return np.array([0.0, 1.0]), np.array([0.4, 3.5])
    presses = np.array([r[0] for r in rows], dtype=np.float64)
    widths = np.array([r[1] for r in rows], dtype=np.float64)
    return presses, widths


def press_to_width_mm(press_mm):
    """
    笔压 -> 实际笔痕宽度（毫米），线性插值 BRUSHES 标定表。

    用途有两个：
      1. 预览时画出"机器人真正会画成多宽"（不是原图的墨宽）
      2. 统计报告里给出笔宽是否够细
    这是诚实的估计：如果原图墨线只有 0.2mm 而最细的笔是 0.4mm，
    预览就会显示线条被加粗，而不是假装能画出来。
    """
    presses, widths = _press_width_table()
    return float(np.interp(float(press_mm), presses, widths))


# =====================================================================
# 1. 墨迹提取
# =====================================================================
def _otsu_ink(gray, roi=None):
    """
    高斯模糊 + Otsu 反向阈值 -> (深墨蒙版, Otsu 阈值, 模糊图)

    roi: (top, bottom, left, right) 像素边距，**只用来计算阈值**，作用范围
         仅限"画面本身"，排除画布留白填充区。

    ★ 为什么必须排除：prepare_image 会按画布长宽比用纯白填充图片。如果拿
      填充后的整幅图去算 Otsu，那 40% 左右的纯白会在直方图里堆成一个巨大
      尖峰，Otsu 为了最大化类间方差会把阈值推到"纸底也算墨"的位置 ——
      结果整幅画（连米黄宣纸底）全被判成墨迹，骨架塌成两三条大直线。

      实测（shuimo.jpg，画布 286x199）：
        不排除填充  ->  墨迹 57.8%，23 条路径塌成 5 条
        排除填充    ->  墨迹  7.0%，路径数正常
    """
    kernel = int(ShuimoConfig.BLUR_KERNEL)
    if kernel % 2 == 0:
        kernel += 1
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    # 只拿画面本身算阈值
    source = blurred
    if roi:
        p_top, p_bottom, p_left, p_right = (int(v) for v in roi)
        height, width = blurred.shape[:2]
        if (p_top + p_bottom) < height and (p_left + p_right) < width:
            source = blurred[p_top:height - p_bottom, p_left:width - p_right]
    if source.size == 0:
        source = blurred

    otsu_gray, _ = cv2.threshold(source, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 再把算出来的阈值作用到整幅图（含填充区）—— 填充区本来就是白的，
    # 不会变成墨，所以对结果没有影响。
    _, strong_ink = cv2.threshold(blurred, float(otsu_gray), 255,
                                  cv2.THRESH_BINARY_INV)
    return strong_ink, float(otsu_gray), blurred


def _recover_light_ink(blurred, strong_ink, otsu_gray):
    """
    以深墨为种子做滞后连接，把同一笔迹里的浅墨段接回来。

    背景：一张兰花图里一笔叶子常常是"根浓尖淡"，Otsu 只会保留浓的部分，
    淡墨段被丢掉 -> 一片叶子变成几截 -> 机械臂反复抬笔。
    这里用一个更宽松的阈值取"弱墨"，然后只接纳与深墨种子相连（或非常接近）
    的弱墨连通域，既能穿过淡墨区，又不会把白纸上的 JPEG 噪声变成轨迹。
    """
    configured = float(ShuimoConfig.LIGHT_INK_GRAY)
    weak_limit = int(np.clip(max(otsu_gray + 12.0, configured), 0.0, 250.0))
    weak_ink = (blurred < weak_limit).astype(np.uint8)

    close_px = max(1, int(ShuimoConfig.LIGHT_INK_CLOSE_PX))
    if close_px % 2 == 0:
        close_px += 1
    if close_px >= 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        weak_ink = cv2.morphologyEx(weak_ink, cv2.MORPH_CLOSE, kernel)

    d = max(1, int(ShuimoConfig.SUPPORT_DILATE_PX))
    support = cv2.dilate(
        (strong_ink > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(weak_ink, connectivity=8)
    recovered = np.zeros_like(strong_ink)
    minimum = int(ShuimoConfig.MIN_COMPONENT_PX)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < minimum:
            continue
        component = labels == label
        if np.any(support[component] > 0):
            recovered[component] = 255
    return recovered


# =====================================================================
# 2. 断笔续接（把碎段接成长笔触，机械臂才不会一直抬笔）
# =====================================================================
def _join_collinear_branches(paths):
    """
    只在同一墨迹的交叉点按切线方向续接，模拟毛笔自然顺势运笔。

    判据：端点间距 <= COLLINEAR_GAP_PX，且"来向"与"去向"夹角余弦 >= COLLINEAR_COS
    （约 60° 以内）。贪心取 gap + 角度惩罚最小的那一对。
    """
    result = [list(path) for path in paths]
    max_gap = float(ShuimoConfig.COLLINEAR_GAP_PX)
    cosine_min = float(ShuimoConfig.COLLINEAR_COS)

    while len(result) > 1:
        best = None
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                for reverse_a in (False, True):
                    for reverse_b in (False, True):
                        a = result[i][::-1] if reverse_a else result[i]
                        b = result[j][::-1] if reverse_b else result[j]
                        if len(a) < 2 or len(b) < 2:
                            continue
                        gap = math.hypot(a[-1][0] - b[0][0], a[-1][1] - b[0][1])
                        if gap > max_gap:
                            continue
                        incoming = np.array(a[-1], float) - np.array(a[-2], float)
                        outgoing = np.array(b[1], float) - np.array(b[0], float)
                        denominator = float(np.linalg.norm(incoming) *
                                            np.linalg.norm(outgoing))
                        if denominator <= 1e-9:
                            continue
                        cosine = float(np.dot(incoming, outgoing) / denominator)
                        if cosine < cosine_min:
                            continue
                        score = gap + (1.0 - cosine) * 5.0
                        if best is None or score < best[0]:
                            best = (score, i, j, reverse_a, reverse_b)
        if best is None:
            break
        _, i, j, reverse_a, reverse_b = best
        a = result[i][::-1] if reverse_a else result[i]
        b = result[j][::-1] if reverse_b else result[j]
        result[i] = a + b[1:] if a[-1] == b[0] else a + b
        result.pop(j)
    return result


def _merge_directional_paths(paths, max_gap_px, gray):
    """
    按端点切线续接长线缺口，**拒绝仅因空间相近就误合笔**。

    这是整个水墨模块最关键的一步：没有它，扫描/压缩留下的肉眼看不出的
    小缺口会让机械臂每 2~3mm 就抬一次笔，画面变成一串墨点而不是一条线。

    三重约束，缺一不可：
      1. 缺口长度在 max_gap_px 以内
      2. 来向 -> 缺口方向 -> 去向，三段方向都要连续（夹角余弦 >= 限值）
      3. 缺口中央确实残留了一部分浅墨证据（bridge_has_ink）
    """
    merged = [list(path) for path in paths]
    cosine_limit = float(np.cos(np.deg2rad(ShuimoConfig.DIRECTIONAL_ANGLE_DEG)))
    min_length = float(ShuimoConfig.DIRECTIONAL_MIN_LENGTH_MM)
    ink_ratio = float(ShuimoConfig.BRIDGE_INK_RATIO)
    step_px = max(1.0, float(ShuimoConfig.BRIDGE_STEP_PX))
    height, width = gray.shape[:2]

    def path_length(path):
        return float(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                         for a, b in zip(path, path[1:])))

    def unit(vector):
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-6 else None

    def bridge_has_ink(start, end):
        """缺口中央允许褪成白纸，但沿途至少要有一部分浅墨证据"""
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        samples = max(3, int(math.ceil(distance)))
        values = []
        for index in range(1, samples):
            ratio = index / float(samples)
            x = int(round(start[0] * (1.0 - ratio) + end[0] * ratio))
            y = int(round(start[1] * (1.0 - ratio) + end[1] * ratio))
            if 0 <= x < width and 0 <= y < height:
                values.append(float(gray[y, x]))
        if not values:
            return False
        return float(np.mean(np.asarray(values) < 248.0)) >= ink_ratio

    while len(merged) > 1:
        lengths = [path_length(path) for path in merged]
        best = None
        for i in range(len(merged)):
            if lengths[i] < min_length:
                continue
            for j in range(i + 1, len(merged)):
                if lengths[j] < min_length:
                    continue
                for reverse_a in (False, True):
                    for reverse_b in (False, True):
                        a = merged[i][::-1] if reverse_a else merged[i]
                        b = merged[j][::-1] if reverse_b else merged[j]
                        if len(a) < 3 or len(b) < 3:
                            continue
                        gap_vector = np.array(b[0], float) - np.array(a[-1], float)
                        gap = float(np.linalg.norm(gap_vector))
                        if gap < 0.5 or gap > max_gap_px:
                            continue
                        # 用端点附近多点平均切线，避免骨架单像素锯齿把
                        # 本应连续的叶子误判成转角
                        window_a = min(5, len(a) - 1)
                        window_b = min(5, len(b) - 1)
                        incoming = unit(np.array(a[-1], float) -
                                        np.array(a[-1 - window_a], float))
                        outgoing = unit(np.array(b[window_b], float) -
                                        np.array(b[0], float))
                        gap_direction = unit(gap_vector)
                        if incoming is None or outgoing is None or gap_direction is None:
                            continue
                        align_a = float(np.dot(incoming, gap_direction))
                        align_b = float(np.dot(gap_direction, outgoing))
                        align_ab = float(np.dot(incoming, outgoing))
                        if min(align_a, align_b, align_ab) < cosine_limit:
                            continue
                        if not bridge_has_ink(a[-1], b[0]):
                            continue
                        score = gap + (3.0 - align_a - align_b - align_ab) * max_gap_px
                        if best is None or score < best[0]:
                            best = (score, i, j, reverse_a, reverse_b, gap)
        if best is None:
            break
        _, i, j, reverse_a, reverse_b, gap = best
        first = merged[i][::-1] if reverse_a else merged[i]
        second = merged[j][::-1] if reverse_b else merged[j]
        steps = max(1, int(math.ceil(gap / step_px)))
        bridge = []
        for step in range(1, steps):
            ratio = step / float(steps)
            bridge.append((
                int(round(first[-1][0] * (1 - ratio) + second[0][0] * ratio)),
                int(round(first[-1][1] * (1 - ratio) + second[0][1] * ratio)),
            ))
        merged[i] = first + bridge + second
        merged.pop(j)
    return merged


# =====================================================================
# 3. 密度与排序
# =====================================================================
def _local_density_map(skeleton, mm_per_px):
    """
    骨架局部密度：每个骨架点邻域内的骨架像素数。

    单条线在邻域内的计数接近低分位；多条皴线、交叉线集中时明显增大。
    归一化后用于"减压 + 加速"，而不是删掉密集处的细节 —— 这正是
    国画"密处不黑、枯笔透气"的效果。
    """
    radius_mm = max(0.0, float(ShuimoConfig.DENSE_RADIUS_MM))
    radius_px = max(1, int(round(radius_mm / max(1e-6, mm_per_px))))
    kernel = radius_px * 2 + 1
    mask = (skeleton > 0).astype(np.float32)
    return cv2.boxFilter(mask, -1, (kernel, kernel), normalize=False,
                         borderType=cv2.BORDER_REPLICATE)


def _width_at(width_map, x, y, height, width, fallback=1.0):
    """安全取某个骨架点的原始墨宽（distanceTransform 值 = 半宽）"""
    xi = int(x)
    yi = int(y)
    if 0 <= xi < width and 0 <= yi < height:
        value = float(width_map[yi, xi])
        if value > 0:
            return value
    return float(fallback)


def _order_paths(paths, width_map):
    """
    按人类作画次序排序：主体长轮廓 -> 中等结构线 -> 短皴线；
    同层按自上而下、自左而右推进。

    顺带把每笔的方向定下来：从**粗**的一端运向细的一端
    （兰叶从叶根写到叶尖），宽度接近时退回"从左上开始"的稳定规则。
    """
    height, width = width_map.shape[:2]
    lengths = []
    for path in paths:
        lengths.append(float(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                                 for a, b in zip(path, path[1:]))))
    if not lengths:
        return [], 0.0
    max_length = max(lengths) if max(lengths) > 0 else 1.0
    long_ratio = float(ShuimoConfig.LAYER_LONG_RATIO)
    mid_ratio = float(ShuimoConfig.LAYER_MID_RATIO)

    entries = []
    for path, length in zip(paths, lengths):
        sample_count = max(1, min(6, len(path) // 4))
        start_width = float(np.mean([
            _width_at(width_map, x, y, height, width)
            for x, y in path[:sample_count]]))
        end_width = float(np.mean([
            _width_at(width_map, x, y, height, width)
            for x, y in path[-sample_count:]]))
        if end_width > start_width * 1.08:
            path = list(reversed(path))
        elif start_width <= end_width * 1.08:
            if (path[-1][1], path[-1][0]) < (path[0][1], path[0][0]):
                path = list(reversed(path))

        ratio = length / max_length
        layer = 0 if ratio >= long_ratio else (1 if ratio >= mid_ratio else 2)
        entries.append((layer,
                        min(p[1] for p in path),
                        min(p[0] for p in path),
                        -length,
                        path))
    entries.sort(key=lambda item: item[:4])
    return entries, max_length


# =====================================================================
# 4. 笔法：弧长重采样 + 逐点笔压/速度
# =====================================================================
def _smooth_profile(profile, window):
    """
    沿弧长做滑动平均（边缘用端点值填充，避免笔触两端被拉细）。

    为什么必须做：distanceTransform 在单像素宽骨架上只有 1.0 和 1.414 两档
    取值，逐点直接拿来算笔压，笔痕边缘就会一跳一跳（锯齿）。
    毛笔的实际接触宽度在相邻轨迹点（间距 1.5mm）之间不可能突变。
    """
    values = np.asarray(profile, dtype=np.float64)
    if values.size < 3:
        return values
    size = int(window)
    if size < 3:
        return values
    size = min(size, values.size)
    if size % 2 == 0:
        size += 1
    if size < 3:
        return values
    kernel = np.ones(size, dtype=np.float64) / float(size)
    pad = size // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _smooth_path(xs, ys, mm_per_px):
    """
    对重采样后的轨迹点做滑动平均，把骨架的"像素台阶"抹成平滑曲线。

    【为什么必须做】骨架是像素阶梯，简化后变成"一段段直线 + 突然拐弯"：
        实测兰花 380 个顶点里，转折角中位 10°、51% 的顶点 >10°、最大 81°，
        转角序列长这样： 0 0 0 0 47 36 0 0 0 0 21 22 ...
      机械臂照着走就是硬邦邦的折线 —— 这正是"没有水墨笔触感"的主因。

    【端点处理】首尾各留一段"渐变区"不做平滑，保证起笔点和收笔点不动
    （否则笔触会缩短，而且和相邻笔触接不上）。
    """
    window_mm = float(getattr(ShuimoConfig, "PATH_SMOOTH_MM", 0.0))
    n = int(xs.size)
    if window_mm <= 1e-6 or n < 5:
        return xs, ys

    step_px = float(np.hypot(xs[1] - xs[0], ys[1] - ys[0]))
    if step_px <= 1e-9:
        return xs, ys
    w = int(round((window_mm / max(1e-9, mm_per_px)) / step_px))
    if w % 2 == 0:
        w += 1
    w = max(3, min(w, n if n % 2 == 1 else n - 1))
    if w < 3:
        return xs, ys

    kernel = np.ones(w, dtype=np.float64) / float(w)
    pad = w // 2
    sx = np.convolve(np.pad(xs, pad, mode="edge"), kernel, mode="valid")
    sy = np.convolve(np.pad(ys, pad, mode="edge"), kernel, mode="valid")

    # 端点渐变：两端各 pad 个点从"原样"过渡到"平滑后"，中间完全用平滑值
    t = np.arange(n, dtype=np.float64)
    ramp = np.minimum(t, (n - 1) - t) / float(max(1, pad))
    ramp = np.clip(ramp, 0.0, 1.0)
    return xs * (1.0 - ramp) + sx * ramp, ys * (1.0 - ramp) + sy * ramp


def _resample_profile(path, width_map, density_map, mm_per_px):
    """
    按**弧长**把路径重采样成固定间距的点，并把原始墨宽/密度沿弧长插值过来。

    为什么要按弧长而不是按下标：起笔/收笔 ramp 是"占笔长的 12%"，
    按下标算的话，点少的短笔几乎整条都在 ramp 里，点多的长笔又几乎没有 ramp。
    """
    points = np.asarray(path, dtype=np.float64)
    if points.shape[0] < 2:
        return None
    height, width = width_map.shape[:2]
    segment = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    total_px = float(cumulative[-1])
    if total_px <= 1e-6:
        return None

    step_mm = max(1e-3, float(ShuimoConfig.TRAJ_STEP_MM))
    step_px = max(1e-3, step_mm / max(1e-6, mm_per_px))
    max_points = max(3, int(ShuimoConfig.MAX_POINTS_PER_STROKE))
    count = int(np.clip(round(total_px / step_px) + 1, 3, max_points))
    stations = np.linspace(0.0, total_px, count)

    xs = np.interp(stations, cumulative, points[:, 0])
    ys = np.interp(stations, cumulative, points[:, 1])

    # 先在原始骨架像素上取墨宽/密度，再沿弧长插值 —— 避免重采样点落到
    # 骨架之外（骨架是单像素宽，直接查表会取到 0）
    sample_x = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, width - 1)
    sample_y = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, height - 1)
    widths = width_map[sample_y, sample_x].astype(np.float64)
    densities = density_map[sample_y, sample_x].astype(np.float64)
    width_profile = np.interp(stations, cumulative, widths)
    density_profile = np.interp(stations, cumulative, densities)

    # 沿弧长平滑：把笔痕边缘的锯齿去掉，让 taper 干净
    smooth_mm = max(0.0, float(ShuimoConfig.PROFILE_SMOOTH_MM))
    window = int(round(smooth_mm / step_mm)) if smooth_mm > 0 else 0
    if window >= 3:
        width_profile = _smooth_profile(width_profile, window)
        density_profile = _smooth_profile(density_profile, window)

    return xs, ys, width_profile, density_profile, total_px


def _stroke_from_path(path, width_map, density_map, mm_per_px, stats,
                      pal_index, layer, t_ratio):
    """
    一条骨架路径 -> 一条带逐点笔压/速度的笔触。

    笔压曲线（对齐旧引擎 landscape_painting.py:775-864）：
      压力       = 原始墨宽归一化 -> [PRESS_MIN, PRESS_MAX]
      密集区修正 = 压力按密度下调（密处不黑，出枯笔）
      起笔 ramp  = 前 12% 由轻到重，速度 0.5 -> 1.0 倍（藏锋）
      行笔       = 叠加正弦微起伏；重压处慢、细线处快
      收笔 ramp  = 后 12% 由重退回轻触，速度 1.0 -> 0.5 倍（收锋）
    """
    resampled = _resample_profile(path, width_map, density_map, mm_per_px)
    if resampled is None:
        return None
    xs, ys, width_profile, density_profile, total_px = resampled
    length_mm = total_px * mm_per_px
    if length_mm < float(ShuimoConfig.MIN_STROKE_MM):
        return None

    # ★ 把骨架的像素台阶抹成平滑曲线 —— 消除"直来直去"的关键一步
    xs, ys = _smooth_path(xs, ys, mm_per_px)

    press_min = float(ShuimoConfig.PRESS_MIN_MM)
    press_max = float(ShuimoConfig.PRESS_MAX_MM)
    span = max(0.1, press_max - press_min)
    base_speed = float(ShuimoConfig.SPEED_BASE)
    # ★ 点状/串珠模式：每段很短（DAB_LENGTH_MM），如果还用 12% 的斜坡，
    #   4mm 的段里有 3mm 都是满压 -> 画出来是一条短横线，不是"珠子"。
    #   改成 50% 斜坡：两端笔压为 0、正中间最大 -> 标准的透镜形墨点，
    #   一串连起来才有参考图那种斑驳颗粒感。
    if float(getattr(ShuimoConfig, "DAB_LENGTH_MM", 0.0) or 0.0) > 1e-6:
        ramp = 0.5
    else:
        ramp = float(np.clip(ShuimoConfig.RAMP, 0.01, 0.45))
    undulation = float(ShuimoConfig.LINE_UNDULATION)
    dense_reduce = float(np.clip(ShuimoConfig.DENSE_PRESS_REDUCTION, 0.0, 1.0))
    dense_boost = max(0.0, float(ShuimoConfig.DENSE_SPEED_BOOST))

    count = xs.shape[0]
    # 手抖扰动：仿人的自然抖动。两端各留一个点不动（保住起笔/收笔的准确位置）
    jitter_mm = max(0.0, float(getattr(ShuimoConfig, "HAND_JITTER_MM", 0.0)))
    jitter_px = (jitter_mm / max(1e-9, mm_per_px)) if jitter_mm > 0 else 0.0

    # ★ 点状/串珠模式：下压量的【地板】从 press_min 降到 0。
    #   连续笔触必须锁 press_min（行笔中永不抬离纸面，否则墨线断成几截）；
    #   但点状模式要的正好相反 —— 每小段两端"笔尖刚好触纸"、中间满压，
    #   才能形成一颗颗"两端细、中间鼓"的墨珠。
    dab_mode = float(getattr(ShuimoConfig, "DAB_LENGTH_MM", 0.0) or 0.0) > 1e-6
    press_floor = 0.0 if dab_mode else press_min

    # ★★ 点状/串珠：先按弧长切段，再【逐段】算笔压 ★★
    #   顺序至关重要：
    #     先算整条笔压再切 -> 每小段拿到的只是长笔触上那一小截的笔压（几乎平的），
    #                        两端斜坡落不到小段上，珠子形状出不来
    #     先切、再逐段算   -> 每小段各自从 0 升到满压再回到 0 = 一颗颗墨珠 ✓
    segments = _dab_segments(xs, ys, width_profile, density_profile, mm_per_px)

    strokes_out = []
    for seg_xs, seg_ys, seg_w, seg_d, seg_len_mm in segments:
        n = int(seg_xs.shape[0])
        if n < 2:
            continue
        points, presses, speeds = [], [], []
        for index in range(n):
            t = index / float(n - 1)
            strength = float(np.clip(
                (seg_w[index] - stats["width_low"]) / stats["width_span"],
                0.0, 1.0))
            density = float(np.clip(
                (seg_d[index] - stats["density_low"]) / stats["density_span"],
                0.0, 1.0))

            press = press_min + strength * span
            # 只压缩"超过最轻档"的部分，保证密集区最多退回轻压，不会变成负压力
            press = press_min + (press - press_min) * (1.0 - dense_reduce * density)

            if t < ramp:
                factor = t / ramp
                offset = press_floor * (1.0 - factor) + press * factor
                speed = base_speed * (0.5 + 0.5 * factor)
            elif t < 1.0 - ramp:
                offset = press + undulation * math.sin(t * math.pi * 2.0)
                speed = base_speed * (1.05 - 0.25 * press / max(1e-6, press_max))
            else:
                factor = (1.0 - t) / ramp
                offset = press * factor + press_floor * (1.0 - factor)
                speed = base_speed * (0.5 + 0.5 * factor)

            # 连续模式：行笔中永不抬离纸面（下限 press_min）
            # 点状模式：允许压到 0（下限 0），两端才收得住、珠子才成形
            offset = float(np.clip(offset, press_floor, press_max))
            speed *= 1.0 + dense_boost * density

            px, py = float(seg_xs[index]), float(seg_ys[index])
            if jitter_px > 0 and 0 < index < n - 1:
                px += _JITTER.uniform(-jitter_px, jitter_px)
                py += _JITTER.uniform(-jitter_px, jitter_px)
            points.append((px, py))
            presses.append(offset)
            speeds.append(float(np.clip(speed, 5.0, 100.0)))

        strokes_out.append({
            "points": points,
            "press": presses,
            "speed": speeds,
            "pass": 1,
            "pal_index": int(pal_index),
            "layer": int(layer),
            "name": STROKE_NAME,
            "brush": ShuimoConfig.DEFAULT_BRUSH,
            "lift_mm": float(ShuimoConfig.LIFT_MM),
            "width_mm": float(np.mean([press_to_width_mm(p) for p in presses])),
            "length_mm": float(seg_len_mm),
            "press_mean": float(np.mean(presses)),
            "press_max": float(np.max(presses)),
            "t_ratio": float(t_ratio),
        })
    return strokes_out


def _dab_segments(xs, ys, width_profile, density_profile, mm_per_px):
    """
    按弧长把重采样后的序列切成小段（点状/串珠模式用）。

    相邻两段【共享端点】（第 i 段结束的那个点就是第 i+1 段的起点），
    所以画到纸上墨点是重叠的、不会断开。

    DAB_LENGTH_MM = 0 时原样返回一整段（= 不切）。
    返回 [(xs_c, ys_c, width_c, density_c, length_mm), ...]
    """
    dab_mm = float(getattr(ShuimoConfig, "DAB_LENGTH_MM", 0.0) or 0.0)
    n = int(xs.shape[0])
    if dab_mm <= 1e-6 or mm_per_px <= 1e-9 or n < 3:
        seg_len = float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))) * mm_per_px
        return [(xs, ys, width_profile, density_profile, seg_len)]

    dab_px = dab_mm / mm_per_px
    step = np.hypot(np.diff(xs), np.diff(ys))     # 每段的像素长度

    out = []
    start = 0
    acc = 0.0
    for i in range(1, n):
        acc += float(step[i - 1])
        last = (i == n - 1)
        if acc >= dab_px or last:
            if i - start >= 1:
                sl = slice(start, i + 1)
                out.append((xs[sl], ys[sl], width_profile[sl],
                            density_profile[sl],
                            float(np.sum(step[start:i])) * mm_per_px))
            if not last:
                start = i          # 下一段和这一段共享端点 i
                acc = 0.0
    return out


# =====================================================================
# 5. 总入口
# =====================================================================
def _split_into_dabs(strokes, mm_per_px):
    """
    把每条长笔触按【弧长】切成小段，复现"串珠 / 点苔"那种斑驳笔触。

    【为什么会长成串珠】
      每一小段都是一次独立的"落笔 -> 行笔 -> 抬笔"。抬笔前墨会在笔尖
      积一下（就是 job.stroke_waypoints 里用"出锋"修掉的那个收笔墨疙瘩），
      所以每段的两端都会留下一个墨点；一串连起来就是参考图里的"串珠"。

    【实现】
      沿点序列累加弧长，每到 DAB_LENGTH_MM 就切一刀。
      相邻两段【共享端点】（第 i 段结束的那个点就是第 i+1 段的起点），
      所以墨点之间是重叠的、不会断开。

    DAB_LENGTH_MM = 0 时原样返回。
    """
    dab_mm = float(getattr(ShuimoConfig, "DAB_LENGTH_MM", 0.0) or 0.0)
    if dab_mm <= 1e-6 or mm_per_px <= 1e-9:
        return strokes
    dab_px = dab_mm / mm_per_px

    out = []
    for stroke in strokes:
        pts = np.asarray(stroke["points"], dtype=np.float64)
        n = int(pts.shape[0])
        if n < 4:
            out.append(stroke)
            continue
        presses = list(stroke["press"])
        speeds = list(stroke["speed"])
        start = 0
        acc = 0.0
        for i in range(1, n):
            acc += float(np.hypot(pts[i, 0] - pts[i - 1, 0],
                                  pts[i, 1] - pts[i - 1, 1]))
            last = (i == n - 1)
            if acc >= dab_px or last:
                if i - start >= 1:
                    piece = dict(stroke)
                    piece["points"] = [(float(p[0]), float(p[1]))
                                       for p in pts[start:i + 1]]
                    piece["press"] = presses[start:i + 1]
                    piece["speed"] = speeds[start:i + 1]
                    # 长度按弧长重算（出锋/蘸墨统计要用）
                    seg = pts[start:i + 1]
                    piece["length_mm"] = float(
                        np.sum(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1])))
                    ) * mm_per_px
                    out.append(piece)
                if not last:
                    start = i          # 下一段和这一段共享端点 i
                    acc = 0.0
    return out


def build_framework_strokes(work_bgr, mm_per_px, pad=None):
    """
    工作图 -> (strokes, stats)

    pad: (top, bottom, left, right) 画布留白填充的边宽（像素），用于把这些
         填充区排除出"白底占比"统计。不排除的话，一张米黄宣纸的画会被
         白边抬高占比、误走"浅墨滞后连接"分支。
    """
    _JITTER.seed(20240607)
    if work_bgr is None or work_bgr.size == 0:
        raise ValueError("工作图为空")
    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    # ---- 画布留白填充区：所有"看整幅图"的统计都必须排除它 ----
    pad_applied = False
    roi = None
    if pad:
        p_top, p_bottom, p_left, p_right = (int(v) for v in pad)
        if (p_top + p_bottom) < height and (p_left + p_right) < width \
                and (p_top + p_bottom + p_left + p_right) > 0:
            roi = (p_top, p_bottom, p_left, p_right)
            pad_applied = True

    # ★ Otsu 阈值也必须只在画面本身上算（见 _otsu_ink 的说明），
    #   否则纯白填充会把阈值推歪，整幅画被当成墨块。
    strong_ink, otsu_gray, blurred = _otsu_ink(gray, roi)

    # ---- 白底占比：同样排除填充区 ----
    if roi is None:
        stat_region = blurred
    else:
        stat_region = blurred[roi[0]:height - roi[1], roi[2]:width - roi[3]]
    white_ratio = float(np.mean(stat_region > 245)) if stat_region.size else 0.0

    use_light_ink = white_ratio >= float(ShuimoConfig.WHITE_BG_RATIO)
    ink = _recover_light_ink(blurred, strong_ink, otsu_gray) if use_light_ink \
        else strong_ink
    # 不做开运算：大量短皴线本来就很细，开运算会把它们当噪声删掉

    ink_px = int(np.count_nonzero(ink))
    ink_ratio = ink_px / float(max(1, ink.size))
    ink_sparse = ink_ratio <= float(ShuimoConfig.INK_SPARSE_RATIO)
    # 缺口桥接的启用判据：白底墨线图，或"墨迹只占画面一小部分"的线稿。
    # 后者是为了宣纸画 —— 米黄纸底的白底占比接近 0，只看白底占比的话
    # 满纸的短碎笔永远接不起来（实测 shuimo.jpg: 267 条路径 -> 158 条短笔）。
    bridge_ok = bool(use_light_ink or ink_sparse)

    # ---- 骨架 + 墨宽图 ----
    skeleton = vision._thin(ink)
    width_map = cv2.distanceTransform(ink, cv2.DIST_L2, 5)

    # ---- 路径追踪：逐连通域，严禁跨空白区域合笔 ----
    epsilon = float(ShuimoConfig.PATH_EPSILON_PX)
    minimum = int(ShuimoConfig.MIN_COMPONENT_PX)
    count, labels, comp_stats, _ = cv2.connectedComponentsWithStats(
        (skeleton > 0).astype(np.uint8), connectivity=8)
    paths = []
    for label in range(1, count):
        if comp_stats[label, cv2.CC_STAT_AREA] < minimum:
            continue
        component = np.zeros_like(skeleton)
        component[labels == label] = 255
        component_paths = vision._trace_paths(component, epsilon=epsilon)
        paths.extend(_join_collinear_branches(component_paths))
    paths = [p for p in paths if len(p) >= 2]

    # ---- 缺口桥接 ----
    directional_gap_px = max(0.0, float(ShuimoConfig.DIRECTIONAL_GAP_MM) /
                             max(1e-6, mm_per_px))
    if bridge_ok and directional_gap_px > 0 and paths:
        before = len(paths)
        paths = _merge_directional_paths(paths, directional_gap_px, blurred)
        merged_away = before - len(paths)
    else:
        merged_away = 0

    if not paths:
        return [], {
            "white_ratio": white_ratio, "pad_excluded": pad_applied,
            "use_light_ink": use_light_ink, "bridge_ok": bridge_ok,
            "ink_ratio": ink_ratio, "merged_away": 0,
            "otsu_gray": otsu_gray,
            "ink_px": ink_px,
            "skeleton_px": int(np.count_nonzero(skeleton)),
            "paths": 0, "strokes": 0, "length_mm": 0.0,
            "layers": {0: 0, 1: 0, 2: 0}, "regions": [],
            "mean_width_mm": 0.0, "max_width_mm": 0.0, "mean_press_mm": 0.0,
        }

    # ---- 排序（先轮廓后皴）----
    entries, max_length_px = _order_paths(paths, width_map)
    ordered = [item[4] for item in entries]
    layers = {0: 0, 1: 0, 2: 0}
    for item in entries:
        layers[int(item[0])] = layers.get(int(item[0]), 0) + 1

    # ---- 全局归一化基准：笔宽与密度分位 ----
    center_widths = []
    for path in ordered:
        for x, y in path:
            value = _width_at(width_map, x, y, height, width, fallback=0.0)
            if value > 0:
                center_widths.append(value)
    if center_widths:
        width_low = float(np.percentile(center_widths,
                                        float(ShuimoConfig.WIDTH_PCT_LOW)))
        width_high = float(np.percentile(center_widths,
                                         float(ShuimoConfig.WIDTH_PCT_HIGH)))
    else:
        width_low, width_high = 1.0, 2.0

    density_map = _local_density_map(skeleton, mm_per_px)
    density_samples = density_map[skeleton > 0]
    if density_samples.size:
        density_low = float(np.percentile(density_samples,
                                          float(ShuimoConfig.DENSITY_PCT_LOW)))
        density_high = float(np.percentile(density_samples,
                                           float(ShuimoConfig.DENSITY_PCT_HIGH)))
    else:
        density_low, density_high = 0.0, 1.0

    stats = {
        "white_ratio": white_ratio,
        "pad_excluded": pad_applied,
        "use_light_ink": use_light_ink,
        "bridge_ok": bridge_ok,
        "ink_ratio": ink_ratio,
        "merged_away": merged_away,
        "otsu_gray": otsu_gray,
        "ink_px": ink_px,
        "skeleton_px": int(np.count_nonzero(skeleton)),
        "paths": len(ordered),
        "layers": layers,
        "width_low": width_low,
        "width_span": max(0.1, width_high - width_low),
        "density_low": density_low,
        "density_span": max(1.0, density_high - density_low),
        "mm_per_px": float(mm_per_px),
    }

    # ---- 逐笔生成 ----
    # 注意：_stroke_from_path 现在返回【一个列表】——
    #       点状/串珠模式下一条路径会被切成很多小段，每段一条笔触。
    pal_index = ink_palette_index()
    strokes = []
    for layer, min_y, min_x, neg_length, path in entries:
        t_ratio = (-neg_length) / max_length_px if max_length_px > 0 else 0.0
        pieces = _stroke_from_path(path, width_map, density_map, mm_per_px,
                                   stats, pal_index, layer, t_ratio)
        strokes.extend(pieces or [])

    if strokes:
        stats["mean_width_mm"] = float(np.mean([s["width_mm"] for s in strokes]))
        stats["max_width_mm"] = float(np.max([s["width_mm"] for s in strokes]))
        stats["mean_press_mm"] = float(np.mean([s["press_mean"] for s in strokes]))
        stats["max_press_mm"] = float(np.max([s["press_max"] for s in strokes]))
    else:
        stats["mean_width_mm"] = stats["max_width_mm"] = 0.0
        stats["mean_press_mm"] = stats["max_press_mm"] = 0.0
    stats["strokes"] = len(strokes)
    stats["length_mm"] = float(sum(s["length_mm"] for s in strokes))
    stats["regions"] = []
    return strokes, stats


# =====================================================================
# 6. 预览
# =====================================================================
def _brush_footprint(sim, points_px, widths_px, color):
    """
    按"笔刷足迹"画一条变宽笔触：逐段用两点平均宽度画线，
    并在每个落点上补一个圆形成圆头，模拟毛笔扫过的痕迹。

    cv2.polylines 只能画等宽线，画不出"一笔之内有粗细"，
    所以水墨预览必须自己画。
    """
    for index in range(len(points_px) - 1):
        x0, y0 = points_px[index]
        x1, y1 = points_px[index + 1]
        thickness = int(max(1, round((widths_px[index] + widths_px[index + 1]) * 0.5)))
        cv2.line(sim, (x0, y0), (x1, y1), color, thickness, cv2.LINE_8)
        radius = max(1, thickness // 2)
        cv2.circle(sim, (x0, y0), radius, color, -1, cv2.LINE_8)


def render_framework_preview(source_bgr, strokes, mm_per_px, out_path, mapper=None, waypoints=None,
                             diagnostics=False):
    """
    原图、层级路径、最终轨迹接触估计，以及加宽 0.5 mm 的敏感性对照。
    BRUSHES 仍是估值；不把软件估计描述为实测落纸效果。
    默认只写一张拼版图；diagnostics=True 才另存 4 张分面板图。
    """
    # Use exactly the same waypoint builder as execution; legacy callers also
    # receive tails and pressure clipping in the preview.
    from geometry import CanvasMap
    import job
    import trajectory_preview
    if mapper is None:
        mapper = CanvasMap((source_bgr.shape[1], source_bgr.shape[0]))
    if waypoints is None:
        waypoints = [wp for s in strokes for wp in job.stroke_waypoints(mapper, s)]
    return trajectory_preview.render(source_bgr, strokes, mapper, waypoints, out_path,
                                     diagnostics=diagnostics)


def _legacy_framework_preview(source_bgr, strokes, mm_per_px, out_path):
    """Historical stroke-only renderer, retained for diagnostic A/B comparisons."""
    import imgio

    height, width = source_bgr.shape[:2]
    sim = np.full((height, width, 3), 255, dtype=np.uint8)
    scale = max(1e-6, float(mm_per_px))

    for stroke in strokes:
        entry = PALETTE[stroke["pal_index"]]
        rgb = entry["rgb"]
        color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        points = [(int(round(p[0])), int(round(p[1]))) for p in stroke["points"]]
        if len(points) < 2:
            continue
        presses = stroke.get("press") or []
        if len(presses) != len(points):
            presses = [float(ShuimoConfig.BRUSHES[ShuimoConfig.DEFAULT_BRUSH]
                             ["press_mm"])] * len(points)
        widths = [max(1.0, press_to_width_mm(p) / scale) for p in presses]
        _brush_footprint(sim, points, widths, color)

    left = source_bgr.copy()
    bar = np.full((24, width, 3), 245, dtype=np.uint8)
    cv2.putText(bar, "SOURCE", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(bar, "FRAMEWORK (pass 1, what the robot will paint)",
                (width // 2 + 8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (60, 60, 60), 1, cv2.LINE_AA)
    left = cv2.vconcat([bar, left])
    right = cv2.vconcat([bar, sim])
    combined = cv2.hconcat([left,
                            np.full((left.shape[0], 4, 3), 200, dtype=np.uint8),
                            right])
    if not imgio.imwrite(out_path, combined):
        raise IOError("预览图写出失败: %s" % out_path)
    return out_path


# =====================================================================
# 7. 报告
# =====================================================================
def print_stats(stats):
    strokes = int(stats.get("strokes", 0))
    if stats.get("detail") == "legacy-source":
        print("[山水旧版融合] 源图骨架 %d 条；白区断开 %d 处；短于 2mm 省略 %d 条" %
              (stats["source_paths"], stats["split_white_gaps"], stats["omitted_under_2mm"]))
        print("[山水旧版融合] 层级 %s；%d 笔，总长 %.1f mm；分层最大压深 %s mm" %
              (stats["layers"], strokes, stats["length_mm"], stats["pressure_caps_mm"]))
        print("[山水旧版融合] 旧版墨宽/密度仅用于相对强弱；当前笔宽仍须在同笔同纸上实测")
        return
    if stats.get("profile") == "mountain":
        print("[山水] 高分辨率固定阈值 210 / 主轮廓优先 / 下行筛选 / 笔宽留白 / 路径内提锋")
        print("[山水] 层级 %s；%d 笔，总长 %.1f mm，平均 %.1f mm/笔" %
              (stats["layers"], strokes, stats["length_mm"], stats["length_mm"] / max(1, strokes)))
        print("[山水] 峰值下压 %.2f mm；笔宽表未实测，预览为估计，含 +0.5 mm 加宽对照" %
              stats["max_press_mm"])
        return
    print("[水墨] 白底占比 %.1f%%（%s）  Otsu 阈值 %.0f  浅墨滞后连接=%s"
          % (stats.get("white_ratio", 0.0) * 100.0,
             "已排除画布留白填充" if stats.get("pad_excluded") else "未填充",
             stats.get("otsu_gray", 0.0),
             "启用" if stats.get("use_light_ink") else "关闭"))
    print("[水墨] 墨迹像素 %d（占画面 %.1f%%）-> 骨架 %d -> 路径 %d 条"
          % (stats.get("ink_px", 0), stats.get("ink_ratio", 0.0) * 100.0,
             stats.get("skeleton_px", 0), stats.get("paths", 0)))

    # 墨迹占比是判断"这张图适不适合水墨路线"最直接的指标：
    #   水墨线稿   2~10%   （浅底 + 少量深墨线）
    #   偏密的画   10~35%
    #   > 35%      基本不是线稿，而是彩色插画/照片 —— Otsu 在切颜色边界，
    #              提取出来的"骨架"是色块轮廓，画出来毫无意义。
    ink_ratio = float(stats.get("ink_ratio", 0.0))
    if ink_ratio > 0.35:
        print("[诊断] 墨迹占比 %.0f%% 太高，这张图多半不是「浅底 + 深墨线」的水墨画，"
              % (ink_ratio * 100.0))
        print("       而是彩色插画或照片 —— Otsu 会把颜色边界当成墨线，")
        print("       提取出来的骨架是色块轮廓，不是你要画的线条。")
        print("       这种图请改用【色块路线】：去掉 --mode shuimo 即可。")
    print("[水墨] 缺口桥接=%s（白底 %s / 线稿稀疏 %s），接掉了 %d 段断笔"
          % ("启用" if stats.get("bridge_ok") else "关闭",
             "是" if stats.get("use_light_ink") else "否",
             "是" if stats.get("ink_ratio", 1.0) <=
             float(ShuimoConfig.INK_SPARSE_RATIO) else "否",
             stats.get("merged_away", 0)))
    layers = stats.get("layers", {})
    print("[水墨] 分层：主体长轮廓 %d / 中等结构线 %d / 短皴线 %d"
          % (layers.get(0, 0), layers.get(1, 0), layers.get(2, 0)))
    if strokes == 0:
        print("[诊断] 一条笔触都没生成。可能的原因：")
        print("       1) 图里没有墨线（是彩色照片或纯色块图）——水墨模式要的是墨线图")
        print("       2) 墨线太淡，Otsu 没分出来 —— 试着调低 ShuimoConfig.LIGHT_INK_GRAY")
        print("       3) 笔触都短于 %.1fmm 被丢了 —— 调低 ShuimoConfig.MIN_STROKE_MM"
              % float(ShuimoConfig.MIN_STROKE_MM))
        return
    print("[水墨] 笔触 %d 条，总笔长 %.0f mm，平均 %.1f mm/笔"
          % (strokes, stats.get("length_mm", 0.0),
             stats.get("length_mm", 0.0) / max(1, strokes)))
    print("[水墨] 笔压 %.2f~%.2f mm（均值 %.2f）  笔宽 %.2f~%.2f mm（均值 %.2f）"
          % (float(ShuimoConfig.PRESS_MIN_MM),
             stats.get("max_press_mm", 0.0),
             stats.get("mean_press_mm", 0.0),
             float(min(p["width_mm"] for p in ShuimoConfig.BRUSHES.values())),
             stats.get("max_width_mm", 0.0),
             stats.get("mean_width_mm", 0.0)))
    finest = float(min(p["width_mm"] for p in ShuimoConfig.BRUSHES.values()))
    if stats.get("mean_width_mm", 0.0) > finest * 2.0:
        print("[诊断] 平均笔宽 %.2fmm 明显粗于最细笔档 %.2fmm："
              % (stats.get("mean_width_mm", 0.0), finest))
        print("       说明原图墨线比你的笔还细，这是物理限制。")
        print("       处理办法：换更细的狼毫，或接受'线条被加粗'的观感。")
    if strokes and stats.get("length_mm", 0.0) / strokes < 8.0:
        print("[诊断] 平均笔长只有 %.1fmm，偏短（会在纸上不停抬笔点按）。"
              % (stats.get("length_mm", 0.0) / strokes))
        print("       处理办法：调大 ShuimoConfig.DIRECTIONAL_GAP_MM，"
              "或调低 BRIDGE_INK_RATIO 让缺口更容易被判为同一笔。")


if __name__ == "__main__":
    print("shuimo.py 是模块，不单独运行。用法：")
    print("  python main.py --mode shuimo --image 范图.png")
