"""Vendored legacy landscape generator from the user-supplied project.

Only its white-background ink path is integrated into the current project.
The original color fallback needs optional dependencies and is not used here.
No robot driver is imported by this module.
"""

"""
landscape_painting.py - 山水画生成器
从图片提取纹理方向，生成具有笔墨意趣的线条
"""

import cv2
import numpy as np
# The integrated ink path does not need the legacy color dependencies.
# Keep explicit sentinels so an accidental call into the color fallback fails.
KMeans = None
sobel = None
import os
import random


class LandscapeBrushGenerator:
    """山水画笔触生成器"""

    def __init__(self, config):
        self.cfg = config

    def image_to_landscape_strokes(self, image_path, num_layers=4, stroke_density=1.0):
        """主入口：图像 → 山水画笔触列表"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        h, w = img.shape[:2]

        # 缩小图片加速
        if max(h, w) > 600:
            scale = 600 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # sb.jpg 这类白底水墨线稿应沿墨线生成连续路径。旧版按色块随机
        # 采样，只会得到少量短笔触，机械臂表现为不断点按和抬笔。
        line_strokes = self._extract_continuous_ink_strokes(gray)
        if line_strokes:
            return line_strokes

        # 提取纹理方向
        structure_tensor = self._compute_structure_tensor(gray)

        # 颜色聚类（分层）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        colors, labels = self._extract_color_regions(img_rgb, num_layers)

        all_strokes = []

        for layer_idx in range(num_layers):
            mask = (labels == layer_idx).astype(np.uint8) * 255

            if np.sum(mask) < 200:
                continue

            strokes = self._generate_landscape_strokes(
                mask, layer_idx, colors, structure_tensor, stroke_density
            )
            all_strokes.extend(strokes)

        return all_strokes

    def _extract_continuous_ink_strokes(self, gray):
        """从白底墨线图中提取连续、平滑且有序的长笔触。"""
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        otsu_gray, strong_ink = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # 浅墨滞后连接适用于大面积白纸上的墨线。满幅灰调山水若使用同一
        # 策略，会把背景层次误连成巨大色块，因此自动回退到原 Otsu 蒙版。
        white_background_ratio = float(np.mean(blurred > 245))
        if white_background_ratio >= 0.70:
            ink = self._recover_light_ink(blurred, strong_ink, otsu_gray)
        else:
            ink = strong_ink
        # 不做开运算：sb.jpg 中大量短皴线本来就很细，开运算会把它们
        # 当作噪声删除。后续仅按极小面积排除真正的孤立像素。

        # 将粗墨线细化成单像素中心骨架，避免沿一根线的两侧重复描绘。
        skeleton = self._thin_binary(ink)
        width_map = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        # Diagnostic source footprint: use the unclosed, unblurred input. The
        # recovered light-ink mask intentionally grows thin lines and therefore
        # must not be advertised as their original physical width.
        _, raw_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        raw_radius = cv2.distanceTransform(raw_ink, cv2.DIST_L2, 5)
        raw_radius = cv2.dilate(raw_radius, np.ones((3, 3), np.uint8))
        h, w = gray.shape
        paths = self._trace_skeleton_paths_by_component(skeleton)

        # 扫描/压缩会让视觉上连续的墨线留下小缺口。将近邻端点桥接，
        # 避免机械臂在每个小段后抬笔，形成真正的长线条。
        mm_per_pixel = float(getattr(self.cfg, 'LANDSCAPE_SCALE', 0.12)) * (
            1024.0 / max(h, w)
        )

        # 对跨连通域的小缺口做方向约束续接。两段必须都具有一定长度，并且
        # “第一段末端切线 → 缺口 → 第二段起始切线”方向连续，才视为同一
        # 片兰叶或同一条山石轮廓。连接后整条路径只在最终首尾抬笔。
        directional_gap_px = max(0.0, float(getattr(
            self.cfg, 'LANDSCAPE_DIRECTIONAL_JOIN_GAP_MM', 3.0
        ))) / mm_per_pixel
        directional_min_px = max(0.0, float(getattr(
            self.cfg, 'LANDSCAPE_DIRECTIONAL_JOIN_MIN_LENGTH_MM', 1.5
        ))) / mm_per_pixel
        directional_angle = float(getattr(
            self.cfg, 'LANDSCAPE_DIRECTIONAL_JOIN_ANGLE_DEG', 38.0
        ))
        if directional_gap_px > 0 and white_background_ratio >= 0.70:
            paths = self._merge_directional_paths(
                paths, directional_gap_px, directional_angle,
                directional_min_px, blurred
            )

        # 统计骨架局部密度。普通单线在邻域内的像素数接近低分位值；
        # 多条皴线、交叉线集中时计数明显增大。将其归一化为 0~1，后续
        # 用于减小下压并提高速度，而不是直接删除密集处的细节。
        dense_radius_mm = max(
            0.0, float(getattr(self.cfg, 'LANDSCAPE_DENSE_RADIUS_MM', 3.0))
        )
        dense_radius_px = max(1, int(round(dense_radius_mm / mm_per_pixel)))
        dense_kernel = dense_radius_px * 2 + 1
        skeleton_mask = (skeleton > 0).astype(np.float32)
        local_density = cv2.boxFilter(
            skeleton_mask, -1, (dense_kernel, dense_kernel),
            normalize=False, borderType=cv2.BORDER_REPLICATE
        )
        density_samples = local_density[skeleton > 0]
        density_low = (
            float(np.percentile(density_samples, 30))
            if density_samples.size else 0.0
        )
        density_high = (
            float(np.percentile(density_samples, 90))
            if density_samples.size else density_low + 1.0
        )
        density_span = max(1.0, density_high - density_low)

        join_gap_px = float(getattr(self.cfg, 'LANDSCAPE_JOIN_GAP_MM', 8.0)) / mm_per_pixel
        if join_gap_px > 0:
            paths = self._merge_nearby_paths(paths, join_gap_px)

        if not paths:
            return []

        # 人类山水画顺序：主体长轮廓 → 中等结构线 → 短皴线；同层按
        # 上到下、左到右推进，不再按“离上一笔最近”这种无语义顺序。
        lengths = [
            sum(np.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:]))
            for path in paths
        ]
        max_length = max(lengths) if lengths else 1.0
        semantic_paths = []
        for path, length in zip(paths, lengths):
            # 兰叶等一笔长线应从较粗的叶根运向较细的叶尖。比较两端一小段
            # 的原始墨宽；宽度接近时才回退到稳定的上/左排序规则。
            sample_count = max(1, min(6, len(path) // 4))
            start_width = float(np.mean([
                width_map[y, x] for x, y in path[:sample_count]
            ]))
            end_width = float(np.mean([
                width_map[y, x] for x, y in path[-sample_count:]
            ]))
            if end_width > start_width * 1.08:
                path = list(reversed(path))
            elif start_width <= end_width * 1.08:
                start_key = (path[0][1], path[0][0])
                end_key = (path[-1][1], path[-1][0])
                if end_key < start_key:
                    path = list(reversed(path))
            ratio = length / max_length
            layer = 0 if ratio >= 0.35 else (1 if ratio >= 0.12 else 2)
            semantic_paths.append((layer, min(p[1] for p in path),
                                   min(p[0] for p in path), -length, path))
        semantic_paths.sort(key=lambda item: item[:4])
        ordered = [item[4] for item in semantic_paths]

        center_widths = [
            float(width_map[y, x])
            for path in ordered for x, y in path
            if 0 <= x < w and 0 <= y < h and width_map[y, x] > 0
        ]
        width_low = float(np.percentile(center_widths, 10)) if center_widths else 1.0
        width_high = float(np.percentile(center_widths, 90)) if center_widths else 2.0
        width_span = max(0.1, width_high - width_low)

        strokes = []
        for path in ordered:
            trajectory = []
            source_width_1024 = []
            for x, y in path:
                nx = int((x / w) * 1024)
                ny = int((y / h) * 1024)
                width = float(width_map[y, x]) if 0 <= x < w and 0 <= y < h else width_low
                # Allow one source pixel of skeleton displacement. Subtract
                # one pixel because EDT measures to background pixel centres.
                raw_width = max(0.0, 2.0 * float(raw_radius[y, x]) - 1.0)
                source_width_1024.append(raw_width * 1024.0 / w)
                pressure_strength = float(np.clip((width-width_low)/width_span, 0.0, 1.0))
                density_strength = float(np.clip(
                    (float(local_density[y, x]) - density_low) / density_span,
                    0.0, 1.0
                ))
                trajectory.append([
                    nx, ny, pressure_strength, density_strength
                ])
            strokes.append({
                'color': [0, 0, 0],
                'trajectory': trajectory,
                'source_width_1024': source_width_1024,
                'type': 'LINE',
                'layer': next(item[0] for item in semantic_paths if item[4] is path),
            })
        return strokes

    def _recover_light_ink(self, blurred, strong_ink, otsu_gray):
        """以深墨为种子进行滞后连接，保留同一笔迹中的浅墨段。"""
        configured = int(getattr(self.cfg, 'LANDSCAPE_LIGHT_INK_GRAY', 238))
        # 浅墨阈值必须比本图的深墨阈值宽松，同时避开接近纯白的 JPEG 噪声。
        weak_limit = int(np.clip(max(float(otsu_gray) + 12.0, configured), 0, 250))
        weak_ink = (blurred < weak_limit).astype(np.uint8)

        close_px = max(1, int(getattr(
            self.cfg, 'LANDSCAPE_LIGHT_INK_CLOSE_PX', 3
        )))
        if close_px % 2 == 0:
            close_px += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        weak_ink = cv2.morphologyEx(weak_ink, cv2.MORPH_CLOSE, kernel)

        # 只接纳与深墨种子相连（或非常接近）的弱墨连通域。这样能够沿兰叶
        # 穿过淡墨区，又不会把白纸上的压缩噪声全部变成轨迹。
        support = cv2.dilate(
            (strong_ink > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            weak_ink, connectivity=8
        )
        recovered = np.zeros_like(strong_ink)
        for label in range(1, count):
            component = labels == label
            if stats[label, cv2.CC_STAT_AREA] < 2:
                continue
            if np.any(support[component] > 0):
                recovered[component] = 255
        return recovered

    @classmethod
    def _trace_skeleton_paths_by_component(cls, skeleton):
        """逐个原始墨迹连通区域提取，严禁跨空白区域错误合笔。"""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (skeleton > 0).astype(np.uint8), connectivity=8
        )
        paths = []
        for label in range(1, count):
            # 仅排除不足两个像素、机械臂无法表达的孤立噪点。
            if stats[label, cv2.CC_STAT_AREA] < 2:
                continue
            component = np.zeros_like(skeleton)
            component[labels == label] = 255
            component_paths = cls._trace_skeleton_paths(component)
            paths.extend(cls._join_collinear_branches(component_paths))
        return paths

    @staticmethod
    def _join_collinear_branches(paths):
        """只在同一墨迹交叉点按切线方向续接，模拟毛笔自然顺势运笔。"""
        result = [list(path) for path in paths]
        while len(result) > 1:
            best = None
            # Only endpoints within 2.5 source pixels can ever join. Index
            # them instead of repeatedly copying every pair of long paths.
            buckets = {}
            for j, path in enumerate(result):
                if len(path) < 2:
                    continue
                for x, y in (path[0], path[-1]):
                    buckets.setdefault((int(np.floor(x / 3)), int(np.floor(y / 3))), set()).add(j)
            for i in range(len(result)):
                if len(result[i]) < 2:
                    continue
                nearby = set()
                for x, y in (result[i][0], result[i][-1]):
                    gx, gy = int(np.floor(x / 3)), int(np.floor(y / 3))
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            nearby.update(buckets.get((gx+dx, gy+dy), ()))
                # Preserve original pair order and its exact tie-breaking.
                for j in sorted(k for k in nearby if k > i):
                    for reverse_a in (False, True):
                        for reverse_b in (False, True):
                            a, b = result[i], result[j]
                            ae, ap = (a[0], a[1]) if reverse_a else (a[-1], a[-2])
                            bs, bn = (b[-1], b[-2]) if reverse_b else (b[0], b[1])
                            gap = np.hypot(ae[0]-bs[0], ae[1]-bs[1])
                            if gap > 2.5:
                                continue
                            incoming = np.array(ae, float) - np.array(ap, float)
                            outgoing = np.array(bn, float) - np.array(bs, float)
                            denominator = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
                            if denominator == 0:
                                continue
                            cosine = float(np.dot(incoming, outgoing) / denominator)
                            # 夹角小于约 60° 才视为同一自然笔势。
                            if cosine < 0.5:
                                continue
                            score = gap + (1.0 - cosine) * 5.0
                            if best is None or score < best[0]:
                                best = (score, i, j, reverse_a, reverse_b)
            if best is None:
                break
            _, i, j, reverse_a, reverse_b = best
            a = result[i][::-1] if reverse_a else result[i]
            b = result[j][::-1] if reverse_b else result[j]
            if a[-1] == b[0]:
                combined = a + b[1:]
            else:
                combined = a + b
            result[i] = combined
            result.pop(j)
        return result

    @staticmethod
    def _merge_nearby_paths(paths, max_gap):
        """贪心连接最近的路径端点，并为缺口补充平滑过渡点。"""
        merged = [list(path) for path in paths]
        while len(merged) > 1:
            best = None
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    a, b = merged[i], merged[j]
                    choices = (
                        (np.hypot(a[-1][0]-b[0][0], a[-1][1]-b[0][1]), False, False),
                        (np.hypot(a[-1][0]-b[-1][0], a[-1][1]-b[-1][1]), False, True),
                        (np.hypot(a[0][0]-b[0][0], a[0][1]-b[0][1]), True, False),
                        (np.hypot(a[0][0]-b[-1][0], a[0][1]-b[-1][1]), True, True),
                    )
                    distance, reverse_a, reverse_b = min(choices, key=lambda item: item[0])
                    if best is None or distance < best[0]:
                        best = (distance, i, j, reverse_a, reverse_b)

            if best is None or best[0] > max_gap:
                break

            distance, i, j, reverse_a, reverse_b = best
            first = merged[i][::-1] if reverse_a else merged[i]
            second = merged[j][::-1] if reverse_b else merged[j]

            # 每约 5 像素补一点，控制器沿这些点连续插补，不会在缺口抬笔。
            bridge = []
            steps = max(1, int(distance / 5.0))
            for step in range(1, steps):
                ratio = step / steps
                bridge.append((
                    int(round(first[-1][0] * (1-ratio) + second[0][0] * ratio)),
                    int(round(first[-1][1] * (1-ratio) + second[0][1] * ratio)),
                ))
            combined = first + bridge + second
            merged[i] = combined
            merged.pop(j)

        return merged

    @staticmethod
    def _merge_directional_paths(paths, max_gap, max_angle_deg,
                                 min_length, gray):
        """按端点切线续接长线缺口，拒绝仅因空间相近而误合笔。"""
        merged = [list(path) for path in paths]
        cosine_limit = float(np.cos(np.deg2rad(max_angle_deg)))

        def path_length(path):
            return sum(
                np.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(path, path[1:])
            )

        def unit(vector):
            norm = float(np.linalg.norm(vector))
            return vector / norm if norm > 1e-6 else None

        def bridge_has_ink(start, end):
            distance = float(np.hypot(end[0]-start[0], end[1]-start[1]))
            samples = max(3, int(np.ceil(distance)))
            values = []
            h, w = gray.shape
            for index in range(1, samples):
                ratio = index / samples
                x = int(round(start[0] * (1-ratio) + end[0] * ratio))
                y = int(round(start[1] * (1-ratio) + end[1] * ratio))
                if 0 <= x < w and 0 <= y < h:
                    values.append(float(gray[y, x]))
            if not values:
                return False
            # 允许缺口中央完全褪白，但附近至少应留下一部分浅墨证据。
            return float(np.mean(np.asarray(values) < 248.0)) >= 0.18

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
                            if gap < 0.5 or gap > max_gap:
                                continue

                            # 使用端点附近多点平均切线，避免骨架单像素锯齿左右
                            # 摇摆导致本应连续的兰叶被判为转角。
                            window_a = min(5, len(a) - 1)
                            window_b = min(5, len(b) - 1)
                            incoming = unit(
                                np.array(a[-1], float) - np.array(a[-1-window_a], float)
                            )
                            outgoing = unit(
                                np.array(b[window_b], float) - np.array(b[0], float)
                            )
                            gap_direction = unit(gap_vector)
                            if incoming is None or outgoing is None or gap_direction is None:
                                continue
                            alignment_a = float(np.dot(incoming, gap_direction))
                            alignment_b = float(np.dot(gap_direction, outgoing))
                            alignment_ab = float(np.dot(incoming, outgoing))
                            if min(alignment_a, alignment_b, alignment_ab) < cosine_limit:
                                continue
                            if not bridge_has_ink(a[-1], b[0]):
                                continue
                            score = (
                                gap + (3.0-alignment_a-alignment_b-alignment_ab)
                                * max_gap
                            )
                            if best is None or score < best[0]:
                                best = (score, i, j, reverse_a, reverse_b, gap)

            if best is None:
                break
            _, i, j, reverse_a, reverse_b, gap = best
            first = merged[i][::-1] if reverse_a else merged[i]
            second = merged[j][::-1] if reverse_b else merged[j]
            bridge = []
            steps = max(1, int(np.ceil(gap / 3.0)))
            for step in range(1, steps):
                ratio = step / steps
                bridge.append((
                    int(round(first[-1][0]*(1-ratio) + second[0][0]*ratio)),
                    int(round(first[-1][1]*(1-ratio) + second[0][1]*ratio)),
                ))
            merged[i] = first + bridge + second
            merged.pop(j)
        return merged

    @staticmethod
    def _thin_binary(binary):
        """Zhang-Suen 中轴细化，保留细节且避免形态学骨架的伪分叉。"""
        image = (binary > 0).astype(np.uint8)
        changed = True
        while changed:
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
                count = p2+p3+p4+p5+p6+p7+p8+p9
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
                remove = (
                    (center == 1) & (count >= 2) & (count <= 6) &
                    (transitions == 1) & (side_a == 0) & (side_b == 0)
                )
                if np.any(remove):
                    center[remove] = 0
                    changed = True
        return image * 255

    @staticmethod
    def _trace_skeleton_paths(skeleton):
        """将骨架图的每条边恰好提取一次，保留分叉和短皴线。"""
        ys, xs = np.where(skeleton > 0)
        pixels = set(zip(xs.tolist(), ys.tolist()))
        if not pixels:
            return []

        def links(point):
            x, y = point
            adjacent = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    candidate = (x + dx, y + dy)
                    if not (dx or dy) or candidate not in pixels:
                        continue
                    # 若对角像素可经一个正交像素连接，就忽略这条对角捷径。
                    # 这样可消除骨架拐角处的小三角，避免产生海量伪分叉。
                    if dx and dy and (
                        (x + dx, y) in pixels or (x, y + dy) in pixels
                    ):
                        continue
                    adjacent.append(candidate)
            return adjacent

        graph = {point: links(point) for point in pixels}
        nodes = {point for point, adjacent in graph.items() if len(adjacent) != 2}
        visited = set()
        raw_paths = []

        def edge(a, b):
            return tuple(sorted((a, b)))

        def walk(start, following):
            path = [start]
            previous, current = start, following
            visited.add(edge(previous, current))
            while True:
                path.append(current)
                if current in nodes and current != start:
                    break
                choices = [
                    point for point in graph[current]
                    if point != previous and edge(current, point) not in visited
                ]
                if not choices:
                    break
                following = choices[0]
                visited.add(edge(current, following))
                previous, current = current, following
            return path

        # 先提取端点/分叉点之间的段，再补充没有节点的闭环。
        for node in nodes:
            for adjacent in graph[node]:
                if edge(node, adjacent) not in visited:
                    raw_paths.append(walk(node, adjacent))
        for point, adjacent_points in graph.items():
            for adjacent in adjacent_points:
                if edge(point, adjacent) not in visited:
                    raw_paths.append(walk(point, adjacent))

        smoothed = []
        for path in raw_paths:
            if len(path) < 2:
                continue
            contour = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
            simplified = cv2.approxPolyDP(contour, epsilon=0.65, closed=False)
            points = [(int(p[0][0]), int(p[0][1])) for p in simplified]
            if len(points) == 2:
                a, b = points
                points.insert(1, ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2))
            if len(points) >= 3:
                smoothed.append(points)
        return smoothed

    def _compute_structure_tensor(self, gray):
        """计算结构张量，提取纹理方向"""
        grad_x = sobel(gray.astype(float), axis=1)
        grad_y = sobel(gray.astype(float), axis=0)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)

        # 低梯度区域设为无效
        magnitude_threshold = np.percentile(magnitude, 20)
        angle[magnitude < magnitude_threshold] = np.nan

        return {
            'angle': angle,
            'magnitude': magnitude,
            'grad_x': grad_x,
            'grad_y': grad_y
        }

    def _extract_color_regions(self, img, num_colors):
        """K-Means 颜色聚类"""
        pixels = img.reshape(-1, 3)

        if len(pixels) > 50000:
            sample_idx = np.random.choice(len(pixels), 50000, replace=False)
            pixels_sample = pixels[sample_idx]
        else:
            pixels_sample = pixels

        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=5)
        labels = kmeans.fit_predict(pixels_sample)

        if len(pixels) > 50000:
            labels_full = np.zeros(len(pixels), dtype=int)
            for i, p in enumerate(pixels):
                dists = np.linalg.norm(p - kmeans.cluster_centers_, axis=1)
                labels_full[i] = np.argmin(dists)
            labels = labels_full

        colors = kmeans.cluster_centers_.astype(int)
        return colors, labels.reshape(img.shape[:2])

    def _generate_landscape_strokes(self, mask, layer_idx, colors, tensor, density):
        """在色块内沿纹理方向生成笔触"""
        strokes = []
        h, w = mask.shape

        y_idxs, x_idxs = np.where(mask > 0)
        if len(y_idxs) < 100:
            return strokes

        # 采样起点
        num_strokes = int(15 * density)
        start_points = self._sample_start_points(mask, num_strokes)

        for start_x, start_y in start_points:
            if np.isnan(tensor['angle'][start_y, start_x]):
                continue

            angle = tensor['angle'][start_y, start_x]
            magnitude = tensor['magnitude'][start_y, start_x]

            # 根据梯度强度决定笔触长度
            max_mag = np.max(tensor['magnitude'])
            stroke_length = int(20 + 40 * (magnitude / max_mag if max_mag > 0 else 0.5))

            # 生成笔触路径
            path = self._generate_stroke_path(
                start_x, start_y, angle, stroke_length, mask
            )

            if len(path) > 5:
                stroke_type = self._classify_stroke_type(path)
                trajectory = self._to_trajectory_format(path, h, w)

                strokes.append({
                    'color': colors[layer_idx].tolist(),
                    'trajectory': trajectory,
                    'type': stroke_type,
                    'layer': layer_idx
                })

        return strokes

    def _sample_start_points(self, mask, num_points):
        """在色块内均匀采样起点"""
        h, w = mask.shape
        y_idxs, x_idxs = np.where(mask > 0)

        if len(y_idxs) == 0:
            return []

        grid_size = int(np.sqrt(h * w / num_points))
        grid_size = max(10, grid_size)

        points = []
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                offset_x = random.randint(-grid_size//3, grid_size//3)
                offset_y = random.randint(-grid_size//3, grid_size//3)
                px = min(w-1, max(0, x + offset_x))
                py = min(h-1, max(0, y + offset_y))
                if mask[py, px] > 0:
                    points.append((px, py))

        random.shuffle(points)
        return points[:num_points]

    def _generate_stroke_path(self, start_x, start_y, angle, length, mask):
        """沿纹理方向生成笔触路径"""
        h, w = mask.shape
        path = []

        dx = np.cos(angle)
        dy = np.sin(angle)

        for i in range(length):
            noise = 0.5 * np.sin(i * 0.3) + 0.3 * np.random.randn()
            px = int(start_x + i * dx + noise * dy)
            py = int(start_y + i * dy + noise * dx)

            if 0 <= px < w and 0 <= py < h:
                if mask[py, px] > 0:
                    path.append((px, py))
                else:
                    break
            else:
                break

        # 如果太短，反向尝试
        if len(path) < 5:
            path = []
            for i in range(length):
                noise = 0.5 * np.sin(i * 0.3) + 0.3 * np.random.randn()
                px = int(start_x - i * dx + noise * dy)
                py = int(start_y - i * dy + noise * dx)
                if 0 <= px < w and 0 <= py < h:
                    if mask[py, px] > 0:
                        path.append((px, py))
                    else:
                        break
                else:
                    break

        return path

    def _classify_stroke_type(self, path):
        """根据路径特征分类笔触类型"""
        if len(path) < 10:
            return 'DOT'

        pts = np.array(path)
        if len(pts) > 5:
            vectors = np.diff(pts, axis=0)
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            angle_diffs = np.abs(np.diff(angles))
            mean_curvature = np.mean(angle_diffs) if len(angle_diffs) > 0 else 0

            if mean_curvature < 0.3:
                return 'LINE'
            elif mean_curvature < 0.8:
                return 'CUN'
            else:
                return 'DOT'

        return 'LINE'

    def _to_trajectory_format(self, pts, h, w):
        """转换为轨迹格式 (0-1024 坐标)"""
        trajectory = []
        for x, y in pts:
            nx = int((x / w) * 1024)
            ny = int((y / h) * 1024)
            trajectory.append([nx, ny])
        return trajectory


class LandscapeBrushEngine:
    """山水画笔触引擎"""

    def __init__(self, config):
        self.cfg = config
        # 笔压和提笔高度都是相对纸面的偏移量，不能直接作为绝对 Z 下发。
        self.lift_offset = float(config.LIFT_Z)
        self.base_z = float(getattr(config, 'LANDSCAPE_BASE_Z', config.BASE_Z))

        self.stroke_params = {
            'LINE': {'pressure': 0.8, 'speed': 0.6, 'variation': 0.3},
            'CUN': {'pressure': 0.6, 'speed': 0.4, 'variation': 0.5},
            'DOT': {'pressure': 1.2, 'speed': 0.3, 'variation': 0.1}
        }

    def generate_waypoints(self, strokes, offset_x=0, offset_y=0):
        """生成机械臂轨迹点"""
        all_waypoints = []
        # 笔画列表的当前顺序就是最终笔顺。生成器已经给出初始顺序，轨迹
        # 编辑器还允许人工上移/下移；这里不能再次按 layer 排序覆盖人工结果。
        for stroke in strokes:
            if not stroke.get('enabled', True):
                continue
            trajectory = stroke['trajectory']
            stroke_type = stroke.get('type', 'LINE')
            num_pts = len(trajectory)

            if num_pts < 3:
                continue

            params = self.stroke_params.get(stroke_type, self.stroke_params['LINE'])
            pressure_scale = float(np.clip(
                stroke.get('pressure_scale', 1.0), 0.0, 2.0
            ))
            speed_scale = float(np.clip(
                stroke.get('speed_scale', 1.0), 0.2, 3.0
            ))
            stroke_points = []

            for i, pt in enumerate(trajectory):
                t = i / (num_pts - 1) if num_pts > 1 else 0

                scale = float(getattr(self.cfg, 'LANDSCAPE_SCALE', self.cfg.FONT_SCALE))
                x = offset_x + (1024 - pt[1]) * scale
                y = offset_y + pt[0] * scale

                if len(pt) >= 3:
                    min_pressure = float(getattr(
                        self.cfg, 'LANDSCAPE_PRESS_MIN', 0.25
                    ))
                    pressure = (
                        min_pressure +
                        float(pt[2]) * (
                            float(getattr(self.cfg, 'LANDSCAPE_PRESS_MAX', 1.8)) -
                            min_pressure
                        )
                    )
                else:
                    min_pressure = 0.0
                    pressure = params['pressure']

                density_strength = (
                    float(np.clip(pt[3], 0.0, 1.0)) if len(pt) >= 4 else 0.0
                )
                dense_reduction = float(np.clip(getattr(
                    self.cfg, 'LANDSCAPE_DENSE_PRESS_REDUCTION', 0.55
                ), 0.0, 1.0))
                # 只压缩“超过最轻档”的部分，保证密集区域最多退回轻压，
                # 不会因简单减法变成负压力或意外抬笔。
                pressure = min_pressure + (pressure - min_pressure) * (
                    1.0 - dense_reduction * density_strength
                )
                # 每笔可编辑倍率只作用于最轻接触以上的下压量，避免倍率为零
                # 时越过纸面基准；最终仍会由机械安全上限统一限幅。
                pressure = min_pressure + (pressure - min_pressure) * pressure_scale

                ramp = 0.12
                if t < ramp:
                    factor = t / ramp
                    # 横向行笔期间始终保持接触纸面。旧实现从 LIFT_Z=-8
                    # 沿前 12% 路径斜着落笔，实际会漏掉笔画开头约一成。
                    z_offset = min_pressure * (1 - factor) + pressure * factor
                    speed = params['speed'] * (0.5 + 0.5 * factor)

                elif t < 1.0 - ramp:
                    if stroke_type == 'LINE':
                        z_offset = pressure + 0.08 * np.sin(t * np.pi * 2)
                        # 重压处略慢，细线处略快，模拟毛笔行笔。
                        max_press = float(getattr(self.cfg, 'LANDSCAPE_PRESS_MAX', 1.8))
                        speed = params['speed'] * (1.05 - 0.25 * pressure / max_press)
                    elif stroke_type == 'CUN':
                        # 在图像宽度和人工压力倍率的结果上叠加皴笔起伏，
                        # 不再用固定参数覆盖编辑器设置。
                        pressure += (
                            params['variation'] * np.sin(t * np.pi * 5) * 0.5
                            * pressure_scale
                        )
                        pressure = max(min_pressure, pressure)
                        z_offset = pressure
                        speed = params['speed'] * (0.5 + 0.5 * np.abs(np.sin(t * np.pi * 4)))
                    else:
                        # DOT 同样保留图像估计与人工倍率；只有旧式二维轨迹
                        # 没有宽度信息时，前面才会使用该类型的默认压力。
                        z_offset = pressure
                        speed = params['speed']

                else:
                    factor = (1 - t) / ramp
                    # 收锋只把压力退回轻触，不在横向移动中抬离纸面；真正
                    # 抬笔会在末端同一 XY 位置以独立垂直动作完成。
                    z_offset = pressure * factor + min_pressure * (1 - factor)
                    speed = params['speed'] * (0.5 + 0.5 * factor)

                # 正方向为下压，限制在现场最粗线标定值以内；LINE 中段的
                # 轻微正弦起伏也不能越过这一机械安全上限。
                max_pressure = float(getattr(
                    self.cfg, 'LANDSCAPE_PRESS_MAX', 1.8
                ))
                # 当前循环生成的全是实际行笔点，最低只能退到轻触纸面；
                # 不能被 LINE 的正弦变化带到负偏移而产生细小断墨。
                z_offset = float(np.clip(
                    z_offset, min_pressure, max_pressure
                ))

                dense_speed_boost = max(0.0, float(getattr(
                    self.cfg, 'LANDSCAPE_DENSE_SPEED_BOOST', 0.45
                )))
                speed *= 1.0 + dense_speed_boost * density_strength
                speed *= speed_scale

                x += np.random.uniform(-0.05, 0.05)
                y += np.random.uniform(-0.05, 0.05)

                pause = 0.05 if stroke_type == 'DOT' and (t < 0.1 or t > 0.9) else 0

                stroke_points.append({
                    'x': float(x),
                    'y': float(y),
                    'z': float(self.base_z + z_offset),
                    'pause': pause,
                    'velocity': self.cfg.BASE_SPEED_RATIO * speed,
                    'ink_density': density_strength,
                })

            if stroke_points:
                # 落笔和抬笔都在笔画端点原地垂直完成。所有图像轨迹点均为
                # 连续落笔，避免以斜向升降吃掉兰叶或山石轮廓的首尾。
                first = stroke_points[0]
                last = stroke_points[-1]
                approach = {
                    'x': first['x'], 'y': first['y'],
                    'z': float(self.base_z + self.lift_offset),
                    'pause': 0.02, 'velocity': 30,
                    'pen_up': True, 'motion_phase': 'approach',
                }
                for point in stroke_points:
                    point['pen_up'] = False
                    point['motion_phase'] = 'draw'
                lift = {
                    'x': last['x'], 'y': last['y'],
                    'z': float(self.base_z + self.lift_offset),
                    'pause': 0.02, 'velocity': 30,
                    'pen_up': True, 'motion_phase': 'lift',
                }
                all_waypoints.append(approach)
                all_waypoints.extend(stroke_points)
                all_waypoints.append(lift)

        return all_waypoints


# =================================================================================
# 公共接口
# =================================================================================

def generate_landscape_trajectory(image_path, config, num_layers=4, stroke_density=1.0):
    """生成山水画轨迹"""
    if not os.path.exists(image_path):
        print(f"错误: 图片不存在 - {image_path}")
        return []

    print(f"正在生成山水画轨迹: {image_path}")
    print(f"  层次数: {num_layers}")
    print(f"  笔触密度: {stroke_density}")

    generator = LandscapeBrushGenerator(config)
    strokes = generator.image_to_landscape_strokes(image_path, num_layers, stroke_density)

    print(f"  生成 {len(strokes)} 个笔触")

    if not strokes:
        return []

    engine = LandscapeBrushEngine(config)
    # 使用纸面标定起点；旧代码默认从用户坐标系 (0, 0) 开始。
    waypoints = engine.generate_waypoints(
        strokes,
        offset_x=float(config.START_X),
        offset_y=float(config.START_Y),
    )

    print(f"  生成 {len(waypoints)} 个轨迹点")

    return waypoints


# =================================================================================
# 测试代码 - 直接使用你的 sb.jpg
# =================================================================================

if __name__ == "__main__":
    class TestConfig:
        FONT_SCALE = 0.1
        BASE_Z = 0.0
        LIFT_Z = 10.0
        BASE_SPEED_RATIO = 50
        START_X = 20.0
        START_Y = 50.0

    config = TestConfig()

    # 直接用你的图片
    YOUR_IMAGE = "sb.jpg"
    
    print(f"正在测试图片: {YOUR_IMAGE}")
    
    if not os.path.exists(YOUR_IMAGE):
        print(f"❌ 错误: 找不到 {YOUR_IMAGE}")
        print("请确保 sb.jpg 在当前目录下")
    else:
        print("\n生成山水画轨迹...")
        waypoints = generate_landscape_trajectory(YOUR_IMAGE, config, num_layers=4, stroke_density=1.0)

        if waypoints:
            print(f"\n✅ 成功生成 {len(waypoints)} 个轨迹点")
            # 打印Z轴范围
            z_values = [wp['z'] for wp in waypoints if wp['z'] < 5]
            if z_values:
                print(f"Z轴范围: {min(z_values):.2f} ~ {max(z_values):.2f}")
            print("\n可以运行 calligraphy.py 选择模式3开始绘制")
        else:
            print("❌ 生成失败，请检查图片格式")
