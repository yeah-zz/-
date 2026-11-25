# -*- coding: utf-8 -*-
"""
geometry.py - 图像像素 <-> 机械臂毫米 的映射，以及可达范围检查
====================================================================

【为什么需要单应变换】
  照片/图片里的"像素坐标"和纸面上的"机械臂坐标"之间是一个平面到平面的
  透视关系。只要给出 4 组对应点（图像上的 4 个角 -> 纸面 4 个角），就能
  用一个 3x3 矩阵把任意像素点换算成机械臂坐标，并且自动包含纸张的旋转
  和轻微倾斜。

【与旧项目的区别】
  旧项目是硬编码 (1024 - y) * FONT_SCALE 这种写法，两个地方还不一致。
  本模块是唯一入口，所有坐标都必须经过 CanvasMap.to_robot()。
"""

import math

import cv2
import numpy as np

from config import ArmConfig, CanvasConfig, ReachConfig, z_for_lift, z_for_press


class CanvasMap:
    """图像像素 -> 用户坐标系毫米（含旋转/缩放/透视）"""

    def __init__(self, image_size, image_corners=None):
        """
        image_size: (width, height) 像素，必须是"最终用于绘制的那张图"的尺寸
                    （即经过等比缩放和留白填充之后的图）
        image_corners: 4 个像素点（左上/右上/右下/左下）；None = 用整张图的四角
        """
        width, height = int(image_size[0]), int(image_size[1])
        if width < 2 or height < 2:
            raise ValueError("图像尺寸非法: %r" % (image_size,))

        if image_corners is None:
            src = [(0.0, 0.0), (width - 1.0, 0.0),
                   (width - 1.0, height - 1.0), (0.0, height - 1.0)]
        else:
            src = [(float(p[0]), float(p[1])) for p in image_corners]

        dst = [(float(p[0]), float(p[1])) for p in CanvasConfig.ROBOT_CORNERS]
        self.matrix = cv2.getPerspectiveTransform(
            np.array([src], dtype=np.float32), np.array([dst], dtype=np.float32))
        self.image_size = (width, height)
        self.image_corners = src
        self.robot_corners = dst
        self.mm_per_px = self._estimate_mm_per_px()

    # ------------------------------------------------------------------
    def _estimate_mm_per_px(self):
        """用四条边的毫米长度 / 像素长度估算平均比例（用于换算笔宽、行距）"""
        src = self.image_corners
        dst = self.robot_corners
        ratios = []
        for i in range(4):
            j = (i + 1) % 4
            px = math.hypot(src[j][0] - src[i][0], src[j][1] - src[i][1])
            mm = math.hypot(dst[j][0] - dst[i][0], dst[j][1] - dst[i][1])
            if px > 1e-6:
                ratios.append(mm / px)
        if not ratios:
            return 1.0
        return float(sum(ratios) / len(ratios))

    def to_robot(self, points_px):
        """(N, 2) 像素坐标 -> (N, 2) 机械臂 mm"""
        arr = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(arr, self.matrix)
        return out.reshape(-1, 2).astype(np.float64)

    def to_robot_point(self, point_px):
        """单个点 -> (x, y)"""
        out = self.to_robot([point_px])
        return (float(out[0][0]), float(out[0][1]))

    def canvas_size_mm(self):
        """纸面绘制区域的 (宽, 高) 毫米"""
        dst = self.robot_corners
        top = math.hypot(dst[1][0] - dst[0][0], dst[1][1] - dst[0][1])
        bottom = math.hypot(dst[2][0] - dst[3][0], dst[2][1] - dst[3][1])
        left = math.hypot(dst[3][0] - dst[0][0], dst[3][1] - dst[0][1])
        right = math.hypot(dst[2][0] - dst[1][0], dst[2][1] - dst[1][1])
        return (0.5 * (top + bottom), 0.5 * (left + right))

    def summary(self):
        w_mm, h_mm = self.canvas_size_mm()
        return ("画布 %.1f x %.1f mm，图像 %d x %d px，比例 %.4f mm/px"
                % (w_mm, h_mm, self.image_size[0], self.image_size[1], self.mm_per_px))


# =====================================================================
# 可达范围检查
# =====================================================================
def check_waypoints(waypoints, limit=8):
    """
    检查所有轨迹点是否落在 ReachConfig 允许的范围内。
    返回 (ok, problems)；problems 是最多 limit 条文字描述。
    """
    problems = []
    ok = True
    for i, wp in enumerate(waypoints):
        for axis, key in (("X", "x"), ("Y", "y"), ("Z", "z")):
            value = float(wp[key])
            if not math.isfinite(value):
                ok = False
                if len(problems) < limit:
                    problems.append("第 %d 点 %s 不是有限数值: %r" % (i, axis, wp[key]))
                continue
            low, high = _limits(axis)
            if value < low or value > high:
                ok = False
                if len(problems) < limit:
                    problems.append("第 %d 点 %s=%.1f 超出 [%.1f, %.1f]"
                                    % (i, axis, value, low, high))
    return ok, problems


def _limits(axis):
    if axis == "X":
        return ReachConfig.X_MIN, ReachConfig.X_MAX
    if axis == "Y":
        return ReachConfig.Y_MIN, ReachConfig.Y_MAX
    return ReachConfig.Z_MIN, ReachConfig.Z_MAX


def check_canvas_reach(mapper):
    """检查画布四角 + 中心点是否可达（上机前先跑一次最省事）"""
    corners = [tuple(p) for p in mapper.robot_corners]
    centre = (sum(p[0] for p in corners) / 4.0, sum(p[1] for p in corners) / 4.0)
    points = corners + [centre]
    names = ["左上", "右上", "右下", "左下", "中心"]
    problems = []
    for name, (x, y) in zip(names, points):
        for axis, value in (("X", x), ("Y", y), ("Z", ArmConfig.PAPER_Z)):
            low, high = _limits(axis)
            if value < low or value > high:
                problems.append("画布%s点 %s=%.1f 超出 [%.1f, %.1f]"
                                % (name, axis, value, low, high))
    return problems


def press_to_z(press_mm):
    """便捷转发：下压量 -> 绝对 Z"""
    return z_for_press(press_mm)


def lift_to_z(height_mm):
    """便捷转发：离纸高度 -> 绝对 Z"""
    return z_for_lift(height_mm)
