# -*- coding: utf-8 -*-
"""笔触轨迹后处理：平滑 B 样条 + 曲率速度 + 压力/线宽耦合。

参考思路（不是照搬代码）：
- Calligraph / long smoothing B-splines：用长而平滑的曲线表达一笔，曲率越高、
  速度越慢，宽度变化可映射为笔压；这样机械臂不会在转折处过冲、堆墨。
- CLIPascene：把"忠实度"和"简洁度"分开控制；这里先落到"曲率/速度/笔压"，
  后续可在 stroke_plan 层做重要性剪枝。
- CoFRIDA：机器人约束要进入生成与执行之间的仿真闭环；这里把每点速度、
  压力、曲率与预计时长写进轨迹，供导出与实机诊断使用。
"""

import math

import numpy as np


def _bspline(control, samples=7):
    """均匀三次 B 样条；端点重复，保证起收笔不跑偏。"""
    p = [tuple(map(float, q)) for q in control]
    if len(p) < 3:
        return p
    ext = [p[0]] + p + [p[-1]]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for j in range(samples):
            t = j / samples
            t2, t3 = t * t, t * t * t
            b0 = (1 - t) ** 3 / 6.0
            b1 = (3 * t3 - 6 * t2 + 4) / 6.0
            b2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0
            b3 = t3 / 6.0
            out.append((p0[0] * b0 + p1[0] * b1 + p2[0] * b2 + p3[0] * b3,
                        p0[1] * b0 + p1[1] * b1 + p2[1] * b2 + p3[1] * b3))
    out.append(p[-1])
    return out


def _resample(points, step=1.8):
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    carry = 0.0
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg = math.hypot(dx, dy)
        if seg < 1e-6:
            continue
        pos = step - carry
        while pos <= seg:
            t = pos / seg
            out.append((a[0] + dx * t, a[1] + dy * t))
            pos += step
        carry = seg - (pos - step)
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def _curvature(points):
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 4:
        return np.zeros(len(p), dtype=np.float32)
    dx = np.gradient(p[:, 0])
    dy = np.gradient(p[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    den = np.power(dx * dx + dy * dy, 1.5)
    k = np.abs(dx * ddy - dy * ddx) / np.maximum(den, 1e-6)
    kp = np.pad(k, 3, mode="edge")
    k = np.convolve(kp, np.ones(7) / 7.0, mode="same")[3:3 + len(k)]
    return k.astype(np.float32)


def optimize(points, pressure, base_speed=40.0, min_speed=18, max_speed=100,
             curvature_slow=0.40, press_gain=0.07):
    """输入中心线、归一化压力，输出平滑点、压力、逐点速度。"""
    if len(points) < 3:
        n = max(2, len(points))
        press = np.asarray(pressure, dtype=np.float32)
        if len(press) != n:
            press = np.full(n, float(press[0]) if len(press) else .8, dtype=np.float32)
        speed = np.full(n, float(base_speed), dtype=np.float32)
        return [tuple(map(float, q)) for q in points], press, speed
    # 用较稀的控制点拟合长曲线，避免把细碎抖动也当成笔形
    stride = max(1, len(points) // 26)
    control = points[::stride]
    if control[-1] != points[-1]:
        control.append(points[-1])
    smooth = _resample(_bspline(control, samples=8), step=1.8)
    if len(smooth) < 2:
        smooth = list(map(tuple, points))
    n = len(smooth)
    old_t = np.linspace(0, 1, len(points))
    new_t = np.linspace(0, 1, n)
    press = np.interp(new_t, old_t, np.asarray(pressure, dtype=np.float32).reshape(-1))
    if press.size != n:
        press = np.resize(press, n)
    k = _curvature(smooth)
    hi = float(np.percentile(k, 92)) if len(k) else 0.0
    kn = np.clip(k / max(hi, 1e-6), 0, 1.0) if hi > 1e-8 else np.zeros(n, dtype=np.float32)
    # 曲率大 -> 慢；压力略增，补偿慢速造成的墨量变化。
    kn = np.resize(kn, n)
    speed = float(base_speed) * (1.0 - float(curvature_slow) * kn)
    speed *= (0.94 + 0.10 * np.clip(press, 0, 1))
    speed = np.clip(speed, min_speed, max_speed)
    press = np.clip(press * (1.0 + float(press_gain) * kn), 0.02, 1.0)
    return smooth, press.astype(np.float32), speed.astype(np.float32)