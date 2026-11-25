"""V6 source-guided mountain faces. Interior paths follow the image skeleton;
no horizontal hatching or procedural stripe patterns are synthesized.
Pressure and contact widths are still uncalibrated experimental settings.
"""
from copy import deepcopy
import cv2
import numpy as np
from config import ShuimoConfig
from ink_control import refresh_stats
from mountain import resample, length, runs, join_junctions
import vision
from shuimo import press_to_width_mm


def simplify_contour(points, scale):
    # RDP retains the global summit by simplifying either side separately.
    p = np.asarray(points, float)
    peak = int(np.argmin(p[:, 1]))
    pieces = []
    for part in (p[:peak+1], p[peak:]):
        if len(part) > 1:
            pieces.append(cv2.approxPolyDP(part.astype(np.float32).reshape(-1, 1, 2),
                                         .32 / scale, False).reshape(-1, 2))
    if not pieces: return p
    return resample(np.vstack([part if i == 0 else part[1:] for i, part in enumerate(pieces)]), scale)


def make_stroke(template, points, scale, cap, role, group, depth, layer):
    s = deepcopy(template)
    p = resample(points, scale)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] * scale
    # No initial pressure step; release progressively within the drawn line.
    end = 3.5 if role != "silhouette" else 2.5
    envelope = np.minimum(np.clip(arc / 1.8, 0, 1), np.clip((arc[-1]-arc) / end, 0, 1))
    # Long texture fades gradually, while outline retains its body.
    if role != "silhouette": envelope *= 1.0 - .3 * arc / max(arc[-1], 1e-6)
    cap = min(cap, float(ShuimoConfig.PRESS_MAX_MM))
    s.update(points=p.tolist(), press=(cap * envelope).tolist(),
             speed=[float(ShuimoConfig.SPEED_BASE)] * len(p), layer=layer,
             role=role, mountain_group=group, depth_band=depth, tail_mm=0.,
             design_origin="source-envelope" if role == "silhouette" else "source-skeleton",
             name={"silhouette": "山体轮廓", "face_ridge": "山面长脊", "face_texture": "山面短皴"}[role])
    return refresh_stats(s, scale)


def restructure(base, stats, work, scale, pad=None):
    shape = work.shape[:2]
    outlines = [s for s in base if s["role"] == "silhouette"]
    if not outlines:
        return [], dict(stats, detail="structured")
    # Depth is an explicit composition heuristic based on ridge position,
    # not a claim that image coordinates contain measured 3D depth.
    heights = [min(p[1] for p in s["points"]) for s in outlines]
    lo, hi = min(heights), max(heights)
    result, contours = [], []
    occupied = np.zeros(shape, np.uint8)

    def reserve(s):
        width = max(press_to_width_mm(p) for p in s["press"]) + 1.1
        cv2.polylines(occupied, [np.rint(s["points"]).astype(np.int32)], False,
                      255, max(1, int(np.ceil(width / scale))))

    for group, source in enumerate(outlines):
        p = simplify_contour(source["points"], scale)
        relative = (p[:, 1].min()-lo) / max(hi-lo, 1)
        depth = "far" if relative < .30 else "near" if relative > .68 else "middle"
        shallow = np.ptp(p[:, 1]) * scale < 5
        if shallow: depth = "far"
        cap = {"far": .12, "middle": .18, "near": .24}[depth]
        # --press-min still scales this experimental profile predictably.
        factor = float(ShuimoConfig.PRESS_MIN_MM) / .15
        s = make_stroke(source, p, scale, cap * factor, "silhouette", group, depth, 0)
        result.append(s)
        reserve(s)
        contours.append(s)

    ink = (cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) < 210).astype(np.uint8)*255
    if pad:
        t, b, l, r = pad
        if t: ink[:t] = 0
        if b: ink[-b:] = 0
        if l: ink[:, :l] = 0
        if r: ink[:, -r:] = 0
    skeleton = vision._thin(ink)
    skeleton[occupied > 0] = 0
    paths = join_junctions(vision._trace_paths(skeleton, epsilon=.12/scale))
    face_counts = {}
    for raw in sorted(paths, key=lambda p: -length(p)):
        path = resample(raw, scale)
        if path[-1, 1] < path[0, 1]: path = path[::-1]
        total = length(path)*scale
        delta = np.diff(path[:, 1])
        # Follow downhill rock contours, never horizontal bars across a face.
        if total < 4.5 or (path[-1, 1]-path[0, 1])*scale < .48*total: continue
        if np.maximum(-delta, 0).sum() > .12*max(np.maximum(delta, 0).sum(), 1): continue
        ij = np.rint(path).astype(int)
        for piece in runs(path, occupied[ij[:, 1], ij[:, 0]] == 0):
            size = length(piece)*scale
            if size < 4.5: continue
            if (piece[-1, 1]-piece[0, 1])*scale < .48*size: continue
            contour = min(contours, key=lambda c: np.linalg.norm(
                np.asarray(c["points"])-piece[0], axis=1).min())
            outline = np.asarray(contour["points"])
            nearest = int(np.argmin(np.linalg.norm(outline-piece[0], axis=1)))
            root_distance = np.linalg.norm(outline[nearest]-piece[0])*scale
            if root_distance > 12 or piece[0, 1] < outline[nearest, 1]-.5/scale: continue
            depth = contour["depth_band"]
            # Budget per physical patch, with fewer marks for distant hills.
            patch = tuple(np.floor(piece[0]*scale/14).astype(int))
            key = (contour["mountain_group"], patch)
            if face_counts.get(key, 0) >= (3 if depth == "far" else 6): continue
            ridge = size >= 10
            cap = {"far": .08, "middle": .12, "near": .16}[depth]
            if not ridge: cap *= .8
            s = make_stroke(contour, piece, scale, cap*factor,
                            "face_ridge" if ridge else "face_texture",
                            contour["mountain_group"], depth, 1 if ridge else 2)
            s["face_root_px"] = outline[nearest].tolist()
            result.append(s)
            face_counts[key] = face_counts.get(key, 0)+1
            reserve(s)
    result.sort(key=lambda s: (s["layer"], s["mountain_group"]))
    summary = dict(stats, detail="structured", strokes=len(result),
                   texture_origin="source skeleton; no synthesized hatching",
                   depth_model="vertical composition heuristic; not measured depth",
                   length_mm=sum(s["length_mm"] for s in result),
                   layers={k: sum(s["layer"] == k for s in result) for k in range(3)},
                   mean_width_mm=float(np.mean([s["width_mm"] for s in result])),
                   max_width_mm=max(press_to_width_mm(s["press_max"]) for s in result),
                   mean_press_mm=float(np.mean([s["press_mean"] for s in result])),
                   max_press_mm=max(s["press_max"] for s in result))
    return result, summary
