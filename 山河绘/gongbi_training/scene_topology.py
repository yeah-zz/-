"""Assign tree and rock ownership to visible ink paths, with an unknown state."""

from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.ndimage import gaussian_filter1d


OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
ROCK, TREE, UNKNOWN = 1, 2, 255
CONTRAST = 0.023
MARGIN = 0.18
STRONG_MARGIN = 0.4
MIN_ALIGNMENT = 0.84
ALIGNMENT_GAP = 0.12
SMOOTH_SIGMA = 4.0


def ink_paths(skeleton):
    """Partition the foreground edges into paths between endpoints and junctions."""
    pixels = set(map(tuple, np.argwhere(skeleton)))
    neighbors = {}
    for y, x in pixels:
        neighbors[y, x] = [
            (y + dy, x + dx) for dy, dx in OFFSETS
            if (y + dy, x + dx) in pixels
            and not (dy and dx and ((y + dy, x) in pixels or (y, x + dx) in pixels))
        ]
    nodes = {point for point, nearby in neighbors.items() if len(nearby) != 2}
    visited = set()
    paths = []
    for start in sorted(nodes) + sorted(pixels - nodes):
        for following in neighbors[start]:
            previous, point = start, following
            path = [start]
            while (previous, point) not in visited:
                visited.update(((previous, point), (point, previous)))
                path.append(point)
                if point in nodes or point == start:
                    break
                a, b = neighbors[point]
                previous, point = point, (b if a == previous else a)
            if len(path) > 1:
                paths.append(np.asarray(path, dtype=np.int32))
    return paths


def continuations(paths):
    """Match only mutual, unambiguous continuations through a junction."""
    incident = defaultdict(list)
    for index, path in enumerate(paths):
        if len(path) < 5 or np.array_equal(path[0], path[-1]):
            continue
        for end, points in enumerate((path, path[::-1])):
            offset = points[min(6, len(points) - 1)] - points[0]
            incident[tuple(points[0])].append(((index, end), offset / np.linalg.norm(offset)))
    matches = {}
    for ends in incident.values():
        options = defaultdict(list)
        for (a, u), (b, v) in combinations(ends, 2):
            score = -float(u @ v)
            options[a].append((score, b))
            options[b].append((score, a))
        preferred = {}
        for end, alternatives in options.items():
            ranked = sorted(alternatives, reverse=True)
            score, partner = ranked[0]
            if score >= MIN_ALIGNMENT and (len(ranked) == 1 or score - ranked[1][0] >= ALIGNMENT_GAP):
                preferred[end] = partner
        matches.update((a, b) for a, b in preferred.items() if preferred.get(b) == a)
    return matches


def infer_ownership(rgb, probabilities, cv2, skeletonize):
    """Fuse probabilities with continuations over a fixed original-image ink support."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    smooth = cv2.GaussianBlur(gray, (0, 0), .7)
    contrast = cv2.GaussianBlur(smooth, (0, 0), 7) - smooth
    skeleton = skeletonize(np.ascontiguousarray(contrast >= CONTRAST))
    paths = ink_paths(skeleton)
    matches = continuations(paths)
    margin = probabilities[TREE] - probabilities[ROCK]
    relevant = probabilities[[ROCK, TREE]].max(axis=0) >= probabilities[[0, 3]].max(axis=0)
    scores = [margin[tuple(path.T)] for path in paths]
    maps = {mode: np.full(skeleton.shape, UNKNOWN, dtype=np.uint8)
            for mode in ("semantic", "path", "junction")}
    assigned = {mode: np.zeros(skeleton.shape, dtype=np.uint8) for mode in ("path", "junction")}
    maps["semantic"][skeleton & relevant & (margin >= MARGIN)] = TREE
    maps["semantic"][skeleton & relevant & (margin <= -MARGIN)] = ROCK
    segments = []
    for index, (path, local) in enumerate(zip(paths, scores)):
        y, x = path.T
        context = []
        for end in (0, 1):
            other = matches.get((index, end))
            extra = np.empty(0)
            if other:
                partner, side = other
                values = scores[partner] if side == 0 else scores[partner][::-1]
                extra = values[1:13]
            context.append(extra)
        a, b = context
        joined = np.concatenate((a[::-1], local, b))
        smoothed = {
            "path": gaussian_filter1d(local, SMOOTH_SIGMA, mode="nearest"),
            "junction": gaussian_filter1d(joined, SMOOTH_SIGMA, mode="nearest")[len(a):len(a)+len(local)]
        }
        for mode, fused in smoothed.items():
            fused = np.where(np.abs(local) >= STRONG_MARGIN, local, fused)
            allowed = relevant[y, x]
            np.bitwise_or.at(assigned[mode], (y, x),
                             np.where(allowed & (fused >= MARGIN), 2,
                                      np.where(allowed & (fused <= -MARGIN), 1, 4)).astype(np.uint8))
        segments.append({"yx": path.tolist(), "links": [matches.get((index, end)) for end in (0, 1)]})
    for mode, votes in assigned.items():
        maps[mode][votes == 1] = ROCK
        maps[mode][votes == 2] = TREE
    return skeleton, maps, segments
