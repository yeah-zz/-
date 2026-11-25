"""Locate and follow a woody limb from a junction within a horizontal branch crop."""
import cv2
import numpy as np
from scipy.ndimage import label
from skimage.morphology import skeletonize

from gongbi_training.automatic_ink_routing import _overlap
from gongbi_training.structural_ink_routing import route_owned_ink


def woody_support(rgb, ink, owner):
    """Use source color to restrict routing in a colored tree crop."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    lightness, red, yellow = np.moveaxis(lab, -1, 0)
    return (ink > 0) & (owner > 0) & (lightness < 78) & (red > -1) & (yellow > 2)


def attachment_candidates(support, *, trunk_side, radius=7):
    """Find distinct skeleton arms toward the crown, trunk base, and branch."""
    if trunk_side not in ('left', 'right'):
        raise ValueError('trunk_side must be left or right')
    skeleton = skeletonize(np.ascontiguousarray(support, dtype=bool))
    padded = np.pad(skeleton, radius)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    distance_sq = xx**2 + yy**2
    disk = distance_sq <= radius**2
    ring = disk & (distance_sq >= (radius - 2)**2)
    neighbors = np.ones((3, 3))
    inward = 1 if trunk_side == 'left' else -1
    width = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3)
    outer_band = min(support.shape[1] / 3, 4 * radius)
    candidates = []
    for y, x in np.argwhere(skeleton):
        edge_distance = x if inward == 1 else skeleton.shape[1] - 1 - x
        if edge_distance >= outer_band:
            continue
        side = 2 * radius + 1
        patch = padded[y:y + side, x:x + side]
        components, _ = label(patch & disk, neighbors)
        reach = ring & (components == components[radius, radius])
        arcs, count = label(reach, neighbors)
        arms = [np.argwhere(arcs == i).mean(axis=0) - radius for i in range(1, count + 1)
                if np.count_nonzero(arcs == i) >= 2]
        crown = [(-dy, i) for i, (dy, dx) in enumerate(arms) if dy < -.4 * radius]
        base = [(dy, i) for i, (dy, dx) in enumerate(arms) if dy > .4 * radius]
        limb = [(inward * dx, i) for i, (dy, dx) in enumerate(arms)
                if inward * dx > .45 * radius and abs(dy) < .4 * radius]
        strengths = [a + b + c for a,i in crown for b,j in base for c,k in limb
                     if len({i, j, k}) == 3]
        if strengths:
            score = max(strengths) / radius + .28 * float(width[y, x])
            candidates.append((score, (int(x), int(y))))
    candidates.sort(key=lambda item: item[0], reverse=True)
    separate = []
    for score, point in candidates:
        if all(np.linalg.norm(np.subtract(point, prior)) >= radius for _, prior in separate):
            separate.append((score, point))
    return separate


def trace_branch_attachment(image, ink, owner, *, crop, hints=None):
    """Find a trunk-side junction and trace the supported limb within the crop."""
    x0, y0, x1, y1 = crop
    padding = 7
    l, t = max(0, x0 - padding), max(0, y0 - padding)
    r, b = min(ink.shape[1], x1 + padding), min(ink.shape[0], y1 + padding)
    region = np.s_[t:b, l:r]
    local_image = image[region]
    support = woody_support(local_image, ink[region], owner[region])
    matches = sorted(((score, side, (x, y)) for side in ('left', 'right')
                      for score, (x, y) in attachment_candidates(support, trunk_side=side)
                      if y0 <= y + t < y1 and x0 <= x + l < x1), reverse=True)
    rows, cols = np.ogrid[t:b, l:r]
    inset = max(2, (x1 - x0) // 10)
    local_crop = (x0 - l, y0 - t, x1 - l, y1 - t)
    routes = []
    for score, side, (x, y) in matches:
        distal_center = x1 - 1 - inset if side == 'left' else x0 + inset
        distal = (abs(cols - distal_center) <= max(2, inset - 1)) & (y0 <= rows) & (rows < y1)
        joint = (cols - l - x)**2 + (rows - t - y)**2 <= 9
        start, end = (joint, distal) if side == 'left' else (distal, joint)
        route = route_owned_ink(local_image, support, owner[region], crop=local_crop,
                                start_zone=start, end_zone=end,
                                hints=hints[region] if hints is not None else None)
        for candidate in route['routes']:
            points = candidate['points'] + [l, t]
            near, far = (points[0], points[-1]) if side == 'left' else (points[-1], points[0])
            endpoint_cost = .025 * np.sum((near - [x + l, y + t])**2)
            endpoint_cost += .035 * (far[0] - distal_center)**2
            routes.append(dict(points=points, cost=candidate['cost'] + endpoint_cost + max(0., 3.5 - score) * .6,
                               junction=(x + l, y + t), trunk_side=side))
    routes.sort(key=lambda route:route['cost'])
    junctions = [dict(score=score, junction=(x + l, y + t), trunk_side=side) for score, side, (x, y) in matches]
    if not routes:
        return dict(points=np.empty((0, 2), dtype=np.int32), status='no_supported_branch',
                    alternatives=[], junctions=junctions)
    best = routes[0]
    alternatives = [best]
    for route in routes[1:]:
        if route['cost'] > 1.12 * best['cost']:
            break
        if _overlap((best['points'] - [l, t])[:, ::-1], (route['points'] - [l, t])[:, ::-1], support.shape) < .70:
            alternatives.append(route)
            break
    return dict(best, status='ambiguous' if len(alternatives) > 1 else 'route_found',
                alternatives=alternatives, junctions=junctions)
