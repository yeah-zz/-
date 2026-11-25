"""Offline detail candidates for weak source ink and tree/rock disagreement."""
from dataclasses import replace

import cv2
import numpy as np

from source_ink import InkConfig, extract


def supplemental_ink(bgr, baseline):
    """Try a weaker seed pass, retaining coherent paths supported by image contrast."""
    weaker = extract(bgr, replace(InkConfig(), strong_contrast=16, weak_contrast=9))
    distance = cv2.distanceTransform((baseline['raster'] == 0).astype(np.uint8),
                                     cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    additions = []
    for row in weaker['paths']:
        if row['length_px'] < 12 or row['closed']:
            continue
        p = np.asarray(row['points'])
        arcs = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        stations = np.linspace(0, arcs[-1], int(arcs[-1]) + 1)
        x, y = np.rint([np.interp(stations, arcs, p[:, i]) for i in range(2)]).astype(int)
        if np.mean(distance[y, x] > 1.5) < .55 or np.mean(weaker['contrast'][y, x] >= 9) < .80:
            continue
        # Defer more circuitous paths for later review.
        if np.linalg.norm(p[-1] - p[0]) / row['length_px'] >= .65:
            additions.append(row)
    return weaker, additions


def guarded_tree_hints(full, local128, local224):
    """Keep the frozen map, and flag uncertain or locally supported tree candidates."""
    rock, tree = 1, 2
    full_class = full.argmax(0)
    locals_agree = (local128.argmax(0) == tree) & (local224.argmax(0) == tree)
    smallest_support = np.minimum(local128[tree], local224[tree])
    candidate = (full_class == rock) & locals_agree & (smallest_support >= .60)
    # Agreement can still be a false positive at a rock face. Keep a separate
    # confidence gate and expose the remaining candidates as an uncertainty layer.
    accepted = candidate & (full[rock] - full[tree] <= .35)
    return candidate, accepted
