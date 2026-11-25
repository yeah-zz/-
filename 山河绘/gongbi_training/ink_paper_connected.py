"""Limit seam edits to source paper reachable from a broad background area."""
import cv2
import numpy as np

from gongbi_training.ink_paper_context import separate_contextual_paper


def separate_connected_paper(image, roles):
    """Keep interior bright flecks unless they have a route to exterior paper."""
    _, proposed, _ = separate_contextual_paper(image, roles)
    if not proposed.any():
        return roles.copy(), proposed
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    paper_level = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    paper = (paper_level - gray <= 6.0).astype(np.uint8)
    # Seed only from clear, broad background. A tiny hole in the initial mask
    # is not grounds to clear a bright patch contained inside a dry brushmark.
    exterior = (roles == 0).astype(np.uint8)
    seed = (cv2.distanceTransform(exterior, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) >= 3) & (paper > 0)
    count, areas = cv2.connectedComponents(paper, connectivity=4)
    reachable = np.zeros(count, dtype=bool)
    reachable[areas[seed]] = True
    reachable[0] = False
    removed = proposed & reachable[areas]
    result = roles.copy()
    result[removed] = 0
    return result, removed
