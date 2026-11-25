"""Limit additional paper removal to the bright center of narrow valleys."""
import cv2
import numpy as np

from gongbi_training.ink_paper_flanks import separate_flank_paper


def separate_narrow_paper(image, roles, *, bridge=False):
    """Keep the prior paper entrance and constrain the additional flank edits."""
    flank_roles, _, extra = separate_flank_paper(image, roles, strict=False)
    if not extra.any():
        return flank_roles, extra, np.zeros_like(extra)

    original_flank_ink = flank_roles.copy()
    original_flank_ink[extra] = roles[extra]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bright_reference = cv2.dilate(cv2.medianBlur(gray, 5), np.ones((19, 19), np.uint8))
    y, x = np.indices(gray.shape, dtype=np.float32)

    def sample(matrix, dx, dy, interpolation):
        return cv2.remap(matrix, x + dx, y + dy, interpolation, borderMode=cv2.BORDER_REPLICATE)

    bounded, joined = np.zeros_like(extra), np.zeros_like(extra)
    for angle in np.linspace(0, np.pi, 12, endpoint=False):
        nx, ny = float(np.cos(angle)), float(np.sin(angle))
        normal = [
            [sample(gray, sign * d * nx, sign * d * ny, cv2.INTER_LINEAR)
             for d in (1.0, 1.5, 2.0, 2.5)]
            for sign in (-1, 1)
        ]
        left, right = normal
        # Retain a center that is bright on both sides of the local cross
        # section, with a darker flank at most 2.5 pixels away on each side.
        centered = (gray >= left[0] - 1) & (gray >= right[0] - 1)
        narrow = gray - np.maximum(np.minimum.reduce(left), np.minimum.reduce(right)) >= 18
        supported = centered & narrow & (gray >= bright_reference - 24)
        seeds = supported & extra
        bounded |= seeds
        if bridge:
            tx, ty = -ny, nx
            seeds = seeds.astype(np.uint8)
            before = sample(seeds, -tx, -ty, cv2.INTER_NEAREST) > 0
            after = sample(seeds, tx, ty, cv2.INTER_NEAREST) > 0
            # One missing sample between two retained centers can be joined
            # only if it has its own opposing-ink and paper-brightness evidence.
            joined |= supported & before & after

    joined &= np.isin(original_flank_ink, [1, 2]) & ~bounded
    original_flank_ink[bounded | joined] = 0
    return original_flank_ink, bounded, joined
