"""Filter faint-ink continuations by source direction and anchor connectivity."""
import cv2
import numpy as np

from gongbi_training.faint_ink_continuation import _scan


def _direction_field(gray):
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.75)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    xx = cv2.GaussianBlur(gx * gx, (7, 7), 1.75)
    yy = cv2.GaussianBlur(gy * gy, (7, 7), 1.75)
    xy = cv2.GaussianBlur(gx * gy, (7, 7), 1.75)
    total = xx + yy + 1e-6
    coherence = np.sqrt((xx - yy)**2 + 4 * xy**2) / total
    return coherence, xx / total, yy / total


def _anchored_candidates(candidates, roles, limit):
    result = np.zeros_like(candidates)
    along = np.zeros((3, 3), dtype=np.uint8)
    along[0] = along[2] = 1
    for role in (1, 2):
        seed = roles == role
        eligible = candidates[..., role-1]
        reached = seed
        for _ in range(limit):
            adjacent = cv2.dilate(reached.astype(np.uint8), along, borderType=cv2.BORDER_CONSTANT).astype(bool)
            reached = seed | (eligible & adjacent)
        result[..., role-1] = reached & eligible
    return result


def filter_faint_ink_direction(image, roles, proposals, *, min_coherence=0.45, aligned=True):
    """Retain source-supported proposals without disconnecting them from ink."""
    additions = (roles == 0) & np.isin(proposals, [1, 2])
    if not additions.any():
        return roles.copy(), additions, additions.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    coherence, across_vertical, across_horizontal = _direction_field(gray)
    coherent = coherence >= min_coherence
    if aligned:
        allow = np.stack([additions & (proposals == role) for role in (1, 2)], axis=-1)
        vertical = _scan(gray, roles, 6, True, 12)
        horizontal = _scan(gray.T, roles.T, 6, True, 12)
        v = vertical & allow & (coherent & (across_vertical >= 0.48))[..., None]
        h = horizontal & allow.transpose(1, 0, 2) & (coherent & (across_horizontal >= 0.48)).T[..., None]
        accepted = _anchored_candidates(v, roles, 6).any(axis=-1)
        accepted |= _anchored_candidates(h, roles.T, 6).any(axis=-1).T
    else:
        accepted = additions & coherent
    result = roles.copy()
    result[accepted] = proposals[accepted]
    return result, accepted, additions & ~accepted
