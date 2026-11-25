"""Verify short bridges using original ink beyond their detected endpoints."""
import cv2
import numpy as np

from gongbi_training.faint_ink_anchor_guard import _neighbors
from gongbi_training.faint_ink_bridge_sections import _cross_section
from gongbi_training.faint_ink_direction_filter import _direction_field


def _ink_contrast(gray):
    padded = np.pad(gray, ((0, 0), (3, 3)), mode='edge')
    cols = gray.shape[1]
    left = np.maximum(padded[:, :cols], padded[:, 1:cols+1])
    right = np.maximum(padded[:, 5:cols+5], padded[:, 6:cols+6])
    return np.minimum(left, right) - gray


def _outward_support(source, direction):
    # Curved ends may move sideways two pixels per cross-section.
    observed = cv2.dilate(source.astype(np.uint8), np.ones((1, 5), np.uint8)).astype(bool)
    shifted = np.roll(observed, direction, axis=0)
    shifted[0 if direction == 1 else -1] = False
    return shifted


def _scan_endpoints(anchors, source, gap, direction, limit):
    supported = source & _outward_support(source, direction)
    inner = anchors & _outward_support(supported, direction)
    pair = anchors & _neighbors(inner, direction)
    step = gap & _neighbors(pair, direction)
    reached = [step]
    for _ in range(1, limit):
        step = gap & _neighbors(step, direction)
        reached.append(step)
    return reached


def _scan(gray, roles, proposals, coherence, orientation, texture, limit):
    anchor_support, source = _cross_section(gray)
    gap_support = source & (coherence >= .55) & (orientation >= .48)
    gap_support &= _ink_contrast(gray) >= 3 * texture
    keep = np.zeros_like(roles, dtype=bool)
    for role in (1, 2):
        anchors = anchor_support & (roles == role)
        gap = gap_support & (roles == 0) & (proposals == role)
        front = _scan_endpoints(anchors, source, gap, 1, limit)
        back = np.logical_or.accumulate(_scan_endpoints(anchors, source, gap, -1, limit), axis=0)
        for index, step in enumerate(front):
            keep |= step & back[limit-index-1]
    return keep


def bridge_faint_ink_verified_endpoints(image, roles, proposals, *, limit=6):
    """Keep source-backed paths bounded by sustained ink at both endpoints."""
    if limit < 1:
        raise ValueError('limit must be positive')
    if min(roles.shape) < 7:
        return np.zeros_like(roles, dtype=bool)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    coherence, vertical, horizontal = _direction_field(gray)
    residual = np.abs(gray - cv2.GaussianBlur(gray, (3, 3), .75))
    texture = cv2.blur(cv2.medianBlur(residual, 5), (9, 9))
    return (_scan(gray, roles, proposals, coherence, vertical, texture, limit)
            | _scan(gray.T, roles.T, proposals.T, coherence.T, horizontal.T, texture.T, limit).T)
