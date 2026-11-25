"""Recover short ink omissions with source-supported ends in both directions."""
import cv2
import numpy as np

from gongbi_training.faint_ink_anchor_guard import _neighbors
from gongbi_training.faint_ink_continuation import _ridge_evidence


def _ink_ridge(gray):
    ridge, _ = _ridge_evidence(gray)
    columns = np.arange(gray.shape[1])
    flank_left = np.maximum(gray[:, np.clip(columns-2, 0, None)], gray[:, np.clip(columns-3, 0, None)])
    flank_right = np.maximum(gray[:, np.clip(columns+2, None, gray.shape[1]-1)],
                             gray[:, np.clip(columns+3, None, gray.shape[1]-1)])
    return ridge & (np.minimum(flank_left, flank_right)-gray >= 5)


def _reach(anchors, gap, direction, limit):
    stable = anchors & _neighbors(anchors, direction)
    step = _neighbors(stable, direction) & gap
    reached = [step]
    for _ in range(1, limit):
        step = _neighbors(step, direction) & gap
        reached.append(step)
    return reached


def _bounded_paths(gray, roles, proposals, limit):
    ridge = _ink_ridge(gray)
    bridged = np.zeros_like(roles, dtype=bool)
    for role in (1, 2):
        anchors = ridge & (roles == role)
        gap = ridge & (roles == 0) & (proposals == role)
        top = _reach(anchors, gap, 1, limit)
        from_bottom = _reach(anchors, gap, -1, limit)
        # At index i, that many earlier gap pixels have already been counted.
        bottom_reachable = np.logical_or.accumulate(from_bottom, axis=0)
        for i, front in enumerate(top):
            bridged |= front & bottom_reachable[limit-i-1]
    return bridged


def bridge_faint_ink(image, roles, proposals, *, limit=6):
    """Return proposal pixels on a short ridge between two runs of the same role."""
    if limit < 1:
        raise ValueError('limit must be positive')
    if min(roles.shape) < 7:
        return np.zeros_like(roles, dtype=bool)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return (_bounded_paths(gray, roles, proposals, limit)
            | _bounded_paths(gray.T, roles.T, proposals.T, limit).T)
