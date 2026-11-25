"""Recover short, source-supported continuations of detected faint ink."""
import cv2
import numpy as np


def _ridge_evidence(gray):
    width = gray.shape[1]
    columns = np.arange(width)
    # Smooth along the scan direction to reduce isolated printing dots.
    tone = cv2.GaussianBlur(gray, (1, 5), 0.9)
    left = [tone[:, np.clip(columns - offset, 0, width - 1)] for offset in (1, 2, 3)]
    right = [tone[:, np.clip(columns + offset, 0, width - 1)] for offset in (1, 2, 3)]
    contrast = np.minimum(np.maximum(left[1], left[2]), np.maximum(right[1], right[2])) - tone
    ridge = (tone <= left[0] + 2) & (tone <= right[0] + 2) & (contrast >= 5)
    ridge[:, :3] = ridge[:, -3:] = False
    return ridge, contrast


def _scan(gray, roles, limit, allow_tips, min_path_contrast):
    ridge, contrast = _ridge_evidence(gray)
    proposals = np.zeros((*roles.shape, 2), dtype=bool)
    nearby = np.ones((1, 3), np.uint8)
    along = cv2.dilate(ridge.astype(np.uint8), nearby).astype(bool)
    for role in (1, 2):
        anchors = ridge & (roles == role)
        adjacent = cv2.dilate(anchors.astype(np.uint8), nearby).astype(bool)
        for direction in (-1, 1):
            starts = anchors & np.roll(adjacent, direction, axis=0) & ~np.roll(adjacent, -direction, axis=0)
            starts[:2] = starts[-2:] = False
            for row, column in np.argwhere(starts):
                row, column = int(row), int(column)
                path = []
                connected = False
                for _ in range(limit + 1):
                    next_row = row + direction
                    if not 0 < next_row < roles.shape[0] - 1:
                        break
                    neighbors = np.arange(max(0, column-1), min(column+2, roles.shape[1]))
                    choices = neighbors[ridge[next_row, neighbors]]
                    viable = choices[along[next_row+direction, choices]]
                    terminal = allow_tips and min_path_contrast > 0 and len(path) >= 2 and not viable.size
                    if terminal:
                        viable = choices
                    if not viable.size:
                        break
                    next_col = viable[np.argmax(contrast[next_row, viable])]
                    if roles[next_row, next_col] != 0:
                        connected = anchors[next_row, next_col]
                        break
                    if len(path) == limit:
                        break
                    path.append((next_row, next_col))
                    row, column = next_row, next_col
                    if terminal:
                        break
                if connected or (allow_tips and len(path) >= 2):
                    if path:
                        ys, xs = np.asarray(path).T
                        if min_path_contrast <= contrast[ys, xs].mean():
                            proposals[ys, xs, role-1] = True
    return proposals


def recover_faint_ink(image, roles, *, limit=5, allow_tips=False, min_path_contrast=0):
    """Restore ridge centers of short omissions without widening detected ink."""
    if min(roles.shape) < 7:
        return roles.copy(), np.zeros(roles.shape, dtype=bool)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    proposed = _scan(gray, roles, limit, allow_tips, min_path_contrast)
    proposed |= _scan(gray.T, roles.T, limit, allow_tips, min_path_contrast).transpose(1, 0, 2)
    restored = (proposed.sum(axis=-1) == 1) & (roles == 0)
    result = roles.copy()
    result[restored] = proposed[restored].argmax(axis=-1) + 1
    return result, restored
