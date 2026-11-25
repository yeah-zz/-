"""Extend an already opened paper gap while both flanking ink lines remain visible."""
import cv2
import numpy as np


def _extend_scan(gray, paper, seeds, limit):
    columns = np.arange(gray.shape[1])
    offsets = np.arange(1, 4)
    left = gray[:, np.clip(columns[None] - offsets[:, None], 0, columns.size - 1)]
    right = gray[:, np.clip(columns[None] + offsets[:, None], 0, columns.size - 1)]
    minimum_contrast = np.minimum(gray - 25, paper - 50)
    left_dark = left <= minimum_contrast[:, None]
    right_dark = right <= minimum_contrast[:, None]
    left_edge = left_dark.argmax(axis=1)
    right_edge = right_dark.argmax(axis=1)
    left_tone, right_tone = left.min(axis=1), right.min(axis=1)
    candidate = (
        left_dark.any(axis=1) & right_dark.any(axis=1)
        & (gray >= paper - 44)
        & (gray >= left[:, 0]) & (gray >= right[:, 0])
        & (columns - left_edge - 1 >= 0) & (columns + right_edge + 1 < columns.size)
    )
    anchors = seeds & candidate
    edits = np.zeros_like(anchors)
    around = np.ones((1, 3), np.uint8)
    expanded = cv2.dilate(anchors.astype(np.uint8), around).astype(bool)

    for direction in (-1, 1):
        # A single stray open pixel cannot start an extension. Require an
        # existing run of at least three neighboring bright centers.
        pairs = anchors & np.roll(expanded, direction, axis=0)
        pairs[:2] = pairs[-2:] = False
        expanded_pairs = cv2.dilate(pairs.astype(np.uint8), around).astype(bool)
        ends = anchors & np.roll(expanded_pairs, direction, axis=0) & ~np.roll(expanded, -direction, axis=0)
        ends[:3] = ends[-3:] = False

        for row, column in np.argwhere(ends):
            row, column = int(row), int(column)
            for _ in range(limit):
                next_row = row + direction
                if not 0 <= next_row < gray.shape[0]:
                    break
                following = np.arange(max(column - 1, 0), min(column + 2, columns.size))
                following = following[
                    candidate[next_row, following]
                    & (np.abs(left_edge[next_row, following] - left_edge[row, column]) <= 1)
                    & (np.abs(right_edge[next_row, following] - right_edge[row, column]) <= 1)
                    & (np.abs(left_tone[next_row, following] - left_tone[row, column]) <= 60)
                    & (np.abs(right_tone[next_row, following] - right_tone[row, column]) <= 60)
                ]
                if not following.size or following[-1] - following[0] > 1:
                    break
                next_col = following[np.argmax(gray[next_row, following])]
                if seeds[next_row, next_col]:
                    break
                edits[next_row, next_col] = True
                row, column = next_row, next_col
    return edits


def extend_narrow_paper(image, original_roles, narrow_roles, *, limit=6):
    """Return one-sided extensions starting from existing opened gap tips."""
    seeds = np.isin(original_roles, [1, 2]) & (narrow_roles == 0)
    if min(seeds.shape) < 7 or not seeds.any():
        return narrow_roles.copy(), np.zeros_like(seeds)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    paper = cv2.dilate(cv2.medianBlur(gray, 5), np.ones((19, 19), np.uint8))
    edits = _extend_scan(gray, paper, seeds, limit)
    edits |= _extend_scan(gray.T, paper.T, seeds.T, limit).T
    edits &= np.isin(narrow_roles, [1, 2])
    output = narrow_roles.copy()
    output[edits] = 0
    return output, edits
