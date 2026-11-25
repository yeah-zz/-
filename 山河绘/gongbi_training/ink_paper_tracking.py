"""Recover short interruptions between already observed paper-gap segments."""
import cv2
import numpy as np


def _scan_between_ridges(gray, paper, anchors, max_gap, nearest):
    width = gray.shape[1]
    columns = np.arange(width)
    offsets = np.arange(1, 5)
    left_samples = gray[:, np.clip(columns[None] - offsets[:, None], 0, width - 1)]
    right_samples = gray[:, np.clip(columns[None] + offsets[:, None], 0, width - 1)]
    if nearest:
        # A deeper trough farther away can belong to a neighboring stroke.
        # Select the first sample with enough contrast to be a ridge.
        ceiling = np.minimum(gray - 18, paper - 45)
        left_dark = left_samples <= ceiling[:, None]
        right_dark = right_samples <= ceiling[:, None]
        left_index = np.where(left_dark.any(axis=1), left_dark.argmax(axis=1), left_samples.argmin(axis=1))
        right_index = np.where(right_dark.any(axis=1), right_dark.argmax(axis=1), right_samples.argmin(axis=1))
    else:
        left_index = left_samples.argmin(axis=1)
        right_index = right_samples.argmin(axis=1)
    left = np.take_along_axis(left_samples, left_index[:, None], axis=1)[:, 0]
    right = np.take_along_axis(right_samples, right_index[:, None], axis=1)[:, 0]
    left_step, right_step = left_index + 1, right_index + 1
    left_x = columns - left_step
    right_x = columns + right_step

    # A track may cross a slightly shaded paper patch. Each crossing still
    # needs a local brightness peak and visible opposing ink ridges.
    candidate = ((gray >= left_samples[:, 0] - 1) & (gray >= right_samples[:, 0] - 1)
                 & (gray >= np.maximum(left, right) + 18) & (gray >= paper - 36)
                 & (left_x >= 0) & (right_x < width) & (right_x - left_x <= 7))
    anchor = anchors & candidate
    unreachable = max_gap + 1

    def distances(rows):
        result = np.full(gray.shape, unreachable, dtype=np.int16)
        for previous, current in zip(rows, rows[1:]):
            result[previous, anchor[previous]] = 0
            for shift in (-1, 0, 1):
                start, stop = max(0, -shift), min(width, width - shift)
                ahead = slice(start + shift, stop + shift)
                behind = slice(start, stop)
                same_ridges = (
                    (np.abs(left_x[current, ahead] - left_x[previous, behind] - shift) <= 1)
                    & (np.abs(right_x[current, ahead] - right_x[previous, behind] - shift) <= 1)
                    & (np.abs(left[current, ahead] - left[previous, behind]) <= 48)
                    & (np.abs(right[current, ahead] - right[previous, behind]) <= 48)
                )
                edge = same_ridges & candidate[current, ahead]
                distance = result[current, ahead]
                np.minimum(distance, np.where(edge, result[previous, behind] + 1, unreachable), out=distance)
        result[anchor] = 0
        return result

    after_anchor = distances(range(gray.shape[0]))
    before_anchor = distances(range(gray.shape[0] - 1, -1, -1))
    return candidate & (after_anchor + before_anchor <= max_gap + 1)


def track_narrow_paper(image, original_roles, narrow_roles, *, max_gap=6, nearest=False):
    """Return the narrowed prediction plus short, double-anchored gap tracks."""
    anchors = np.isin(original_roles, [1, 2]) & (narrow_roles == 0)
    if not anchors.any():
        return narrow_roles.copy(), np.zeros_like(anchors)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    paper = cv2.dilate(cv2.medianBlur(gray, 5), np.ones((19, 19), np.uint8))
    tracked = _scan_between_ridges(gray, paper, anchors, max_gap, nearest)
    tracked |= _scan_between_ridges(gray.T, paper.T, anchors.T, max_gap, nearest).T
    tracked &= np.isin(narrow_roles, [1, 2])
    revised = narrow_roles.copy()
    revised[tracked] = 0
    return revised, tracked
