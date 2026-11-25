"""Follow faint ink across individual cross-sections of a short omission."""
import cv2
import numpy as np

from gongbi_training.faint_ink_bridge import _reach


def _cross_section(gray):
    # Two to three pixels away avoid judging the flank by the other half of a wide stroke.
    padded = np.pad(gray, ((0, 0), (3, 3)), mode='edge')
    width = gray.shape[1]
    left = np.maximum(padded[:, :width], padded[:, 1:width+1])
    right = np.maximum(padded[:, 5:width+5], padded[:, 6:width+6])
    contrast = np.minimum(left, right) - gray
    supported = contrast >= 5
    supported[:, :3] = supported[:, -3:] = False
    center = supported & (gray <= padded[:, 2:width+2] + 2) & (gray <= padded[:, 4:width+4] + 2)
    cross = np.ones((1, 3), np.uint8)
    near_ridge = cv2.dilate(center.astype(np.uint8), cross).astype(bool)
    center_gray = cv2.erode(np.where(center, gray, 256), cross)
    return supported & near_ridge, supported & (gray - center_gray <= 8)


def _cross_section_paths(gray, roles, proposals, limit):
    anchor_support, gap_support = _cross_section(gray)
    result = np.zeros_like(roles, dtype=bool)
    for role in (1, 2):
        anchor = anchor_support & (roles == role)
        gap = gap_support & (roles == 0) & (proposals == role)
        forward = _reach(anchor, gap, 1, limit)
        backward = np.logical_or.accumulate(_reach(anchor, gap, -1, limit), axis=0)
        for i, step in enumerate(forward):
            result |= step & backward[limit-i-1]
    return result


def bridge_faint_ink_sections(image, roles, proposals, *, limit=6):
    """Accept short, two-ended proposals backed by the original ink cross-sections."""
    if limit < 1:
        raise ValueError('limit must be positive')
    if min(roles.shape) < 7:
        return np.zeros_like(roles, dtype=bool)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return (_cross_section_paths(gray, roles, proposals, limit)
            | _cross_section_paths(gray.T, roles.T, proposals.T, limit).T)
