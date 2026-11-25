"""Test short continuations beside a nearby ink line with a narrower profile."""
import cv2
import numpy as np

from gongbi_training.faint_ink_anchor_guard import _neighbors
from gongbi_training.faint_ink_bridge_endpoints import _ink_contrast, _scan_endpoints
from gongbi_training.faint_ink_bridge_sections import _cross_section
from gongbi_training.faint_ink_color_texture import _printing_pattern, _roughness
from gongbi_training.faint_ink_direction_filter import _direction_field


def _scan(gray, roles, proposed, coherent, alignment, texture, limit, allow_new):
    wide_anchor, wide_center = _cross_section(gray)
    padded = np.pad(gray, ((0, 0), (1, 1)), mode='edge')
    narrow_contrast = np.minimum(padded[:, :-2], padded[:, 2:]) - gray
    narrow = narrow_contrast >= 3
    narrow[:, 0] = narrow[:, -1] = False
    center = wide_center | narrow
    near_narrow = cv2.dilate(narrow.astype(np.uint8), np.ones((1, 3), np.uint8)).astype(bool)
    contrast = np.maximum(_ink_contrast(gray), narrow_contrast)
    valid = center & coherent & (alignment >= .48) & (contrast >= 3 * texture)
    made = np.zeros((*roles.shape, 2), dtype=bool)
    for role in (1, 2):
        anchors = (wide_anchor | near_narrow) & (roles == role)
        proposed_gap = (proposed == role) & (roles == 0)
        nearby = _neighbors(proposed_gap, 1) | _neighbors(proposed_gap, -1)
        extra = allow_new & nearby & (proposed == 0) & narrow
        gap = valid & (roles == 0) & (proposed_gap | extra)
        forward = _scan_endpoints(anchors, center, gap, 1, limit)
        backward = np.logical_or.accumulate(_scan_endpoints(anchors, center, gap, -1, limit), axis=0)
        for step, branch in enumerate(forward):
            made[...,role-1] |= branch & backward[limit-step-1]
    return made


def bridge_faint_ink_narrow(image, roles, proposed, *, allow_new=False, limit=6):
    """Return short two-ended patches supported by a narrow ink profile."""
    if limit < 1:
        raise ValueError('limit must be positive')
    if min(roles.shape) < 7:
        return np.zeros_like(roles)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    coherence, vertical, horizontal = _direction_field(gray)
    residual = gray - cv2.GaussianBlur(gray, (3, 3), .75)
    stable = (coherence >= .55) & ~_printing_pattern(residual)
    texture = _roughness(residual)
    paths = _scan(gray, roles, proposed, stable, vertical, texture, limit, allow_new)
    paths |= _scan(gray.T, roles.T, proposed.T, stable.T, horizontal.T, texture.T, limit, allow_new).transpose(1, 0, 2)
    unique = paths.sum(axis=-1)==1
    return np.where(unique, paths.argmax(axis=-1)+1, 0).astype(roles.dtype)
