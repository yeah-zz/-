"""Require original-image stroke evidence at faint-ink continuation anchors."""
import cv2
import numpy as np

from gongbi_training.faint_ink_continuation import _ridge_evidence, _scan
from gongbi_training.faint_ink_direction_filter import _direction_field


def _neighbors(mask, direction):
    nearby = cv2.dilate(mask.astype(np.uint8), np.ones((1, 3), np.uint8)).astype(bool)
    shifted = np.roll(nearby, direction, axis=0)
    shifted[0 if direction == 1 else -1] = False
    return shifted


def _reach_from_source(candidates, roles, evidence, min_anchor_length):
    accepted = np.zeros_like(candidates)
    for role in (1, 2):
        anchors = evidence & (roles == role)
        eligible = candidates[..., role-1]
        for direction in (-1, 1):
            backed = anchors
            for _ in range(min_anchor_length-1):
                backed = anchors & _neighbors(backed, direction)
            reached = backed
            for _ in range(6):
                reached = backed | (eligible & _neighbors(reached, direction))
            accepted[..., role-1] |= reached & eligible
    return accepted


def guard_faint_ink_anchors(
    image, roles, proposed, *, min_anchor_length=3, min_ink_depth=0,
    texture_weight=0, strong_ridge_floor=None, dark_core_floor=None,
):
    """Keep existing proposals only if connected to a supported anchor run."""
    additions = (roles == 0) & np.isin(proposed, [1, 2])
    if not additions.any():
        return roles.copy(), additions, additions.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    paper = cv2.dilate(cv2.medianBlur(gray, 5), np.ones((19, 19), np.uint8))
    residual = np.abs(gray - cv2.GaussianBlur(gray, (3, 3), .75))
    texture = cv2.blur(cv2.medianBlur(residual, 5), (9, 9))
    source_depth = paper - gray
    depth = source_depth - texture_weight * texture
    coherent, across_vertical, across_horizontal = _direction_field(gray)
    allow = np.stack([additions & (proposed == role) for role in (1, 2)], axis=-1)
    accepted = np.zeros(roles.shape, dtype=bool)
    axes = (
        (gray, roles, allow, across_vertical, depth, source_depth, coherent, False),
        (gray.T, roles.T, allow.transpose(1,0,2), across_horizontal.T, depth.T, source_depth.T, coherent.T, True),
    )
    for tone, label, mask, axis_share, active_depth, raw_depth, consistency, transpose in axes:
        ridges, contrast = _ridge_evidence(tone)
        strong = active_depth >= min_ink_depth
        if strong_ridge_floor is not None:
            strong |= contrast >= strong_ridge_floor
        source_support = ridges & strong
        if dark_core_floor is not None:
            # A wide dark line can have a usable anchor one pixel off its
            # center. The direction and shared-role requirements still apply.
            source_support |= raw_depth >= dark_core_floor
        evidence = source_support & (axis_share >= .48) & (consistency >= .35)
        traced = _scan(tone, label, 6, True, 12) & mask
        kept = _reach_from_source(traced, label, evidence, min_anchor_length).any(axis=-1)
        accepted |= kept.T if transpose else kept
    output = roles.copy()
    output[accepted] = proposed[accepted]
    return output, accepted, additions & ~accepted
