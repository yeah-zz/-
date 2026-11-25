"""Estimate texture along candidate ink and guard against repeating print patterns."""
import cv2
import numpy as np

from gongbi_training.faint_ink_bridge_endpoints import _ink_contrast, _scan_endpoints
from gongbi_training.faint_ink_bridge_sections import _cross_section
from gongbi_training.faint_ink_direction_filter import _direction_field


def _roughness(residual):
    return cv2.blur(cv2.medianBlur(np.abs(residual), 5), (9, 9))


def _repeat_correlation(residual, x_step, y_step):
    rows, cols = residual.shape
    padded = np.pad(residual, ((0, 2), (0, 2)), mode='edge')
    shifted = padded[y_step:y_step+rows, x_step:x_step+cols]
    numerator = cv2.blur(residual * shifted, (15, 15))
    energy = cv2.blur(residual * residual, (15, 15))
    shifted_energy = cv2.blur(shifted * shifted, (15, 15))
    return numerator / np.sqrt(energy * shifted_energy + 1e-6)


def _printing_pattern(residual):
    alternating = (_repeat_correlation(residual, 1, 0) < -.35)
    alternating &= _repeat_correlation(residual, 0, 1) < -.35
    repeated = (_repeat_correlation(residual, 2, 0) > .10)
    repeated |= _repeat_correlation(residual, 0, 2) > .10
    return alternating & repeated


def _texture_budget(gray, chroma, isotropic):
    # Use the along-stroke estimate in regions whose two flanks have similar
    # pigment. This reduces the contribution from transverse ink transitions.
    along = _roughness(gray - cv2.GaussianBlur(gray, (1, 3), .75))
    cols = gray.shape[1]
    padded = np.pad(chroma, ((0, 0), (3, 3), (0, 0)), mode='edge')
    left = (padded[:, :cols] + padded[:, 1:cols+1]) * .5
    right = (padded[:, 5:cols+5] + padded[:, 6:cols+6]) * .5
    same_pigment = np.sum((left-right)**2, axis=-1) <= 3**2
    return np.where(same_pigment, np.minimum(isotropic, (isotropic + along) * .5), isotropic)


def _scan(gray, roles, proposals, alignment, coherent, texture, limit):
    anchor_support, source = _cross_section(gray)
    supported = source & coherent & (alignment >= .48) & (_ink_contrast(gray) >= 3 * texture)
    kept = np.zeros_like(roles, dtype=bool)
    for role in (1, 2):
        anchors = anchor_support & (roles == role)
        gap = supported & (roles == 0) & (proposals == role)
        forward = _scan_endpoints(anchors, source, gap, 1, limit)
        reverse = np.logical_or.accumulate(_scan_endpoints(anchors, source, gap, -1, limit), axis=0)
        for index, step in enumerate(forward):
            kept |= step & reverse[limit-index-1]
    return kept


def bridge_faint_ink_color_texture(image, roles, proposals, *, limit=6, reject_periodic=True):
    """Verify a bridge with orientation-aware texture and original ink support."""
    if limit < 1:
        raise ValueError('limit must be positive')
    if min(roles.shape) < 7:
        return np.zeros_like(roles, dtype=bool)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(image.astype(np.float32)/255, cv2.COLOR_RGB2LAB)
    chroma = cv2.medianBlur(lab[..., 1:3].copy(), 3)
    coherence, vertical, horizontal = _direction_field(gray)
    residual = gray - cv2.GaussianBlur(gray, (3, 3), .75)
    coherent = coherence >= .55
    if reject_periodic:
        coherent &= ~_printing_pattern(residual)
    isotropic = _roughness(residual)
    texture_v = _texture_budget(gray, chroma, isotropic)
    texture_h = _texture_budget(gray.T, chroma.transpose(1, 0, 2), isotropic.T)
    return (_scan(gray, roles, proposals, vertical, coherent, texture_v, limit)
            | _scan(gray.T, roles.T, proposals.T, horizontal.T, coherent.T, texture_h, limit).T)
