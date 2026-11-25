"""Limit narrow continuation to a short run of existing candidates."""
import cv2
import numpy as np

from gongbi_training.faint_ink_narrow import bridge_faint_ink_narrow


def supported_proposals(roles, proposals, *, minimum_span):
    """Keep candidate components spanning the required number of image rows or columns."""
    if minimum_span < 1:
        raise ValueError('minimum_span must be positive')
    supported = roles.copy()
    for role in (1, 2):
        missing = (roles == 0) & (proposals == role)
        _, component, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), connectivity=8)
        span = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        qualify = span >= minimum_span
        qualify[0] = False
        supported[qualify[component]] = role
    return supported


def bridge_faint_ink_narrow_supported(image, roles, proposals, *, allow_new=False, limit=6, minimum_span=3):
    """Bridge narrow gaps only where incoming proposals contain a coherent run."""
    supported = supported_proposals(roles,proposals,minimum_span=minimum_span)
    return bridge_faint_ink_narrow(image,roles,supported,allow_new=allow_new,limit=limit)
