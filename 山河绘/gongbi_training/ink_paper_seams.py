"""Propose paper seams between close ink bands on the original pixel grid."""
import cv2
import numpy as np


def separate_visible_paper(image, roles):
    """Return a copy of roles and the pixels removed for visible paper evidence.

    The directional evidence uses only the image. The mask limits edits to
    existing foreground. Darkness and continuity are checked at every edited
    pixel, so support elsewhere cannot cut across a dark junction.
    """
    if image.ndim != 3 or image.shape[2] != 3 or roles.shape != image.shape[:2]:
        raise ValueError('Expected aligned native RGB image and 2D role raster')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if min(gray.shape) < 25:
        return roles.copy(), np.zeros_like(roles, dtype=bool)
    y, x = np.indices(gray.shape, dtype=np.float32)

    def shift(matrix, dx, dy, interpolation=cv2.INTER_LINEAR):
        return cv2.remap(matrix, x + dx, y + dy, interpolation, borderMode=cv2.BORDER_REPLICATE)

    # Keep candidates close to the local paper level. Check both nearer and
    # broader neighborhoods to protect faded ink inside narrow white channels.
    bright = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    near_paper = (bright - gray) <= 6.0
    candidate = np.zeros(gray.shape, dtype=bool)
    for angle in np.linspace(0, np.pi, 12, endpoint=False):
        nx, ny = float(np.cos(angle)), float(np.sin(angle))
        tx, ty = -ny, nx
        left = np.minimum.reduce([shift(gray, -nx * k, -ny * k) for k in (2, 3, 4)])
        right = np.minimum.reduce([shift(gray, nx * k, ny * k) for k in (2, 3, 4)])
        corridor = near_paper & ((gray - np.maximum(left, right)) >= 12.0)
        supported = corridor.astype(np.uint8)
        # Require several nearby cross-sections on both sides of the candidate.
        # A one-pixel normal tolerance accommodates rasterized oblique lines.
        close_support = np.maximum.reduce([
            supported,
            shift(supported, nx, ny, cv2.INTER_NEAREST),
            shift(supported, -nx, -ny, cv2.INTER_NEAREST),
        ])
        left_votes = sum(shift(close_support, -tx * k, -ty * k, cv2.INTER_NEAREST) for k in (1, 2, 3, 4))
        right_votes = sum(shift(close_support, tx * k, ty * k, cv2.INTER_NEAREST) for k in (1, 2, 3, 4))
        candidate |= corridor & (left_votes >= 3) & (right_votes >= 3)
    candidate[:8, :] = candidate[-8:, :] = False
    candidate[:, :8] = candidate[:, -8:] = False
    removed = candidate & np.isin(roles, [1, 2])
    result = roles.copy()
    result[removed] = 0
    return result, removed
