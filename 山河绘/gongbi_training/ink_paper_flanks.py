"""Trace narrow paper valleys using paper on the outer sides of ink ridges."""
import cv2
import numpy as np

from gongbi_training.ink_paper_source import separate_source_paper, source_paper_support


def separate_flank_paper(image, roles, *, strict=True, evidence=None):
    """Supplement the source-entrance output with locally supported narrow gaps."""
    revised, previous = separate_source_paper(image, roles)
    if min(roles.shape) < 25:
        return revised, previous, np.zeros_like(previous)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Use independently established broad paper as a reference on the outside of
    # the pair. Do not seed its center from the model's background mask.
    paper = cv2.dilate(source_paper_support(image).astype(np.uint8), np.ones((3, 3), np.uint8))
    # Ignore isolated highlights when estimating the surrounding bright tone.
    robust_bright = cv2.dilate(cv2.medianBlur(gray, 5), np.ones((19, 19), np.uint8))
    y, x = np.indices(gray.shape, dtype=np.float32)

    def sample(matrix, dx, dy, interpolation):
        return cv2.remap(matrix, x + dx, y + dy, interpolation, borderMode=cv2.BORDER_REPLICATE)

    added = np.zeros(roles.shape, dtype=bool)
    for theta in np.linspace(0, np.pi, 12, endpoint=False):
        nx, ny = float(np.cos(theta)), float(np.sin(theta))
        tx, ty = -ny, nx
        ridges, shoulders, local_shoulders = [], [], []
        for side in (-1, 1):
            dx, dy = side * nx, side * ny
            ridges.append(np.minimum.reduce([sample(gray, dx*d, dy*d, cv2.INTER_LINEAR) for d in (1, 2, 3, 4)]))
            offsets = (5, 6, 7, 8, 9)
            outer_values = np.stack([sample(gray, dx*d, dy*d, cv2.INTER_LINEAR) for d in offsets])
            outer_paper = np.stack([sample(paper, dx*d, dy*d, cv2.INTER_NEAREST) > 0 for d in offsets])
            outer = np.where(outer_paper, outer_values, -np.inf)
            # A pair of supported exterior samples makes a lone bright pixel
            # insufficient. The second strongest sample is the paper estimate.
            shoulders.append(np.partition(outer, -2, axis=0)[-2])
            local_shoulders.append(np.partition(outer_values, -2, axis=0)[-2])
        local_paper = 0.5 * (shoulders[0] + shoulders[1])
        paired_ridges = gray - np.maximum(*ridges) >= 18
        both_sides = np.isfinite(local_paper)
        if strict:
            allowed = both_sides
            left, right = shoulders
        else:
            left, right = local_shoulders
            local_paper = 0.5 * (left + right)
            # Reject valleys that only look bright inside a broad dark wash,
            # while tolerating a small photograph-induced paper gradient.
            allowed = (gray >= robust_bright - 24) & (local_paper >= robust_bright - 24)
        valleys = (allowed & (local_paper - gray <= 14) & paired_ridges
                   & (left - ridges[0] >= 18) & (right - ridges[1] >= 18))
        current = valleys.astype(np.uint8)
        neighboring = np.maximum.reduce([current,
            sample(current, nx, ny, cv2.INTER_NEAREST), sample(current, -nx, -ny, cv2.INTER_NEAREST)])
        forward = sum(sample(neighboring, tx*d, ty*d, cv2.INTER_NEAREST) for d in (1, 2, 3))
        backward = sum(sample(neighboring, -tx*d, -ty*d, cv2.INTER_NEAREST) for d in (1, 2, 3))
        continuous = valleys & (forward >= 1) & (backward >= 1) & (forward + backward >= 4)
        added |= continuous
        if evidence is not None:
            for name, mask in [('paired_ridges', paired_ridges), ('pair_with_outer_paper', paired_ridges & both_sides),
                               ('local_paper_valley', valleys), ('direction_support', continuous)]:
                evidence[name] = evidence.get(name, np.zeros_like(added)) | mask
    added[:10, :] = added[-10:, :] = False
    added[:, :10] = added[:, -10:] = False
    added &= np.isin(revised, [1, 2])
    revised[added] = 0
    return revised, previous | added, added
