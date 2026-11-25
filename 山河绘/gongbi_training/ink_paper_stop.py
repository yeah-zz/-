"""Stop one-sided paper-gap edits when the source no longer shows two rails."""
import cv2
import numpy as np

from gongbi_training.ink_paper_extension import extend_narrow_paper


def _guard_scan(gray, seeds, proposed, lookahead):
    """Accept only proposed pixels reachable along a corroborated scan direction."""
    columns = np.arange(gray.shape[1])
    offsets = np.arange(1, 4)
    left = gray[:, np.clip(columns[None] - offsets[:, None], 0, columns.size - 1)]
    right = gray[:, np.clip(columns[None] + offsets[:, None], 0, columns.size - 1)]
    left_dark = left <= gray[:, None] - 25
    right_dark = right <= gray[:, None] - 25
    left_edge = left_dark.argmax(axis=1) + 1
    right_edge = right_dark.argmax(axis=1) + 1
    contrast = gray - np.maximum(left.min(axis=1), right.min(axis=1))
    center = (
        left_dark.any(axis=1) & right_dark.any(axis=1)
        & (gray >= left[:, 0]) & (gray >= right[:, 0])
        & (columns - left_edge >= 0) & (columns + right_edge < columns.size)
    )
    seeds = seeds & center
    proposed = proposed & center
    accepted = np.zeros_like(proposed)

    for direction in (-1, 1):
        # At a merge or fading end, a single section may still look like a
        # valley. Require a plausible pair in the next sections as well.
        links = []
        for shift in (-1, 0, 1):
            def forward(array):
                return np.roll(array, (-direction, -shift), axis=(0, 1))

            same_pair = (
                center & forward(center)
                & (np.abs(forward(left_edge) - left_edge) <= 1)
                & (np.abs(forward(right_edge) - right_edge) <= 1)
                & (forward(contrast) >= 0.60 * contrast)
                & (np.abs(forward(gray) - gray) <= 32)
            )
            same_pair[-1 if direction == 1 else 0] = False
            if shift:
                same_pair[:, -1 if shift == 1 else 0] = False
            links.append((shift, same_pair))
        supported = center
        for _ in range(lookahead):
            supported = np.logical_or.reduce([
                linked & np.roll(supported, (-direction, -shift), axis=(0, 1))
                for shift, linked in links
            ])
        eligible = proposed & supported
        # Grow only through eligible proposals. A later plausible pixel cannot
        # restart an extension beyond a rejected section.
        rows = range(gray.shape[0]) if direction == 1 else range(gray.shape[0] - 1, -1, -1)
        reached = np.zeros_like(proposed)
        for previous, row in zip(rows, rows[1:]):
            previous_active = seeds[previous] | reached[previous]
            next_active = np.logical_or.reduce([
                np.roll(previous_active & linked[previous], shift)
                for shift, linked in links
            ])
            reached[row] = eligible[row] & next_active
        accepted |= reached
    return accepted


def guarded_paper_extension(image, original_roles, narrow_roles, *, lookahead=2):
    """Filter the frozen extension using forward evidence on the source image."""
    _, proposed = extend_narrow_paper(image, original_roles, narrow_roles)
    out = narrow_roles.copy()
    if not proposed.any():
        return out, proposed, proposed.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    seeds = np.isin(original_roles, [1, 2]) & (narrow_roles == 0)
    accepted = _guard_scan(gray, seeds, proposed, lookahead)
    accepted |= _guard_scan(gray.T, seeds.T, proposed.T, lookahead).T
    out[accepted] = 0
    return out, accepted, proposed & ~accepted
