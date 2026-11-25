"""Continue through a one-section shadow when the same paper gap reappears."""
import cv2
import numpy as np

from gongbi_training.ink_paper_extension import extend_narrow_paper


def _matching_sections(center, left_edge, right_edge, direction):
    links = {}
    for shift in (-1, 0, 1):
        step = (-direction, -shift)
        link = (
            center & np.roll(center, step, axis=(0, 1))
            & (np.abs(np.roll(left_edge, step, axis=(0, 1)) - left_edge) <= 1)
            & (np.abs(np.roll(right_edge, step, axis=(0, 1)) - right_edge) <= 1)
        )
        link[-1 if direction == 1 else 0] = False
        if shift:
            link[:, -1 if shift == 1 else 0] = False
        links[shift] = link
    return links


def _confirmed_sections(gray, contrast, geometry, direction):
    # A well-defined weaker rail may lose relative contrast quickly where
    # sampling shifts along the slope. Persistent low contrast still stops.
    contrast_floor = np.minimum(0.60 * contrast, 60)
    brightness_floor = gray - 32
    links = {}
    for shift, current in geometry.items():
        step = (-direction, -shift)
        stable = (
            (np.roll(gray, step, axis=(0, 1)) >= brightness_floor)
            & (np.roll(contrast, step, axis=(0, 1)) >= contrast_floor)
        )
        rebound = np.zeros_like(current)
        for following_shift, following in geometry.items():
            second = (-2 * direction, -shift - following_shift)
            rebound |= (
                np.roll(following, step, axis=(0, 1))
                & (np.roll(gray, second, axis=(0, 1)) >= brightness_floor)
                & (np.roll(contrast, second, axis=(0, 1)) >= contrast_floor)
            )
        links[shift] = current & (stable | rebound)
    return links


def _scan(gray, seeds, proposed):
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
    anchors = seeds & center
    kept = np.zeros_like(proposed)
    for direction in (-1, 1):
        geometric_links = _matching_sections(center, left_edge, right_edge, direction)
        links = _confirmed_sections(gray, contrast, geometric_links, direction)
        eligible = center
        for _ in range(2):
            eligible = np.logical_or.reduce([
                link & np.roll(eligible, (-direction, -shift), axis=(0, 1))
                for shift, link in links.items()
            ])
        order = range(gray.shape[0]) if direction == 1 else range(gray.shape[0] - 1, -1, -1)
        reached = np.zeros_like(proposed)
        for previous, row in zip(order, order[1:]):
            prior = anchors[previous] | reached[previous]
            next_active = np.logical_or.reduce([
                np.roll(prior & link[previous], shift)
                for shift, link in links.items()
            ])
            reached[row] = proposed[row] & eligible[row] & next_active
        kept |= reached
    return kept


def persistent_paper_extension(image, original_roles, narrow_roles):
    """Keep only extension proposals supported by a continuing pair of ink lines."""
    _, proposed = extend_narrow_paper(image, original_roles, narrow_roles)
    out = narrow_roles.copy()
    if not proposed.any():
        return out, proposed, proposed.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    seeds = np.isin(original_roles, [1, 2]) & (narrow_roles == 0)
    accepted = _scan(gray, seeds, proposed)
    accepted |= _scan(gray.T, seeds.T, proposed.T).T
    out[accepted] = 0
    return out, accepted, proposed & ~accepted
