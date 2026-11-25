"""Distinguish dark continuations of paper gaps from dark, joined ink."""
import cv2
import numpy as np

from gongbi_training.ink_paper_persistent_stop import _matching_sections


def _bright_reach(bright, links, direction, steps):
    """Find sections that lead to a brighter part of the same gap."""
    reached = bright
    for _ in range(steps):
        reached = bright | np.logical_or.reduce([
            link & np.roll(reached, (-direction, -shift), axis=(0, 1))
            for shift, link in links.items()
        ])
    return reached


def _scan(gray, paper, seeds, proposed, span):
    columns = np.arange(gray.shape[1])
    offsets = np.arange(1, 4)
    left = gray[:, np.clip(columns[None] - offsets[:, None], 0, columns.size - 1)]
    right = gray[:, np.clip(columns[None] + offsets[:, None], 0, columns.size - 1)]
    left_dark = left <= gray[:, None] - 25
    right_dark = right <= gray[:, None] - 25
    left_edge = left_dark.argmax(axis=1) + 1
    right_edge = right_dark.argmax(axis=1) + 1
    center = (
        left_dark.any(axis=1) & right_dark.any(axis=1)
        & (gray >= left[:, 0]) & (gray >= right[:, 0])
        & (columns - left_edge >= 0) & (columns + right_edge < columns.size)
    )
    bright = center & (gray >= paper - 24)
    anchors = seeds & center
    links = {direction: _matching_sections(center, left_edge, right_edge, direction)
             for direction in (-1, 1)}
    # A darker section can belong to a temporary shadow. A nearby bright part
    # must be reachable on both sides without changing the pair of ink lines.
    bracketed = _bright_reach(bright, links[-1], -1, span) & _bright_reach(bright, links[1], 1, span)
    eligible = proposed & center & bracketed
    accepted = np.zeros_like(proposed)
    for direction in (-1, 1):
        order = range(gray.shape[0]) if direction == 1 else range(gray.shape[0] - 1, -1, -1)
        reached = np.zeros_like(proposed)
        for previous, row in zip(order, order[1:]):
            prior = anchors[previous] | reached[previous]
            incoming = np.logical_or.reduce([
                np.roll(prior & link[previous], shift)
                for shift, link in links[direction].items()
            ])
            reached[row] = eligible[row] & incoming
        accepted |= reached
    return accepted


def protect_ink_joins(image, original_roles, narrow_roles, persistent_roles, *, span=3, check_dark_edge=False):
    """Keep pale, connected gap proposals and dark proposals bracketed by paper."""
    proposed = (narrow_roles > 0) & (persistent_roles == 0)
    if not proposed.any():
        return persistent_roles.copy(), proposed, proposed.copy()
    seeds = np.isin(original_roles, [1, 2]) & (narrow_roles == 0)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    paper = cv2.dilate(cv2.medianBlur(gray, 5), np.ones((19, 19), np.uint8))
    candidates = proposed
    if check_dark_edge:
        nearby_light = cv2.dilate(gray, np.ones((3, 3), np.uint8))
        dark_edge = (gray < paper - 24) & (gray < nearby_light - 44)
        candidates = proposed & ~dark_edge
    accepted = _scan(gray, paper, seeds, candidates, span)
    accepted |= _scan(gray.T, paper.T, seeds.T, candidates.T, span).T
    rejected = proposed & ~accepted
    output = persistent_roles.copy()
    output[rejected] = narrow_roles[rejected]
    return output, accepted, rejected
