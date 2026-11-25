"""Reject candidate seams that are bright only inside a darker ink area."""
import cv2
import numpy as np

from gongbi_training.ink_paper_seams import separate_visible_paper


def separate_contextual_paper(image, roles):
    """Split only seams whose brightness also agrees with a broader neighborhood.

    Returns the revised roles, the removed pixels, and the proposed but
    withheld pixels so callers can review the source evidence separately.
    """
    _, proposed = separate_visible_paper(image, roles)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Use the same six-level paper tolerance as the original detector. This
    # longer radius checks for wash that encloses both dark flanks.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    paper_level = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    enough_evidence = (paper_level - gray) <= 6.0
    removed = proposed & enough_evidence
    result = roles.copy()
    result[removed] = 0
    return result, removed, proposed & ~enough_evidence
