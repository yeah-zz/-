"""Find broad paper entrances using the source image rather than model labels."""
import cv2
import numpy as np

from gongbi_training.ink_paper_context import separate_contextual_paper


def source_paper_support(image):
    """Return paper-like pixels connected to a broad source-image paper patch."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bright = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    paper = (bright - gray <= 6.0).astype(np.uint8)
    # Pad the crop with non-paper so the image boundary alone cannot create
    # an arbitrarily wide entrance. Keep the previous three-pixel radius.
    distance = cv2.distanceTransform(np.pad(paper, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    count, labels = cv2.connectedComponents(paper, connectivity=4)
    has_entrance = np.zeros(count, dtype=bool)
    has_entrance[labels[distance >= 3]] = True
    has_entrance[0] = False
    return has_entrance[labels]


def separate_source_paper(image, roles):
    """Revise only contextual seam candidates with a source-paper entrance."""
    _, proposed, _ = separate_contextual_paper(image, roles)
    if not proposed.any():
        return roles.copy(), proposed
    removed = proposed & source_paper_support(image)
    result = roles.copy()
    result[removed] = 0
    return result, removed
