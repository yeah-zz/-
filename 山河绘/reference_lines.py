"""Reference-derived contours; paper tone and broad washes are not ink fills."""
import cv2
import numpy as np
import watercolor


def normalize_paper(work):
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    eligible = (hsv[:, :, 1] < 65) & (gray > 170)
    # Exclude synthetic pure-white padding from estimating the actual paper.
    eligible &= np.min(work, axis=2) < 253
    if np.count_nonzero(eligible) < 100:
        return work.copy()
    threshold = np.percentile(gray[eligible], 80)
    samples = work[eligible & (gray >= threshold)]
    paper = np.median(samples, axis=0).astype(np.float32)
    return np.clip(work.astype(np.float32) * (255. / np.maximum(paper, 1)), 0, 255).astype(np.uint8)


def extract(work, scale, pad=None):
    normalized = normalize_paper(work)
    smooth = cv2.bilateralFilter(normalized, 5, 18, 3)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    radius = max(3, int(round(.9 / scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,)*2)
    local_dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # Very dark source ink, plus locally darker thin lines. A flat gray wash
    # never becomes a large solid region just because it is below gray=238.
    dark = ((gray < 105) & (hsv[:, :, 1] < 130)).astype(np.uint8)
    detail = ((local_dark > 13) & (gray < 220) & (hsv[:, :, 1] < 75)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    lines = detail * 255
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 4:
            continue
        region = (labels == label).astype(np.uint8)
        distance = cv2.distanceTransform(region, cv2.DIST_L2, 5)
        if distance.max() * scale > 1.0:
            # Preserve the shape of broad branches instead of replacing their
            # silhouette with an arbitrary medial-axis network.
            lines[region > 0] = 0
            contours, _ = cv2.findContours(region, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                if cv2.arcLength(contour, True) * scale >= 1.:
                    cv2.drawContours(lines, [contour], -1, 255, 1)
        else:
            lines[region > 0] = 255
    # Keep image-supported edges of tonal mountains; suppress faint paper grain.
    edges = cv2.Canny(gray, 35, 85)
    edges[(hsv[:, :, 1] > 100)] = 0
    lines |= edges
    # Highly chromatic petals have explicit source boundaries; do not outline
    # every weak hue variation in a watercolor mountain or paper background.
    chromatic = ((hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 80)).astype(np.uint8)
    chromatic = cv2.morphologyEx(chromatic, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(chromatic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if cv2.contourArea(contour) * scale * scale >= .5:
            cv2.drawContours(lines, [contour], -1, 255, 1)
    # Remove isolated specks, not short connected source strokes.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    keep = np.zeros(count, np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 4).astype(np.uint8)
    lines = keep[labels] * 255
    if pad is not None:
        top, bottom, left, right = pad
        h, w = lines.shape
        # A cropped painting meeting white padding is not an ink frame.
        lines[max(0, top-1):top+2, :] = 0
        lines[max(0, h-bottom-2):h-bottom+1, :] = 0
        lines[:, max(0, left-1):left+2] = 0
        lines[:, max(0, w-right-2):w-right+1] = 0
    ink, fields = watercolor.analyze(normalized, scale)
    return cv2.cvtColor(255-lines, cv2.COLOR_GRAY2BGR), (ink, fields)
