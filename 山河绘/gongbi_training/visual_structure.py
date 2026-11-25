"""Multiscale source-image candidates for dark marks, light ridges and paired contours."""

import numpy as np

from gongbi_training.scene_topology import CONTRAST


def _sample(field, x, y, dx, dy, cv2):
    return cv2.remap(field, x + np.float32(dx), y + np.float32(dy), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def detect(rgb, cv2, skeletonize):
    """Return cue scores and inferred sections in source coordinates."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    smooth = cv2.GaussianBlur(gray, (0, 0), .7)
    dark = skeletonize(np.ascontiguousarray(cv2.GaussianBlur(smooth, (0, 0), 7) - smooth >= CONTRAST))
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB) / 100
    y, x = np.indices(gray.shape, dtype=np.float32)
    ridge = np.zeros_like(gray)
    for sigma in (1.0, 2.0, 3.8):
        field = cv2.GaussianBlur(lab[:, :, 0], (0, 0), sigma)
        xx = cv2.Sobel(field, cv2.CV_32F, 2, 0, ksize=3) * (sigma**2 / 4)
        yy = cv2.Sobel(field, cv2.CV_32F, 0, 2, ksize=3) * (sigma**2 / 4)
        xy = cv2.Sobel(field, cv2.CV_32F, 1, 1, ksize=3) * (sigma**2 / 4)
        gap = np.hypot(xx - yy, 2 * xy)
        normal = .5 * np.arctan2(2 * xy, xx - yy) + np.pi / 2
        dx, dy = np.cos(normal) * sigma, np.sin(normal) * sigma
        across = (xx + yy - gap) / 2
        along = (xx + yy + gap) / 2
        fit = np.maximum(-across - np.abs(along), 0)
        sides = [_sample(field, x, y, sign*dx, sign*dy, cv2) for sign in (-1, 1)]
        peak = fit * (field >= np.maximum(*sides) - .001)
        ridge = np.maximum(ridge, peak)

    pair = np.zeros_like(gray)
    chroma = np.zeros_like(gray)
    half_width = np.zeros_like(gray, dtype=np.uint8)
    angle = np.zeros_like(gray, dtype=np.uint8)
    for blur, radii in ((.85, (2, 4, 6)), (1.5, (8, 11, 15))):
        field = cv2.GaussianBlur(lab, (0, 0), blur)
        for index in range(8):
            radians = np.pi * index / 8
            nx, ny = np.cos(radians), np.sin(radians)
            for radius in radii:
                edges = [_sample(field, x, y, sign*radius*nx, sign*radius*ny, cv2) for sign in (-1, 1)]
                outs = [_sample(field, x, y, sign*(radius+2)*nx, sign*(radius+2)*ny, cv2) for sign in (-1, 1)]
                ins = [_sample(field, x, y, sign*radius*.55*nx, sign*radius*.55*ny, cv2) for sign in (-1, 1)]
                # Each visible contour has a trough on both sides. The two interior
                # samples should also agree in color across the proposed body.
                outlines = [np.minimum(inner[:,:,0]-edge[:,:,0], outer[:,:,0]-edge[:,:,0])
                            for inner, edge, outer in zip(ins, edges, outs)]
                agree = np.exp(-np.linalg.norm(ins[0]-ins[1], axis=2) / .13)
                direction = min(4., radius*.7)
                stripes = [field[:,:,1:] - outside[:,:,1:] for outside in outs]
                same_color = np.sum(stripes[0]*stripes[1], axis=2) > 0
                color_score = np.minimum(*(np.linalg.norm(side, axis=2) for side in stripes))
                color_score *= same_color * agree
                score = np.maximum(np.minimum(*outlines), 0) * agree
                score = np.maximum(score, color_score*.68)
                adjacent = [_sample(score, x, y, -sign*ny*direction, sign*nx*direction, cv2) for sign in (-1, 1)]
                score = np.minimum(score, np.maximum(*adjacent))
                local_max = np.maximum(*[_sample(score, x, y, sign*nx, sign*ny, cv2) for sign in (-1, 1)])
                score = np.where(score >= local_max-.0002, score, 0)
                replace = score > pair
                half_width[replace] = radius
                angle[replace] = index
                chroma[replace] = color_score[replace]
                pair = np.maximum(pair, score)
    # Leave an outer margin where a bilateral source measurement is impossible.
    for field in (pair, ridge):
        field[:17] = field[-17:] = 0
        field[:, :17] = field[:, -17:] = 0
    paired = skeletonize(np.ascontiguousarray(pair >= .034))
    light = skeletonize(np.ascontiguousarray(ridge >= .027))
    return dict(dark=dark, light=light, paired=paired, pair_score=pair, ridge_score=ridge,
                half_width=half_width, angle=angle, chroma_score=chroma)


def sections(cues, keep):
    """Export selected body sections, with paired edge endpoints and width."""
    result = []
    for y, x in np.argwhere(keep):
        radius = int(cues["half_width"][y, x])
        theta = float(cues["angle"][y, x]) * np.pi / 8
        dx, dy = radius*np.cos(theta), radius*np.sin(theta)
        result.append(dict(center_xy=[int(x), int(y)],
                           sides_xy=[[round(x-dx, 2), round(y-dy, 2)], [round(x+dx, 2), round(y+dy, 2)]],
                           width_px=2*radius, score=float(cues["pair_score"][y, x]),
                           source="paired_contours_and_lab",
                           chroma_score=float(cues["chroma_score"][y, x])))
    return result
