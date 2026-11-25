"""Original-image support for offline, area-assisted ink routing."""
import cv2
import numpy as np

from gongbi_training.structural_ink_routing import route_owned_ink
from gongbi_training.structure_candidate_rank import rank_owned_candidates


def source_ridge_support(rgb, contrast, weak_mask, start_box, end_box):
    """Find weak dark ridges that can survive without a strong-ink seed."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    start = np.asarray(start_box).reshape(2, 2).mean(axis=0)
    end = np.asarray(end_box).reshape(2, 2).mean(axis=0)
    delta = end - start
    norm = np.linalg.norm(delta)
    if norm < 1:
        raise ValueError('Endpoint areas require different centers')
    dx, dy = -delta[1] / norm, delta[0] / norm
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    extra = np.zeros(gray.shape, dtype=bool)
    for radius in (2.5, 4.0):
        a = cv2.remap(gray, xx + dx*radius, yy + dy*radius, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        b = cv2.remap(gray, xx - dx*radius, yy - dy*radius, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        extra |= ((a+b)*.5-gray >= 2.) & (np.minimum(a,b)-gray >= -1.)
    near = cv2.dilate((weak_mask>0).astype(np.uint8),np.ones((3,3),np.uint8))>0
    return (weak_mask>0) | (extra & near & (contrast>=6.))


def route_between_areas(rgb, evidence, *, crop, start_box, end_box):
    """Return source-supported area-to-area routes and explicitly flag competing paths."""
    def zone(box):
        x0,y0,x1,y1 = box
        mask = np.zeros(evidence.shape,np.uint8)
        mask[y0:y1,x0:x1] = 1
        return mask
    owner = zone(crop)
    axis = 0 if crop[2]-crop[0] >= crop[3]-crop[1] else 1
    if start_box[axis]+start_box[axis+2] > end_box[axis]+end_box[axis+2]:
        start_box,end_box = end_box,start_box
    proposal = route_owned_ink(rgb,evidence,owner,crop=crop,start_zone=zone(start_box),
                               end_zone=zone(end_box))
    if not proposal['routes']:
        return proposal
    routes = []
    for item in proposal['routes']:
        points = item['points']
        travel = np.linalg.norm(np.diff(points,axis=0),axis=1).sum()
        span = np.linalg.norm(points[-1]-points[0])
        if travel <= 1.7*span:
            routes.append(item)
    return rank_owned_candidates([{'routes':routes}],owner,start_box,end_box)
