"""Source checks for diffuse rock boundaries and terminal branch direction."""
import cv2
import numpy as np

from gongbi_training.automatic_ink_routing import _overlap
from gongbi_training.continuity_attachment import route_between_areas


def rock_boundary_support(rgb, start_box, end_box, *, lighter_side):
    """Find a sustained tonal change across the visible rock face."""
    if lighter_side not in (-1, 1):
        raise ValueError('lighter_side must be -1 or 1 along the route normal')
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    start = np.asarray(start_box).reshape(2,2).mean(0)
    end = np.asarray(end_box).reshape(2,2).mean(0)
    travel = end-start
    distance = np.linalg.norm(travel)
    if distance < 1:
        raise ValueError('Endpoint areas must be distinct')
    tx,ty = travel/distance
    nx,ny = -ty,tx
    yy,xx = np.indices(gray.shape,dtype=np.float32)
    smooth = cv2.GaussianBlur(gray,(0,0),1.1)

    def offset(image, x, y):
        return cv2.remap(image,xx+x,yy+y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

    def across(radius):
        return lighter_side*(offset(smooth,nx*radius,ny*radius)-offset(smooth,-nx*radius,-ny*radius))

    fine = across(2)
    medium = across(4)
    broad = across(7)
    longitudinal = (offset(medium,tx*3,ty*3)+offset(medium,-tx*3,-ty*3))*.5
    response = np.minimum(medium,broad*.75)
    aligned = (fine>=1.) & (medium>=4.) & (broad>=5.) & (longitudinal>=2.)
    peak = response >= np.maximum(offset(response,nx,ny),offset(response,-nx,-ny))-.35
    return aligned, aligned & peak, response


def trace_rock_boundary(rgb, *, crop, start_box, end_box, lighter_side):
    """Search the measured boundary and retain rival routes for review."""
    evidence, peaks, strength = rock_boundary_support(rgb,start_box,end_box,lighter_side=lighter_side)
    result = route_between_areas(rgb,evidence,crop=crop,start_box=start_box,end_box=end_box)
    if not result['routes']:
        return result, evidence
    scored = []
    for row in result['routes']:
        x,y = row['points'].T
        peak_share = float(peaks[y,x].mean())
        face = float(np.clip(strength[y,x]/16,0,1).mean())
        score = row['cost'] + .6*(1-peak_share) + .5*(1-face)
        scored.append(dict(row,cost=score,edge_peak_share=peak_share))
    scored.sort(key=lambda row:row['cost'])
    best = scored[0]
    rival = next((row for row in scored[1:] if row['cost'] <= 1.12*best['cost'] and
                  _overlap(best['points'][:,::-1],row['points'][:,::-1],evidence.shape)<.70),None)
    candidates = [best,rival] if rival else [best]
    return dict(points=best['points'],status='ambiguous' if rival else 'route_found',
                alternatives=candidates,routes=scored), evidence


def check_branch_entry(points, trunk_box, downstream, *, minimum_cosine=.55):
    """Check the approach direction of an existing route against a reviewed trunk axis."""
    points = np.asarray(points,dtype=float)
    if len(points)<2:
        return dict(status='no_route',cosine=None)
    x0,y0,x1,y1 = trunk_box
    def in_box(p):
        return x0<=p[0]<x1 and y0<=p[1]<y1
    first,last = in_box(points[0]),in_box(points[-1])
    if first == last:
        return dict(status='attachment_uncertain',cosine=None)
    if first:
        points=points[::-1]
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if arc[-1]<5:
        return dict(status='attachment_uncertain',cosine=None)
    axis = np.subtract(downstream[1],downstream[0])
    if not np.any(axis):
        raise ValueError('Downstream trunk axis must have nonzero length')
    cosine=[]
    for distance in (5.,9.):
        tip = np.array([np.interp(arc[-1]-distance,arc,points[:,i]) for i in range(2)])
        approach = points[-1]-tip
        if np.linalg.norm(approach)<1:
            return dict(status='attachment_uncertain',cosine=None)
        cosine.append(float(approach@axis/(np.linalg.norm(approach)*np.linalg.norm(axis))))
    status='direction_supported' if min(cosine)>=minimum_cosine else 'direction_conflict'
    return dict(status=status,cosine=cosine)
