"""Rank connected routes by ownership and attachment-area support."""
import numpy as np

from gongbi_training.automatic_ink_routing import _overlap


def _attachment_distance(point, box):
    corners = np.asarray(box,dtype=float).reshape(2,2)
    center = corners.mean(axis=0)-.5
    radius = (corners[1]-corners[0])/2
    return float(np.sum(((point-center)/radius)**2))


def _inside(point, box):
    left,top,right,bottom = box
    x,y = point
    return left<=x<right and top<=y<bottom


def rank_owned_candidates(pools, owner, start_box, end_box, *, attachment_weight=.8):
    """Re-rank candidate paths using broad structural regions without a target trace."""
    ranked = []
    seen = set()
    for pool in pools:
        for route in pool.get('routes',[]):
            points = route['points']
            key = tuple(map(tuple,points))
            if key in seen:
                continue
            seen.add(key)
            xs,ys = points.T
            if not owner[ys,xs].all() or not _inside(points[0],start_box) or not _inside(points[-1],end_box):
                continue
            attach = _attachment_distance(points[0],start_box)+_attachment_distance(points[-1],end_box)
            ranked.append(dict(points=points,cost=route['cost']+attachment_weight*attach,
                               ink_cost=route['cost'],attachment_cost=attach))
    if not ranked:
        return dict(points=np.empty((0,2),dtype=np.int32),status='no_owned_candidate',alternatives=[],routes=[])
    ranked.sort(key=lambda row:row['cost'])
    best = ranked[0]
    competitor = next((row for row in ranked[1:] if row['cost']<=1.12*best['cost']
                       and _overlap(best['points'][:,::-1],row['points'][:,::-1],owner.shape)<.70),None)
    status = 'ambiguous' if competitor else 'route_found'
    return dict(points=best['points'],status=status,routes=ranked,
                alternatives=[best,competitor] if competitor else [best])
