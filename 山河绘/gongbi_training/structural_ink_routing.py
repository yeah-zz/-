"""Trace existing ink between structural neighborhoods of one object."""
import numpy as np
from skimage.graph import MCP_Geometric

from gongbi_training.automatic_ink_routing import _gate_candidates, _overlap, _role_cost
from gongbi_training.ink_band_reconstruction import _source_cost


def route_owned_ink(image, ink, owner, *, crop, start_zone, end_zone, hints=None):
    """Select points from broad structural regions and trace inside one owner."""
    left,top,right,bottom = crop
    region = np.s_[top:bottom,left:right]
    permitted = (ink[region]>0) & (owner[region]>0)
    empty = np.empty((0,2),dtype=np.int32)
    if not permitted.any():
        return dict(points=empty,status='no_owned_ink',alternatives=[],routes=[])
    values = _source_cost(image[region],permitted,.35)
    if hints is not None:
        values += _role_cost(hints[region]>0)
    dim = int(permitted.shape[1]>=permitted.shape[0])
    start, end = [], []
    for target, zone in ((start,start_zone),(end,end_zone)):
        zoned = (zone[region]>0) & permitted
        if zoned.any():
            extremes = np.argwhere(zoned)[:,dim]
            center = (int(extremes.min())+int(extremes.max()))/2
            width = max(float(extremes.max()-center),1)
            target.extend(_gate_candidates(values,zoned,dim,center,8,width,2,.015))
    if not start or not end:
        return dict(points=empty,status='attachment_not_supported',alternatives=[],routes=[])
    costs = np.where(permitted,values,np.inf)
    routes = []
    for a in start:
        solver = MCP_Geometric(costs,fully_connected=True)
        distances,_ = solver.find_costs([a],end,find_all_ends=True)
        for b in end:
            span = b[dim]-a[dim]
            if span<=0 or not np.isfinite(distances[b]):
                continue
            path = np.asarray(solver.traceback(b),dtype=np.int32)
            score = float(distances[b]/span + .08*(values[a]+values[b]))
            routes.append((score,path))
    if not routes:
        return dict(points=empty,status='no_owned_connection',alternatives=[],routes=[])
    routes.sort(key=lambda item:item[0])
    best,path = routes[0]
    competitor = next((item for item in routes[1:] if _overlap(path,item[1],permitted.shape)<.70),None)
    ambiguous = competitor is not None and competitor[0]<=1.12*best
    def global_route(item):
        return dict(cost=item[0],points=item[1][:,::-1]+[left,top])
    alternatives = [global_route(routes[0])]
    if competitor is not None:
        alternatives.append(global_route(competitor))
    return dict(points=alternatives[0]['points'],status='ambiguous' if ambiguous else 'route_found',
                alternatives=alternatives,routes=[global_route(item) for item in routes])
