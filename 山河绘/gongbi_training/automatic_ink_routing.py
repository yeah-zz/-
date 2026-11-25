"""Choose endpoints in a local ink region and keep competing branch routes visible."""
import cv2
import numpy as np
from skimage.graph import MCP_Geometric

from gongbi_training.ink_band_reconstruction import _source_cost


def _gate_candidates(cost, mask, dimension, center, limit, width, separation, centering):
    yx = np.argwhere(mask)
    axis = yx[:,dimension]
    select = np.abs(axis-center)<=width
    position = yx[select]
    values = cost[position[:,0],position[:,1]] + centering*np.abs(position[:,dimension]-center)
    chosen = []
    for point in position[np.argsort(values,kind='stable')]:
        if any(abs(point[1-dimension]-p[1-dimension])<separation for p in chosen):
            continue
        chosen.append(tuple(int(value) for value in point))
        if len(chosen)==limit:
            break
    return chosen


def _role_cost(hints):
    if not hints.any():
        return np.zeros(hints.shape,dtype=np.float32)
    distance = cv2.distanceTransform((~hints).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return .72*np.minimum(distance/4,1)


def _overlap(first, second, shape):
    mask = np.zeros(shape,np.uint8)
    mask[first[:,0],first[:,1]] = 1
    near_first = cv2.dilate(mask,np.ones((3,3),np.uint8)).astype(bool)
    return float(near_first[second[:,0],second[:,1]].mean())


def automatic_ink_route(image, ink, *, crop, hints=None, ownership_cost=None,
                        gate_options=(4,2,4,.10), include_routes=False):
    """Search between automatically detected gate points in an inspected object region."""
    left,top,right,bottom = crop
    region = np.s_[top:bottom,left:right]
    permitted = ink[region]>0
    empty = np.empty((0,2),dtype=np.int32)
    if min(permitted.shape)<5 or not permitted.any():
        return dict(points=empty,status='empty_region',alternatives=[])
    dim = int(permitted.shape[1]>=permitted.shape[0])
    count = permitted.shape[dim]
    centers = [int(round(fraction*(count-1))) for fraction in (.14,.86)]
    values = _source_cost(image[region],permitted,.35)
    if hints is not None:
        values += _role_cost(hints[region]>0)
    if ownership_cost is not None:
        values += ownership_cost[region]
    start,end = [_gate_candidates(values,permitted,dim,center,*gate_options) for center in centers]
    if not start or not end:
        return dict(points=empty,status='no_gate_ink',alternatives=[])
    costs = np.where(permitted,values,np.inf)
    routes = []
    for a in start:
        mcp = MCP_Geometric(costs,fully_connected=True)
        distance,_ = mcp.find_costs([a],end,find_all_ends=True)
        for b in end:
            if not np.isfinite(distance[b]):
                continue
            route = np.asarray(mcp.traceback(b),dtype=np.int32)
            span = b[dim]-a[dim]
            score = float(distance[b]/span + .08*(values[a]+values[b]))
            routes.append((score,route))
    if not routes:
        return dict(points=empty,status='no_connected_route',alternatives=[])
    routes.sort(key=lambda item:item[0])
    best,path = routes[0]
    competitor = next((entry for entry in routes[1:] if _overlap(path,entry[1],permitted.shape)<.70),None)
    ambiguous = competitor is not None and competitor[0]<=best*1.12
    evidence = [dict(cost=best,points=path[:,::-1]+[left,top])]
    if competitor is not None:
        evidence.append(dict(cost=competitor[0],points=competitor[1][:,::-1]+[left,top]))
    result = dict(points=evidence[0]['points'],status='ambiguous' if ambiguous else 'route_found',
                  alternatives=evidence,gate_counts=[len(start),len(end)])
    if include_routes:
        result['routes'] = [dict(cost=cost,points=route[:,::-1]+[left,top]) for cost,route in routes]
    return result
