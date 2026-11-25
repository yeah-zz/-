"""Read a continuous, source-weighted center path from an existing ink mask."""
import cv2
import numpy as np
from skimage.graph import MCP_Geometric

from gongbi_training.local_rock_patch import contrast_field


def _snap(point, permitted, radius):
    x,y = np.asarray(point,dtype=float)
    left = max(0,int(np.floor(x-radius)))
    right = min(permitted.shape[1],int(np.ceil(x+radius+1)))
    top = max(0,int(np.floor(y-radius)))
    bottom = min(permitted.shape[0],int(np.ceil(y+radius+1)))
    pixels = np.argwhere(permitted[top:bottom,left:right]) + [top,left]
    if not len(pixels):
        return None
    distances = np.linalg.norm(pixels-[y,x],axis=1)
    winner = int(distances.argmin())
    return tuple(pixels[winner]) if distances[winner]<=radius else None


def _source_cost(image, permitted, ridge_weight):
    ridge = np.maximum(contrast_field(image),0)
    normalizer = max(float(np.percentile(ridge[permitted],90)),.02)
    dark_ridge = np.minimum(ridge/normalizer,1)
    width = cv2.distanceTransform(permitted.astype(np.uint8),cv2.DIST_L2,3)
    interior = np.minimum(width/3,1)
    return .45 + ridge_weight*(1-dark_ridge) + .45*(1-interior)


def reconstruct_ink_path(image, mask, start, end, *, crop, snap_radius=2, source_weighted=True, ridge_weight=.35):
    """Return an endpoint-constrained path, or no path for disconnected evidence."""
    if ridge_weight<0:
        raise ValueError('ridge_weight must be nonnegative')
    left,top,right,bottom = crop
    permitted = mask[top:bottom,left:right]>0
    a = _snap(np.asarray(start)-[left,top],permitted,snap_radius)
    b = _snap(np.asarray(end)-[left,top],permitted,snap_radius)
    if a is None or b is None or a==b:
        return np.empty((0,2),dtype=np.int32)
    _,components = cv2.connectedComponents(permitted.astype(np.uint8),connectivity=8)
    if components[a]!=components[b]:
        return np.empty((0,2),dtype=np.int32)
    permitted = components==components[a]
    if source_weighted:
        weights = _source_cost(image[top:bottom,left:right],permitted,ridge_weight)
    else:
        weights = np.ones(permitted.shape,dtype=np.float32)
    solver = MCP_Geometric(np.where(permitted,weights,np.inf),fully_connected=True)
    cost,_ = solver.find_costs([a],[b],find_all_ends=False)
    if not np.isfinite(cost[b]):
        return np.empty((0,2),dtype=np.int32)
    yx = np.asarray(solver.traceback(b),dtype=np.int32)
    return yx[:,::-1]+[left,top]
