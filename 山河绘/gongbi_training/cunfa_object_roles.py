"""Object context is required before a local mark may be called rock cun."""
import numpy as np
UNKNOWN=255
OBJECT_IDS={'paper':0,'rock':1,'tree_trunk_branch':2,'tree_foliage':3,'architecture':4,'water':5,'tool_or_hand':6,'unknown':255}
ROCK_ROLES={'structure':1,'cun':2}

def constrain_rock_roles(raw_roles,object_targets,object_verified):
    """Abstain outside confirmed rock. This is a guard, not an object detector.

    Unknown/tree pixels are NOT converted into background. The raw model output
    must be retained separately for evaluation and must never be overwritten.
    """
    raw=np.asarray(raw_roles);objects=np.asarray(object_targets);verified=np.asarray(object_verified,dtype=bool)
    if raw.shape!=objects.shape or raw.shape!=verified.shape:
        raise ValueError('role/object/verified shapes must agree')
    if not set(np.unique(raw)).issubset({0,1,2,255}):raise ValueError('Unexpected rock role id')
    allowed=verified&(objects==OBJECT_IDS['rock'])
    result=np.full(raw.shape,UNKNOWN,dtype=np.uint8);result[allowed]=raw[allowed]
    return result

def forbidden_cun_reference(object_targets,object_verified):
    """Known non-rock objects supply exclusion, not background labels."""
    objects=np.asarray(object_targets);verified=np.asarray(object_verified,dtype=bool)
    return verified&np.isin(objects,[OBJECT_IDS[k] for k in ['tree_trunk_branch','tree_foliage','architecture','water','tool_or_hand']])
