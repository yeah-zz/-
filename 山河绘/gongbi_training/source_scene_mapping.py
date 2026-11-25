"""Map original source IDs, never recover open scene from collapsed legacy labels."""
import numpy as np
OPEN_SCENE_IDS=(1,4,10)
OTHER_OBJECT_IDS=(5,6,7,8,9,11,12,14)
def source_to_scene(raw):
    if raw.ndim!=2 or not set(np.unique(raw))<=(set(range(15))|{255}):raise ValueError('Expected original source-ID mask, not RGB or collapsed labels')
    target=np.full(raw.shape,255,'uint8')
    target[np.isin(raw,OPEN_SCENE_IDS)]=0
    target[np.isin(raw,[2,13])]=1
    target[raw==3]=2
    target[np.isin(raw,OTHER_OBJECT_IDS)]=3
    return target
