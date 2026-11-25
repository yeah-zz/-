"""Scene/object labels must not silently become visible-ink labels."""
import numpy as np
from PIL import Image,ImageDraw
SCENE_CLASSES={'open_scene':0,'rock':1,'tree':2,'other_object':3}
def sparse_scene_reference(size,regions):
    scene=Image.new('L',tuple(size),255)
    # A broad uncertain area never supplies a negative label. Explicit inner cores
    # are narrower confirmed evidence and take precedence over broad uncertain zones.
    for r in regions:
        if r['kind']!='core_polygon' or r['object'] not in SCENE_CLASSES:continue
        ImageDraw.Draw(scene).polygon([tuple(p) for p in r['points']],fill=SCENE_CLASSES[r['object']])
    # Neither blank-looking object interiors nor open scene imply measured ink truth.
    ink=np.full((size[1],size[0]),255,'uint8')
    return np.array(scene),ink

def legacy_object_to_scene(target):
    out=np.full_like(target,255);out[target==0]=3;out[target==1]=1;out[target==2]=2
    return out
