"""A serializable digital stroke and adapter to the existing ink brush."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

from master_studio import brush


@dataclass(frozen=True)
class DigitalStroke:
    id: str
    object_id: str
    role: str
    technique: str
    xy: list
    widths_px: list
    rgb: tuple
    opacity: float
    dry: float
    spread_px: float
    layer: int
    after: tuple
    provenance: dict
    opacity_profile: tuple = ()

    def record(self):
        values = asdict(self)
        xy = np.asarray(self.xy)
        tangents = np.gradient(xy, axis=0)
        heading = np.unwrap(np.arctan2(tangents[:,1], tangents[:,0]))
        width = np.asarray(self.widths_px)
        base = max(float(width.max()), .8)
        values["heading_rad"] = heading.tolist()
        values["turn_per_sample_rad"] = np.gradient(heading).tolist()
        values["virtual_pressure"] = np.clip((width/base-.08)/1.05, .02, 1.).tolist()
        return values


def replay(path, frames=False):
    """Render an exported sequence from blank paper."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document["coordinate_system"] != "xy pixels":
        raise ValueError("Digital strokes must use xy pixel coordinates")
    raw_fields = {field.name for field in fields(DigitalStroke)}
    strokes = [DigitalStroke(**{key:value for key,value in record.items() if key in raw_fields})
               for record in document["strokes"]]
    return render(strokes, tuple(document["canvas_wh"]), seed=document.get("seed",24), frames=frames,
                  renderer=document.get("renderer","legacy"))


def _ribbon_tone(layer, centers, widths, arc, profile, stroke, color, phase):
    """Vary ink along and across the ribbon using its measured centerline."""
    from scipy.spatial import cKDTree

    alpha = np.array(layer.getchannel("A"))
    y,x = np.where(alpha)
    if not len(y):
        return layer
    points=np.c_[x,y]
    _,nearest=cKDTree(centers).query(points)
    tangent=np.gradient(centers,axis=0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1),1e-6)[:,None]
    normal=np.c_[-tangent[:,1],tangent[:,0]]
    transverse=np.sum((points-centers[nearest])*normal[nearest],axis=1)
    across=transverse/np.maximum(.65,widths[nearest]*.5)
    progress=arc[nearest]/max(arc[-1],1e-6)
    # The same paper-direction fiber can persist over several successive samples.
    fiber=np.sin(transverse*3.2+0.37*np.sin(arc[nearest]*.13+phase)+phase)
    fiber=np.maximum(fiber-.20,0)
    grain=np.clip(1.0-stroke.dry*fiber*(.48+.28*np.abs(across)),.22,1.0)
    edge=np.clip(1.12-.29*np.abs(across)**2,.45,1.0)
    endpoints=(.79+.21*np.clip(progress/.075,0,1))*(.53+.47*np.clip((1-progress)/.11,0,1))
    tone=np.clip(profile[nearest],0,1)*grain*edge*endpoints
    alpha[y,x]=np.clip(alpha[y,x]*tone,0,255).astype(np.uint8)
    layer=Image.new("RGBA",layer.size,(*color,0))
    layer.putalpha(Image.fromarray(alpha))
    return layer


def render(strokes, size, seed=24, frames=False, renderer="legacy"):
    """Replay the supplied order from blank paper using the existing digital ink."""
    from PIL import ImageFilter

    paper = Image.new("RGBA", size, (246, 242, 232, 255))
    if renderer not in ("legacy","ribbon"):
        raise ValueError(f"Unknown digital renderer {renderer}")
    rng = random.Random(seed)
    drawn = set()
    sequence = [paper.convert("RGB")] if frames else []
    for stroke in strokes:
        if stroke.id in drawn or not set(stroke.after) <= drawn:
            raise ValueError(f"Invalid drawing dependencies for {stroke.id}")
        xy = np.asarray(stroke.xy, dtype=float)
        widths = np.asarray(stroke.widths_px, dtype=float)
        if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or len(widths)!=len(xy):
            raise ValueError(f"Invalid width profile for {stroke.id}")
        if not np.isfinite(xy).all() or not np.isfinite(widths).all() or np.any(widths<=0):
            raise ValueError(f"Invalid geometry for {stroke.id}")
        centers = np.asarray(brush.resample(xy, step=1.4))
        if len(centers) < 2:
            raise ValueError(f"Empty stroke {stroke.id}")
        distance = lambda curve: np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
        arc=distance(centers)
        original_arc=distance(xy)
        width_profile = np.interp(arc,original_arc,widths)
        base = max(.8, float(width_profile.max()))
        pressure = np.clip((width_profile/base-.08)/1.05, .02, 1.)
        layer = Image.new("RGBA", size)
        color = tuple(stroke.rgb)
        alpha=stroke.opacity*255
        brush.draw_ink_stroke(layer, centers, base, color,
                              round(alpha if renderer=="ribbon" else alpha*(1-.86*stroke.dry)), pressure, rng,
                              widths_px=width_profile)
        if renderer=="ribbon":
            profile=np.asarray(stroke.opacity_profile,dtype=float)
            if profile.size and (profile.shape!=(len(xy),) or not np.isfinite(profile).all()):
                raise ValueError(f"Invalid opacity profile for {stroke.id}")
            sampled=np.interp(arc,original_arc,profile) if profile.size else np.ones_like(arc)
            layer=_ribbon_tone(layer,centers,width_profile,arc,sampled,stroke,color,rng.random()*6.28)
        elif stroke.dry > .02:
            texture = Image.new("RGBA", size)
            brush.draw_stroke(ImageDraw.Draw(texture), centers, base, color,
                              round(stroke.opacity*255*stroke.dry*.7), pressure, rng,
                              dry=stroke.dry, end_pool=False)
            layer.alpha_composite(texture)
        if stroke.spread_px:
            blur_alpha = layer.getchannel("A").filter(ImageFilter.GaussianBlur(stroke.spread_px))
            halo = Image.new("RGBA",size,(*stroke.rgb,0))
            halo.putalpha(blur_alpha.point(lambda alpha: round(alpha*.45)))
            paper.alpha_composite(halo)
        paper.alpha_composite(layer)
        drawn.add(stroke.id)
        if frames:
            sequence.append(paper.convert("RGB"))
    return paper.convert("RGB"), sequence
