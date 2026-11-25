"""D original 600px ink extraction, adapted to current calibrated execution.

Only image/relative brush logic is imported. Site coordinates, pressure ceiling,
ink loading and robot execution stay in the host project.
"""
import cv2
import numpy as np
from config import ShuimoConfig
from ink_control import refresh_stats
from shuimo import ink_palette_index, press_to_width_mm
from original_d_tracer import LandscapeBrushGenerator


class TraceConfig:
    LANDSCAPE_SCALE = .12
    LANDSCAPE_JOIN_GAP_MM = 0.
    LANDSCAPE_DENSE_RADIUS_MM = 3.
    LANDSCAPE_DENSE_PRESS_REDUCTION = .55
    LANDSCAPE_DENSE_SPEED_BOOST = .45
    LANDSCAPE_LIGHT_INK_GRAY = 238
    LANDSCAPE_LIGHT_INK_CLOSE_PX = 3
    LANDSCAPE_DIRECTIONAL_JOIN_GAP_MM = 4.
    LANDSCAPE_DIRECTIONAL_JOIN_ANGLE_DEG = 38.
    LANDSCAPE_DIRECTIONAL_JOIN_MIN_LENGTH_MM = 1.5


def trace_original(work, pad=None):
    top,bottom,left,right = pad or (0,0,0,0)
    h,w=work.shape[:2]
    crop=work[top:h-bottom if bottom else h,left:w-right if right else w]
    ch,cw=crop.shape[:2]
    if min(ch,cw)<3: raise ValueError('Empty source crop')
    small=crop
    if max(ch,cw)>600:
        ratio=600/max(ch,cw)
        small=cv2.resize(crop,(int(cw*ratio),int(ch*ratio)))
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    return LandscapeBrushGenerator(TraceConfig())._extract_continuous_ink_strokes(gray),(cw,ch,left,top)


def build_strokes(work,scale,pad=None,source_image=None):
    if not np.isfinite(scale) or scale<=0: raise ValueError('Invalid scale')
    if source_image is None:
        traced,(cw,ch,left,top)=trace_original(work,pad)
    else:
        traced,_=trace_original(source_image)
        top,bottom,left,right=pad or (0,0,0,0)
        ch=work.shape[0]-top-bottom;cw=work.shape[1]-left-right
    lo=float(ShuimoConfig.PRESS_MIN_MM); hi=float(ShuimoConfig.PRESS_MAX_MM)
    if not 0<=lo<hi: raise ValueError('Invalid site pressure range')
    strokes=[]
    for index,old in enumerate(traced):
        raw=np.asarray(old['trajectory'],float)
        if len(raw)<3:continue
        xy=raw[:,:2]/1024*[cw,ch]+[left,top]
        t=np.linspace(0,1,len(raw))
        density=np.clip(raw[:,3],0,1)
        strength=np.clip(raw[:,2],0,1)*(1-.55*density)
        # Preserve D's 12% start/end ramps and relative LINE variation.
        ramp=np.minimum(np.clip(t/.12,0,1),np.clip((1-t)/.12,0,1))
        middle=(t>=.12)&(t<.88)
        shaped=strength*ramp
        shaped[middle]+=(.08/1.9756)*np.sin(t[middle]*np.pi*2)
        press=lo+(hi-lo)*np.clip(shaped,0,1)
        speed=np.where(middle,1.05-.25*strength,.5+.5*ramp)
        speed=np.clip(ShuimoConfig.SPEED_BASE*speed*(1+.45*density),5,100)
        layer=int(old['layer'])
        s=dict(points=xy.tolist(),press=press.tolist(),speed=speed.tolist(),pass_=1,
               pal_index=ink_palette_index(),layer=layer,name='D原版山水',role='original_d_line',
               design_origin='D-original-600px',source_stroke_id=index,tail_mm=0.,lift_mm=4.)
        strokes.append(refresh_stats(s,scale))
    stats=dict(profile='mountain',detail='original-d',algorithm='D-original-600px',
        source_paths=len(traced),strokes=len(strokes),layers={k:sum(s['layer']==k for s in strokes) for k in range(3)},
        length_mm=sum(s['length_mm'] for s in strokes),
        mean_width_mm=float(np.mean([s['width_mm'] for s in strokes])) if strokes else 0.,
        max_width_mm=max((press_to_width_mm(s['press_max']) for s in strokes),default=0.),
        mean_press_mm=float(np.mean([s['press_mean'] for s in strokes])) if strokes else 0.,
        max_press_mm=max((s['press_max'] for s in strokes),default=0.),
        pressure_mapping='D relative width/density and ramps remapped to current site limits',
        pressure_caps_mm={str(k):hi for k in range(3)},brush_width_calibrated_for_this_job=False)
    return strokes,stats
