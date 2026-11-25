# -*- coding: utf-8 -*-
"""Bounded curve fitting and image-only brush appearance estimates. No robot."""
import sys
sys.dont_write_bytecode=True
from rock_v15_common import read,jsave,digest
import split_detail_v4_frozen as v4
import numpy as np,cv2,math
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from pathlib import Path
from PIL import Image
CONFIG=dict(version='bounded_bezier_image_brush_r2_regularized',maximum_error_px=.6,curve_sample_step=.25,corner_angle_degrees=70,profile_step_px=1.,profile_radius_at_496=10.,profile_sample_step=.25,maximum_measured_width_at_496=10.,minimum_density=.025,width_smoothing_sigma_at_496=2.,density_smoothing_sigma_at_496=1.5,short_patch_length_to_width=1.8,unknown_preview_width_at_496=.85,unknown_preview_density=.32,render_supersampling=3,physical_z=None,physical_speed=None,water_content=None)

def resample(p,step):
    p=np.asarray(p,np.float64)
    if len(p)<2:return p.copy()
    good=np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-7];p=p[good]
    if len(p)<2:return p.copy()
    a=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];t=np.linspace(0,a[-1],max(2,int(math.ceil(a[-1]/step))+1));return np.column_stack([np.interp(t,a,p[:,j]) for j in [0,1]])

def bezier(c,t):
    t=np.asarray(t)[:,None];return (1-t)**3*c[0]+3*(1-t)**2*t*c[1]+3*(1-t)*t*t*c[2]+t**3*c[3]

def unit(v):return v/max(float(np.linalg.norm(v)),1e-9)

def sample_curve(c):
    bound=float(np.linalg.norm(np.diff(c,axis=0),axis=1).sum());return bezier(c,np.linspace(0,1,max(5,int(math.ceil(bound/CONFIG['curve_sample_step']))+1)))

def deviation(a,b):
    if not len(a) or not len(b):return float('inf')
    return max(float(cKDTree(a).query(b)[0].max()),float(cKDTree(b).query(a)[0].max()))

def fit_piece(p,t0,t1,depth=0):
    if len(p)<2:return []
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];length=float(arc[-1])
    if length<1e-8:return []
    t=arc/length;b0=(1-t)**3;b1=3*(1-t)**2*t;b2=3*(1-t)*t*t;b3=t**3;fixed=(b0+b1)[:,None]*p[0]+(b2+b3)[:,None]*p[-1];rhs=p-fixed;a=np.stack([b1[:,None]*t0,-b2[:,None]*t1],axis=2).reshape(-1,2);target=rhs.reshape(-1)
    try:alpha=np.linalg.lstsq(a,target,rcond=None)[0]
    except np.linalg.LinAlgError:alpha=np.array([length/3,length/3])
    alpha=np.clip(alpha,length*.01,length*2);c=np.array([p[0],p[0]+alpha[0]*t0,p[-1]-alpha[1]*t1,p[-1]]);sample=sample_curve(c);reference=resample(p,.25);error=deviation(reference,sample)
    if error<=CONFIG['maximum_error_px']-.06:return [c]
    if len(p)<=3 or depth>=18:
        return [np.array([a,a+(b-a)/3,a+2*(b-a)/3,b]) for a,b in zip(p[:-1],p[1:]) if np.linalg.norm(b-a)>1e-8]
    distances=cKDTree(sample).query(p)[0];i=int(np.argmax(distances));i=i if 1<i<len(p)-2 else len(p)//2;tangent=unit(p[min(len(p)-1,i+3)]-p[max(0,i-3)])
    return fit_piece(p[:i+1],t0,tangent,depth+1)+fit_piece(p[i:],tangent,t1,depth+1)

def fit_paths(paths,shape):
    raw=v4.raster(shape,paths);_,neighbors=v4.pixel_chains(raw);occupancy=np.zeros(shape,np.uint16)
    for p in paths:occupancy[v4.raster(shape,[p])]+=1
    records=[];fitted=[]
    for ident,pp in enumerate(paths):
        p=np.asarray(pp,np.float64);own=v4.raster(shape,[p]);occupancy[own]-=1;anchors={0,len(p)-1};closed=len(p)>2 and np.linalg.norm(p[0]-p[-1])<.1
        for i,(x,y) in enumerate(np.rint(p).astype(int)):
            if len(neighbors.get((y,x),[]))!=2:anchors.add(i)
        if len(p)>=12:
            smooth=gaussian_filter1d(p,1.2,axis=0,mode='nearest');candidates=[]
            for i in range(4,len(p)-4):
                a=unit(smooth[i]-smooth[i-4]);b=unit(smooth[i+4]-smooth[i]);angle=math.acos(float(np.clip(a@b,-1,1)))
                if angle>math.radians(CONFIG['corner_angle_degrees']):candidates.append((angle,i))
            chosen=[]
            for _,i in sorted(candidates,reverse=True):
                if all(abs(i-j)>5 for j in chosen):chosen.append(i);anchors.add(i)
        if closed and len(anchors)<4:anchors.update([len(p)//4,len(p)//2,3*len(p)//4])
        segments=[];indices=sorted(anchors)
        for i,j in zip(indices[:-1],indices[1:]):
            piece=p[i:j+1]
            if len(piece)<2:continue
            t0=unit(piece[min(4,len(piece)-1)]-piece[0]);t1=unit(piece[-1]-piece[max(0,len(piece)-5)]);segments+=fit_piece(piece,t0,t1)
        q=np.concatenate([sample_curve(c)[:-1] for c in segments]+[segments[-1][-1:]]) if segments else p.copy();error=deviation(resample(p,.25),q);reason=None;mask=v4.raster(shape,[q]);other=occupancy>0;old_touch=(cv2.dilate(own.astype(np.uint8),np.ones((3,3),np.uint8))>0)&other;new_touch=(cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0)&other;unexpected=new_touch&~old_touch
        if error>CONFIG['maximum_error_px']:reason='deviation_bound'
        elif unexpected.any():reason='would_add_neighbor_contact'
        if reason:q=p.copy();segments=[];mask=own;error=0.
        occupancy[mask]+=1;fitted.append(q);records.append(dict(path_id=ident,segments=[c.tolist() for c in segments],polyline_fallback=p.tolist() if not segments else None,max_sampled_hausdorff_px=error,endpoint_start=p[0].tolist(),endpoint_end=p[-1].tolist(),closed=bool(closed),fixed_graph_junctions=True,fallback_reason=reason,stroke_identity_verified=False,direction_verified=False))
    return fitted,records

def remap_rgb(rgb,coords):
    return cv2.remap(rgb.astype(np.float32)/255.,coords[:,:,0].astype(np.float32),coords[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)

def profile_parameters(rgb,p,junction_distance=None):
    scale=max(rgb.shape[:2])/496.;p=resample(p,CONFIG['profile_step_px']);n=len(p)
    if n==0:return dict(points=[],width_px=[],optical_density=[],valid=[],reason=[])
    if n<2:tan=np.array([[1.,0.]])
    else:
        sm=gaussian_filter1d(p,max(.6,1.*scale),axis=0,mode='nearest');tan=np.gradient(sm,axis=0);tan/=np.maximum(np.linalg.norm(tan,axis=1,keepdims=True),1e-8)
    normals=np.column_stack([-tan[:,1],tan[:,0]]);radius=CONFIG['profile_radius_at_496']*scale;offset=np.linspace(-radius,radius,max(25,int(round(2*radius/CONFIG['profile_sample_step']))+1));coords=p[:,None,:]+normals[:,None,:]*offset[None,:,None];values=remap_rgb(rgb,coords);luma=values@np.array([.2126,.7152,.0722],np.float32);outer=max(3,int(.18*len(offset)));left=np.quantile(luma[:,:outer],.8,axis=1);right=np.quantile(luma[:,-outer:],.8,axis=1);t=(offset+radius)/(2*radius);bg=left[:,None]*(1-t)+right[:,None]*t;density=np.maximum(0,-np.log(np.clip((luma+.002)/(bg+.002),.01,1.)));width=[];dark=[];valid=[];reasons=[];center_indices=np.flatnonzero(abs(offset)<=max(.8,.75*scale))
    for i in range(n):
        reason=None;k=int(center_indices[np.argmax(density[i,center_indices])]);d=density[i];peak=float(d[k]);outer_std=float(max(np.std(luma[i,:outer]),np.std(luma[i,-outer:])));min_density=max(CONFIG['minimum_density'],outer_std*3)
        if peak<min_density:reason='weak_or_pigment_ambiguous'
        if junction_distance is not None:
            x,y=np.rint(p[i]).astype(int);x=int(np.clip(x,0,rgb.shape[1]-1));y=int(np.clip(y,0,rgb.shape[0]-1))
            if junction_distance[y,x]<2.*scale:reason='junction_overlap'
        half=peak*.5;l=k;r=k
        while l>0 and d[l]>half:l-=1
        while r<len(d)-1 and d[r]>half:r+=1
        if l==0 or r==len(d)-1:reason=reason or 'profile_search_boundary'
        def cross(a,b):
            fraction=(half-d[a])/max(abs(float(d[b]-d[a])),1e-8)
            if d[b]<d[a]:fraction=(d[a]-half)/max(abs(float(d[b]-d[a])),1e-8)
            return float(offset[a]+np.clip(fraction,0,1)*(offset[b]-offset[a]))
        lo=cross(l,min(l+1,len(d)-1));hi=cross(max(0,r-1),r);w=hi-lo
        peaks,_=find_peaks(d,prominence=max(.025,.12*peak))
        for other in peaks:
            if other==k or abs(offset[other]-offset[k])<1.25*scale or d[other]<peak*.65:continue
            valley=float(d[min(k,other):max(k,other)+1].min())
            if valley>half:reason=reason or 'merged_neighbor_ink_band'
        if not .35<=w<=CONFIG['maximum_measured_width_at_496']*scale:reason=reason or 'width_not_isolated'
        width.append(float(w) if reason is None else None);dark.append(peak if reason is None else None);valid.append(reason is None);reasons.append(reason or 'measured_local_profile')
    return dict(points=p.tolist(),width_px=width,optical_density=dark,valid=valid,reason=reasons,direction_verified=False,pressure_z=None,speed=None,water_content=None)

def render_parameters(shape,parameters):
    supersample=CONFIG['render_supersampling'];h,w=shape;canvas=np.ones((h*supersample,w*supersample),np.float32);scale=max(shape)/496.
    for row in parameters:
        p=np.asarray(row['points'],float)
        if not len(p):continue
        widths=np.array([x if x is not None else CONFIG['unknown_preview_width_at_496']*scale for x in row['width_px']]);ds=np.array([x if x is not None else CONFIG['unknown_preview_density'] for x in row['optical_density']]);alpha=1-np.exp(-ds)
        if len(p)>1:tan=np.gradient(p,axis=0);tan/=np.maximum(np.linalg.norm(tan,axis=1,keepdims=True),1e-8)
        else:tan=np.array([[1.,0.]])
        normals=np.column_stack([-tan[:,1],tan[:,0]]);edge_a=p+normals*widths[:,None]/2;edge_b=p-normals*widths[:,None]/2;lo=np.maximum(0,np.floor(np.minimum(edge_a,edge_b).min(0)*supersample-3)).astype(int);hi=np.minimum([w*supersample,h*supersample],np.ceil(np.maximum(edge_a,edge_b).max(0)*supersample+4)).astype(int);bw,bh=hi-lo
        if bw<=0 or bh<=0:continue
        layer=np.zeros((bh,bw),np.float32)
        for i in range(len(p)-1):
            poly=np.rint((np.array([edge_a[i],edge_a[i+1],edge_b[i+1],edge_b[i]])*supersample)-lo).astype(np.int32);a=np.maximum(0,poly.min(0));b=np.minimum([bw,bh],poly.max(0)+1);ww,hh=b-a
            if ww<=0 or hh<=0:continue
            mask=np.zeros((hh,ww),np.uint8);cv2.fillConvexPoly(mask,poly-a,255);region=layer[a[1]:b[1],a[0]:b[0]];np.maximum(region,(mask/255.)*float((alpha[i]+alpha[i+1])/2),out=region)
        for i in {0,len(p)-1}:
            center=np.rint(p[i]*supersample-lo).astype(int);rad=max(1,int(round(widths[i]*supersample/2)));mask=np.zeros_like(layer,np.uint8);cv2.circle(mask,tuple(center),rad,255,-1);np.maximum(layer,(mask/255.)*float(alpha[i]),out=layer)
        canvas[lo[1]:hi[1],lo[0]:hi[0]]*=1-layer
    im=np.rint(np.clip(canvas,0,1)*255).astype(np.uint8);return cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA)

def regularize_parameters(row,scale):
    """Retain raw estimates; do not map an isolated ink patch to a wide pen."""
    raw_width=list(row['width_px']);raw_density=list(row['optical_density']);row['raw_profile_width_px']=raw_width;row['raw_profile_optical_density']=raw_density
    valid=np.array(row['valid'],bool);p=np.asarray(row['points']);length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()) if len(p)>1 else 0.
    if valid.any():
        widths=np.array([v if v is not None else np.nan for v in raw_width]);median=float(np.nanmedian(widths));radius=max(2,int(round(3*scale)));padded=np.pad(widths,(radius,radius),constant_values=np.nan);windows=np.lib.stride_tricks.sliding_window_view(padded,2*radius+1)
        local=np.array([np.nanmedian(a) if np.isfinite(a).any() else median for a in windows]);spikes=valid&(widths>local*2.5)&(widths>local+2*scale)
        if length<CONFIG['short_patch_length_to_width']*median:spikes|=valid
        for i in np.flatnonzero(spikes):row['valid'][i]=False;row['reason'][i]='ink_patch_not_resolved_as_line_width';row['width_px'][i]=None;row['optical_density'][i]=None
    valid=np.array(row['valid'],bool);weights=valid.astype(float)
    for name,sigma in [('width_px',CONFIG['width_smoothing_sigma_at_496']*scale),('optical_density',CONFIG['density_smoothing_sigma_at_496']*scale)]:
        values=np.array([v if v is not None else 0. for v in row[name]]);den=gaussian_filter1d(weights,max(.5,sigma),mode='nearest');sm=gaussian_filter1d(values*weights,max(.5,sigma),mode='nearest')/np.maximum(den,1e-8)
        row[name]=[float(sm[i]) if valid[i] else None for i in range(len(values))]
    row['appearance_smoothed_along_path']=True;return row

def process(image,path_file,output):
    image=Path(image);output=Path(output);output.mkdir(parents=True,exist_ok=True);rgb=np.asarray(Image.open(image).convert('RGB'));doc=read(path_file);paths=[np.asarray(r['points'],float) for r in doc['paths']];shape=rgb.shape[:2];fitted,curves=fit_paths(paths,shape);raw=v4.raster(shape,paths);_,neighbors=v4.pixel_chains(raw);junction=np.zeros(shape,np.uint8)
    for (y,x),links in neighbors.items():
        if len(links)>2:junction[y,x]=1
    from scipy.ndimage import distance_transform_edt
    jd=distance_transform_edt(junction==0);scale=max(shape)/496.;parameters=[dict(path_id=i,**regularize_parameters(profile_parameters(rgb,p,jd),scale)) for i,p in enumerate(fitted)];preview=render_parameters(shape,parameters);Image.fromarray(preview).save(output/'image_width_tone_preview.png');Image.fromarray(np.where(v4.raster(shape,fitted),0,255).astype(np.uint8)).save(output/'smooth_centerlines.png')
    svg=[]
    for r in curves:
        if r['segments']:
            start=r['segments'][0][0];d=f'M {start[0]:.4f} {start[1]:.4f}'
            for c in r['segments']:d+=' C '+' '.join(f'{v:.4f}' for p in c[1:] for v in p)
        else:
            p=r['polyline_fallback'];d='M '+' L '.join(' '.join(f'{v:.4f}' for v in q) for q in p)
        svg.append('<path d="'+d+'"/>')
    (output/'curves.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {shape[1]} {shape[0]}"><g fill="none" stroke="black" stroke-width="1" stroke-linecap="round">'+''.join(svg)+'</g></svg>',encoding='utf-8');jsave(output/'curves.json',dict(paths=curves,config=CONFIG,source_paths_sha256=digest(path_file),stroke_identity_verified=False,ready_for_robot=False));jsave(output/'image_brush_parameters.json',dict(paths=parameters,source_sha256=digest(image),width_definition='Image optical-density FWHM in pixels',ink_definition='Apparent local optical density from encoded RGB luma; not physical pigment concentration',unknown_rendering=dict(width_px_at_496=CONFIG['unknown_preview_width_at_496'],density=CONFIG['unknown_preview_density']),physical_z=None,physical_speed=None,water_content=None,actually_drawn=False,ready_for_robot=False))
    total=sum(len(r['valid']) for r in parameters);valid=sum(sum(r['valid']) for r in parameters);stats=dict(config_version=CONFIG['version'],paths=len(paths),bezier_segments=sum(len(r['segments']) for r in curves),polyline_fallbacks=sum(bool(r['polyline_fallback']) for r in curves),maximum_sampled_deviation_px=max([r['max_sampled_hausdorff_px'] for r in curves] or [0]),profile_samples=total,valid_profile_samples=valid,valid_profile_fraction=valid/max(1,total),width_px_range=[min([v for r in parameters for v in r['width_px'] if v is not None],default=0),max([v for r in parameters for v in r['width_px'] if v is not None],default=0)],physical_calibration=False,new_training_steps=0);jsave(output/'statistics.json',stats);return stats
