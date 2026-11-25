"""Conservative source-evidence recovery after frozen v4, no neural weights.

Existing selected paths are immutable. Short source chains are proposals, not
artist stroke labels. Multi-direction short-line bundles protect fine details.
"""
import sys
sys.dont_write_bytecode=True
import numpy as np, cv2
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
import split_detail_v4_frozen as v4

CFG=dict(version='v4_local_short_evidence_r2_cycle_guard',fan_window_fraction=.0665,
         fan_min_chains=12,fan_direction_dispersion=.16,fan_min_length=2.4,
         fan_max_length=18.,minimum_novel_fraction=.65,
         minimum_gray_contrast=.012,minimum_supported_fraction=.65,
         standalone_length=8.,standalone_straightness=.82,
         maximum_fan_length=24.,fan_straightness=.82,tested_min_side=400,tested_max_side=600,
         forbid_new_pixel_graph_cycles=True)

def graph_cycles(m):
    a=m[1:,1:];b=m[:-1,:-1];c=m[1:,:-1];d=m[:-1,1:]
    edges=int((m[:,1:]&m[:,:-1]).sum()+(m[1:]&m[:-1]).sum()+(a&b&~c&~d).sum()+(c&d&~a&~b).sum())
    components=cv2.connectedComponents(m.astype(np.uint8),connectivity=8)[0]-1
    return edges-int(m.sum())+components

def samples(gray,p):
    p=np.asarray(p,np.float32)
    return cv2.remap(gray,p[:,0].reshape(-1,1),p[:,1].reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101).ravel()

def evidence(gray,p,scale):
    # Dark center compared with BOTH cross-sectional shoulders. No blank links.
    tangent=np.gradient(p,axis=0);norm=np.linalg.norm(tangent,axis=1,keepdims=True)
    tangent=tangent/np.maximum(norm,1e-6);normal=np.column_stack([-tangent[:,1],tangent[:,0]])
    center=samples(gray,p)
    shoulder=[]
    for radius in [1.5,2.5,4.]:
        a=samples(gray,p+normal*radius*scale);b=samples(gray,p-normal*radius*scale)
        shoulder.append(np.minimum(a,b)-center)
    contrast=np.max(shoulder,axis=0)
    return contrast,center

def recover(rgb,base_paths,raw,cfg=CFG):
    shape=raw.shape;h,w=shape;scale=max(h,w)/496.
    # Only the small-image short-line failure has been evaluated this round.
    # Leave native large-image v4 extraction untouched, including accepted pine.
    if not cfg['tested_min_side']<=max(h,w)<=cfg['tested_max_side']:
        return [],np.zeros(shape,bool),[dict(accepted=False,reason='outside_evaluated_size_range_keep_v4')]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
    chains,neighbors=v4.pixel_chains(raw)
    old=v4.raster(shape,[np.asarray(p['points'],np.float32) for p in base_paths])
    distance=distance_transform_edt(~old)
    count=np.zeros(shape,np.float32);vx=count.copy();vy=count.copy()
    desc=[]
    for p in chains:
        length=v4.plen(p);span=float(np.linalg.norm(p[-1]-p[0]));straight=span/max(length,1e-6)
        contrast,center=evidence(gray,p,scale)
        supported=float(((contrast>=cfg['minimum_gray_contrast'])&(center<.72)).mean())
        ij=np.rint(p).astype(int);novel=float((distance[ij[:,1],ij[:,0]]>1.1).mean())
        desc.append((p,length,straight,supported,novel,float(np.median(contrast))))
        if cfg['fan_min_length']*scale<=length<=cfg['fan_max_length']*scale and straight>=.78 and supported>=.6:
            x,y=np.rint(p.mean(0)).astype(int);t=v4.unit(p[-1]-p[0]);count[y,x]+=1
            vx[y,x]+=t[0]*t[0]-t[1]*t[1];vy[y,x]+=2*t[0]*t[1]
    win=max(9,int(round(cfg['fan_window_fraction']*max(h,w)))|1)
    count=cv2.boxFilter(count,-1,(win,win),normalize=False)
    vx=cv2.boxFilter(vx,-1,(win,win),normalize=False);vy=cv2.boxFilter(vy,-1,(win,win),normalize=False)
    dispersion=1-np.sqrt(vx*vx+vy*vy)/np.maximum(count,1)
    fan=(count>=cfg['fan_min_chains'])&(dispersion>=cfg['fan_direction_dispersion'])
    rad=max(3,int(round(5*scale))|1)
    fan=cv2.dilate(fan.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(rad,rad)))>0
    additions=[];decisions=[];accepted_mask=old.copy();cycle_count=graph_cycles(old)
    for k,(p,length,straight,supported,novel,contrast) in enumerate(desc):
        if novel<cfg['minimum_novel_fraction'] or length<2.4*scale:continue
        # Never recover a closed skeleton ring as if it were a brush centerline.
        if np.linalg.norm(p[-1]-p[0])<2.*scale:continue
        ij=np.rint(p).astype(int);bundle=float(fan[ij[:,1],ij[:,0]].mean())
        in_fan=bundle>=.5 and length<=cfg['maximum_fan_length']*scale and straight>=cfg['fan_straightness']
        isolated=length>=cfg['standalone_length']*scale and straight>=cfg['standalone_straightness']
        accepted=supported>=cfg['minimum_supported_fraction'] and (in_fan or isolated)
        reason='image_profile_and_short_line_geometry' if accepted else 'insufficient_profile_or_geometry'
        if accepted and cfg['forbid_new_pixel_graph_cycles']:
            proposal=accepted_mask.copy();cv2.polylines(proposal.view(np.uint8),[ij.astype(np.int32)],False,1,1,cv2.LINE_8)
            proposed_cycles=graph_cycles(proposal)
            if proposed_cycles>cycle_count:
                accepted=False;reason='would_create_new_pixel_graph_cycle_keep_pending'
            else:accepted_mask=proposal;cycle_count=proposed_cycles
        if accepted:
            additions.append(dict(role='source_supported_short_detail' if in_fan else 'source_supported_short_structure',points=p.tolist(),stroke_identity_verified=False))
        decisions.append(dict(chain=k,length=length,straightness=straight,supported_fraction=supported,novel_fraction=novel,median_cross_section_contrast=contrast,fan_fraction=bundle,accepted=bool(accepted),reason=reason))
    return additions,fan,decisions
