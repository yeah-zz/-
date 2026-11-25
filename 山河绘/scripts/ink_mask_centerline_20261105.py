# -*- coding: utf-8 -*-
"""Read centerlines inside a predicted instance mask and measured source ink.

This is deterministic vectorization, NOT the failed neural coordinate head.
No endpoint joining across components, unknown branch guessing or robot code.
"""
import heapq
import numpy as np,cv2
from scipy.ndimage import binary_propagation,binary_fill_holes,distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from module_stroke_core_20261104 import length

CONFIG=dict(scale=2,model_strong=.60,model_weak=.30,contrast_strong=.008,contrast_weak=.003,
            neutral_strong=.002,neutral_weak=.0005,ridge_strong=.005,ridge_weak=.002,
            max_seed_distance_px=3.,branch_margin=.15,lookahead_px=3.,simplify_px=.18)
CONFIG.update(dark_core_gray=.28,dark_core_distance_px=4.,maximum_dark_hole_area_px=12.,
              dark_hole_gray=.40,posthoc_dark_core_fix=True,ownership_localization_tolerance_px=1.25)

def crop_fields(rgb,prob,origin,side,scale=2):
    origin=np.asarray(origin,float);h,w=rgb.shape[:2]
    x0=max(0,int(np.floor(origin[0])));y0=max(0,int(np.floor(origin[1])))
    x1=min(w,int(np.ceil(origin[0]+side)));y1=min(h,int(np.ceil(origin[1]+side)))
    if x1-x0<3 or y1-y0<3:return None
    yy,xx=np.mgrid[:(y1-y0)*scale,:(x1-x0)*scale]
    gx=x0+(xx+.5)/scale;gy=y0+(yy+.5)/scale
    color=cv2.remap(rgb,gx.astype(np.float32),gy.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
    model=cv2.remap(prob,((gx-origin[0])/side*prob.shape[1]-.5).astype(np.float32),
        ((gy-origin[1])/side*prob.shape[0]-.5).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    rgbf=color.astype(np.float32)/255;gray=cv2.cvtColor(color,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    blurred=cv2.GaussianBlur(rgbf,(0,0),2.2*scale);neutral=np.min(blurred-rgbf,axis=2)
    contrast=cv2.GaussianBlur(gray,(0,0),2.2*scale)-gray
    ridge=np.full(gray.shape,-1,np.float32)
    for r in (1.,2.,4.,6.):
        for dx,dy in ((1.,0.),(0.,1.),(.707,.707),(.707,-.707)):
            a=cv2.remap(gray,(xx+r*scale*dx).astype(np.float32),(yy+r*scale*dy).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
            b=cv2.remap(gray,(xx-r*scale*dx).astype(np.float32),(yy-r*scale*dy).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
            ridge=np.maximum(ridge,np.minimum(a,b)-gray)
    strong=(contrast>CONFIG['contrast_strong'])&(neutral>CONFIG['neutral_strong'])&(ridge>CONFIG['ridge_strong'])
    weak=(contrast>CONFIG['contrast_weak'])&(neutral>CONFIG['neutral_weak'])&(ridge>CONFIG['ridge_weak'])
    return dict(color=color,gray=gray,model=model,strong=strong,weak=weak,ridge=ridge,
                origin=np.array([x0,y0]),scale=scale)

def skeleton_graph(binary):
    sk=skeletonize(binary);yx=np.argwhere(sk);lookup={tuple(p):i for i,p in enumerate(yx)}
    adjacency=[[] for _ in yx]
    for i,(y,x) in enumerate(yx):
        for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
            j=lookup.get((y+dy,x+dx))
            if j is None:continue
            # Avoid diagonal triangles around an orthogonal corner.
            if dy and dx and ((y,x+dx) in lookup or (y+dy,x) in lookup):continue
            adjacency[i].append(j);adjacency[j].append(i)
    return yx[:,::-1].astype(float),adjacency,sk

def dominant_geodesic(xy,adj,start,direction,prob,binary,scale,use_model):
    """A full path through the query neighborhood; ambiguous forks stay flagged.

    Small skeleton spurs must not terminate a visible whole ink ribbon. This
    readout uses only connected skeleton edges; it never fills a source gap.
    """
    distance=np.full(len(xy),np.inf);distance[start]=0;parent=np.full(len(xy),-1,int)
    width=distance_transform_edt(binary);heap=[(0.,start)]
    while heap:
        cost,i=heapq.heappop(heap)
        if cost>distance[i]+1e-9:continue
        for j in adj[i]:
            pos=np.rint(xy[j]).astype(int);owner=float(prob[pos[1],pos[0]]) if use_model else 1.
            step=np.linalg.norm(xy[j]-xy[i])*(1+.4*(1-owner))/(1+.2*width[pos[1],pos[0]])
            value=cost+step
            if value<distance[j]:distance[j]=value;parent[j]=i;heapq.heappush(heap,(value,j))
    endpoints=[i for i in range(len(xy)) if len(adj[i])==1 and np.isfinite(distance[i])]
    if len(endpoints)<2:return [],'cycle_or_no_endpoints',0.
    endpoints=sorted(endpoints,key=lambda i:-distance[i])[:24]
    paths={}
    for e in endpoints:
        path=[e]
        while path[-1]!=start and parent[path[-1]]>=0:path.append(int(parent[path[-1]]))
        paths[e]=path[::-1]
    candidates=[]
    for aidx,a in enumerate(endpoints):
        for b in endpoints[aidx+1:]:
            pa,pb=paths[a],paths[b];common=0
            while common<min(len(pa),len(pb)) and pa[common]==pb[common]:common+=1
            root=common-1;ids=pa[root:][::-1]+pb[root+1:]
            if len(ids)<3:continue
            q=xy[ids];near=int(np.linalg.norm(q-xy[start],axis=1).argmin());snap=float(np.linalg.norm(q[near]-xy[start]))
            if snap>3*scale:continue
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
            lo=max(0,int(np.searchsorted(arc,arc[near]-3*scale)));hi=min(len(q)-1,int(np.searchsorted(arc,arc[near]+3*scale)))
            tangent=q[hi]-q[lo];tangent/=max(np.linalg.norm(tangent),1e-8)
            alignment=abs(float(tangent@direction))
            if alignment<.6:continue
            pos=np.rint(q).astype(int);owner=float(prob[pos[:,1],pos[:,0]].mean()) if use_model else 1.
            score=arc[-1]*(.65+.35*owner)*(.75+.25*alignment)
            candidates.append(dict(ids=ids,score=score,snap=snap,nodes=set(ids)))
    if not candidates:return [],'query_direction_unsupported',0.
    candidates.sort(key=lambda r:-r['score']);best=candidates[0];ambiguous=False
    for other in candidates[1:]:
        # Endpoint variants on tiny spurs are the same main ribbon candidate.
        shared=len(best['nodes']&other['nodes'])/max(1,len(best['nodes']|other['nodes']))
        if shared>.85:continue
        if (best['score']-other['score'])/max(best['score'],1e-8)<CONFIG['branch_margin']:
            ambiguous=True
        break
    return best['ids'],'ambiguous_branch' if ambiguous else 'source_mask_readout',best['snap']/scale

def read_instance(rgb,prob,origin,side,back,front,mode='model_source'):
    """Modes distinguish model-only mask, source-only control and their fusion."""
    f=crop_fields(rgb,np.asarray(prob,np.float32),origin,side,CONFIG['scale'])
    if f is None:return dict(points=np.empty((0,2)),status='empty_crop',eligible_for_replacement=False)
    p=f['model'];scale=f['scale']
    if mode=='model_only':binary=p>.5
    elif mode in ('model_source','source_only'):
        radius=int(np.ceil(CONFIG['ownership_localization_tolerance_px']*scale))
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
        strong_owner=cv2.dilate((p>CONFIG['model_strong']).astype(np.uint8),kernel)>0
        weak_owner=cv2.dilate((p>CONFIG['model_weak']).astype(np.uint8),kernel)>0
        seed=f['strong'] if mode=='source_only' else f['strong']&strong_owner
        # Broad opaque ink may have zero local contrast at its centre. Restore
        # only dark pixels near measured strong ink, never seed uniform darkness.
        nearby=distance_transform_edt(~f['strong'])<CONFIG['dark_core_distance_px']*scale
        dark=(f['gray']<CONFIG['dark_core_gray'])&nearby
        weak=f['weak']|dark
        allowed=weak if mode=='source_only' else weak&weak_owner
        binary=binary_propagation(seed,mask=allowed|seed)
        holes=binary_fill_holes(binary)&~binary
        count,regions,regions_stats,_=cv2.connectedComponentsWithStats(holes.astype(np.uint8),8)
        for hidx in range(1,count):
            region=regions==hidx
            if regions_stats[hidx,cv2.CC_STAT_AREA]<=CONFIG['maximum_dark_hole_area_px']*scale*scale and f['gray'][region].mean()<CONFIG['dark_hole_gray']:
                binary[region]=True
    else:raise ValueError('Unknown readout mode')
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary.astype(np.uint8),8)
    middle=(np.asarray(back)+np.asarray(front))/2
    seedpoint=(middle-f['origin'])*scale-.5
    yy,xx=np.where(binary)
    if not len(xx):return dict(points=np.empty((0,2)),status='no_supported_ink',eligible_for_replacement=False,fields=f,binary=binary)
    coords=np.column_stack([xx,yy]);distances=np.linalg.norm(coords-seedpoint,axis=1)
    nearest=int(distances.argmin())
    if distances[nearest]>CONFIG['max_seed_distance_px']*scale:
        return dict(points=np.empty((0,2)),status='query_not_on_supported_ink',eligible_for_replacement=False,fields=f,binary=binary)
    chosen=labels[yy[nearest],xx[nearest]];component=labels==chosen
    xy,adj,sk=skeleton_graph(component)
    if len(xy)<3:return dict(points=np.empty((0,2)),status='tiny_component',eligible_for_replacement=False,fields=f,binary=component)
    start=int(cKDTree(xy).query(seedpoint)[1]);direction=np.asarray(front,float)-np.asarray(back,float)
    direction/=max(np.linalg.norm(direction),1e-8)
    ids,path_status,seed_snap=dominant_geodesic(xy,adj,start,direction,p,component,scale,mode!='source_only')
    if not ids:return dict(points=np.empty((0,2)),status=path_status,eligible_for_replacement=False,fields=f,binary=component)
    points=f['origin']+(xy[ids]+.5)/scale
    if len(points)>2:
        smooth=cv2.approxPolyDP(points.astype(np.float32),CONFIG['simplify_px'],False).reshape(-1,2)
        if len(smooth)>1:points=smooth.astype(float)
    if len(points)<2 or length(points)<2:
        return dict(points=np.empty((0,2)),status='too_short',eligible_for_replacement=False,fields=f,binary=component)
    ends=xy[[ids[0],ids[-1]]];shape=np.array(binary.shape[::-1])-1
    boundary=bool(np.any((ends<2*scale)|(ends>shape-2*scale)))
    ambiguous=path_status=='ambiguous_branch'
    status='ambiguous_branch' if ambiguous else 'crop_boundary' if boundary else 'source_mask_readout'
    return dict(points=points,status=status,eligible_for_replacement=not ambiguous and not boundary,
        seed_snap_px=seed_snap,readout='connected_skeleton_geodesic_through_query_neighborhood',
        fields=f,binary=component,mode=mode,artist_stroke_identity_verified=False)
