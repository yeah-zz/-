# -*- coding: utf-8 -*-
"""Constrained source-backed joining of unpainted curves. Not a trained policy."""
import math
import numpy as np,cv2
from scipy.spatial import cKDTree

def length(p):return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
def sample(a,p):
    p=np.asarray(p,np.float32)
    return cv2.remap(a,p[:,0:1],p[:,1:2],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).ravel()
def inward(p):
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    at=min(3.,arc[-1]);q=np.array([np.interp(at,arc,p[:,k]) for k in [0,1]])-p[0]
    return q/max(1e-8,np.linalg.norm(q))
def raster(paths,shape):
    a=np.zeros(shape,np.uint8)
    for p in paths:cv2.polylines(a,[np.rint(p).astype(np.int32)],False,1,1)
    return a

def join_remaining(paths,gray,drawn,mm_per_px,maximum_mm=19.8):
    paths=[np.asarray(p,float) for p in paths];lens=np.array([length(p) for p in paths]);shape=gray.shape
    bg=cv2.GaussianBlur(gray,(0,0),3.)
    ink=((bg-gray>.008)|(gray<.4)).astype(np.float32)
    forbidden=cv2.dilate(drawn.astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32)
    ends=[]
    for i,p in enumerate(paths):
        if np.linalg.norm(p[0]-p[-1])<.1 or lens[i]<.7:continue
        for side in [0,1]:ends.append((i,side,p[0] if side==0 else p[-1]))
    candidates=[];rejected={k:0 for k in ['same_curve','angle','axis','white_gap','drawn_region','length','ambiguous','cycle']}
    if ends:
        tree=cKDTree([e[2] for e in ends]);pairs=tree.query_pairs(3.)
    else:pairs=[]
    for a,b in sorted(pairs):
        i,si,x=ends[a];j,sj,y=ends[b]
        if i==j:rejected['same_curve']+=1;continue
        if (lens[i]+lens[j]+np.linalg.norm(x-y))*mm_per_px>maximum_mm:rejected['length']+=1;continue
        p=paths[i] if si==0 else paths[i][::-1];q=paths[j] if sj==0 else paths[j][::-1]
        ta=inward(p);tb=inward(q);facing=float(ta@(-tb))
        if facing<math.cos(math.radians(35)):rejected['angle']+=1;continue
        gap=float(np.linalg.norm(x-y));bridge=np.linspace(x,y,max(3,int(gap*8)+1))
        if sample(forbidden,bridge).max()>.01:rejected['drawn_region']+=1;continue
        axis_score=1.
        if gap>.05:
            axis=(y-x)/gap;axis_score=min(float((-ta)@axis),float(tb@axis))
            if axis_score<math.cos(math.radians(35)):rejected['axis']+=1;continue
            support=float((sample(ink,bridge)>.5).mean())
            if support<.95:rejected['white_gap']+=1;continue
        else:support=1.
        score=.5*facing+.35*axis_score+.15*support-.03*gap
        candidates.append(dict(a=a,b=b,i=i,j=j,si=si,sj=sj,gap=gap,score=score,support=support))
    ranks={}
    for k,r in enumerate(candidates):
        for e in [r['a'],r['b']]:ranks.setdefault(e,[]).append((r['score'],k))
    for e in ranks:ranks[e].sort(reverse=True)
    linked={};decisions=[];parent=list(range(len(paths)));sizes=lens.copy()
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for k,r in sorted(enumerate(candidates),key=lambda v:-v[1]['score']):
        a,b=r['a'],r['b']
        if a in linked or b in linked:continue
        if any(ranks[e][0][1]!=k or len(ranks[e])>1 and ranks[e][0][0]-ranks[e][1][0]<.10 for e in [a,b]):rejected['ambiguous']+=1;continue
        x,y=find(r['i']),find(r['j'])
        if x==y:rejected['cycle']+=1;continue
        if (sizes[x]+sizes[y]+r['gap'])*mm_per_px>maximum_mm:rejected['length']+=1;continue
        parent[y]=x;sizes[x]+=sizes[y]+r['gap'];linked[a]=b;linked[b]=a;decisions.append(r)
    byend={ends[a][:2]:ends[b][:2] for a,b in linked.items()};visited=set();result=[]
    def walk(start):
        current=start;parts=[];ids=[];directions=[]
        while current[0] not in visited:
            i,side=current;visited.add(i);p=paths[i] if side==0 else paths[i][::-1]
            if parts and np.linalg.norm(parts[-1][-1]-p[0])<1e-7:p=p[1:]
            parts.append(p);ids.append(i);directions.append(side)
            if (i,1-side) not in byend:break
            current=byend[(i,1-side)]
        result.append(dict(points=np.concatenate(parts),old_remaining_indices=ids,reversed=directions))
    for i in range(len(paths)):
        for side in [0,1]:
            if i not in visited and (i,side) not in byend:walk((i,side))
    for i in range(len(paths)):
        if i not in visited:walk((i,0))
    assert sorted(i for r in result for i in r['old_remaining_indices'])==list(range(len(paths)))
    old=raster(paths,shape);new=raster([r['points'] for r in result],shape)
    # Appending segments in reverse order must preserve every original pixel.
    # A single-pixel line-raster tie can differ with direction; check .5px-ish
    # neighborhood separately and report the exact raster discrepancy as well.
    exact_missing=int(np.count_nonzero((old>0)&(new==0)))
    near=cv2.dilate(new,np.ones((3,3),np.uint8))
    if np.any((old>0)&(near==0)):raise ValueError('Original geometry coverage lost')
    return result,dict(accepted=decisions,rejections=rejected,candidate_pairs=len(candidates),
        original_paths=len(paths),output_paths=len(result),no_original_segment_deleted=True,
        exact_raster_tie_difference_pixels=exact_missing,coverage_within_one_pixel=True,
        algorithm='geometry/source-supported mutual-best join, not trained complete-stroke recognition',
        artist_stroke_identity_verified=False,maximum_joined_length_mm=maximum_mm)
