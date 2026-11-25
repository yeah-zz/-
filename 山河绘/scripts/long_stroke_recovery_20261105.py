# -*- coding: utf-8 -*-
"""Multi-scale endpoint graph with same-band overlap stitching.

No extraction pruning, semantic relabelling, forced cross-module connection,
pressure calibration or robot control. New bridges remain offline hypotheses.
"""
import math
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from module_stroke_core_20261104 import samples,length
from source_stroke_width_20261104 import sample
from hierarchical_paint_core_20261105 import source_fields,ridge_connector
from landscape_stroke_refine_20261104 import bridge_path

def point_at(p,distance):
    p=np.asarray(p,float);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    keep=np.r_[True,np.diff(arc)>1e-8]
    return np.array([np.interp(distance,arc[keep],p[keep,k]) for k in (0,1)])

def clip_path(p,start=0.,end=0.):
    p=np.asarray(p,float);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    stop=arc[-1]-end
    if start+1e-5>=stop:raise ValueError('Overlap trims consume an entire stroke')
    if start==end==0:return p.copy()
    middle=p[(arc>start+1e-8)&(arc<stop-1e-8)]
    return np.vstack([point_at(p,start),middle,point_at(p,stop)])

def endpoint(p):
    p=np.asarray(p,float);arc=length(p)
    vectors=[]
    for radius in (1.,2.5,5.):
        v=point_at(p,min(radius,arc*.75))-p[0];vectors.append(v/max(np.linalg.norm(v),1e-8))
    mean=np.mean(vectors,axis=0);mean/=max(np.linalg.norm(mean),1e-8)
    return dict(p=p[0],path=p,t=mean,vectors=np.asarray(vectors),arc=arc)

def overlap_stitch(ea,eb,gray,max_overlap=6.):
    """Trim each half of a shared segment, never backtrack over the same ink."""
    qa=samples(ea['path'],.18);qb=samples(eb['path'],.18)
    da,ia=cKDTree(qa).query(eb['p']);db,ib=cKDTree(qb).query(ea['p'])
    aa=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(qa,axis=0),axis=1))]
    ab=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(qb,axis=0),axis=1))]
    oa=float(aa[ia]);ob=float(ab[ib])
    if max(da,db)>.40 or min(oa,ob)<.65 or max(oa,ob)>max_overlap:return None
    if abs(oa-ob)>.7 or oa>ea['arc']*.8 or ob>eb['arc']*.8:return None
    part=qb[:ib+1];dist,ix=cKDTree(qa[:ia+3]).query(part)
    if dist.max()>.48 or (np.diff(ix)<=1).mean()<.98:return None
    # A visible light separation means adjacent strokes, even if very close.
    paired=qa[ix];vals=np.stack([sample(gray,part*(1-t)+paired*t) for t in np.linspace(0,1,9)],1)
    if (vals[:,1:-1].max(1)>np.maximum(vals[:,0],vals[:,-1])+.018).mean()>.02:return None
    ta=oa*.5;tb=ob*.5;a=point_at(ea['path'],ta);b=point_at(eb['path'],tb)
    if np.linalg.norm(a-b)>.48:return None
    return dict(trim_a=ta,trim_b=tb,bridge=np.linspace(a,b,5),overlap_px=(oa+ob)/2,
                maximum_overlap_deviation_px=float(dist.max()))

def connection_evidence(gray,contrast,ink,path):
    q=samples(path,.12)
    support=float(sample(ink.astype(np.float32),q).mean())
    if length(path)<.08:return support,1.
    t=np.gradient(q,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-8)
    normal=np.column_stack([-t[:,1],t[:,0]]);center=sample(gray,q)
    ridge=np.full(len(q),-1.,float)
    for radius in (1.,2.,3.):
        ridge=np.maximum(ridge,np.minimum(sample(gray,q+normal*radius),sample(gray,q-normal*radius))-center)
    return support,float((ridge>.004).mean())

def recover_long_strokes(rgb,rows,max_gap=8.):
    paths=[np.asarray(r['points'],float) for r in rows];gray,contrast,ink=source_fields(rgb)
    ends=[]
    for i,p in enumerate(paths):
        if length(p)<1 or np.linalg.norm(p[0]-p[-1])<.1:continue
        for side in (0,1):ends.append(dict(i=i,side=side,**endpoint(p if not side else p[::-1])))
    pairs=sorted(cKDTree([e['p'] for e in ends]).query_pairs(max_gap)) if ends else []
    degree=Counter();junction_options={}
    for a,b in pairs:
        if ends[a]['i']!=ends[b]['i'] and np.linalg.norm(ends[a]['p']-ends[b]['p'])<.7:degree[a]+=1;degree[b]+=1
        if ends[a]['i']==ends[b]['i'] or rows[ends[a]['i']]['module']!=rows[ends[b]['i']]['module']:continue
        if np.linalg.norm(ends[a]['p']-ends[b]['p'])<.45:
            facing=float(np.mean((ends[a]['vectors']*-ends[b]['vectors']).sum(1)))
            if facing>=math.cos(math.radians(50)):
                junction_options.setdefault(a,[]).append(facing);junction_options.setdefault(b,[]).append(facing)
    ambiguous_junctions={e for e,scores in junction_options.items() if len(scores)>1 and sorted(scores,reverse=True)[0]-sorted(scores,reverse=True)[1]<.18}
    candidates=[];rejected=Counter();pending=[]
    for a,b in pairs:
        ea,eb=ends[a],ends[b];i,j=ea['i'],eb['i']
        if i==j:continue
        if rows[i]['module']!=rows[j]['module']:rejected['different_module']+=1;continue
        # Keep geometric branch ambiguity even when antialiasing makes one
        # branch fail the ink threshold. Filtering evidence first can otherwise
        # turn a symmetric Y into a spurious unique continuation.
        if a in ambiguous_junctions or b in ambiguous_junctions:
            pending.append(dict(i=i,j=j,kind='junction',reason='geometric branch ambiguity before image filtering'));continue
        uncertain=bool(rows[i].get('uncertain_module') or rows[j].get('uncertain_module'))
        gap=float(np.linalg.norm(eb['p']-ea['p']))
        facing=float(np.mean((ea['vectors']*-eb['vectors']).sum(1)))
        corner=gap<.40 and degree[a]==degree[b]==1
        curved=not corner and gap>.55 and facing>=math.cos(math.radians(100)) and facing<math.cos(math.radians(50))
        if not curved and facing<math.cos(math.radians(105 if corner else 50)):
            rejected['inconsistent_direction']+=1;continue
        shared=overlap_stitch(ea,eb,gray) if facing>.94 else None
        if shared:
            bridge=shared['bridge'];trim_a=shared['trim_a'];trim_b=shared['trim_b'];kind='same_band_overlap'
        else:
            if gap>.55:
                axis=(eb['p']-ea['p'])/gap
                if min(-ea['t']@axis,eb['t']@axis)<(.45 if curved else .70):rejected['sideways_or_unverified_overlap']+=1;continue
            bridge=bridge_path(ea['p'],eb['p'],ea['t'],eb['t']) if curved else ridge_connector(ea['p'],eb['p'],ink,contrast)
            if bridge is None:rejected['no_supported_route']+=1;continue
            trim_a=trim_b=0.;kind='curved_source_supported_gap' if curved else 'source_supported_gap' if gap>.18 else 'touching_endpoints'
        support,ridge=connection_evidence(gray,contrast,ink,bridge)
        if curved:
            # Subpixel anti-aliasing gives fractional binary-mask values. Curved
            # candidates need virtually every sample near ink AND a dark ridge.
            q=samples(bridge,.12);support=float((sample(ink.astype(np.float32),q)>.5).mean())
            if support<.98 or ridge<.85:rejected['curve_without_continuous_dark_ridge']+=1;continue
        threshold=.97 if uncertain else .93
        if gap>.15 and support<threshold:rejected['insufficient_source_evidence']+=1;continue
        if uncertain and gap>.4 and ridge<.70:rejected['uncertain_without_dark_ridge']+=1;continue
        # At exact touching points source support may be weak due to subpixel
        # antialiasing; still require adjacent original endpoints to carry ink.
        if gap<=.15:
            probes=np.vstack([point_at(ea['path'],min(.6,ea['arc']/2)),point_at(eb['path'],min(.6,eb['arc']/2))])
            if sample(ink.astype(np.float32),probes).mean()<.7:rejected['touch_without_ink']+=1;continue
        score=.75*facing+.20*support+.05*ridge-.018*gap+(.08 if shared else 0)
        candidates.append(dict(a=a,b=b,i=i,j=j,gap_px=gap,score=score,support=support,ridge_support=ridge,
            kind=kind,trim_a=trim_a,trim_b=trim_b,bridge=bridge,uncertain_region=uncertain,
            overlap=shared))
    ranks={}
    for k,r in enumerate(candidates):
        for e in (r['a'],r['b']):ranks.setdefault(e,[]).append((r['score'],k))
    for v in ranks.values():v.sort(reverse=True)
    parent=list(range(len(paths)));links={};trims={};accepted=[]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for k,z in sorted(enumerate(candidates),key=lambda kv:-kv[1]['score']):
        a,b=z['a'],z['b']
        if a in links or b in links:continue
        if any(ranks[e][0][1]!=k or len(ranks[e])>1 and ranks[e][0][0]-ranks[e][1][0]<.12 for e in (a,b)):
            pending.append(dict(i=z['i'],j=z['j'],kind=z['kind'],reason='ambiguous competing continuations'));continue
        x,y=root(z['i']),root(z['j'])
        if x==y:rejected['new_cycle']+=1;continue
        if any(z[trimkey]+trims.get((ends[e]['i'],1-ends[e]['side']),0)>length(paths[ends[e]['i']])-.5
               for e,trimkey in ((a,'trim_a'),(b,'trim_b'))):
            rejected['overlap_consumes_short_curve']+=1;continue
        free=[e for e in range(len(ends)) if e not in links and e not in (a,b) and root(ends[e]['i']) in (x,y)]
        if len(free)==2 and np.linalg.norm(ends[free[0]]['p']-ends[free[1]]['p'])<.5:
            rejected['would_close_geometry']+=1;continue
        parent[y]=x;links[a]=(b,z['bridge']);links[b]=(a,z['bridge'][::-1])
        trims[ends[a]['i'],ends[a]['side']]=z['trim_a'];trims[ends[b]['i'],ends[b]['side']]=z['trim_b'];accepted.append(z)
    byend={(ends[a]['i'],ends[a]['side']):((ends[b]['i'],ends[b]['side']),p) for a,(b,p) in links.items()}
    visited=set();result=[]
    def walk(start):
        current=start;parts=[];ids=[]
        while current[0] not in visited:
            i,side=current;visited.add(i);ids.append(i)
            p=paths[i] if not side else paths[i][::-1]
            p=clip_path(p,trims.get((i,side),0),trims.get((i,1-side),0));parts.append(p)
            nex=byend.get((i,1-side))
            if nex is None:break
            current,bridge=nex;parts.append(bridge)
        result.append(dict(points=np.concatenate(parts),source_curve_ids=ids,module=rows[start[0]]['module'],
            uncertain_module=any(rows[i].get('uncertain_module',False) for i in ids),
            complete_input_ids=ids,artist_same_stroke_verified=False))
    for i in range(len(paths)):
        for side in (0,1):
            if i not in visited and (i,side) not in byend:walk((i,side))
    for i in range(len(paths)):
        if i not in visited:walk((i,0))
    assert sorted(i for r in result for i in r['source_curve_ids'])==list(range(len(paths)))
    return result,dict(input_count=len(paths),output_count=len(result),accepted=accepted,pending=pending,
        rejected=dict(rejected),types=dict(Counter(z['kind'] for z in accepted)),
        joins_in_uncertain_same_region=sum(z['uncertain_region'] for z in accepted),
        input_deleted=False,cross_module_joins_allowed=False,expert_stroke_identity_verified=False)
