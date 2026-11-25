# -*- coding: utf-8 -*-
"""Image-only ink graph, learned junction ownership and object-locked readout.

No reference lines, old extraction, gap bridging, controller or drawing imports.
The neural model predicts masks; these routines vectorize connected source ink.
Object and structural roles are proposals, not semantic/cun ground truth.
"""
from collections import deque, Counter
import numpy as np
import cv2
from scipy.ndimage import binary_propagation, binary_fill_holes, distance_transform_edt
from scipy.spatial import cKDTree
from ink_mask_centerline_20261105 import crop_fields, skeleton_graph
from module_stroke_core_20261104 import length, samples

CONFIG = dict(min_chain_length=1.25, query_min_length=2.5, query_spacing=32.,
              query_side=(72.,144.), ownership_threshold=.54, ownership_margin=.12,
              ownership_advantage=.08, minimum_votes=1, simplify_native_px=.35,
              short_closed_length=7., native_distance_scale='max_dimension/496',
              no_gap_bridging=True, automatic_prompts_only=True)


def source_ink(rgb):
    """Native-resolution common-channel ridges; does not claim ink ground truth."""
    scale=max(rgb.shape[:2])/496.
    # Source field radii are in the 496px comparison system, evaluated natively.
    work=rgb if scale<=1.01 else cv2.resize(rgb,None,fx=1/scale,fy=1/scale,interpolation=cv2.INTER_AREA)
    f=crop_fields(work,np.ones((96,96),np.float32),(0,0),max(work.shape[:2]),1)
    strong=f['strong'];weak=f['weak'];gray=f['gray']
    dark=(gray<.28)&(distance_transform_edt(~strong)<4)
    binary=binary_propagation(strong,mask=strong|weak|dark)
    holes=binary_fill_holes(binary)&~binary
    n,lab,stats,_=cv2.connectedComponentsWithStats(holes.astype(np.uint8),8)
    for j in range(1,n):
        reg=lab==j
        if stats[j,cv2.CC_STAT_AREA]<=12 and gray[reg].mean()<.40:binary[reg]=True
    # For high resolution sources, localize the coarse foreground to native ink.
    if scale>1.01:
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        rgbf=rgb.astype(np.float32)/255
        bg=cv2.GaussianBlur(rgbf,(0,0),2.2*scale)
        neutral=np.min(bg-rgbf,axis=2)
        contrast=cv2.GaussianBlur(gray,(0,0),2.2*scale)-gray
        foreground=cv2.resize(binary.astype(np.uint8),rgb.shape[1::-1],interpolation=cv2.INTER_NEAREST)>0
        owner=cv2.dilate(foreground.astype(np.uint8),np.ones((3,3),np.uint8))>0
        seed=owner&(contrast>.008)&(neutral>.002)
        weak_native=owner&(((contrast>.003)&(neutral>.0005))|(gray<.28))
        binary=binary_propagation(seed,mask=seed|weak_native)
    return binary.astype(bool)


def ink_graph(binary):
    xy,adj,sk=skeleton_graph(binary)
    degree=np.array([len(a) for a in adj]);node_of={};nodes=[]
    # Collapse adjacent junction pixels, not their outgoing strokes.
    candidates=set(np.flatnonzero(degree>=3).tolist())
    while candidates:
        first=min(candidates);stack=[first];candidates.remove(first);members=[]
        while stack:
            a=stack.pop();members.append(a)
            for b in adj[a]:
                if b in candidates:candidates.remove(b);stack.append(b)
        nid=len(nodes);nodes.append(members)
        for a in members:node_of[a]=nid
    for a in np.flatnonzero(degree<=1):
        node_of[int(a)]=len(nodes);nodes.append([int(a)])
    visited=set();chains=[]
    def edge(a,b):return (min(a,b),max(a,b))
    for nid,members in enumerate(nodes):
        for a in members:
            for b in adj[a]:
                if node_of.get(b)==nid or edge(a,b) in visited:continue
                ids=[a,b];visited.add(edge(a,b));last,cur=a,b
                while cur not in node_of:
                    options=[v for v in adj[cur] if v!=last]
                    if not options:break
                    nxt=options[0]
                    if edge(cur,nxt) in visited:break
                    visited.add(edge(cur,nxt));ids.append(nxt);last,cur=cur,nxt
                chains.append(dict(points=xy[ids],nodes=[nid,node_of.get(cur,-1)],pixel_ids=ids,closed=False))
    # Components containing only degree-two pixels are actual source loops.
    seen_pixels=set(a for r in chains for a in r['pixel_ids'])|set(node_of)
    for start in range(len(xy)):
        if start in seen_pixels:continue
        ids=[start];prev=-1;cur=start
        while True:
            seen_pixels.add(cur);options=[n for n in adj[cur] if n!=prev]
            if not options:break
            nxt=options[0]
            if nxt==start:ids.append(start);break
            if nxt in seen_pixels:break
            ids.append(nxt);prev,cur=cur,nxt
        if len(ids)>2:chains.append(dict(points=xy[ids],nodes=[-1,-1],pixel_ids=ids,closed=ids[-1]==start))
    return dict(xy=xy,adj=adj,nodes=nodes,chains=chains,mask=binary,skeleton=sk)


def automatic_queries(graph,canonical_factor):
    """All hints derived from RGB ink graph; no baseline or manual bbox access."""
    queries=[]
    for cid,row in enumerate(graph['chains']):
        p=row['points'];arc=length(p)/canonical_factor
        if row['closed'] or arc<CONFIG['query_min_length']:continue
        q=samples(p,.5*canonical_factor)
        steps=max(1,int(np.ceil(arc/CONFIG['query_spacing'])))
        indices=np.linspace(0,len(q)-1,steps+2,dtype=int)[1:-1]
        for k in indices:
            back=q[max(0,k-3)];front=q[min(len(q)-1,k+3)]
            if np.linalg.norm(front-back)<.5*canonical_factor:continue
            for side in CONFIG['query_side']:
                queries.append(dict(center=q[k],side=side*canonical_factor,back=back,front=front,
                    chain=cid,origin='source_ink_skeleton',reference_used=False))
    return queries


def mask_at(pred,points):
    p=(np.asarray(points)-pred['origin'])/pred['side']*pred['mask'].shape[0]-.5
    return cv2.remap(pred['mask'].astype(np.float32),p[:,0].astype(np.float32).reshape(1,-1),
        p[:,1].astype(np.float32).reshape(1,-1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).ravel()


def junction_votes(graph,predictions,chain_modules,factor):
    """Require two-sided instance evidence, unambiguous preference and same block.

    Geometry alone is never sufficient for an ownership join. No new ink is made.
    """
    chains=graph['chains'];bychain={};incidence={}
    for p in predictions:
        if p['presence']>=.5:bychain.setdefault(p['query']['chain'],[]).append(p)
    for i,c in enumerate(chains):
        if c['closed']:continue
        for side,n in enumerate(c['nodes']):
            if n>=0:incidence.setdefault(n,[]).append((i,side))
    links={};diagnostics=[]
    for node,ends in incidence.items():
        if len(ends)<2:continue
        # Tiny source spurs remain pending, rather than being used to force forks.
        active=[e for e in ends if length(chains[e[0]]['points'])>=CONFIG['query_min_length']*factor]
        profiles={}
        for e in active:
            p=chains[e[0]]['points'];p=p if e[1]==0 else p[::-1]
            q=samples(p,.5*factor);profiles[e]=q[1:min(len(q),19)]
        votes={}
        for a in active:
            own=profiles[a]
            for b in active:
                if a==b:continue
                key=(a,b)
                if a[0]==b[0] or chain_modules[a[0]]!=chain_modules[b[0]]:votes[key]=0.;continue
                scores=[]
                for pred in bychain.get(a[0],[]):
                    if len(own)==0 or len(profiles[b])==0:continue
                    selfscore=float(np.mean(mask_at(pred,own)))
                    if selfscore<CONFIG['ownership_threshold']:continue
                    scores.append(float(np.mean(mask_at(pred,profiles[b]))))
                votes[key]=max(scores,default=0.)
        proposals=[]
        for ai,a in enumerate(active):
            for b in active[ai+1:]:
                score=min(votes[a,b],votes[b,a])
                alternatives=max([min(votes.get((a,c),0),votes.get((c,a),0)) for c in active if c not in (a,b)]+
                                 [min(votes.get((b,c),0),votes.get((c,b),0)) for c in active if c not in (a,b)]+[0.])
                margin=score-alternatives
                if score>=CONFIG['ownership_threshold'] and margin>=CONFIG['ownership_margin']:
                    proposals.append((score,margin,a,b))
        used=set();accepted=[]
        for score,margin,a,b in sorted(proposals,reverse=True):
            if a in used or b in used:continue
            links[a]=b;links[b]=a;used.update((a,b))
            accepted.append(dict(first=list(a),second=list(b),mutual_owner_score=score,margin=margin))
        diagnostics.append(dict(node=node,incident=len(ends),eligible=len(active),accepted=accepted,
            status='paired_by_frozen_model' if accepted else 'unresolved_junction',
            source_pixels=graph['xy'][graph['nodes'][node]].tolist(),
            votes=[dict(first=list(a),second=list(b),score=s) for (a,b),s in votes.items()]))
    return links,diagnostics


def internal_route(graph,node,start,goal):
    if start==goal:return [start]
    allowed=set(graph['nodes'][node]);queue=deque([start]);parent={start:None}
    while queue:
        a=queue.popleft()
        for b in graph['adj'][a]:
            if b not in allowed or b in parent:continue
            parent[b]=a;queue.append(b)
            if b==goal:
                out=[b]
                while parent[out[-1]] is not None:out.append(parent[out[-1]])
                return out[::-1]
    raise ValueError('Junction route must stay inside the measured connected skeleton')


def read_paths(graph,links,chain_modules,factor):
    chains=graph['chains'];seen=set();rows=[];pending=[]
    def walk(i,side):
        pieces=[];ids=[];uncertain=False
        while i not in seen:
            seen.add(i);r=chains[i];ids.append(i);p=r['points'] if side==0 else r['points'][::-1];pieces.append(p)
            nextend=links.get((i,1-side))
            if nextend is None:break
            j,jside=nextend
            if j in seen:uncertain=True;break
            node=r['nodes'][1-side]
            first=r['pixel_ids'][-1] if side==0 else r['pixel_ids'][0]
            second=chains[j]['pixel_ids'][0] if jside==0 else chains[j]['pixel_ids'][-1]
            route=internal_route(graph,node,first,second);pieces.append(graph['xy'][route]);i,side=j,jside
        p=np.concatenate(pieces)
        p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-8]]
        closed=chains[ids[0]]['closed'] or uncertain
        record=dict(points=p,module=chain_modules[ids[0]],source_curve_ids=ids,
                    uncertain_module=False,closed=closed,artist_stroke_identity_verified=False,
                    learned_junctions=max(0,len(ids)-1),raw_graph_chain_ids=ids)
        if length(p)<CONFIG['min_chain_length']*factor or (closed and length(p)<CONFIG['short_closed_length']*factor):
            record['reason']='short_closed_ink_texture' if closed else 'subpixel_short_ink';pending.append(record)
        else:
            smooth=cv2.approxPolyDP(p.astype(np.float32),CONFIG['simplify_native_px'],closed).reshape(-1,2)
            if closed and len(smooth)>2:smooth=np.vstack([smooth,smooth[0]])
            if len(smooth)>1:record['points']=smooth.astype(float);rows.append(record)
    for i,c in enumerate(chains):
        if i in seen:continue
        if c['closed']:walk(i,0);continue
        free=[side for side in (0,1) if (i,side) not in links]
        if free:walk(i,free[0])
    for i in range(len(chains)):
        if i not in seen:walk(i,0)
    assert sorted(i for r in rows+pending for i in r['source_curve_ids'])==list(range(len(chains)))
    return rows,pending


def curve_metrics(rows,factor=1.):
    arcs=np.array([length(r['points'])/factor for r in rows])
    return dict(count=len(rows),short_under_5px=int((arcs<5).sum()),long_over_25px=int((arcs>25).sum()),
                centerline_length_px=float(arcs.sum()),median_length_px=float(np.median(arcs)) if len(arcs) else 0.)


def compare_geometry(baseline,rows,tolerance=1.5):
    """Coverage is relative to old candidates, NOT true extraction accuracy."""
    if not baseline or not rows:return dict(baseline_reference_only=True,retained_fraction=0.)
    a=np.concatenate([samples(r['points'],.4) for r in baseline]);b=np.concatenate([samples(r['points'],.4) for r in rows])
    return dict(baseline_reference_only=True,not_ground_truth_accuracy=True,tolerance_496px=tolerance,
                old_samples_near_new=float((cKDTree(b).query(a)[0]<=tolerance).mean()),
                new_samples_near_old=float((cKDTree(a).query(b)[0]<=tolerance).mean()))
