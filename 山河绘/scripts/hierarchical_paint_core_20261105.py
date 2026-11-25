# -*- coding: utf-8 -*-
"""Source-supported continuation and object-locked, coarse-to-fine planning.

Image coordinates only. Masks and structural roles are unverified proposals.
No learned cun labels, pressure values, controller connection or execution.
"""
import heapq
import math
from collections import Counter
import cv2
import numpy as np
from scipy.spatial import cKDTree
from module_stroke_core_20261104 import samples, length, raster
from source_stroke_width_20261104 import sample
from landscape_stroke_refine_20261104 import deduplicate

PHASES=['轮廓／骨架候选','内部长线／皴线候选','细节／待定']

def source_fields(rgb):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    contrast=cv2.GaussianBlur(g,(0,0),2.2)-g
    # Relative local evidence protects faint ink; no global dark threshold alone.
    hp=np.abs(g-cv2.GaussianBlur(g,(0,0),.65))
    noise=cv2.GaussianBlur(cv2.medianBlur(hp,5),(0,0),2.)
    ink=(contrast>np.maximum(.004,noise*.9))|(g<.32)
    return g,contrast,ink

def object_ownership(rgb,paths,masks,records):
    """Reuse automatic masks; never invent semantic rock/tree labels."""
    h,w=rgb.shape[:2];g,_,_=source_fields(rgb)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    fg=(g<.88)|(hsv[:,:,1]>35)
    proposals=[]
    for r in records:
        m=cv2.resize(masks[r['key']].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        area=int(m.sum());ys,xs=np.where(m)
        if not len(xs) or not 420<=area<=h*w*.40:continue
        bw=xs.max()-xs.min()+1;bh=ys.max()-ys.min()+1
        if max(bw,bh)/max(1,min(bw,bh))>20 or fg[m].mean()<.25:continue
        if any((m&p['mask']).sum()/max(1,(m|p['mask']).sum())>.88 for p in proposals):continue
        proposals.append(dict(mask=m,area=area,key=r['key']))
    # Remove clearly nested parts, retaining their enclosing object proposal.
    roots=[]
    for i,p in enumerate(proposals):
        if any(q['area']>p['area']*1.15 and q['area']<p['area']*12
               and (p['mask']&q['mask']).sum()/p['area']>.90 for j,q in enumerate(proposals) if i!=j):continue
        roots.append(p)
    roots.sort(key=lambda p:p['area'])
    fields=[cv2.distanceTransform((~r['mask']).astype(np.uint8),cv2.DIST_L2,5) for r in roots]
    owners=[];ambiguous=0
    for p in paths:
        q=samples(p,.7)
        fits=[(float((sample(d,q)<=3).mean()),float(np.mean(sample(d,q))),i) for i,d in enumerate(fields)]
        good=[z for z in fits if z[0]>=.78]
        if good:
            # Tightest supporting region, preserving an entire curve at its rim.
            z=min(good,key=lambda z:roots[z[2]]['area']);owners.append(dict(module=z[2],uncertain=False,fraction=z[0]))
        else:
            relaxed=[z for z in fits if z[0]>=.58 and z[1]<=5.]
            relaxed.sort(key=lambda z:(-z[0],z[1]))
            if relaxed and (len(relaxed)==1 or relaxed[0][0]-relaxed[1][0]>.18):
                z=relaxed[0];owners.append(dict(module=z[2],uncertain=True,fraction=z[0]));ambiguous+=1
            else:owners.append(dict(module=-1,uncertain=True,fraction=0.))
    used=sorted({o['module'] for o in owners if o['module']>=0});remap={m:i for i,m in enumerate(used)}
    modules=[dict(mask=roots[m]['mask'],id=remap[m],sam_key=roots[m]['key'],kind='对象候选',verified=False) for m in used]
    for o in owners:
        if o['module']>=0:o['module']=remap[o['module']]
    # Unassigned curves form explicitly unresolved local work regions, not rocks.
    groups=[]
    for i in sorted([i for i,o in enumerate(owners) if o['module']<0],key=lambda i:-length(paths[i])):
        q=samples(paths[i]);c=q.mean(0)
        choices=[(np.linalg.norm(c-a['anchor']),j) for j,a in enumerate(groups)]
        dist,j=min(choices,default=(1e9,-1))
        if dist<44:groups[j]['ids'].append(i)
        else:groups.append(dict(anchor=c,ids=[i]))
    for a in groups:
        mid=len(modules);mask=raster([paths[i] for i in a['ids']],(h,w),4)>0
        modules.append(dict(id=mid,mask=mask,sam_key=None,kind='归属待定区域',verified=False))
        for i in a['ids']:owners[i]['module']=mid
    for m in modules:
        ids=[i for i,o in enumerate(owners) if o['module']==m['id']]
        q=np.concatenate([samples(paths[i]) for i in ids]);lo=q.min(0);hi=q.max(0)
        m['bbox']=[*lo.tolist(),*hi.tolist()];m['center']=q.mean(0).tolist()
        m['input_count']=len(ids);m['uncertain_curves']=sum(owners[i]['uncertain'] for i in ids)
    return modules,owners

def ridge_connector(a,b,ink,contrast):
    """A* in a narrow corridor. A connector still needs dense source evidence."""
    a=np.asarray(a,float);b=np.asarray(b,float);gap=np.linalg.norm(b-a)
    if gap<.18:return np.array([a,b])
    lo=np.maximum(0,np.floor(np.minimum(a,b)-2).astype(int))
    hi=np.minimum(np.array(ink.shape[::-1])-1,np.ceil(np.maximum(a,b)+2).astype(int))
    start=tuple(np.rint(a).astype(int));goal=tuple(np.rint(b).astype(int))
    v=b-a;v2=max(float(v@v),1e-8)
    def corridor(x,y):
        p=np.array([x,y]);t=np.clip((p-a)@v/v2,0,1)
        return np.linalg.norm(p-(a+t*v))<=1.5
    heap=[(0.,0.,start)];cost={start:0.};parent={};found=False
    while heap:
        _,d,(x,y)=heapq.heappop(heap)
        if d>cost.get((x,y),1e9)+1e-6:continue
        if (x,y)==goal:found=True;break
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx,ny=x+dx,y+dy
            if nx<lo[0] or ny<lo[1] or nx>hi[0] or ny>hi[1] or not corridor(nx,ny):continue
            step=math.hypot(dx,dy)*(1+4*(not ink[ny,nx])+1/(1+max(0,float(contrast[ny,nx]))*100))
            nd=d+step;k=(nx,ny)
            if nd<cost.get(k,1e9):
                cost[k]=nd;parent[k]=(x,y);heapq.heappush(heap,(nd+math.hypot(nx-goal[0],ny-goal[1]),nd,k))
    if not found:return None
    q=[goal]
    while q[-1]!=start:q.append(parent[q[-1]])
    q=np.vstack([a,np.array(q[::-1],float),b]);q=q[np.r_[True,np.linalg.norm(np.diff(q,axis=0),axis=1)>1e-6]]
    return q if length(q)<=gap*1.4+.2 else None

def recover_continuity(rgb,paths,owners,max_gap=6.):
    """No deletion, no endpoint reversal during execution, no cross-object joins."""
    g,c,ink=source_fields(rgb);ends=[]
    for i,p in enumerate(paths):
        if length(p)<1 or np.linalg.norm(p[0]-p[-1])<.1:continue
        for side in [0,1]:
            q=samples(p if not side else p[::-1],.25);k=min(len(q)-1,8)
            v=q[k]-q[0];v=v/max(np.linalg.norm(v),1e-8)
            ends.append(dict(i=i,side=side,p=q[0],t=v))
    pairs=sorted(cKDTree([e['p'] for e in ends]).query_pairs(max_gap)) if ends else []
    degree=Counter()
    for a,b in pairs:
        if ends[a]['i']!=ends[b]['i'] and np.linalg.norm(ends[a]['p']-ends[b]['p'])<1:degree[a]+=1;degree[b]+=1
    candidates=[];rejected=Counter()
    for a,b in pairs:
        ea,eb=ends[a],ends[b];i,j=ea['i'],eb['i']
        if i==j:continue
        if owners[i]['module']!=owners[j]['module']:rejected['different_object']+=1;continue
        # Relaxed ownership cannot justify cross-fragment synthesis.
        if owners[i]['uncertain'] or owners[j]['uncertain']:rejected['uncertain_owner']+=1;continue
        v=eb['p']-ea['p'];gap=float(np.linalg.norm(v));facing=float(ea['t']@-eb['t'])
        corner=gap<.65 and degree[a]==degree[b]==1
        if facing<math.cos(math.radians(105 if corner else 45)):rejected['direction']+=1;continue
        if gap>.65:
            axis=v/gap
            if min(-ea['t']@axis,eb['t']@axis)<.75:rejected['sideways']+=1;continue
        bridge=ridge_connector(ea['p'],eb['p'],ink,c)
        if bridge is None:rejected['no_ridge_route']+=1;continue
        q=samples(bridge,.12);support=float(sample(ink.astype(np.float32),q).mean())
        if gap>.15 and support<.92:rejected['unsupported_gap']+=1;continue
        score=(facing+1)/2-gap*.055+support*.15
        candidates.append(dict(a=a,b=b,i=i,j=j,gap=gap,score=score,support=support,bridge=bridge))
    ranks={}
    for k,z in enumerate(candidates):
        for e in [z['a'],z['b']]:ranks.setdefault(e,[]).append((z['score'],k))
    for v in ranks.values():v.sort(reverse=True)
    parent=list(range(len(paths)));links={};accepted=[];pending=[]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for k,z in sorted(enumerate(candidates),key=lambda x:-x[1]['score']):
        a,b=z['a'],z['b']
        if a in links or b in links:continue
        if any(ranks[e][0][1]!=k or len(ranks[e])>1 and ranks[e][0][0]-ranks[e][1][0]<.14 for e in [a,b]):
            pending.append(dict(i=z['i'],j=z['j'],reason='competing continuation'));continue
        x,y=root(z['i']),root(z['j'])
        if x==y:rejected['new_loop']+=1;continue
        free=[e for e in range(len(ends)) if e not in links and e not in (a,b)
              and root(ends[e]['i']) in (x,y)]
        if len(free)==2 and np.linalg.norm(ends[free[0]]['p']-ends[free[1]]['p'])<.5:
            rejected['would_close_geometry']+=1;continue
        parent[y]=x;links[a]=(b,z['bridge']);links[b]=(a,z['bridge'][::-1]);accepted.append(z)
    byend={(ends[a]['i'],ends[a]['side']):((ends[b]['i'],ends[b]['side']),p) for a,(b,p) in links.items()}
    seen=set();out=[]
    def walk(start):
        cur=start;parts=[];ids=[]
        while cur[0] not in seen:
            i,side=cur;seen.add(i);ids.append(i);parts.append(paths[i] if not side else paths[i][::-1])
            z=byend.get((i,1-side))
            if z is None:break
            cur,bridge=z;parts.append(bridge)
        out.append(dict(points=np.concatenate(parts),source_curve_ids=ids,module=owners[start[0]]['module'],
                        uncertain_module=any(owners[i]['uncertain'] for i in ids)))
    for i in range(len(paths)):
        for side in [0,1]:
            if i not in seen and (i,side) not in byend:walk((i,side))
    for i in range(len(paths)):
        if i not in seen:walk((i,0))
    assert sorted(i for r in out for i in r['source_curve_ids'])==list(range(len(paths)))
    return out,dict(accepted=accepted,pending=pending,rejected=dict(rejected))

def structural_roles(rgb,curves,modules):
    g,c,ink=source_fields(rgb)
    borders={m['id']:cv2.distanceTransform(m['mask'].astype(np.uint8),cv2.DIST_L2,5) for m in modules}
    for row in curves:
        q=samples(row['points'],.7);arc=length(row['points']);extent=float(np.linalg.norm(np.ptp(q,axis=0)))
        boundary=float((sample(borders[row['module']],q)<3).mean())
        supported=float(sample(ink.astype(np.float32),q).mean())
        long=arc>=16 and extent>=8
        phase=0 if long and boundary>=.38 else 1 if arc>=6 and extent>=3 else 2
        row.update(phase=phase,phase_label=PHASES[phase],role_verified=False,
                   source_support=supported,length_px=arc,boundary_fraction=boundary)
    return curves

def hierarchy_schedule(curves,modules):
    remaining=set(range(len(curves)));cursor=np.array([0.,495.]);route=[];order=[]
    # Salience/extent first, travel second. Fixed object lock until all its layers finish.
    available=sorted({r['module'] for r in curves})
    while available:
        def score(mid):
            ids=[i for i in remaining if curves[i]['module']==mid]
            ends=np.concatenate([curves[i]['points'][[0,-1]] for i in ids]);travel=np.linalg.norm(ends-cursor,axis=1).min()
            structure=sum(min(curves[i]['length_px'],60) for i in ids if curves[i]['phase']==0)
            salience=structure+sum(min(curves[i]['length_px'],20) for i in ids)*.15
            return (modules[mid]['sam_key'] is not None,salience/(1+travel/160),-mid)
        mid=max(available,key=score);available.remove(mid);order.append(mid)
        for phase in range(3):
            todo=[i for i in remaining if curves[i]['module']==mid and curves[i]['phase']==phase]
            # Internal strokes form geometric directional bundles. These are
            # NOT traditional cun style labels and are never fused by proximity.
            bundles={};anchors=[]
            if phase==1:
                for i in sorted(todo,key=lambda i:-curves[i]['length_px']):
                    p=curves[i]['points'];v=p[-1]-p[0];norm=np.linalg.norm(v)
                    unit=v/max(norm,1e-8);center=samples(p).mean(0)
                    good=[j for j,a in enumerate(anchors) if norm>1 and abs(unit@a['unit'])>math.cos(math.radians(25))
                          and np.linalg.norm(center-a['center'])<26]
                    if good:j=min(good,key=lambda j:np.linalg.norm(center-anchors[j]['center']))
                    else:j=len(anchors);anchors.append(dict(unit=unit,center=center))
                    bundles[i]=j
            active_bundle=None
            while todo:
                def pick(i):
                    p=curves[i]['points'];d=min(np.linalg.norm(p[0]-cursor),np.linalg.norm(p[-1]-cursor))
                    return curves[i]['length_px']**.65/(1+d/35)
                eligible=[i for i in todo if bundles.get(i)==active_bundle] if phase==1 and active_bundle is not None else []
                i=max(eligible or todo,key=pick)
                if phase==1:active_bundle=bundles[i]
                todo.remove(i);remaining.remove(i);p=curves[i]['points']
                rev=bool(np.linalg.norm(p[-1]-cursor)<np.linalg.norm(p[0]-cursor));p=p[::-1] if rev else p
                route.append(dict(sequence=len(route)+1,stroke_id=f's{i:05}',curve=i,module=mid,phase=phase,
                    texture_bundle=bundles.get(i),reverse=rev,air_travel_px=float(np.linalg.norm(p[0]-cursor)),length_px=curves[i]['length_px']))
                cursor=p[-1]
    assert len(route)==len(curves) and len({r['curve'] for r in route})==len(curves)
    return route,order

def order_metrics(actions):
    seen=set();last=None;reentry=0;switches=0
    for a in actions:
        if a['module']!=last:
            switches+=last is not None;reentry+=a['module'] in seen;seen.add(a['module']);last=a['module']
    return dict(module_switches=int(switches),module_reentries=int(reentry),
                air_travel_px=sum(a.get('air_travel_px',0) for a in actions))
