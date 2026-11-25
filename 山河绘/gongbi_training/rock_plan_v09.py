"""Directional tracing with an experimental stroke-embedding join constraint."""
import math,cv2,numpy as np
from scipy.spatial import cKDTree
from .stroke_streamline_v08 import trace
from .stroke_planner_v08 import endpoint_tangent,render_plan

def repair_instances(paths,density,prediction):
    h,w=density.shape;events=[];rejections=0
    def embedding(path,start):
        p=path[:7] if start else path[-7:];xy=np.rint(p).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);v=prediction[10:14,xy[:,1],xy[:,0]].mean(1);return v/max(1e-5,np.linalg.norm(v))
    for iteration in range(4):
        if len(paths)<2:break
        ends=np.array([p[k] for p in paths for k in (0,-1)]);tree=cKDTree(ends);candidates=[]
        for a,b in tree.query_pairs(9):
            ia,ea=divmod(a,2);ib,eb=divmod(b,2)
            if ia==ib:continue
            pa,pb=paths[ia],paths[ib];va=endpoint_tangent(pa,ea==0);vb=endpoint_tangent(pb,eb==0);gap=ends[b]-ends[a];distance=np.linalg.norm(gap)
            if distance<.4 or va@vb>-.86 or min(va@gap/distance,-vb@gap/distance)<.82:continue
            cosine=float(embedding(pa,ea==0)@embedding(pb,eb==0))
            if cosine<.55:rejections+=1;continue
            t=np.linspace(0,1,max(4,int(distance*2)+1))[:,None];curve=(2*t**3-3*t*t+1)*ends[a]+(t**3-2*t*t+t)*va*distance+(-2*t**3+3*t*t)*ends[b]+(t**3-t*t)*(-vb)*distance
            xy=np.rint(curve).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);xs,ys=xy.T;evidence=density[ys,xs]
            if (evidence>.025).mean()<.85 or evidence.mean()<.055 or np.median(prediction[0,ys,xs])<.25:continue
            theta=.5*np.arctan2(prediction[2,ys,xs],prediction[1,ys,xs]);v=np.gradient(curve,axis=0);v/=np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-5)
            if np.median(np.abs(v[:,0]*np.cos(theta)+v[:,1]*np.sin(theta)))<.78:continue
            candidates.append((distance+2*(1-cosine),ia,ea,ib,eb,curve,cosine,float(evidence.mean())))
        used=set();merged=[]
        for score,ia,ea,ib,eb,curve,cosine,evidence in sorted(candidates,key=lambda r:r[0]):
            if ia in used or ib in used:continue
            a=paths[ia][::-1] if ea==0 else paths[ia];b=paths[ib][::-1] if eb==1 else paths[ib];q=np.concatenate((a,curve[1:-1],b))
            if np.linalg.norm(q[0]-q[-1])<4:continue
            merged.append(q);used.update((ia,ib));events.append(dict(iteration=iteration,gap_pixels=round(float(np.linalg.norm(curve[0]-curve[-1])),2),mean_ink_evidence=evidence,embedding_cosine=cosine,curve=curve.round(2).tolist(),status='experimental ink/direction/embedding supported join, not confirmed human stroke'))
        paths=merged+[p for i,p in enumerate(paths) if i not in used]
        if not used:break
    return paths,events,rejections

def plan(ink,prediction):
    paths,stats=trace(ink,prediction,strict=True);paths,events,rejections=repair_instances(paths,ink[1],prediction);info,view,replay,roles=render_plan(paths,ink,prediction,events);info['streamline_diagnostics']=stats;info['embedding_rejected_join_candidates']=rejections;info['embedding_constraint_status']='experimental, synthetic/assistant supervised; not verified true stroke identity';return info,view,replay,roles
