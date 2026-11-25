"""Experimental subpixel tracing; no per-painting reference paths are inputs."""
import math
import cv2,numpy as np
from .stroke_direction_v07 import ridge_fields
from .stroke_planner_v08 import repair_paths,render_plan

def trace(ink,prediction,strict=False):
    coverage,density=ink;h,w=density.shape;density=cv2.GaussianBlur(density.astype(np.float32),(0,0),.5)
    response,_,_,_=ridge_fields(density)
    g=cv2.GaussianBlur(density,(0,0),1.3);a=cv2.Sobel(g,cv2.CV_32F,2,0)/4;b=cv2.Sobel(g,cv2.CV_32F,0,2)/4;c=cv2.Sobel(g,cv2.CV_32F,1,1)/4
    theta=.5*np.arctan2(2*c,a-b);gap=np.sqrt((a-b)**2+4*c*c);large=(a+b+gap)/2;small=(a+b-gap)/2;coherence=np.clip(1-np.abs(large)/(np.abs(small)+1e-5),0,1)
    model_theta=.5*np.arctan2(prediction[2],prediction[1]);angle=.5*np.arctan2(.6*np.sin(2*model_theta)+.4*coherence*np.sin(2*theta),.6*np.cos(2*model_theta)+.4*coherence*np.cos(2*theta));tx=np.cos(angle);ty=np.sin(angle)
    score=(.50*np.sqrt(np.clip(density,0,1))+.5*prediction[0])*np.clip((coverage-.12)/.4,0,1)
    yy,xx=np.nonzero((response>(.006 if strict else .004))&(prediction[0]>(.22 if strict else .16))&(density>(.10 if strict else .04)));order=np.argsort(-(response[yy,xx]*(.4+prediction[0,yy,xx])))
    visited=np.zeros((h,w),np.uint8);visited_angle=np.zeros((h,w),np.float32);paths=[];rejected=0
    def sample(arr,p):
        x=float(np.clip(p[0],0,w-1));y=float(np.clip(p[1],0,h-1));xi=int(x);yi=int(y);xj=min(w-1,xi+1);yj=min(h-1,yi+1);u=x-xi;v=y-yi
        return float((1-u)*(1-v)*arr[yi,xi]+u*(1-v)*arr[yi,xj]+(1-u)*v*arr[yj,xi]+u*v*arr[yj,xj])
    def occupied(p,vec):
        x=int(round(p[0]));y=int(round(p[1]));x1=max(0,x-1);x2=min(w,x+2);y1=max(0,y-1);y2=min(h,y+2);v=visited[y1:y2,x1:x2];ang=visited_angle[y1:y2,x1:x2]
        return bool(((v>0)&(np.abs(np.cos(ang)*vec[0]+np.sin(ang)*vec[1])>.78)).any())
    def walk(seed,direction):
        out=[seed.copy()];position=seed.copy();previous=direction.copy();weak=0
        for step in range(1000):
            vec=np.array([sample(tx,position),sample(ty,position)]);vec/=max(1e-5,np.linalg.norm(vec))
            if vec@previous<0:vec=-vec
            if vec@previous<.75:break
            vec=.55*previous+.45*vec;vec/=max(1e-5,np.linalg.norm(vec));proposed=position+vec*.9;normal=np.array([-vec[1],vec[0]])
            if min(proposed)<2 or proposed[0]>w-3 or proposed[1]>h-3:break
            candidates=[]
            for offset in np.linspace(-1.4,1.4,7):
                q=proposed+normal*offset;candidates.append((sample(score,q)-.035*offset*offset,q))
            _,candidate=max(candidates,key=lambda r:r[0]);advance=candidate-position
            if advance@previous<.2:break
            ink_level=sample(density,candidate);confidence=sample(prediction[0],candidate)
            if ink_level<(.04 if strict else .023) or confidence<(.18 if strict else .12):
                weak+=1
                if weak>2:break
            else:weak=0
            if step>4 and occupied(candidate,vec):break
            if len(out)>14 and np.min(np.linalg.norm(np.asarray(out[:-10])-candidate,axis=1))<1.4:break
            out.append(candidate);position=candidate;previous=advance/max(1e-5,np.linalg.norm(advance))
        if weak:out=out[:-weak] if len(out)>weak+1 else out[:1]
        return np.asarray(out)
    for k in order:
        seed=np.array([xx[k],yy[k]],float);vec=np.array([tx[yy[k],xx[k]],ty[yy[k],xx[k]]])
        if occupied(seed,vec):continue
        left=walk(seed,-vec);right=walk(seed,vec);p=np.concatenate((left[::-1][:-1],right));length=np.linalg.norm(np.diff(p,axis=0),axis=1).sum()
        if length<10:continue
        span=np.ptp(p,axis=0)
        if np.linalg.norm(p[0]-p[-1])<3 and max(span)<38:rejected+=1;continue
        # At least 90% of final samples must be supported by visible ink.
        if np.mean([sample(density,q)>.02 for q in p])<.90:continue
        if strict:
            levels=np.array([sample(density,q) for q in p])
            strong=np.flatnonzero(levels>.065)
            if not len(strong):continue
            p=p[strong[0]:strong[-1]+1];levels=levels[strong[0]:strong[-1]+1]
            if len(p)<10 or levels.mean()<.10:continue
        paths.append(p)
        tang=np.gradient(p,axis=0);angles=np.arctan2(tang[:,1],tang[:,0])
        for q,ang in zip(np.rint(p).astype(int),angles):
            x,y=q
            if strict:
                radius=min(3,max(1,int(round(prediction[3,y,x]*12/2))))
                cv2.circle(visited,(int(x),int(y)),radius,1,-1);cv2.circle(visited_angle,(int(x),int(y)),radius,float(ang),-1)
            else:visited[y,x]=1;visited_angle[y,x]=ang
    return paths,dict(raw_paths=len(paths),rejected_small_loops=rejected,algorithm='experimental subpixel direction-field tracing; same generic settings on all images',max_local_weak_run_steps=2,minimum_visible_ink_support=.9,conservative_seed_and_width_deduplication=strict)

def plan(ink,prediction,strict=False):
    paths,stats=trace(ink,prediction,strict);paths,events=repair_paths(paths,ink[1],prediction);info,view,replay,roles=render_plan(paths,ink,prediction,events);info['streamline_diagnostics']=stats;return info,view,replay,roles
