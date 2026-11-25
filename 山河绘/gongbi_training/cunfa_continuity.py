"""Track evidence for possible continuation of visible texture fragments."""

import numpy as np
from scipy.ndimage import map_coordinates


def join_marks(marks, gray, cv2):
    """Return traced curves with provenance and inspectable rejected pairs."""
    smooth=cv2.GaussianBlur(gray,(0,0),.55)
    relief=np.maximum(cv2.GaussianBlur(smooth,(0,0),5)-smooth,0)
    ends={}
    for i,(curve,width,*_) in enumerate(marks):
        for side,path in enumerate((curve,curve[::-1])):
            tangent=path[0]-path[min(len(path)-1,5)]
            length=np.linalg.norm(tangent)
            if length>0:
                w=width if side==0 else width[::-1]
                ends[i,side]=(path[0],tangent/length,float(np.median(w[:5])))
    proposals=[]
    options={}
    items=list(ends.items())
    for n,(first,(start,forward,w0)) in enumerate(items):
        for second,(finish,backward,w1) in items[n+1:]:
            if first[0]==second[0]:
                continue
            gap=finish-start
            span=np.linalg.norm(gap)
            if not 1.4<=span<=13. or min(w0,w1)<.40*max(w0,w1):
                continue
            direction=gap/span
            align=min(forward@direction,backward@-direction,forward@-backward)
            if align<.60:
                continue
            t=np.arange(1,max(3,round(span)))/max(3,round(span))
            t=t[:,None]
            mid=(1-t)**3*start + 3*(1-t)**2*t*(start+forward*span/3)
            mid+=3*(1-t)*t*t*(finish+backward*span/3)+t*t*t*finish
            intensity=map_coordinates(relief,mid[:,::-1].T,order=1,mode="nearest")
            # Faint but continuous evidence is admissible. Bright unsupported
            # corridors and near-parallel adjacent lines stay unresolved.
            score=float(.55*align+.45*np.minimum(1,intensity/.035).mean())
            passed=bool(intensity.min()>=.010 and np.mean(intensity>=.022)>=.70)
            record=dict(a=first,b=second,span=float(span),alignment=float(align),score=score,
                        path=mid.tolist(),contrast=intensity.tolist(),evidence_pass=passed,selected=False)
            i=len(proposals)
            proposals.append(record)
            if passed:
                options.setdefault(first,[]).append(i)
                options.setdefault(second,[]).append(i)
    preferred={}
    for end,variants in options.items():
        best=sorted(variants,key=lambda i:proposals[i]["score"],reverse=True)
        if len(best)==1 or proposals[best[0]]["score"]-proposals[best[1]]["score"]>=.045:
            preferred[end]=best[0]
    links={}
    for i,record in enumerate(proposals):
        a,b=record["a"],record["b"]
        if preferred.get(a)==i and preferred.get(b)==i:
            links[a]=(b,record)
            links[b]=(a,record)
            record["selected"]=True
    routes=[]
    used=set()
    starts=[(i,k) for i in range(len(marks)) for k in (0,1) if (i,k) not in links]
    for start in starts+[(i,0) for i in range(len(marks))]:
        if start[0] in used:
            continue
        curves,widths,tones,origins=[],[],[],[]
        index,side=start
        while index not in used:
            used.add(index)
            curve,width,_,_,tone=marks[index]
            step=slice(None) if side==0 else slice(None,None,-1)
            curves.append(curve[step])
            widths.append(width[step])
            tones.append(tone[step])
            origins.append(index)
            link=links.get((index,1-side))
            if link is None or link[0][0] in used:
                break
            (index,side),evidence=link
            mid=np.asarray(evidence["path"])
            if np.linalg.norm(mid[0]-curves[-1][-1])>np.linalg.norm(mid[-1]-curves[-1][-1]):
                mid=mid[::-1]
            strength=map_coordinates(1-gray,mid[:,::-1].T,order=1,mode="nearest")
            target_width=marks[index][1][0 if side==0 else -1]
            widths.append(np.linspace(widths[-1][-1],target_width,len(mid)+2)[1:-1])
            curves.append(mid)
            tones.append(strength)
        routes.append((np.concatenate(curves),np.concatenate(widths),np.concatenate(tones),origins))
    return routes,proposals
