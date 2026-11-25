"""Conservative endpoint repair on directional strokes, no hardware imports."""
import math
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from .stroke_direction_v07 import plan as previous_plan

def endpoint_tangent(p,start=False,span=7):
    q=p if start else p[::-1];dist=np.linalg.norm(q-q[0],axis=1);index=min(len(q)-1,max(1,int(np.searchsorted(dist,span))));v=q[0]-q[index]
    return v/max(1e-6,np.linalg.norm(v))

def repair_paths(paths,density,prediction,max_gap=9):
    paths=[np.asarray(p,float) for p in paths];events=[];h,w=density.shape
    for iteration in range(5):
        if len(paths)<2:break
        endpoints=np.array([p[k] for p in paths for k in (0,-1)]);tree=cKDTree(endpoints);candidates=[]
        for a,b in tree.query_pairs(max_gap):
            ia,ea=divmod(a,2);ib,eb=divmod(b,2)
            if ia==ib:continue
            pa,pb=paths[ia],paths[ib];va=endpoint_tangent(pa,ea==0);vb=endpoint_tangent(pb,eb==0);gap=endpoints[b]-endpoints[a];distance=np.linalg.norm(gap)
            if distance<.4 or va@vb>-.86:continue
            if min(va@gap/distance,-vb@gap/distance)<.82:continue
            t=np.linspace(0,1,max(4,int(distance*2)+1))[:,None]
            # Cubic Hermite interpolation respects both measured tangents.
            p0,p1=endpoints[a],endpoints[b];m0=va*distance;m1=-vb*distance
            curve=(2*t**3-3*t*t+1)*p0+(t**3-2*t*t+t)*m0+(-2*t**3+3*t*t)*p1+(t**3-t*t)*m1
            xy=np.rint(curve).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);xs,ys=xy.T
            evidence=density[ys,xs];prob=prediction[0,ys,xs]
            # Every gap must have visible ink evidence. A strong network score
            # alone is not permission to invent a stroke over blank paper.
            if (evidence>.025).mean()<.85 or evidence.mean()<.055:continue
            if prediction is not None and np.median(prob)<.25:continue
            a_angle=.5*np.arctan2(prediction[2,ys,xs],prediction[1,ys,xs]);g=np.gradient(curve,axis=0);g/=np.maximum(np.linalg.norm(g,axis=1)[:,None],1e-5);alignment=np.abs(g[:,0]*np.cos(a_angle)+g[:,1]*np.sin(a_angle))
            if np.median(alignment)<.78:continue
            score=distance+5*(1+va@vb)+3*(1-float(np.median(prob)))
            candidates.append((score,ia,ea,ib,eb,curve,distance,float(evidence.mean())))
        used=set();merged=[]
        for score,ia,ea,ib,eb,curve,distance,evidence in sorted(candidates,key=lambda v:v[0]):
            if ia in used or ib in used:continue
            a=paths[ia][::-1] if ea==0 else paths[ia];b=paths[ib][::-1] if eb==1 else paths[ib];q=np.concatenate((a,curve[1:-1],b))
            if np.linalg.norm(q[0]-q[-1])<4:continue
            merged.append(q);used.update((ia,ib));events.append(dict(iteration=iteration,gap_pixels=round(float(distance),2),mean_ink_evidence=round(evidence,4),curve=curve.round(2).tolist(),status='heuristic_ink_supported_join_not_confirmed_human_stroke'))
        paths=merged+[p for i,p in enumerate(paths) if i not in used]
        if not used:break
    return paths,events

def render_plan(paths,ink,prediction,events=None):
    coverage,density=ink;h,w=density.shape;view=np.full((h,w,3),255,np.uint8);role_view=view.copy();replay=np.zeros((h,w),np.float32);records=[]
    for p in paths:
        # Resample merged segments to approximately one pixel before measuring
        # width, so a repaired gap does not create a sparse disk or a width jump.
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];keep=np.r_[True,np.diff(arc)>.05];p=p[keep];arc=arc[keep]
        if len(p)<3 or arc[-1]<4:continue
        u=np.linspace(0,arc[-1],max(3,math.ceil(arc[-1])+1));p=np.stack([np.interp(u,arc,p[:,k]) for k in range(2)],1)
        p=np.stack([gaussian_filter1d(p[:,k],.65,mode='nearest') for k in range(2)],1);xy=np.rint(p).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);xs,ys=xy.T
        d=gaussian_filter1d(density[ys,xs],1.5);tangent=np.gradient(p,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-5);normal=np.stack((-tangent[:,1],tangent[:,0]),1);width=np.ones(len(p),np.float32)
        for sign in (-1,1):
            alive=np.ones(len(p),bool)
            for radius in range(1,7):
                q=np.rint(p+normal*radius*sign).astype(int);q[:,0]=np.clip(q[:,0],0,w-1);q[:,1]=np.clip(q[:,1],0,h-1);alive&=density[q[:,1],q[:,0]]>np.maximum(.04,d*.35);width+=alive
        width=np.clip(gaussian_filter1d(.75*width+.25*prediction[3,ys,xs]*12,2),1,8)
        if width[:5].mean()>width[-5:].mean()+.6 or (abs(width[0]-width[-1])<.6 and p[0,1]>p[-1,1]):p=p[::-1];xy=xy[::-1];width=width[::-1];d=d[::-1];xs,ys=xy.T
        width=np.maximum(1,width*np.interp(np.linspace(0,1,len(p)),[0,.08,.92,1],[.35,1,1,.4]));pressure=np.clip((width-1)/7,0,1)
        roles=prediction[4:6,ys,xs].mean(axis=1) if prediction.shape[0]>=6 else np.array([.5,.5]);winner=int(np.argmax(roles));margin=float(abs(roles[0]-roles[1]));role=('outline_candidate' if winner==0 else 'internal_line_candidate') if roles[winner]>.65 and margin>.18 else 'uncertain_line'
        role_color=(25,115,240) if role=='outline_candidate' else (235,100,15) if role=='internal_line_candidate' else (135,135,135)
        color=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[len(records)*37%180,185,170]]]),cv2.COLOR_HSV2RGB)[0,0]);cv2.polylines(view,[xy],False,color,1,cv2.LINE_AA);cv2.polylines(role_view,[xy],False,role_color,1,cv2.LINE_AA)
        scale=3;layer=np.zeros((h*scale,w*scale),np.float32)
        for i in range(len(p)-1):
            dp=p[i+1]-p[i];n=np.array([-dp[1],dp[0]])/max(1e-6,np.linalg.norm(dp));poly=np.array([p[i]+n*width[i]/2,p[i+1]+n*width[i+1]/2,p[i+1]-n*width[i+1]/2,p[i]-n*width[i]/2]);cv2.fillConvexPoly(layer,np.rint(poly*scale).astype(np.int32),float((d[i]+d[i+1])/2))
        replay=np.maximum(replay,cv2.resize(layer,(w,h),interpolation=cv2.INTER_AREA))
        records.append(dict(order=len(records),xy_pixels=p.round(2).tolist(),xy_normalized=(p/[max(1,w-1),max(1,h-1)]).round(6).tolist(),width_pixels=width.round(2).tolist(),relative_pressure=pressure.round(3).tolist(),ink_density=d.round(4).tolist(),role=role,role_candidate_scores=roles.round(4).tolist(),role_scores_calibrated=False,direction_provenance='heuristic thin endpoint or top-to-bottom; original motion unknown',virtual_pen='down'))
    info=dict(image_width=w,image_height=h,planned_strokes=len(records),ink_supported_joins=events or [],coordinate_units='pixels and normalized image coordinates',physical_z=None,physical_speed=None,paper_transform=None,robot_connected=False,actually_drawn=False,ready_for_robot=False,pressure_units='relative 0..1, no physical calibration',semantic_role_status='assistant partial supervision; NOT verified cunfa classification',strokes=records)
    return info,view,np.uint8(np.clip(255*(1-replay),0,255)),role_view

def plan(ink,prediction):
    old,_,_=previous_plan(ink,prediction[:4]);paths=[np.asarray(r['xy_pixels']) for r in old['strokes']];paths,events=repair_paths(paths,ink[1],prediction)
    info,view,replay,roles=render_plan(paths,ink,prediction,events);info['before_repair_strokes']=old['planned_strokes'];return info,view,replay,roles
