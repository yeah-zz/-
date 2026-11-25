# -*- coding: utf-8 -*-
"""Image-evidence continuity and weak ridges; independent frozen v4 control."""
import split_detail_v4_frozen as base
from split_detail_v4_frozen import *
from rock_v15_common import read
from scipy.ndimage import gaussian_filter1d

ROOT=LAB/'stroke_completeness_v5_20261029'
CFG=dict(base.CFG,version='stroke_completeness_v5',max_bridge=11.,margin=.08,weak_ridge_seed=.007,weak_ridge_follow=.0025,max_tiny_void=18,void_min_coherence=.6,void_min_ink_ring=.65,bridge_min_contrast=.004,bridge_min_supported_fraction=.80)

def fill_tiny_ink_voids(ink,rgb,dense,coherence):
    # Only a topology hypothesis for oriented thick ink. Original ink stays saved.
    n,labels,stats,_=cv2.connectedComponentsWithStats((~ink).astype(np.uint8),8)
    topo=ink.copy();events=[];h,w=ink.shape
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if area>CFG['max_tiny_void'] or x==0 or y==0 or x+bw==w or y+bh==h:continue
        ys,xs=np.where(labels[max(0,y-3):min(h,y+bh+3),max(0,x-3):min(w,x+bw+3)]==i);ys+=max(0,y-3);xs+=max(0,x-3)
        if dense[ys,xs].any() or np.median(coherence[ys,xs])<CFG['void_min_coherence']:continue
        local=labels[max(0,y-3):min(h,y+bh+3),max(0,x-3):min(w,x+bw+3)]==i
        ring=(cv2.dilate(local.astype(np.uint8),np.ones((5,5),np.uint8))>0)&~local
        crop=ink[max(0,y-3):min(h,y+bh+3),max(0,x-3):min(w,x+bw+3)]
        if not ring.any() or crop[ring].mean()<CFG['void_min_ink_ring']:continue
        topo[ys,xs]=True;events.append(dict(box=[int(x),int(y),int(bw),int(bh)],pixels=int(area),kind='oriented_ink_tiny_void_hypothesis',expert_verified=False))
    return topo,events

def weak_ridges(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    h,w=gray.shape;yy,xx=np.mgrid[:h,:w].astype(np.float32);best=np.zeros(gray.shape,np.float32);normal=np.zeros((*gray.shape,2),np.float32)
    for sigma in [.7,1.2,2.,3.5]:
        g=cv2.GaussianBlur(gray,(0,0),sigma);factor=sigma**1.5/4
        a=cv2.Sobel(g,cv2.CV_32F,2,0,ksize=3)*factor;b=cv2.Sobel(g,cv2.CV_32F,0,2,ksize=3)*factor;c=cv2.Sobel(g,cv2.CV_32F,1,1,ksize=3)*factor
        delta=np.sqrt((a-b)**2+4*c*c);large=(a+b+delta)*.5;small=(a+b-delta)*.5
        score=np.maximum(large,0)*np.exp(-.5*(small/np.maximum(large*.5,1e-5))**2)
        angle=.5*np.arctan2(2*c,a-b);nx=np.cos(angle);ny=np.sin(angle)
        left=cv2.remap(gray,xx+nx*max(2,3*sigma),yy+ny*max(2,3*sigma),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        right=cv2.remap(gray,xx-nx*max(2,3*sigma),yy-ny*max(2,3*sigma),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        score[(np.minimum(left,right)-gray)<.004]=0
        take=score>best;best[take]=score[take];normal[take,0]=nx[take];normal[take,1]=ny[take]
    nx,ny=normal[:,:,0],normal[:,:,1]
    l=cv2.remap(best,xx+nx,yy+ny,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);r=cv2.remap(best,xx-nx,yy-ny,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    nms=(best>=l)&(best>=r);sk=skeletonize(binary_propagation(nms&(best>CFG['weak_ridge_seed']),mask=nms&(best>CFG['weak_ridge_follow'])))
    return sk,best

def sample(a,p):
    p=np.asarray(p,np.float32);return cv2.remap(a,p[:,0][None],p[:,1][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)[0]

def band_contrast(gray,p):
    t=np.gradient(p,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-6);normal=np.c_[-t[:,1],t[:,0]]
    left=np.maximum.reduce([sample(gray,p+normal*d) for d in [3.,5.,8.]])
    right=np.maximum.reduce([sample(gray,p-normal*d) for d in [3.,5.,8.]])
    return np.minimum(left,right)-sample(gray,p)

def bridge_curve(a,b,ta,tb,gray):
    gap=float(np.linalg.norm(b-a));t=np.linspace(0,1,max(4,int(math.ceil(gap*2))+1))[:,None]
    q=(1-t)**3*a+3*(1-t)**2*t*(a+ta*gap/3)+3*(1-t)*t*t*(b+tb*gap/3)+t**3*b
    tangent=np.gradient(q,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-6);normal=np.c_[-tangent[:,1],tangent[:,0]];off=np.array([-1.,0.,1.],np.float32)
    states=q[:,None]+normal[:,None]*off[None,:,None];values=np.stack([band_contrast(gray,states[:,i]) for i in range(3)],1)
    energy=-np.clip(values/max(CFG['bridge_min_contrast'],1e-5),-2,4)+.2*off[None]**2
    energy[0,[0,2]]=1e6;energy[-1,[0,2]]=1e6;dp=energy[0].copy();back=np.zeros((len(q),3),int);transition=.6*(off[:,None]-off[None,:])**2
    for k in range(1,len(q)):
        cost=dp[:,None]+transition;back[k]=cost.argmin(0);dp=energy[k]+cost.min(0)
    ids=np.zeros(len(q),int);ids[-1]=int(dp.argmin())
    for k in range(len(q)-1,0,-1):ids[k-1]=back[k,ids[k]]
    result=states[np.arange(len(q)),ids];result[[0,-1]]=np.stack([a,b]);contrast=band_contrast(gray,result)
    return result,contrast

def stitch(paths,rgb,ink,dense,cfg=CFG):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    coords=[];descs=[]
    for i,p in enumerate(paths):
        if plen(p)<7 or np.linalg.norm(p[0]-p[-1])<1:continue
        for side in (0,1):
            xy,t=endpoint(p,side)
            if near_region(p[[0 if side==0 else -1]],dense)[0]:continue
            coords.append(xy);descs.append((i,side,t))
    if not coords:return paths,[],[]
    coords=np.asarray(coords);tree=cKDTree(coords);candidates=[];by_end={}
    packed=np.concatenate(paths);owner=np.concatenate([np.full(len(p),i,int) for i,p in enumerate(paths)]);occupied=cKDTree(packed)
    for a,b in sorted(tree.query_pairs(cfg['max_bridge'])):
        ia,sa,ta=descs[a];ib,sb,tb=descs[b]
        if ia==ib:continue
        delta=coords[b]-coords[a];gap=float(np.linalg.norm(delta));alignment=float(ta@(-tb))
        if alignment<cfg['facing']:continue
        axis=1.
        if gap>.8:
            direction=delta/gap;axis=min(float(ta@direction),float(tb@(-direction)))
            cross=max(abs(float(np.cross(ta,delta))),abs(float(np.cross(tb,delta))))
            if axis<cfg['axis'] or cross>cfg['cross_track']:continue
        bridge,contrast=bridge_curve(coords[a],coords[b],ta,tb,gray)
        # Offset states near the image edge may leave the real image. Reflected
        # samples are not evidence for drawing outside it; abstain before indexing.
        h,w=gray.shape
        if not np.isfinite(bridge).all() or np.any(bridge[:,0]<0) or np.any(bridge[:,0]>w-1) or np.any(bridge[:,1]<0) or np.any(bridge[:,1]>h-1):continue
        if near_region(bridge,dense).any():continue
        support=float((contrast>=cfg['bridge_min_contrast']).mean())
        if gap>1.5 and support<cfg['bridge_min_supported_fraction']:continue
        # Do not cross unrelated branches, even when an image valley exists.
        interior=bridge[2:-2]
        if gap>2 and len(interior):
            dist,idx=occupied.query(interior,k=min(8,len(packed)))
            if dist.ndim==1:dist=dist[:,None];idx=idx[:,None]
            if ((dist<.85)&~np.isin(owner[idx],[ia,ib])).any():continue
        score=.65*alignment+.20*axis+.10*support+.05*(1-gap/cfg['max_bridge'])
        k=len(candidates);candidates.append(dict(a=a,b=b,score=score,gap=gap,support=support,bridge=bridge,min_contrast=float(contrast.min())))
        by_end.setdefault(a,[]).append((score,k));by_end.setdefault(b,[]).append((score,k))
    pair={};events=[];bridges={}
    for k,c in enumerate(candidates):
        ranks=[sorted(by_end[e],reverse=True) for e in [c['a'],c['b']]]
        if any(r[0][1]!=k or (len(r)>1 and r[0][0]-r[1][0]<cfg['margin']) for r in ranks):continue
        ia,sa,_=descs[c['a']];ib,sb,_=descs[c['b']];a=(ia,sa);b=(ib,sb)
        if a in pair or b in pair:continue
        connected={ia};stack=[ia]
        while stack:
            i=stack.pop()
            for side in (0,1):
                nxt=pair.get((i,side))
                if nxt is not None and nxt[0] not in connected:connected.add(nxt[0]);stack.append(nxt[0])
        if ib in connected:continue
        pair[a]=b;pair[b]=a;bridges[a]=c['bridge'];bridges[b]=c['bridge'][::-1]
        events.append(dict(a=list(a),b=list(b),gap=c['gap'],image_support=c['support'],score=c['score'],new_bridge_pixels=c['bridge'].tolist(),hypothesis_not_artist_gold=True))
    starts=[(i,s) for i in range(len(paths)) for s in (0,1) if (i,s) not in pair]+[(i,0) for i in range(len(paths))];used=set();out=[];members=[]
    for i,s in starts:
        if i in used:continue
        pieces=[];ids=[]
        while i not in used:
            used.add(i);ids.append(i);pieces.append(paths[i] if s==0 else paths[i][::-1]);nxt=pair.get((i,1-s))
            if nxt is None:break
            pieces.append(bridges[(i,1-s)][1:-1]);i,s=nxt
        out.append(np.concatenate(pieces));members.append(ids)
    return out,events,members

def run(source,output):
    source=Path(source).resolve();output=Path(output)
    if output.drive.upper() not in ['E:','F:'] or not output.is_absolute():raise ValueError('E/F absolute output required')
    rgb=np.asarray(ImageOps.exif_transpose(Image.open(source)).convert('RGB'))
    if max(rgb.shape[:2])>2600:raise ValueError('Use explicit prepared image <=2600px; no silent scaling')
    ink,rawsk=source_ink(rgb);raw,neighbors=pixel_chains(rawsk);dense,coherence=dense_regions(rgb,rawsk,neighbors,CFG)
    topo,void_events=fill_tiny_ink_voids(ink,rgb,dense,coherence);toposk=skeletonize(topo);topopaths,_=pixel_chains(toposk)
    strong,_=ink_ridges(rgb);weak,response=weak_ridges(rgb)
    strongpaths,_=pixel_chains(strong&~dense);weakpaths,_=pixel_chains(weak&~dense)
    # Weak candidates need an existing strong anchor, not just a lower threshold.
    anchors=np.argwhere(strong)[:,::-1];anchor_tree=cKDTree(anchors) if len(anchors) else None
    accepted_weak=[]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    for p in weakpaths:
        length=plen(p)
        if length<14 or anchor_tree is None or np.linalg.norm(p[-1]-p[0])/max(length,1e-6)<.88:continue
        dd=anchor_tree.query(p)[0]
        if (dd<=1.5).mean()>.8:continue
        if min(dd[0],dd[-1])>3.5:continue
        side=0 if dd[0]<=dd[-1] else -1;point=p[side];_,idx=anchor_tree.query(point);neighbors=anchors[anchor_tree.query_ball_point(anchors[idx],7.)]
        if len(neighbors)<4:continue
        cov=np.cov(neighbors.T);vals,vecs=np.linalg.eigh(cov)
        if vals[-1]<2*max(vals[0],1e-6):continue
        direction=unit(p[min(5,len(p)-1)]-p[0]) if side==0 else unit(p[-1]-p[max(0,len(p)-6)])
        if abs(float(direction@vecs[:,-1]))<.85:continue
        if (band_contrast(gray,p)>.004).mean()<.8:continue
        accepted_weak.append(p)
    other,count=consolidate_source_lines(topopaths,strongpaths+accepted_weak,dense)
    joined,events,members=stitch(other,rgb,ink,dense,CFG)
    structures=[];pending=[]
    for p in joined:(structures if plen(p)>=CFG['min_structure_length'] else pending).append(p)
    details=[p for p in raw if near_region(p,dense).any()]
    meta=dict(source=str(source),sha256=digest(source),input_size=[rgb.shape[1],rgb.shape[0]],method='anchored faint-line and bounded curve-bridge experiment',device='CPU OpenCV/SciPy; no neural training',new_training_steps=0,semantic_tree_rock_classification=False,manual_roi_used=False,intact_source_structure_candidates=count,weak_anchor_candidates=len(accepted_weak),tiny_void_hypotheses=len(void_events))
    stats=base.save(output,rgb,ink,rawsk,dense,details,structures,pending,events,meta)
    doc=read(output/'paths.json');doc['config']=CFG;doc['tiny_void_events']=void_events;jsave(output/'paths.json',doc)
    Image.fromarray(np.where(topo,0,255).astype(np.uint8)).save(output/'topology_ink.png')
    Image.fromarray(np.where(weak,0,255).astype(np.uint8)).save(output/'weak_ridge_candidates.png')
    selected=raster(rawsk.shape,details+structures)
    stats.update(weak_anchor_candidates=len(accepted_weak),tiny_void_hypotheses=len(void_events),new_vs_source_skeleton_pixels=int((selected&~rawsk).sum()))
    jsave(output/'statistics.json',stats)
    print('V5_RESULT',source.name,json.dumps(stats),flush=True)
    return stats

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('image');a.add_argument('--output',required=True);args=a.parse_args();run(args.image,args.output)
